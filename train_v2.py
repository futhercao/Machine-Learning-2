"""
Train v2 (and v1) models with seed control + AdamW + warmup-cosine + early stopping.
Supports CUDA. Saves ckpt as <name>_pl<L>_s<seed>.pt for multi-seed ensembling.

Usage:
    python train_v2.py --model timemixer --pred_len 96 --epochs 30 --seed 2026 --device cuda
"""
import argparse
import os
import time
import math
import numpy as np
import torch
import torch.nn as nn

from data_loader import load_train_csv, make_loaders
from models import build_model
from models_v2 import build_v2


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_any(name, seq_len, pred_len, n_channels):
    try:
        return build_v2(name, seq_len, pred_len, n_channels)
    except ValueError:
        return build_model(name, seq_len, pred_len, n_channels)


def evaluate(model, loader, device):
    model.eval()
    tot_sq, tot_n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            p = model(x)
            tot_sq += ((p - y) ** 2).sum().item()
            tot_n += y.numel()
    return tot_sq / tot_n


def warmup_cosine_lr(step, total_steps, warmup_steps, base_lr, min_lr_ratio=0.01):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', default='../TimeSeriesDataset/train/train.csv')
    p.add_argument('--model', required=True)
    p.add_argument('--seq_len', type=int, default=96)
    p.add_argument('--pred_len', type=int, required=True)
    p.add_argument('--n_channels', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--warmup_pct', type=float, default=0.05)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--val_ratio', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--ckpt_dir', default='checkpoints_v2')
    p.add_argument('--device', default='cuda')
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--loss', default='mse', choices=['mse', 'huber', 'smooth_l1'])
    p.add_argument('--clip_grad', type=float, default=1.0)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    data = load_train_csv(args.data_path)
    train_loader, val_loader, train_ds, val_ds = make_loaders(
        data, args.seq_len, args.pred_len,
        batch_size=args.batch_size, val_ratio=args.val_ratio,
        num_workers=args.num_workers,
    )
    print(f"[data] train={len(train_ds)} val={len(val_ds)}")

    model = build_any(args.model, args.seq_len, args.pred_len, args.n_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.model} pl={args.pred_len} seed={args.seed} params={n_params:,}")

    if args.loss == 'mse':
        criterion = nn.MSELoss()
    elif args.loss == 'huber':
        criterion = nn.HuberLoss(delta=0.1)
    else:
        criterion = nn.SmoothL1Loss(beta=0.1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay, betas=(0.9, 0.99))

    steps_per_ep = max(1, len(train_loader))
    total_steps = args.epochs * steps_per_ep
    warmup_steps = int(args.warmup_pct * total_steps)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_path = os.path.join(args.ckpt_dir, f"{args.model}_pl{args.pred_len}_s{args.seed}.pt")
    best_val = float('inf'); bad = 0
    step = 0
    for epoch in range(args.epochs):
        t0 = time.time(); model.train()
        train_loss = 0.0; nb = 0
        for x, y in train_loader:
            lr = warmup_cosine_lr(step, total_steps, warmup_steps, args.lr)
            for g in optimizer.param_groups: g['lr'] = lr
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            train_loss += loss.item(); nb += 1; step += 1
        train_loss /= max(nb, 1)
        val_mse = evaluate(model, val_loader, device)
        dt = time.time() - t0
        mk = ''
        if val_mse < best_val:
            best_val = val_mse; bad = 0
            torch.save({'state_dict': model.state_dict(), 'args': vars(args),
                        'val_mse': val_mse, 'epoch': epoch + 1}, best_path)
            mk = '  *'
        else:
            bad += 1
        print(f"ep {epoch+1:3d}/{args.epochs} | lr {lr:.5f} | train {train_loss:.5f} | val {val_mse:.5f} | {dt:.1f}s{mk}")
        if bad >= args.patience:
            print(f"[stop] no improve for {args.patience}"); break
    print(f"[done] best val mse = {best_val:.6f}  -> {best_path}")


if __name__ == '__main__':
    main()
