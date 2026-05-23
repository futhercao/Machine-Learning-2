"""
Final v2 inference: loads ensemble_config.json and produces pred_{96,192,336,720}.npy.

Usage:
    python predict_v2.py --test_path /path/to/test.npy
"""
import argparse
import json
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch

from models import build_model
from models_v2 import build_v2


def build_any(name, seq_len, pred_len, n_channels):
    try:
        return build_v2(name, seq_len, pred_len, n_channels)
    except ValueError:
        return build_model(name, seq_len, pred_len, n_channels)


def disp_to_builder_path(disp, pred_len, ckpt_dir):
    if '_s' in disp and disp.rsplit('_s', 1)[1].isdigit():
        builder, s = disp.rsplit('_s', 1)
        return builder, os.path.join(ckpt_dir, f"{builder}_pl{pred_len}_s{s}.pt")
    return disp, os.path.join(ckpt_dir, f"{disp}_pl{pred_len}.pt")


def predict_one(model, x_np, device, batch_size=128):
    n = len(x_np); outs = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = torch.from_numpy(x_np[i:i + batch_size]).float().to(device)
            outs.append(model(batch).cpu().numpy())
    return np.concatenate(outs, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--test_path', required=True)
    p.add_argument('--out_dir', default='results_v2')
    p.add_argument('--ckpt_dir', default='checkpoints_v2')
    p.add_argument('--config', default='ensemble_config.json')
    p.add_argument('--device', default='cuda')
    p.add_argument('--seq_len', type=int, default=96)
    p.add_argument('--n_channels', type=int, default=100)
    p.add_argument('--no_clip', action='store_true')
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.config) as f:
        cfg = json.load(f)

    print(f"[load] {args.test_path}")
    test_x = np.load(args.test_path).astype(np.float32)
    print(f"  shape={test_x.shape}, range=[{test_x.min():.4f}, {test_x.max():.4f}]")
    assert test_x.ndim == 3 and test_x.shape[1] == args.seq_len

    for pl in [96, 192, 336, 720]:
        models = cfg[str(pl)]['models']
        accum = None
        used = []
        for disp in models:
            builder, path = disp_to_builder_path(disp, pl, args.ckpt_dir)
            if not os.path.exists(path):
                print(f"  [skip] {path}"); continue
            m = build_any(builder, args.seq_len, pl, args.n_channels).to(device)
            ck = torch.load(path, map_location=device, weights_only=False)
            m.load_state_dict(ck['state_dict'])
            m.eval()
            pred = predict_one(m, test_x, device)
            accum = pred if accum is None else accum + pred
            used.append(disp)
            del m; torch.cuda.empty_cache()
        if accum is None:
            print(f"[pl={pl}] no models available, skipping"); continue
        accum = accum / len(used)
        if not args.no_clip:
            accum = np.clip(accum, 0.0, 1.0)
        out_path = os.path.join(args.out_dir, f"pred_{pl}.npy")
        np.save(out_path, accum.astype(np.float32))
        print(f"[pl={pl}] ensemble of {len(used)}: {used[:4]}{'...' if len(used)>4 else ''}")
        print(f"  -> {out_path}  shape={accum.shape}  range=[{accum.min():.4f}, {accum.max():.4f}]  val_mse={cfg[str(pl)]['val_mse']:.6f}")


if __name__ == '__main__':
    main()
