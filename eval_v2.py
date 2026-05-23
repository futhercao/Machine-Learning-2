"""
Evaluate ALL trained models on validation set and find optimal per-horizon ensemble.
Saves the chosen ensemble config to ensemble_config.json.

Per-horizon, tries:
  1. Mean of all available v1+v2 models (per-seed averaged for v2 first)
  2. Mean of top-K (K=3,5,7) by individual val MSE
  3. Greedy forward selection by val MSE
Reports best.
"""
import json
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from data_loader import load_train_csv, make_loaders
from models import build_model
from models_v2 import build_v2


CKPT_DIR = 'checkpoints_v2'

V1_NAMES = ['dlinear', 'dlinear_k51', 'dlinear_k15', 'dlinear_mlp', 'nlinear',
            'itransformer', 'itransformer_xl']
V2_NAMES = ['tsmixer', 'timemixer', 'patchtst_strong', 'itransformer_strong']
V2_SEEDS = [2026, 42, 1234]


def build_any(name, seq_len, pred_len, n_channels):
    try:
        return build_v2(name, seq_len, pred_len, n_channels)
    except ValueError:
        return build_model(name, seq_len, pred_len, n_channels)


def list_ckpts_for(pred_len):
    """Returns list of (display_name, ckpt_path, builder_name)."""
    out = []
    for n in V1_NAMES:
        p = f"{CKPT_DIR}/{n}_pl{pred_len}.pt"
        if os.path.exists(p):
            out.append((n, p, n))
    for n in V2_NAMES:
        for s in V2_SEEDS:
            p = f"{CKPT_DIR}/{n}_pl{pred_len}_s{s}.pt"
            if os.path.exists(p):
                out.append((f"{n}_s{s}", p, n))
    return out


def predict_full(model, loader, device):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            p = model(x)
            ys.append(y.cpu().numpy())
            preds.append(p.cpu().numpy())
    return np.concatenate(preds, 0), np.concatenate(ys, 0)


def mse(p, y):
    return float(np.mean((p - y) ** 2))


def greedy_select(preds_dict, y, max_k=None):
    """Greedy forward selection: pick model that most improves MSE when averaged in."""
    names = list(preds_dict.keys())
    chosen = []
    chosen_pred = None
    history = []
    best_overall = float('inf')
    while True:
        best_n, best_p, best_m = None, None, float('inf')
        for n in names:
            if n in chosen:
                continue
            if chosen_pred is None:
                cand = preds_dict[n]
            else:
                cand = (chosen_pred * len(chosen) + preds_dict[n]) / (len(chosen) + 1)
            m = mse(cand, y)
            if m < best_m:
                best_m = m; best_n = n; best_p = cand
        if best_n is None: break
        # only keep if improves or first
        if not chosen or best_m < best_overall:
            chosen.append(best_n)
            chosen_pred = best_p
            best_overall = best_m
            history.append((best_n, best_m))
            if max_k and len(chosen) >= max_k: break
        else:
            break
    return chosen, best_overall, history


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = load_train_csv('../TimeSeriesDataset/train/train.csv')

    config = {}
    summary = []

    for pl in [96, 192, 336, 720]:
        print(f"\n========== pred_len = {pl} ==========")
        _, val_loader, _, val_ds = make_loaders(data, 96, pl, batch_size=128, num_workers=2)
        ckpts = list_ckpts_for(pl)
        if not ckpts:
            print("  no checkpoints found"); continue

        preds = {}
        y_true = None
        for disp, path, builder in ckpts:
            try:
                m = build_any(builder, 96, pl, 100).to(device)
                ck = torch.load(path, map_location=device, weights_only=False)
                m.load_state_dict(ck['state_dict'])
                p, y_true = predict_full(m, val_loader, device)
                preds[disp] = p
                print(f"  {disp:<30s} val MSE = {mse(p, y_true):.6f}")
                del m
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  [skip] {disp}: {e}")

        # 1. Mean of all
        all_pred = np.mean(list(preds.values()), axis=0)
        all_mse = mse(all_pred, y_true)
        print(f"\n  [ensemble: mean-of-all ({len(preds)})] MSE = {all_mse:.6f}")

        # 2. Top-K by individual MSE
        sorted_models = sorted(preds.items(), key=lambda kv: mse(kv[1], y_true))
        topk_results = {}
        for K in [3, 5, 7, 10]:
            if K > len(preds): continue
            chosen = [n for n, _ in sorted_models[:K]]
            ens = np.mean([preds[n] for n in chosen], axis=0)
            m = mse(ens, y_true)
            topk_results[K] = (chosen, m)
            print(f"  [ensemble: top-{K}] MSE = {m:.6f}  {chosen[:3]}...")

        # 3. Greedy forward
        greedy_chosen, greedy_mse, history = greedy_select(preds, y_true)
        print(f"  [ensemble: greedy ({len(greedy_chosen)})] MSE = {greedy_mse:.6f}")
        for n, m in history:
            print(f"      + {n:<30s} -> {m:.6f}")

        # Pick best
        candidates = [
            ('all', list(preds.keys()), all_mse),
            ('greedy', greedy_chosen, greedy_mse),
        ]
        for K, (chosen, m) in topk_results.items():
            candidates.append((f'top{K}', chosen, m))
        best = min(candidates, key=lambda x: x[2])
        print(f"\n  ==> BEST: {best[0]} ({len(best[1])} models)  MSE = {best[2]:.6f}")
        config[str(pl)] = {'strategy': best[0], 'models': best[1], 'val_mse': best[2]}
        summary.append((pl, best[2]))

    print("\n========== SUMMARY ==========")
    avg = 0
    for pl, m in summary:
        print(f"  pl={pl}: val MSE = {m:.6f}")
        avg += m
    if summary:
        print(f"  AVG: {avg/len(summary):.6f}")
    config['avg_val_mse'] = avg / max(len(summary), 1)

    with open('ensemble_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("\n[saved] ensemble_config.json")


if __name__ == '__main__':
    main()
