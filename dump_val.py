"""
Materialize the validation split so you can audit / verify it manually.

The val split is the LAST 10% of the training time series (no leakage).
Saves the val window inputs and ground-truth predictions for each horizon
to results_v2/val_truth/, so you can compare them against ensemble preds.

Usage:
    python dump_val.py --train_csv ../TimeSeriesDataset/train/train.csv
        -> writes results_v2/val_truth/x_val.npy        (N_val, 96, 100) inputs
                  results_v2/val_truth/y_val_<L>.npy    (N_val, L, 100) ground truth for each L
                  results_v2/val_truth/info.json
"""
import argparse
import json
import os
import numpy as np

from data_loader import load_train_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train_csv', default='../TimeSeriesDataset/train/train.csv')
    p.add_argument('--seq_len', type=int, default=96)
    p.add_argument('--val_ratio', type=float, default=0.1)
    p.add_argument('--out_dir', default='results_v2/val_truth')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = load_train_csv(args.train_csv)
    n = len(data)
    n_val = int(n * args.val_ratio)
    val_start = n - n_val
    print(f"data: {data.shape}, val_start={val_start} (last {n_val} rows held out)")

    info = {'val_start': int(val_start), 'val_rows': int(n_val),
            'seq_len': args.seq_len, 'per_horizon': {}}

    for L in [96, 192, 336, 720]:
        # build val windows (no leakage: input window is fully in val region)
        val_data = data[val_start - args.seq_len:]   # (n_val + seq_len, 100)
        n_samples = len(val_data) - args.seq_len - L + 1
        if n_samples <= 0:
            print(f"pred_len={L}: not enough val data, skipping")
            continue
        xs = np.zeros((n_samples, args.seq_len, data.shape[1]), dtype=np.float32)
        ys = np.zeros((n_samples, L, data.shape[1]), dtype=np.float32)
        for i in range(n_samples):
            xs[i] = val_data[i:i + args.seq_len]
            ys[i] = val_data[i + args.seq_len:i + args.seq_len + L]
        np.save(os.path.join(args.out_dir, f'x_val_pl{L}.npy'), xs)
        np.save(os.path.join(args.out_dir, f'y_val_pl{L}.npy'), ys)
        info['per_horizon'][L] = {'n_samples': int(n_samples),
                                  'x_shape': list(xs.shape),
                                  'y_shape': list(ys.shape)}
        print(f"pred_len={L}: {n_samples} val windows -> x_val_pl{L}.npy, y_val_pl{L}.npy")

    with open(os.path.join(args.out_dir, 'info.json'), 'w') as f:
        json.dump(info, f, indent=2)
    print(f"\n[done] artifacts in {args.out_dir}/")


if __name__ == '__main__':
    main()
