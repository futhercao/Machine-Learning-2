"""
Evaluate predictions in results/ against ground truth in test_demo/.
"""
import argparse
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", default="results")
    p.add_argument("--gt_dir", default="../TimeSeriesDataset/test_demo")
    args = p.parse_args()

    mses = []
    for L in [96, 192, 336, 720]:
        true = np.load(f"{args.gt_dir}/pred_{L}.npy")
        pred = np.load(f"{args.pred_dir}/pred_{L}.npy")
        mse = float(np.mean((true - pred) ** 2))
        mses.append(mse)
        print(f"pred_len={L:3d}: MSE = {mse:.6f}")
    avg = float(np.mean(mses))
    print(f"\nAverage MSE: {avg:.6f}")
    if avg < 0.005:
        print("MSE < 0.005 -> +10 bonus")
    elif avg < 0.006:
        print("MSE < 0.006 -> +5 bonus")
    elif avg < 0.01:
        print("MSE < 0.01 -> base bonus only")
    else:
        print("MSE >= 0.01 -> no bonus")


if __name__ == "__main__":
    main()
