import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class TSDataset(Dataset):
    """Sliding-window dataset on the single long series."""

    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int):
        self.data = data.astype(np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_samples = len(data) - seq_len - pred_len + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.from_numpy(x), torch.from_numpy(y)


def load_train_csv(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    return df.values  # (20000, 100), already in [0, 1]


def make_loaders(data: np.ndarray, seq_len: int, pred_len: int,
                 batch_size: int = 64, val_ratio: float = 0.1, num_workers: int = 0):
    """Split with no leakage: train labels stay strictly before val region."""
    n = len(data)
    n_val = int(n * val_ratio)
    val_start = n - n_val          # first index of validation labels

    # Train: any window whose label end <= val_start
    # i.e. start + seq_len + pred_len <= val_start, so train data is data[:val_start]
    # but the last train window uses data[val_start - seq_len - pred_len : val_start]
    train_data = data[: val_start]            # labels strictly < val_start
    # Val: first sample input starts at val_start - seq_len, label at val_start
    val_data = data[val_start - seq_len:]     # length = seq_len + n_val

    train_ds = TSDataset(train_data, seq_len, pred_len)
    val_ds = TSDataset(val_data, seq_len, pred_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    return train_loader, val_loader, train_ds, val_ds


if __name__ == "__main__":
    data = load_train_csv("../TimeSeriesDataset/train/train.csv")
    print("Data shape:", data.shape, "range:", data.min(), data.max())
    for L in [96, 192, 336, 720]:
        train_loader, val_loader, train_ds, val_ds = make_loaders(data, 96, L)
        print(f"pred_len={L}: train_samples={len(train_ds)}, val_samples={len(val_ds)}")
        x, y = next(iter(train_loader))
        print(f"  x={x.shape}, y={y.shape}")
