"""
Model zoo for multivariate time series forecasting.

Implements:
  - DLinear: simple linear baseline with trend-seasonal decomposition
  - NLinear: linear with last-step normalization
  - PatchTST: transformer on patched channels (channel-independent)
  - iTransformer: transformer over variables (inverted)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------- DLinear ---------------------
class MovingAvg(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x: (B, L, C)
        pad = (self.kernel_size - 1) // 2
        front = x[:, :1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size - 1 - pad, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        out = self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)
        return out


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinear(nn.Module):
    """Channel-independent DLinear with optional RevIN."""

    def __init__(self, seq_len, pred_len, n_channels, kernel_size=25,
                 individual=False, use_revin=True):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.individual = individual
        self.use_revin = use_revin
        self.decomp = SeriesDecomp(kernel_size)

        if individual:
            self.linear_s = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(n_channels)])
            self.linear_t = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(n_channels)])
        else:
            self.linear_s = nn.Linear(seq_len, pred_len)
            self.linear_t = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: (B, L, C)
        if self.use_revin:
            mean = x.mean(dim=1, keepdim=True).detach()
            std = x.std(dim=1, keepdim=True).detach() + 1e-5
            x = (x - mean) / std

        seasonal, trend = self.decomp(x)
        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)
        if self.individual:
            s_out = torch.stack([self.linear_s[i](seasonal[:, i, :]) for i in range(self.n_channels)], dim=1)
            t_out = torch.stack([self.linear_t[i](trend[:, i, :]) for i in range(self.n_channels)], dim=1)
        else:
            s_out = self.linear_s(seasonal)
            t_out = self.linear_t(trend)
        out = (s_out + t_out).permute(0, 2, 1)  # (B, pred_len, C)

        if self.use_revin:
            out = out * std + mean
        return out


# --------------------- NLinear ---------------------
class NLinear(nn.Module):
    def __init__(self, seq_len, pred_len, n_channels):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        last = x[:, -1:, :]
        x_n = x - last
        x_n = x_n.permute(0, 2, 1)
        out = self.linear(x_n).permute(0, 2, 1)
        return out + last


# --------------------- iTransformer ---------------------
class iTransformer(nn.Module):
    """Inverted Transformer: tokens are variables, attention is across channels.
    Reference: Liu et al., ICLR 2024.
    """

    def __init__(self, seq_len, pred_len, n_channels, d_model=128, n_heads=8,
                 e_layers=2, d_ff=256, dropout=0.1, use_norm=True):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_norm = use_norm

        self.embed = nn.Linear(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.proj = nn.Linear(d_model, pred_len)

    def forward(self, x):
        # x: (B, L, C)
        if self.use_norm:
            mean = x.mean(dim=1, keepdim=True).detach()
            std = x.std(dim=1, keepdim=True).detach() + 1e-5
            x = (x - mean) / std

        # invert: treat each channel as a token
        x_t = x.permute(0, 2, 1)            # (B, C, L)
        tok = self.embed(x_t)               # (B, C, d_model)
        enc = self.encoder(tok)             # (B, C, d_model)
        out = self.proj(enc)                # (B, C, pred_len)
        out = out.permute(0, 2, 1)          # (B, pred_len, C)

        if self.use_norm:
            out = out * std + mean
        return out


# --------------------- PatchTST ---------------------
class PatchTST(nn.Module):
    """Channel-independent PatchTST (Nie et al., ICLR 2023, simplified)."""

    def __init__(self, seq_len, pred_len, n_channels, patch_len=16, stride=8,
                 d_model=64, n_heads=4, e_layers=2, d_ff=128, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.n_channels = n_channels

        # padding so number of patches is consistent
        self.pad = stride
        self.n_patches = (seq_len + self.pad - patch_len) // stride + 1

        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.head = nn.Linear(self.n_patches * d_model, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape
        # RevIN-style instance norm
        mean = x.mean(dim=1, keepdim=True).detach()
        std = x.std(dim=1, keepdim=True).detach() + 1e-5
        x = (x - mean) / std

        # to channel-independent: (B*C, L)
        x = x.permute(0, 2, 1).reshape(B * C, L)
        # pad at the end
        x = F.pad(x, (0, self.pad), mode="replicate")
        # patch via unfold: (B*C, n_patches, patch_len)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = self.embed(x) + self.pos       # (B*C, n_patches, d_model)
        x = self.dropout(x)
        x = self.encoder(x)                # (B*C, n_patches, d_model)
        x = x.reshape(B * C, -1)
        x = self.head(x)                   # (B*C, pred_len)
        x = x.reshape(B, C, self.pred_len).permute(0, 2, 1)  # (B, pred_len, C)

        x = x * std + mean
        return x


class DLinearMLP(nn.Module):
    """DLinear with extra hidden layer for more capacity (channel-shared)."""

    def __init__(self, seq_len, pred_len, n_channels, kernel_size=25,
                 hidden=256, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.decomp = SeriesDecomp(kernel_size)
        self.head_s = nn.Sequential(
            nn.Linear(seq_len, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, pred_len),
        )
        self.head_t = nn.Sequential(
            nn.Linear(seq_len, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, pred_len),
        )

    def forward(self, x):
        # x: (B, L, C)
        mean = x.mean(dim=1, keepdim=True).detach()
        std = x.std(dim=1, keepdim=True).detach() + 1e-5
        x = (x - mean) / std

        seasonal, trend = self.decomp(x)
        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)
        out = (self.head_s(seasonal) + self.head_t(trend)).permute(0, 2, 1)
        return out * std + mean


def build_model(name: str, seq_len: int, pred_len: int, n_channels: int):
    name = name.lower()
    if name == "dlinear":
        return DLinear(seq_len, pred_len, n_channels)
    if name == "dlinear_indiv":
        return DLinear(seq_len, pred_len, n_channels, individual=True)
    if name == "dlinear_k51":
        return DLinear(seq_len, pred_len, n_channels, kernel_size=51)
    if name == "dlinear_k15":
        return DLinear(seq_len, pred_len, n_channels, kernel_size=15)
    if name == "dlinear_mlp":
        return DLinearMLP(seq_len, pred_len, n_channels)
    if name == "nlinear":
        return NLinear(seq_len, pred_len, n_channels)
    if name == "itransformer":
        return iTransformer(seq_len, pred_len, n_channels)
    if name == "itransformer_xl":
        return iTransformer(seq_len, pred_len, n_channels,
                            d_model=192, n_heads=8, e_layers=3, d_ff=384)
    if name == "patchtst":
        return PatchTST(seq_len, pred_len, n_channels)
    raise ValueError(f"Unknown model: {name}")


if __name__ == "__main__":
    x = torch.randn(4, 96, 100)
    for name in ["dlinear", "nlinear", "itransformer", "patchtst"]:
        for pl in [96, 720]:
            m = build_model(name, 96, pl, 100)
            y = m(x)
            n_params = sum(p.numel() for p in m.parameters())
            print(f"{name} pred_len={pl}: out={y.shape}, params={n_params:,}")
