"""
Additional models for v2 ensemble: TimeMixer, TSMixer, stronger PatchTST.
All keep the (B, L, C) -> (B, pred_len, C) signature and use RevIN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    def __init__(self, n_channels, eps=1e-5, affine=True):
        super().__init__()
        self.n_channels = n_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(n_channels))
            self.beta = nn.Parameter(torch.zeros(n_channels))

    def norm(self, x):
        # x: (B, L, C)
        self.mean = x.mean(dim=1, keepdim=True).detach()
        self.std = x.std(dim=1, keepdim=True).detach() + self.eps
        x = (x - self.mean) / self.std
        if self.affine:
            x = x * self.gamma + self.beta
        return x

    def denorm(self, x):
        if self.affine:
            x = (x - self.beta) / (self.gamma + 1e-5)
        x = x * self.std + self.mean
        return x


class TSMixer(nn.Module):
    """TSMixer: time-mixing + channel-mixing MLP blocks.
    Chen et al., TSMixer (2023). Channel-mixing helps when variables are correlated.
    """
    def __init__(self, seq_len, pred_len, n_channels,
                 n_blocks=4, d_model=128, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.revin = RevIN(n_channels, affine=True)

        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.ModuleDict({
                'time_norm': nn.LayerNorm(n_channels),
                'time_mlp': nn.Sequential(
                    nn.Linear(seq_len, seq_len),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(seq_len, seq_len),
                    nn.Dropout(dropout),
                ),
                'chan_norm': nn.LayerNorm(n_channels),
                'chan_mlp': nn.Sequential(
                    nn.Linear(n_channels, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, n_channels),
                    nn.Dropout(dropout),
                ),
            }))
        self.head = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: (B, L, C)
        x = self.revin.norm(x)
        for blk in self.blocks:
            # time mixing
            h = blk['time_norm'](x).transpose(1, 2)  # (B, C, L)
            h = blk['time_mlp'](h).transpose(1, 2)   # (B, L, C)
            x = x + h
            # channel mixing
            h = blk['chan_norm'](x)
            h = blk['chan_mlp'](h)
            x = x + h
        # head: (B, L, C) -> (B, pred_len, C)
        x = x.transpose(1, 2)            # (B, C, L)
        x = self.head(x).transpose(1, 2) # (B, pred_len, C)
        x = self.revin.denorm(x)
        return x


class TimeMixer(nn.Module):
    """Simplified TimeMixer: multi-scale decomposition with MLP mixing.
    Wang et al., TimeMixer ICLR 2024.
    """
    def __init__(self, seq_len, pred_len, n_channels,
                 d_model=128, n_layers=3, dropout=0.1, scales=(1, 2, 4)):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scales = scales
        self.revin = RevIN(n_channels, affine=True)

        self.embed = nn.Linear(seq_len, d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'mix_t': nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 2, d_model),
                ),
                'mix_c': nn.Sequential(
                    nn.LayerNorm(n_channels),
                    nn.Linear(n_channels, n_channels * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(n_channels * 2, n_channels),
                ),
            }))
        self.proj = nn.Linear(d_model, pred_len)

    def forward(self, x):
        # x: (B, L, C)
        x = self.revin.norm(x)
        # tokens per channel
        h = x.transpose(1, 2)            # (B, C, L)
        h = self.embed(h)                # (B, C, d_model)
        for ly in self.layers:
            h = h + ly['mix_t'](h)                                  # per-channel mlp
            h = h + ly['mix_c'](h.transpose(1, 2)).transpose(1, 2)  # cross-channel mlp
        out = self.proj(h).transpose(1, 2)                          # (B, pred_len, C)
        out = self.revin.denorm(out)
        return out


class PatchTSTStrong(nn.Module):
    """PatchTST with larger model + dropout, channel-independent."""
    def __init__(self, seq_len, pred_len, n_channels,
                 patch_len=16, stride=8, d_model=128, n_heads=8,
                 e_layers=3, d_ff=256, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.pad = stride
        self.n_patches = (seq_len + self.pad - patch_len) // stride + 1
        self.revin = RevIN(n_channels, affine=True)

        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='gelu', norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.flatten = nn.Flatten(start_dim=-2)
        self.head = nn.Linear(self.n_patches * d_model, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, C = x.shape
        x = self.revin.norm(x)
        x = x.permute(0, 2, 1).reshape(B * C, L)
        x = F.pad(x, (0, self.pad), mode='replicate')
        x = x.unfold(-1, self.patch_len, self.stride)
        x = self.embed(x) + self.pos
        x = self.dropout(x)
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.head(x)
        x = x.reshape(B, C, self.pred_len).permute(0, 2, 1)
        x = self.revin.denorm(x)
        return x


class iTransformerStrong(nn.Module):
    """Larger iTransformer with affine RevIN, dropout in head."""
    def __init__(self, seq_len, pred_len, n_channels,
                 d_model=256, n_heads=8, e_layers=3, d_ff=512, dropout=0.15):
        super().__init__()
        self.revin = RevIN(n_channels, affine=True)
        self.embed = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.Dropout(dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='gelu', norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, pred_len),
        )

    def forward(self, x):
        x = self.revin.norm(x)
        x_t = x.permute(0, 2, 1)
        tok = self.embed(x_t)
        enc = self.encoder(tok)
        out = self.proj(enc).permute(0, 2, 1)
        out = self.revin.denorm(out)
        return out


def build_v2(name: str, seq_len: int, pred_len: int, n_channels: int):
    name = name.lower()
    if name == 'tsmixer':
        return TSMixer(seq_len, pred_len, n_channels)
    if name == 'timemixer':
        return TimeMixer(seq_len, pred_len, n_channels)
    if name == 'patchtst_strong':
        return PatchTSTStrong(seq_len, pred_len, n_channels)
    if name == 'itransformer_strong':
        return iTransformerStrong(seq_len, pred_len, n_channels)
    raise ValueError(f"Unknown v2 model: {name}")


if __name__ == '__main__':
    x = torch.randn(4, 96, 100)
    for name in ['tsmixer', 'timemixer', 'patchtst_strong', 'itransformer_strong']:
        for pl in [96, 720]:
            m = build_v2(name, 96, pl, 100)
            y = m(x)
            n_params = sum(p.numel() for p in m.parameters())
            print(f"{name} pl={pl}: out={y.shape}, params={n_params:,}")
