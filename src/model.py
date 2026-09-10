import math

import torch
from torch import nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -scale)
        emb = t.float()[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class TinyUNet(nn.Module):
    """Compact time-conditioned U-Net; default parameter count is below 5M."""
    def __init__(self, base: int = 32, time_dim: int = 128):
        super().__init__()
        self.time = nn.Sequential(
            SinusoidalEmbedding(time_dim), nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(), nn.Linear(time_dim * 4, time_dim),
        )
        self.in_conv = nn.Conv2d(1, base, 3, padding=1)
        self.down1 = ResBlock(base, base, time_dim)
        self.down2 = ResBlock(base, base * 2, time_dim)
        self.pool = nn.MaxPool2d(2)
        self.mid = ResBlock(base * 2, base * 4, time_dim)
        self.up = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.up1 = ResBlock(base * 4, base * 2, time_dim)
        self.up2 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.up3 = ResBlock(base * 2, base, time_dim)
        self.out = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, 1, 3, padding=1))

    def forward(self, x, t):
        te = self.time(t)
        x0 = self.in_conv(x)
        s1 = self.down1(x0, te)
        s2 = self.down2(self.pool(s1), te)
        h = self.mid(self.pool(s2), te)
        h = self.up(h)
        h = self.up1(torch.cat((h, s2), dim=1), te)
        h = self.up2(h)
        h = self.up3(torch.cat((h, s1), dim=1), te)
        return self.out(h)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
