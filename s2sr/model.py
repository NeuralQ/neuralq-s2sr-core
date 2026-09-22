"""Reconstructed S2SR network architecture and checkpoint loading.

``S2SRNet`` maps ``N x 50 x H x W`` normalized reflectance stacks (five
dates x ten Sentinel-2 bands, date-major, reflectance DN/10000 in 0..1) to
``N x 10 x 10H x 10W`` super-resolved outputs (10 bands, 10×, clamped 0..1)
through:

  * **deformer** — 50→160 conv + 7× grouped deformable conv (DCNv1, groups=5,
    90 offsets/block). Groups=5 ≡ one deformable group per input date, so the
    network learns per-date sub-pixel offsets to compensate residual
    misregistration after UTM WarpedVRT. DCNv1 (no modulation) is used.
  * **encoder** — 160→160 conv + 23× RRDB (3 dense blocks each, growth 80,
    residual scale 0.2, ESRGAN-style). Depth gives large receptive field for
    texture synthesis; 0.2 scaling stabilizes residual-in-residual.
  * **generator** — 10× via nearest-exact 2×2×2×1.25 + conv (no transposed
    conv / PixelShuffle) to avoid checkerboard. Final conv is bias-free +
    LeakyReLU to enforce non-negative reflectance without DC shift.

105,055,800 parameters, ``params_ema`` (EMA) for inference stability. Science:
multi-image SR fuses complementary aliasing across 5 dates; single-image SR
would hallucinate. ``load_model`` does strict, assign=True load and ``eval()``.
"""
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import DeformConv2d


class S2SRResidualDenseBlock(nn.Module):
    """Five-conv dense block with 0.2 residual scaling (ESRGAN RDB)."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), dim=1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))
        return x5 * 0.2 + x


class S2SRRRDB(nn.Module):
    """Residual-in-Residual Dense Block: 3× RDB + 0.2 residual."""

    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = S2SRResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = S2SRResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = S2SRResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: Tensor) -> Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class S2SRResample(nn.Module):
    """3×3 conv (+ optional nearest-exact up / bilinear down) + LeakyReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: float = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: Tensor) -> Tensor:
        if self.scale > 1:
            x = F.interpolate(x, scale_factor=self.scale, mode="nearest-exact")

        x = self.conv(x)

        if self.scale < 1:
            x = F.interpolate(
                x,
                scale_factor=self.scale,
                mode="bilinear",
                antialias=True,
                align_corners=True,
            )

        return self.lrelu(x)


class S2SRDeformationBlock(nn.Module):
    """DCNv1 block: predicts 90 offsets (5 groups × 9 kernel) then deformable conv + ReLU."""

    def __init__(self, num_feat: int, kernel_size: int = 3, padding: int = 1) -> None:
        super().__init__()
        groups = 5
        offset_channels = 2 * groups * kernel_size * kernel_size
        self.offset_conv = nn.Conv2d(
            num_feat, offset_channels, kernel_size, padding=padding
        )
        self.deform_conv = DeformConv2d(
            num_feat,
            num_feat,
            kernel_size,
            padding=padding,
            groups=groups,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        offset = self.offset_conv(x)
        out = self.deform_conv(x, offset)
        return self.relu(out)


class S2SRNet(nn.Module):
    """105M S2SR: deformer (temporal alignment) → encoder (RRDB) → generator (10×)."""

    def __init__(self) -> None:
        super().__init__()
        num_feat = 160
        num_grow_ch = 80

        self.deformer = nn.Sequential(
            S2SRResample(50, num_feat),
            *(S2SRDeformationBlock(num_feat) for _ in range(7)),
        )
        self.encoder = nn.Sequential(
            S2SRResample(num_feat, num_feat),
            *(S2SRRRDB(num_feat, num_grow_ch) for _ in range(23)),
        )
        self.generator = _make_generator(num_feat)

    def forward(self, x: Tensor) -> Tensor:
        out = self.deformer(x)
        out = self.encoder(out)
        return self.generator(out)


def _make_generator(num_feat: int) -> nn.Sequential:
    return nn.Sequential(
        S2SRResample(num_feat, 80, scale=2),
        S2SRResample(80, 40, scale=2),
        S2SRResample(40, 20, scale=2),
        S2SRResample(20, 10, scale=1.25),
        S2SRResample(10, 10, bias=False),
    )


def load_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    *,
    use_ema: bool = True,
) -> S2SRNet:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_key = "params_ema" if use_ema else "params"
    if not isinstance(checkpoint, dict) or state_key not in checkpoint:
        raise ValueError(f"Checkpoint does not contain {state_key!r}")

    model = S2SRNet()
    model.load_state_dict(checkpoint[state_key], strict=True, assign=True)
    model.to(device)
    model.eval()
    return model
