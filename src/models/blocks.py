"""Reusable building blocks for the U-Net-style encoder-decoder."""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import nn


class ConvBNReLU(nn.Module):
    """3x3 convolution -> BatchNorm -> ReLU.

    Padding equals the dilation rate, so the spatial size is preserved for any dilation of a
    3x3 kernel (effective kernel extent = 3 + 2*(dilation-1)). Dilation adds NO parameters.
    """

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1) -> None:
        super().__init__()
        if dilation < 1:
            raise ValueError(f"dilation must be >= 1, got {dilation}")
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1,
            padding=dilation, dilation=dilation, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DoubleConv(nn.Module):
    """Two consecutive ConvBNReLU layers; ``dilations`` gives the dilation of each of the two."""

    def __init__(self, in_channels: int, out_channels: int, dilations: Sequence[int] = (1, 1)) -> None:
        super().__init__()
        if len(dilations) != 2:
            raise ValueError("DoubleConv expects exactly two dilation rates.")
        self.conv1 = ConvBNReLU(in_channels, out_channels, dilation=dilations[0])
        self.conv2 = ConvBNReLU(out_channels, out_channels, dilation=dilations[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class EncoderBlock(nn.Module):
    """DoubleConv followed by 2x2 max pooling. Returns (skip_features, pooled_features)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        return skip, self.pool(skip)


class DecoderBlock(nn.Module):
    """Transposed-conv upsampling (x2), concatenation with the skip tensor, then DoubleConv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            raise RuntimeError(
                f"Spatial mismatch between upsampled {tuple(x.shape[-2:])} and skip "
                f"{tuple(skip.shape[-2:])} features."
            )
        return self.conv(torch.cat([skip, x], dim=1))
