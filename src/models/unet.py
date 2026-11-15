"""Generic compact U-Net. The two experimental models only differ in ``bottleneck_dilations``."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .blocks import DecoderBlock, DoubleConv, EncoderBlock


class UNet(nn.Module):
    """Three-level U-Net (encoder x3, bottleneck, decoder x3) returning raw logits.

    Channels for ``base_channels=32``: encoder 32/64/128, bottleneck 256, decoder 128/64/32.
    Input spatial size must be divisible by 8.
    """

    DOWNSAMPLING_FACTOR = 8

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 21,
        base_channels: int = 32,
        bottleneck_dilations: Sequence[int] = (1, 1),
    ) -> None:
        super().__init__()
        c1, c2, c3, cb = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.bottleneck_dilations = tuple(bottleneck_dilations)

        self.enc1 = EncoderBlock(in_channels, c1)
        self.enc2 = EncoderBlock(c1, c2)
        self.enc3 = EncoderBlock(c2, c3)
        self.bottleneck = DoubleConv(c3, cb, dilations=self.bottleneck_dilations)
        self.dec1 = DecoderBlock(cb, c3, c3)
        self.dec2 = DecoderBlock(c3, c2, c2)
        self.dec3 = DecoderBlock(c2, c1, c1)
        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        if height % self.DOWNSAMPLING_FACTOR or width % self.DOWNSAMPLING_FACTOR:
            raise ValueError(
                f"Input size {height}x{width} must be divisible by {self.DOWNSAMPLING_FACTOR}."
            )
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        x = self.bottleneck(x)
        x = self.dec1(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec3(x, skip1)
        return self.head(x)  # raw logits, shape [B, num_classes, H, W]
