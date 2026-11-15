"""MODEL A - baseline: both bottleneck 3x3 convolutions use dilation = 1."""
from __future__ import annotations

from .unet import UNet

# (dilation of 1st bottleneck conv, dilation of 2nd bottleneck conv)
BOTTLENECK_DILATIONS = (1, 1)


class BaselineUNet(UNet):
    def __init__(self, in_channels: int = 3, num_classes: int = 21, base_channels: int = 32) -> None:
        super().__init__(in_channels, num_classes, base_channels, BOTTLENECK_DILATIONS)
