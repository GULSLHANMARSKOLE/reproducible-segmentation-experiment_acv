"""MODEL B - dilated: identical to the baseline except for ONE mechanism.

The ONLY difference: the SECOND 3x3 convolution of the bottleneck uses dilation = 2
(padding = 2, so the feature-map size is unchanged) instead of dilation = 1.
No weights are added: parameter count and tensor shapes equal the baseline's.
"""
from __future__ import annotations

from .unet import UNet

# (dilation of 1st bottleneck conv, dilation of 2nd bottleneck conv)  <-- the single difference
BOTTLENECK_DILATIONS = (1, 2)


class DilatedUNet(UNet):
    def __init__(self, in_channels: int = 3, num_classes: int = 21, base_channels: int = 32) -> None:
        super().__init__(in_channels, num_classes, base_channels, BOTTLENECK_DILATIONS)
