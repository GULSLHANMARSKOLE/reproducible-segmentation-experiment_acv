"""Model construction from the experiment configuration."""
from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from .baseline import BaselineUNet
from .dilated import DilatedUNet

MODEL_NAMES = ("baseline", "dilated")
_REGISTRY = {"baseline": BaselineUNet, "dilated": DilatedUNet}


def build_model(name: str, config: Mapping[str, Any]) -> nn.Module:
    """Build ``baseline`` or ``dilated`` using ``model.base_channels`` and ``dataset.num_classes``.

    Weights are randomly initialised (never pretrained). Both models consume the RNG identically,
    so for the same seed they start from identical weights.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Valid names: {', '.join(MODEL_NAMES)}")
    if config["model"].get("pretrained", False):
        raise ValueError("Pretrained weights are not allowed in this experiment.")
    return _REGISTRY[name](
        in_channels=3,
        num_classes=int(config["dataset"]["num_classes"]),
        base_channels=int(config["model"]["base_channels"]),
    )
