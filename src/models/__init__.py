"""Segmentation models: a compact U-Net baseline and its single-modification dilated variant."""
from .model_factory import MODEL_NAMES, build_model

__all__ = ["MODEL_NAMES", "build_model"]
