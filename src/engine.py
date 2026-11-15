"""Evaluation loop shared by training (per-epoch validation) and the final evaluation script."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import (
    ConfusionMatrix,
    iou_from_confusion,
    mean_iou,
    per_sample_miou,
    pixel_accuracy,
)


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    criterion: Optional[nn.Module] = None,
    collect_per_image: bool = False,
    progress_desc: Optional[str] = None,
) -> Dict[str, Any]:
    """Run ``model`` over ``loader`` (eval mode, no gradients).

    Returns loss (pixel-weighted mean), confusion matrix, per-class IoU, mIoU, pixel accuracy and,
    optionally, per-image statistics in loader order (loader must not shuffle in that case).
    """
    model.eval()
    confusion = ConfusionMatrix(num_classes, ignore_index)
    loss_sum, valid_total = 0.0, 0
    per_image: list = []
    iterator = tqdm(loader, desc=progress_desc, leave=False) if progress_desc else loader
    for images, masks in iterator:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        logits = model(images)
        if criterion is not None:
            valid = int((masks != ignore_index).sum())
            if valid > 0:
                loss_sum += float(criterion(logits, masks)) * valid
                valid_total += valid
        predictions = logits.argmax(dim=1)
        confusion.update(predictions, masks)
        if collect_per_image:
            per_image.extend(per_sample_miou(predictions, masks, num_classes, ignore_index))
    iou = iou_from_confusion(confusion.matrix)
    return {
        "loss": loss_sum / valid_total if (criterion is not None and valid_total > 0) else float("nan"),
        "confusion_matrix": confusion.matrix,
        "per_class_iou": iou,
        "miou": mean_iou(iou),
        "pixel_accuracy": pixel_accuracy(confusion.matrix),
        "per_image": per_image,
    }


def load_model_from_checkpoint(
    model_name: str, checkpoint_path, config: Mapping[str, Any], device: torch.device
) -> tuple[nn.Module, Dict[str, Any]]:
    """Load a trained model from a checkpoint written by ``src.train``.

    Raises FileNotFoundError (with training instructions) if the checkpoint does not exist and
    ValueError if it was produced for a different model/architecture/dataset setting.
    """
    from pathlib import Path

    from .models import build_model

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            f"Train the model first, e.g.:  python -m src.train --model {model_name}"
        )
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    required = {"model_state_dict", "epoch", "optimizer_state_dict", "config", "seed", "val_miou", "model_name"}
    missing = required - set(ckpt)
    if missing:
        raise ValueError(f"Checkpoint {path} lacks required keys: {sorted(missing)}")
    if ckpt["model_name"] != model_name:
        raise ValueError(f"Checkpoint {path} holds model '{ckpt['model_name']}', expected '{model_name}'.")
    for section, key in (("model", "base_channels"), ("dataset", "num_classes"), ("dataset", "image_size")):
        if ckpt["config"][section][key] != config[section][key]:
            raise ValueError(
                f"Checkpoint {path} was trained with {section}.{key}={ckpt['config'][section][key]} but the "
                f"current config has {config[section][key]}."
            )
    model = build_model(model_name, config)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), ckpt
