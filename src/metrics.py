"""Segmentation metrics implemented from scratch (confusion matrix, IoU, mIoU), plus parameter
count and inference-latency measurement.

Conventions
-----------
* ``confusion[gt, pred]`` counts pixels; pixels whose ground truth equals ``ignore_index`` are excluded.
* IoU_c = TP / (TP + FP + FN) = diag / (row_sum + col_sum - diag).
* A class whose union is zero (absent from both ground truth and prediction) is **NaN** (undefined)
  and is excluded from the mean. Note that a class absent from the ground truth but predicted
  somewhere has union > 0 and therefore IoU = 0.
"""
from __future__ import annotations

import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn


def confusion_matrix_from_batch(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int
) -> torch.Tensor:
    """Confusion matrix ``[num_classes, num_classes]`` (int64) for one batch; rows = ground truth."""
    pred = pred.reshape(-1).to(torch.int64)
    target = target.reshape(-1).to(torch.int64)
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same number of pixels: {pred.shape} vs {target.shape}")
    valid = target != ignore_index
    target, pred = target[valid], pred[valid]
    if target.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.int64, device=pred.device)
    for name, values in (("target", target), ("pred", pred)):
        if int(values.min()) < 0 or int(values.max()) >= num_classes:
            raise ValueError(f"{name} contains class ids outside [0, {num_classes - 1}] (excluding ignore_index).")
    index = target * num_classes + pred
    return torch.bincount(index, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


class ConfusionMatrix:
    """Accumulates a confusion matrix over batches."""

    def __init__(self, num_classes: int, ignore_index: int) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        batch = confusion_matrix_from_batch(pred, target, self.num_classes, self.ignore_index)
        self.matrix += batch.cpu().numpy()


def iou_from_confusion(confusion: np.ndarray) -> np.ndarray:
    """Per-class IoU (float64); NaN where TP + FP + FN == 0."""
    confusion = np.asarray(confusion, dtype=np.float64)
    tp = np.diag(confusion)
    union = confusion.sum(axis=1) + confusion.sum(axis=0) - tp
    iou = np.full(confusion.shape[0], np.nan, dtype=np.float64)
    defined = union > 0
    iou[defined] = tp[defined] / union[defined]
    return iou


def mean_iou(iou: np.ndarray) -> float:
    """Mean over classes with a defined IoU (NaN classes excluded). NaN if none is defined."""
    iou = np.asarray(iou, dtype=np.float64)
    defined = ~np.isnan(iou)
    return float(iou[defined].mean()) if defined.any() else float("nan")


def pixel_accuracy(confusion: np.ndarray) -> float:
    total = float(np.asarray(confusion).sum())
    return float(np.trace(confusion) / total) if total > 0 else float("nan")


def per_sample_miou(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int
) -> List[Dict[str, float]]:
    """Per-image mIoU over the classes present in that image's ground truth or prediction.

    Returns one dict per image: ``miou`` (NaN if no valid pixels), ``has_foreground`` (GT contains
    any class 1..num_classes-1) and ``valid_pixels``.
    """
    results = []
    for p, t in zip(pred, target):
        conf = confusion_matrix_from_batch(p, t, num_classes, ignore_index).cpu().numpy()
        results.append({
            "miou": mean_iou(iou_from_confusion(conf)),
            "has_foreground": float(conf[1:].sum() > 0),
            "valid_pixels": float(conf.sum()),
        })
    return results


# --------------------------------------------------------------------------------------
# Model size and speed
# --------------------------------------------------------------------------------------
def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_latency(
    model: nn.Module,
    device: torch.device,
    input_shape: Tuple[int, int, int, int],
    warmup_iterations: int,
    timed_iterations: int,
    seed: int = 0,
) -> Dict[str, float]:
    """Forward-pass latency in milliseconds (model in eval mode, ``torch.inference_mode()``).

    Only the forward pass of an already prepared input tensor is timed; image loading and
    preprocessing are excluded. On CUDA the device is synchronised before starting and after
    every timed forward pass.
    """
    model = model.to(device).eval()
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(input_shape, generator=generator).to(device)
    timings: List[float] = []
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            model(x)
        _synchronize(device)
        for _ in range(timed_iterations):
            _synchronize(device)
            start = time.perf_counter()
            model(x)
            _synchronize(device)
            timings.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(timings, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "batch_size": int(input_shape[0]),
        "input_shape": list(input_shape),
        "warmup_iterations": int(warmup_iterations),
        "timed_iterations": int(timed_iterations),
        "device": str(device),
        "torch_num_threads": int(torch.get_num_threads()),
    }
