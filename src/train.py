"""Reproducible training pipeline.

Usage (from the project root):
    python -m src.train --model baseline
    python -m src.train --model dilated

Both models use the same seed, data, optimizer, loss, epochs and evaluation procedure. All settings
come from configs/experiment.yaml. Fixed stopping rule: exactly ``training.epochs`` completed epochs.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import VOCSegmentationDataset
from .engine import evaluate_model
from .environment import build_experiment_record
from .metrics import count_trainable_parameters
from .models import MODEL_NAMES, build_model
from .utils import (
    ConfigError, ensure_dir, load_config, make_generator, resolve_path, save_json, seed_worker,
    select_device, set_seed,
)

LOG_FIELDS = [
    "epoch", "train_loss", "val_loss", "val_miou", "learning_rate",
    "epoch_time_sec", "cumulative_time_sec", "is_best",
]


def train_one_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module,
    device: torch.device, ignore_index: int, desc: Optional[str] = None,
) -> tuple[float, int]:
    """One pass over the training loader. Returns (pixel-weighted mean loss, optimisation steps)."""
    model.train()
    loss_sum, valid_total, steps = 0.0, 0, 0
    iterator = tqdm(loader, desc=desc, leave=False) if desc else loader
    for images, masks in iterator:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), masks)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss ({float(loss)}) at step {steps + 1}.")
        loss.backward()
        optimizer.step()
        steps += 1
        valid = int((masks != ignore_index).sum())
        loss_sum += float(loss.detach()) * valid
        valid_total += valid
    return loss_sum / max(valid_total, 1), steps


def _checkpoint_payload(model, optimizer, epoch, val_miou, val_loss, config, model_name) -> Dict[str, Any]:
    return {
        "model_name": model_name,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "seed": config["experiment"]["seed"],
        "val_miou": None if math.isnan(val_miou) else float(val_miou),
        "val_loss": float(val_loss),
    }


def run_training(
    config: Mapping[str, Any], model_name: str, overwrite: bool = False,
    device: Optional[torch.device] = None, show_progress: bool = True,
) -> Dict[str, Any]:
    """Train one model for the fixed budget and return the training summary dict."""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model '{model_name}'. Valid: {MODEL_NAMES}")
    config = dict(config)
    tr, ds, ev = config["training"], config["dataset"], config["evaluation"]
    seed = int(config["experiment"]["seed"])
    device = device or select_device()

    ckpt_dir = ensure_dir(resolve_path(config["paths"]["checkpoints_dir"]))
    log_dir = ensure_dir(resolve_path(config["paths"]["training_logs_dir"]))
    best_path, last_path = ckpt_dir / f"{model_name}_best.pt", ckpt_dir / f"{model_name}_last.pt"
    log_path = log_dir / f"{model_name}_training_log.csv"
    summary_path = log_dir / f"{model_name}_training_summary.json"
    existing = [p for p in (best_path, last_path, log_path, summary_path) if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Previous outputs exist for this model: "
            + ", ".join(str(p.name) for p in existing)
            + ". Re-run with --overwrite to replace them (so runs are never silently mixed)."
        )

    print(f"[train] model={model_name} | device={device} | seed={seed}")
    set_seed(seed, deterministic=tr["deterministic"])

    train_set = VOCSegmentationDataset.from_config(config, "train", train=True)
    val_set = VOCSegmentationDataset.from_config(config, "val", train=False)
    overlap = set(train_set.image_ids) & set(val_set.image_ids)
    if overlap:
        raise RuntimeError(f"Train/validation overlap detected ({len(overlap)} IDs, e.g. {sorted(overlap)[:3]}).")

    # Model is created right after seeding => baseline and dilated start from identical weights.
    model = build_model(model_name, config).to(device)
    n_params = count_trainable_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr["learning_rate"], weight_decay=tr["weight_decay"])
    criterion = nn.CrossEntropyLoss(ignore_index=ds["ignore_index"])

    # Re-seed the data RNGs so augmentation/shuffling streams are identical for both models.
    random.seed(seed)
    np.random.seed(seed)
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_set, batch_size=tr["batch_size"], shuffle=True, num_workers=tr["num_workers"],
        worker_init_fn=seed_worker, generator=make_generator(seed), pin_memory=pin, drop_last=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=ev["validation_batch_size"], shuffle=False, num_workers=tr["num_workers"],
        worker_init_fn=seed_worker, generator=make_generator(seed), pin_memory=pin,
    )

    record = build_experiment_record(config, model_name, model, device)
    (log_dir / f"{model_name}_architecture.txt").write_text(
        f"{model}\n\nTrainable parameters: {n_params:,}\n", encoding="utf-8"
    )
    epochs = int(tr["epochs"])
    batches_per_epoch = len(train_loader)
    print(f"[train] train images={len(train_set)} | val images={len(val_set)} | "
          f"batches/epoch={batches_per_epoch} | epochs={epochs} | params={n_params:,}")

    summary: Dict[str, Any] = {
        "status": "running",
        **record,
        "trainable_parameters": n_params,
        "budget": {
            "epochs_planned": epochs,
            "training_images": len(train_set),
            "validation_images": len(val_set),
            "batches_per_epoch": batches_per_epoch,
            "total_optimization_steps_planned": batches_per_epoch * epochs,
            "batch_size": tr["batch_size"],
            "input_resolution": record["input_resolution"],
            "total_training_samples_planned": len(train_set) * epochs,
            "validation_frequency": "every epoch",
            "hardware": record["environment"]["hardware"],
        },
        "interruptions_or_deviations": "none recorded",
    }

    best_miou, best_epoch, total_steps, epochs_done = -math.inf, 0, 0, 0
    last_row: Dict[str, Any] = {}
    start = time.perf_counter()
    status, error = "completed", None
    try:
        with open(log_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
            writer.writeheader()
            for epoch in range(1, epochs + 1):
                epoch_start = time.perf_counter()
                train_loss, steps = train_one_epoch(
                    model, train_loader, optimizer, criterion, device, ds["ignore_index"],
                    desc=f"{model_name} epoch {epoch}/{epochs} [train]" if show_progress else None,
                )
                total_steps += steps
                val = evaluate_model(
                    model, val_loader, device, ds["num_classes"], ds["ignore_index"], criterion,
                    progress_desc=f"{model_name} epoch {epoch}/{epochs} [val]" if show_progress else None,
                )
                now = time.perf_counter()
                val_miou = val["miou"]
                improved = (not math.isnan(val_miou)) and val_miou > best_miou
                if improved:
                    best_miou, best_epoch = val_miou, epoch
                    torch.save(_checkpoint_payload(model, optimizer, epoch, val_miou, val["loss"], config, model_name), best_path)
                torch.save(_checkpoint_payload(model, optimizer, epoch, val_miou, val["loss"], config, model_name), last_path)
                last_row = {
                    "epoch": epoch, "train_loss": f"{train_loss:.6f}", "val_loss": f"{val['loss']:.6f}",
                    "val_miou": f"{val_miou:.6f}", "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch_time_sec": f"{now - epoch_start:.2f}", "cumulative_time_sec": f"{now - start:.2f}",
                    "is_best": int(improved),
                }
                writer.writerow(last_row)
                handle.flush()
                epochs_done = epoch
                print(f"[train] {model_name} epoch {epoch:>3}/{epochs} | train_loss={train_loss:.4f} | "
                      f"val_loss={val['loss']:.4f} | val_mIoU={val_miou:.4f} | best={best_miou:.4f}@{best_epoch}")
    except KeyboardInterrupt:
        status, error = "interrupted", "KeyboardInterrupt"
        raise
    except BaseException as exc:  # record the failure, then propagate
        status, error = "failed", repr(exc)
        raise
    finally:
        elapsed = time.perf_counter() - start
        summary["status"] = status
        summary["budget"].update({
            "epochs_completed": epochs_done,
            "total_optimization_steps_completed": total_steps,
            "total_training_samples_processed": len(train_set) * epochs_done,
            "training_time_sec": round(elapsed, 2),
        })
        summary["best_epoch"] = best_epoch or None
        summary["best_val_miou"] = None if best_epoch == 0 else best_miou
        summary["final_epoch_val_miou"] = last_row.get("val_miou")
        if status != "completed":
            summary["interruptions_or_deviations"] = f"Run {status} after {epochs_done}/{epochs} epochs: {error}"
        save_json(summary, summary_path)
    print(f"[train] done: status={status} | time={elapsed / 60:.1f} min | best val mIoU={best_miou:.4f} "
          f"(epoch {best_epoch}) | checkpoints in {ckpt_dir}")
    return summary


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Train one segmentation model with the fixed experiment budget.")
    parser.add_argument("--model", required=True, choices=MODEL_NAMES, help="Which model to train.")
    parser.add_argument("--config", default=None, help="Experiment YAML (default: configs/experiment.yaml).")
    parser.add_argument("--data-root", default=None, help="Override dataset.root (path to VOCdevkit/VOC2012).")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing checkpoints/logs of this model.")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, data_root=args.data_root)
        run_training(config, args.model, overwrite=args.overwrite)
    except (ConfigError, FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"[train] ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
