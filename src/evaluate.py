"""One evaluation script for both models (identical conditions).

    python -m src.evaluate

Loads the best checkpoints, evaluates on the official VOC2012 validation split, and writes
metrics.csv / metrics.json / comparison.csv (+ per-class IoU, confusion matrices, per-image mIoU,
figures) to the configured results/figures directories. Fails with a clear message if a
checkpoint is missing - it never fabricates results.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import VOC_CLASS_NAMES, VOCSegmentationDataset
from .engine import evaluate_model, load_model_from_checkpoint
from .environment import collect_environment
from .metrics import count_trainable_parameters, measure_latency
from .models import MODEL_NAMES
from .predict import generate_qualitative_figures, plot_per_class_iou, plot_training_curves
from .utils import (
    ConfigError, ensure_dir, load_config, make_generator, resolve_path, save_json, seed_worker,
    select_device, set_seed, to_jsonable,
)

LATENCY_NOTE = (
    "Latency = model forward pass only, on an already prepared random input tensor "
    "(batch size 1, eval mode, torch.inference_mode()). Image loading and preprocessing are excluded."
)


def _fmt(value: Optional[float]) -> str:
    return "" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.6f}"


def run_evaluation(
    config: Mapping[str, Any], model_names: Sequence[str] = MODEL_NAMES,
    device: Optional[torch.device] = None, make_figures: bool = True, show_progress: bool = True,
) -> Dict[str, Any]:
    ds, ev, tr = config["dataset"], config["evaluation"], config["training"]
    seed = int(config["experiment"]["seed"])
    device = device or select_device()
    ckpt_dir = resolve_path(config["paths"]["checkpoints_dir"])
    results_dir = ensure_dir(resolve_path(config["paths"]["results_dir"]))
    figures_dir = ensure_dir(resolve_path(config["paths"]["figures_dir"]))

    missing = [ckpt_dir / f"{n}_best.pt" for n in model_names if not (ckpt_dir / f"{n}_best.pt").is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing checkpoint(s):\n  " + "\n  ".join(str(m) for m in missing)
            + "\nTrain the models first (results are never fabricated):\n"
            + "\n".join(f"  python -m src.train --model {n}" for n in model_names)
        )

    print(f"[evaluate] device={device} | split=official validation | models={list(model_names)}")
    set_seed(seed, deterministic=tr["deterministic"])
    val_set = VOCSegmentationDataset.from_config(config, "val", train=False)
    loader = DataLoader(
        val_set, batch_size=ev["validation_batch_size"], shuffle=False, num_workers=tr["num_workers"],
        worker_init_fn=seed_worker, generator=make_generator(seed), pin_memory=device.type == "cuda",
    )
    criterion = nn.CrossEntropyLoss(ignore_index=ds["ignore_index"])
    environment = collect_environment(device)
    latency_shape = (ev["batch_size"], 3, ds["image_size"], ds["image_size"])

    models: Dict[str, nn.Module] = {}
    results: Dict[str, Dict[str, Any]] = {}
    per_image_raw: Dict[str, list] = {}
    for name in model_names:
        model, ckpt = load_model_from_checkpoint(name, ckpt_dir / f"{name}_best.pt", config, device)
        out = evaluate_model(
            model, loader, device, ds["num_classes"], ds["ignore_index"], criterion, collect_per_image=True,
            progress_desc=f"evaluating {name}" if show_progress else None,
        )
        latency = measure_latency(model, device, latency_shape, ev["warmup_iterations"], ev["timed_iterations"], seed)
        models[name] = model
        per_image_raw[name] = out["per_image"]
        results[name] = {
            "checkpoint": str(ckpt_dir / f"{name}_best.pt"),
            "checkpoint_epoch": ckpt["epoch"],
            "checkpoint_logged_val_miou": ckpt["val_miou"],
            "validation_images": len(val_set),
            "val_loss": out["loss"],
            "miou": out["miou"],
            "pixel_accuracy": out["pixel_accuracy"],
            "per_class_iou": {VOC_CLASS_NAMES[i]: float(v) for i, v in enumerate(out["per_class_iou"])},
            "undefined_classes": [VOC_CLASS_NAMES[i] for i, v in enumerate(out["per_class_iou"]) if np.isnan(v)],
            "trainable_parameters": count_trainable_parameters(model),
            "latency": latency,
            "bottleneck_dilations": list(getattr(model, "bottleneck_dilations", [])),
        }
        np.savetxt(results_dir / f"confusion_matrix_{name}.csv", out["confusion_matrix"], fmt="%d", delimiter=",")
        print(f"[evaluate] {name}: mIoU={out['miou']:.4f} | params={results[name]['trainable_parameters']:,} | "
              f"latency mean={latency['mean_ms']:.2f} ms")

    # ---------------- machine-readable outputs ----------------
    payload = {
        "evaluation_settings": {
            "split": ds["val_split"], "validation_images": len(val_set), "image_size": ds["image_size"],
            "ignore_index": ds["ignore_index"], "validation_batch_size": ev["validation_batch_size"],
            "latency_batch_size": ev["batch_size"], "warmup_iterations": ev["warmup_iterations"],
            "timed_iterations": ev["timed_iterations"], "latency_note": LATENCY_NOTE,
            "undefined_class_policy": "IoU is NaN (null) when TP+FP+FN == 0 and is excluded from the mean",
            "checkpoint_selection": "best validation mIoU during training (same split as reported here)",
        },
        "environment": environment,
        "models": results,
    }
    save_json(payload, results_dir / "metrics.json")
    save_json(environment, results_dir / "environment.json")

    with open(results_dir / "metrics.csv", "w", newline="", encoding="utf-8") as handle:
        fields = ["model", "checkpoint_epoch", "miou", "pixel_accuracy", "val_loss", "trainable_parameters",
                  "latency_mean_ms", "latency_median_ms", "latency_p95_ms", "device"] + [f"iou_{c}" for c in VOC_CLASS_NAMES]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, r in results.items():
            row = {"model": name, "checkpoint_epoch": r["checkpoint_epoch"], "miou": _fmt(r["miou"]),
                   "pixel_accuracy": _fmt(r["pixel_accuracy"]), "val_loss": _fmt(r["val_loss"]),
                   "trainable_parameters": r["trainable_parameters"],
                   "latency_mean_ms": _fmt(r["latency"]["mean_ms"]), "latency_median_ms": _fmt(r["latency"]["median_ms"]),
                   "latency_p95_ms": _fmt(r["latency"]["p95_ms"]), "device": r["latency"]["device"]}
            row.update({f"iou_{c}": _fmt(v) for c, v in r["per_class_iou"].items()})
            writer.writerow(row)

    names = list(results)
    with open(results_dir / "per_class_iou.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_name"] + names + (["delta_dilated_minus_baseline"] if len(names) == 2 else []))
        for i, cname in enumerate(VOC_CLASS_NAMES):
            vals = [results[n]["per_class_iou"][cname] for n in names]
            row = [i, cname] + [_fmt(v) for v in vals]
            if len(names) == 2:
                row.append(_fmt(vals[1] - vals[0]))
            writer.writerow(row)

    if len(names) == 2 and set(names) == {"baseline", "dilated"}:
        b, d = results["baseline"], results["dilated"]
        metric_rows = [
            ("mIoU", b["miou"], d["miou"]), ("pixel_accuracy", b["pixel_accuracy"], d["pixel_accuracy"]),
            ("val_loss", b["val_loss"], d["val_loss"]),
            ("trainable_parameters", b["trainable_parameters"], d["trainable_parameters"]),
            ("latency_mean_ms", b["latency"]["mean_ms"], d["latency"]["mean_ms"]),
            ("latency_median_ms", b["latency"]["median_ms"], d["latency"]["median_ms"]),
            ("latency_p95_ms", b["latency"]["p95_ms"], d["latency"]["p95_ms"]),
            ("best_checkpoint_epoch", b["checkpoint_epoch"], d["checkpoint_epoch"]),
        ]
        with open(results_dir / "comparison.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric", "baseline", "dilated", "difference_dilated_minus_baseline"])
            for metric, bv, dv in metric_rows:
                is_int = isinstance(bv, int) and isinstance(dv, int)
                writer.writerow([metric, bv if is_int else _fmt(float(bv)), dv if is_int else _fmt(float(dv)),
                                 dv - bv if is_int else _fmt(float(dv) - float(bv))])

    per_image: Dict[str, Dict[str, float]] = {}
    with open(results_dir / "per_image_miou.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "has_foreground"] + [f"{n}_miou" for n in names])
        for k, image_id in enumerate(val_set.image_ids):
            fg = int(per_image_raw[names[0]][k]["has_foreground"])
            mious = [per_image_raw[n][k]["miou"] for n in names]
            writer.writerow([image_id, fg] + [_fmt(m) for m in mious])
            per_image[image_id] = {"has_foreground": bool(fg), **{n: m for n, m in zip(names, mious)}}

    # ---------------- figures ----------------
    figures = []
    if make_figures:
        figures.append(str(plot_per_class_iou({n: list(results[n]["per_class_iou"].values()) for n in names},
                                              figures_dir / "per_class_iou.png")))
        curves = plot_training_curves(resolve_path(config["paths"]["training_logs_dir"]), names, figures_dir / "training_curves.png")
        if curves:
            figures.append(str(curves))
        qual = generate_qualitative_figures(config, val_set, models, per_image, device)
        figures.extend(qual["figures"])
    payload["figures"] = figures
    print(f"[evaluate] results written to {results_dir}")
    for fig in figures:
        print(f"[evaluate] figure: {fig}")
    return payload


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate trained models on the official VOC2012 validation split.")
    parser.add_argument("--config", default=None, help="Experiment YAML (default: configs/experiment.yaml).")
    parser.add_argument("--data-root", default=None, help="Override dataset.root (path to VOCdevkit/VOC2012).")
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation.")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, data_root=args.data_root)
        run_evaluation(config, args.models, make_figures=not args.no_figures)
        try:
            from .report_tables import write_report_tables
            path = write_report_tables(config)
            print(f"[evaluate] report tables: {path}  (run `python -m src.report_tables --update-report` "
                  f"to insert them into reports/experiment_report.md)")
        except FileNotFoundError as exc:
            print(f"[evaluate] note: report tables not written: {exc}")
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"[evaluate] ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
