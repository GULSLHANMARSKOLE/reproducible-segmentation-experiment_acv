"""Prediction and visualisation: qualitative comparison figures, failure analysis, plots.

    python -m src.predict --ids 2007_000033 2007_000042      # custom comparison figure
    python -m src.predict --auto                              # rebuild success/failure figures from
                                                              # reports/results/per_image_miou.csv
Predictions are produced by the trained checkpoints; nothing is simulated.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend (works without a display, also on Windows)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image  # noqa: E402

from .dataset import VOC_CLASS_NAMES, VOCSegmentationDataset, denormalize_image, voc_colormap  # noqa: E402
from .engine import load_model_from_checkpoint  # noqa: E402
from .utils import (  # noqa: E402
    ConfigError, ensure_dir, load_config, resolve_path, save_json, select_device, MODEL_NAMES,
)

COLORMAP = voc_colormap(256)


def colorize(mask: np.ndarray) -> np.ndarray:
    """Map an index mask (values 0..20 or 255) to an RGB image with the VOC palette."""
    return COLORMAP[np.asarray(mask, dtype=np.uint8)]


@torch.inference_mode()
def predict_masks(model: torch.nn.Module, images: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    return model(images.to(device)).argmax(dim=1).cpu().numpy().astype(np.uint8)


# --------------------------------------------------------------------------------------
# Factual error description (derived from the masks only - no interpretation is invented)
# --------------------------------------------------------------------------------------
def describe_errors(gt: np.ndarray, pred: np.ndarray, ignore_index: int = 255, top_k: int = 2,
                    min_fp_fraction: float = 0.01) -> List[str]:
    """Quantitative description of the main errors of one prediction.

    Reports, for the largest ground-truth classes (foreground preferred), how much of each region was
    predicted correctly and which class took its place; plus classes predicted but absent from GT.
    """
    valid = gt != ignore_index
    total = int(valid.sum())
    if total == 0:
        return ["No valid (non-void) ground-truth pixels."]
    g, p = gt[valid].astype(np.int64), pred[valid].astype(np.int64)
    counts = np.bincount(g, minlength=len(VOC_CLASS_NAMES))
    order = [c for c in np.argsort(-counts) if counts[c] > 0]
    foreground = [c for c in order if c != 0]
    focus = (foreground or order)[:top_k]
    lines: List[str] = []
    for c in focus:
        in_region = p[g == c]
        hist = np.bincount(in_region, minlength=len(VOC_CLASS_NAMES))
        correct = hist[c] / in_region.size
        wrong = hist.copy()
        wrong[c] = 0
        sentence = (f"Ground-truth '{VOC_CLASS_NAMES[c]}' covers {counts[c] / total:.1%} of valid pixels; "
                    f"{correct:.1%} of it was predicted correctly")
        if wrong.sum() > 0:
            w = int(wrong.argmax())
            sentence += f", {wrong[w] / in_region.size:.1%} was predicted as '{VOC_CLASS_NAMES[w]}'"
        lines.append(sentence + ".")
    pred_counts = np.bincount(p, minlength=len(VOC_CLASS_NAMES))
    for c in np.argsort(-pred_counts):
        if counts[c] == 0 and pred_counts[c] / total >= min_fp_fraction:
            lines.append(f"'{VOC_CLASS_NAMES[c]}' was predicted on {pred_counts[c] / total:.1%} of valid pixels "
                         f"but is absent from the ground truth.")
    return lines


def select_examples(per_image: Mapping[str, Mapping[str, float]], model_names: Sequence[str],
                    n_success: int, n_failure: int) -> tuple[List[str], List[str]]:
    """Best / worst validation images by mean per-image mIoU across the models.

    Only images whose ground truth contains at least one foreground class are considered (an
    all-background image scores trivially high and says little about segmentation quality).
    """
    scored = []
    for image_id, stats in per_image.items():
        values = [stats[m] for m in model_names]
        if not stats.get("has_foreground") or any(v is None or np.isnan(v) for v in values):
            continue
        scored.append((float(np.mean(values)), image_id))
    scored.sort(key=lambda t: (t[0], t[1]))
    failures = [i for _, i in scored[:n_failure]]
    successes = [i for _, i in reversed(scored[-n_success:])] if n_success > 0 else []
    return successes, failures


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def make_comparison_figure(rows: List[Dict[str, Any]], model_names: Sequence[str], title: str, path: Path) -> Path:
    """Grid figure: source image | ground truth | one column per model."""
    n_cols = 2 + len(model_names)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(2.7 * n_cols, 2.7 * len(rows) + 1.6), squeeze=False)
    for r, row in enumerate(rows):
        panels = [(row["image"], f"VOC ID {row['id']}"), (colorize(row["gt"]), "Ground truth")]
        for name in model_names:
            label = f"{name}\nimage mIoU = {row['miou'][name]:.3f}"
            panels.append((colorize(row["pred"][name]), label))
        for c, (img, label) in enumerate(panels):
            ax = axes[r][c]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(label, fontsize=8)
    handles = [Patch(facecolor=COLORMAP[i] / 255.0, edgecolor="k", label=name) for i, name in enumerate(VOC_CLASS_NAMES)]
    handles.append(Patch(facecolor=COLORMAP[255] / 255.0, edgecolor="k", label="Void (255)"))
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.subplots_adjust(hspace=0.4)  # room for the per-panel titles between rows
    ensure_dir(path.parent)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _collect_rows(ids: Sequence[str], dataset: VOCSegmentationDataset, models: Mapping[str, torch.nn.Module],
                  device: torch.device, mean: Sequence[float], std: Sequence[float]) -> List[Dict[str, Any]]:
    from .metrics import iou_from_confusion, mean_iou, confusion_matrix_from_batch

    samples = [dataset[dataset.index_of(i)] for i in ids]
    images = torch.stack([s[0] for s in samples])
    gts = [s[1].numpy().astype(np.uint8) for s in samples]
    preds = {name: predict_masks(model, images, device) for name, model in models.items()}
    rows = []
    for k, image_id in enumerate(ids):
        miou = {}
        for name in models:
            conf = confusion_matrix_from_batch(torch.from_numpy(preds[name][k]), torch.from_numpy(gts[k]),
                                               dataset.num_classes, dataset.ignore_index).numpy()
            miou[name] = mean_iou(iou_from_confusion(conf))
        rows.append({
            "id": image_id, "image": denormalize_image(images[k], mean, std), "gt": gts[k],
            "pred": {n: preds[n][k] for n in models}, "miou": miou,
        })
    return rows


def save_prediction_pngs(rows: Sequence[Dict[str, Any]], predictions_dir: Path) -> None:
    """Save predicted masks as indexed PNGs (VOC palette) under outputs/predictions/<model>/."""
    for row in rows:
        for name, pred in row["pred"].items():
            out = ensure_dir(predictions_dir / name) / f"{row['id']}.png"
            img = Image.fromarray(pred.astype(np.uint8))
            img.putpalette(COLORMAP.reshape(-1).tolist())
            img.save(out)


def generate_qualitative_figures(
    config: Mapping[str, Any], dataset: VOCSegmentationDataset, models: Mapping[str, torch.nn.Module],
    per_image: Mapping[str, Mapping[str, float]], device: torch.device,
) -> Dict[str, Any]:
    """Success + failure figures, auto-generated failure facts and saved prediction PNGs."""
    ev, ds = config["evaluation"], config["dataset"]
    names = list(models)
    figures_dir = ensure_dir(resolve_path(config["paths"]["figures_dir"]))
    results_dir = ensure_dir(resolve_path(config["paths"]["results_dir"]))
    predictions_dir = ensure_dir(resolve_path(config["paths"]["predictions_dir"]))
    successes, failures = select_examples(per_image, names, ev["num_success_examples"], ev["num_failure_examples"])
    outputs: Dict[str, Any] = {"success_ids": successes, "failure_ids": failures, "figures": []}

    for label, ids, fname, title in (
        ("success", successes, "qualitative_success.png",
         "Best-segmented validation images (highest mean per-image mIoU across models)"),
        ("failure", failures, "qualitative_failure.png",
         "Worst-segmented validation images (lowest mean per-image mIoU across models)"),
    ):
        if not ids:
            continue
        rows = _collect_rows(ids, dataset, models, device, ds["mean"], ds["std"])
        outputs["figures"].append(str(make_comparison_figure(rows, names, title, figures_dir / fname)))
        save_prediction_pngs(rows, predictions_dir)
        if label == "failure":
            cases = []
            for row in rows:
                cases.append({
                    "image_id": row["id"],
                    "per_image_miou": {n: row["miou"][n] for n in names},
                    "error_facts": {n: describe_errors(row["gt"], row["pred"][n], ds["ignore_index"]) for n in names},
                })
            save_json(cases, results_dir / "failure_cases.json")
            md = ["# Failure cases (auto-generated facts)", "",
                  "The statements below are computed directly from the ground-truth and predicted masks. "
                  "They are NOT a visual interpretation. Add your own observations after inspecting "
                  "`reports/figures/qualitative_failure.png`.", ""]
            for case in cases:
                md.append(f"## VOC image ID `{case['image_id']}`")
                md.append("")
                for n in names:
                    md.append(f"**{n}** - per-image mIoU = {case['per_image_miou'][n]:.3f}")
                    md.append("")
                    md.extend(f"- {line}" for line in case["error_facts"][n])
                    md.append("")
                md += ["", "_Visual interpretation (to be written by the author after viewing the figure):_ TODO", ""]
            (results_dir / "failure_cases.md").write_text("\n".join(md), encoding="utf-8")
            outputs["failure_cases_file"] = str(results_dir / "failure_cases.md")
    return outputs


def plot_per_class_iou(per_class_iou: Mapping[str, Sequence[float]], path: Path) -> Path:
    names = list(per_class_iou)
    y = np.arange(len(VOC_CLASS_NAMES))
    height = 0.8 / max(len(names), 1)
    fig, ax = plt.subplots(figsize=(7, 7))
    for k, name in enumerate(names):
        vals = np.array([np.nan if v is None else v for v in per_class_iou[name]], dtype=float)
        ax.barh(y + k * height, np.nan_to_num(vals), height=height, label=name)
    ax.set_yticks(y + height * (len(names) - 1) / 2)
    ax.set_yticklabels(VOC_CLASS_NAMES, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("IoU (validation split)")
    ax.set_title("Per-class IoU")
    ax.legend()
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_training_curves(log_dir: Path, model_names: Sequence[str], path: Path) -> Optional[Path]:
    """Training/validation loss and validation mIoU per epoch, from the real training logs."""
    logs = {}
    for name in model_names:
        log_path = log_dir / f"{name}_training_log.csv"
        if log_path.is_file():
            with open(log_path, newline="", encoding="utf-8") as handle:
                logs[name] = list(csv.DictReader(handle))
    if not logs:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name, rows in logs.items():
        epochs = [int(r["epoch"]) for r in rows]
        axes[0].plot(epochs, [float(r["train_loss"]) for r in rows], label=name)
        axes[1].plot(epochs, [float(r["val_loss"]) for r in rows], label=name)
        axes[2].plot(epochs, [float(r["val_miou"]) for r in rows], label=name)
    for ax, title in zip(axes, ("Training loss", "Validation loss", "Validation mIoU")):
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _load_models(config: Mapping[str, Any], names: Sequence[str], device: torch.device) -> Dict[str, torch.nn.Module]:
    ckpt_dir = resolve_path(config["paths"]["checkpoints_dir"])
    return {n: load_model_from_checkpoint(n, ckpt_dir / f"{n}_best.pt", config, device)[0] for n in names}


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Qualitative predictions from trained checkpoints.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None, help="Override dataset.root.")
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ids", nargs="+", help="Validation image IDs to compare (e.g. 2007_000033).")
    group.add_argument("--auto", action="store_true",
                       help="Rebuild success/failure figures from reports/results/per_image_miou.csv.")
    parser.add_argument("--output", default="custom_comparison.png",
                        help="File name for --ids figure (saved in outputs/predictions/).")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, data_root=args.data_root)
        device = select_device()
        print(f"[predict] device={device}")
        models = _load_models(config, args.models, device)
        dataset = VOCSegmentationDataset.from_config(config, "val", train=False)
        if args.ids:
            ds = config["dataset"]
            rows = _collect_rows(args.ids, dataset, models, device, ds["mean"], ds["std"])
            out_dir = ensure_dir(resolve_path(config["paths"]["predictions_dir"]))
            save_prediction_pngs(rows, out_dir)
            fig = make_comparison_figure(rows, list(models), "Custom comparison (validation images)", out_dir / args.output)
            print(f"[predict] saved {fig}")
        else:
            csv_path = resolve_path(config["paths"]["results_dir"]) / "per_image_miou.csv"
            if not csv_path.is_file():
                raise FileNotFoundError(f"{csv_path} not found. Run `python -m src.evaluate` first.")
            per_image = {}
            with open(csv_path, newline="", encoding="utf-8") as handle:
                for rec in csv.DictReader(handle):
                    per_image[rec["image_id"]] = {
                        "has_foreground": rec["has_foreground"] == "1",
                        **{m: float(rec[f"{m}_miou"]) for m in args.models},
                    }
            out = generate_qualitative_figures(config, dataset, models, per_image, device)
            print(f"[predict] saved {out['figures']}")
    except (ConfigError, FileNotFoundError, KeyError, ValueError) as exc:
        print(f"[predict] ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
