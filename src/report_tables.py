"""Turn the REAL result files into Markdown tables for reports/experiment_report.md.

    python -m src.report_tables                  # writes <results_dir>/report_tables.md
    python -m src.report_tables --update-report  # also fills the AUTO blocks in the report

Reads metrics.json and the training summaries/logs. If those files do not exist the script stops
with an error; it never fills in numbers by itself.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .dataset import VOC_CLASS_NAMES
from .utils import ConfigError, load_config, load_json, resolve_path

MODELS = ("baseline", "dilated")


def _f(value: Any, digits: int = 4) -> str:
    return "undefined" if value is None else f"{value:.{digits}f}"


def _table(header: list, rows: list) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def build_tables(config: Mapping[str, Any]) -> Dict[str, str]:
    results_dir = resolve_path(config["paths"]["results_dir"])
    log_dir = resolve_path(config["paths"]["training_logs_dir"])
    metrics_path = results_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"{metrics_path} not found - run training and `python -m src.evaluate` first.")
    metrics = load_json(metrics_path)
    models = [m for m in MODELS if m in metrics["models"]]
    r = metrics["models"]
    tables: Dict[str, str] = {}

    tables["per_class_iou"] = _table(
        ["ID", "Class"] + models + (["Δ (dilated − baseline)"] if len(models) == 2 else []),
        [[i, c] + [_f(r[m]["per_class_iou"][c]) for m in models]
         + ([_f(None if None in (r["baseline"]["per_class_iou"][c], r["dilated"]["per_class_iou"][c])
                else r["dilated"]["per_class_iou"][c] - r["baseline"]["per_class_iou"][c])] if len(models) == 2 else [])
         for i, c in enumerate(VOC_CLASS_NAMES)],
    )

    summaries = {m: load_json(log_dir / f"{m}_training_summary.json") for m in models
                 if (log_dir / f"{m}_training_summary.json").is_file()}
    rows = []
    for m in models:
        s = summaries.get(m, {})
        final = s.get("final_epoch_val_miou")
        rows.append([m, r[m]["checkpoint_epoch"], _f(r[m]["miou"]), _f(r[m]["pixel_accuracy"]),
                     "n/a" if final is None else _f(float(final))])
    tables["miou"] = _table(
        ["Model", "Best epoch", "mIoU (best checkpoint, validation)", "Pixel accuracy", "mIoU at final epoch (training log)"], rows)
    if len(models) == 2:
        tables["miou"] += (f"\n\nDifference (dilated − baseline) in mIoU: "
                           f"{r['dilated']['miou'] - r['baseline']['miou']:+.4f} (single seed; see Limitations).")

    tables["params"] = _table(["Model", "Trainable parameters", "Bottleneck dilations (conv1, conv2)"],
                              [[m, f"{r[m]['trainable_parameters']:,}", tuple(r[m]["bottleneck_dilations"])] for m in models])

    lat_rows = [[m, _f(r[m]["latency"]["mean_ms"], 2), _f(r[m]["latency"]["median_ms"], 2), _f(r[m]["latency"]["p95_ms"], 2)]
                for m in models]
    lat = r[models[0]]["latency"]
    tables["latency"] = _table(["Model", "Mean (ms)", "Median (ms)", "95th percentile (ms)"], lat_rows) + (
        f"\n\nDevice: {lat['device']} ({metrics['environment']['hardware']}); input {lat['input_shape']}; "
        f"batch size {lat['batch_size']}; warm-up {lat['warmup_iterations']}; timed iterations {lat['timed_iterations']}; "
        f"torch threads {lat['torch_num_threads']}. Forward pass only - image loading and preprocessing are excluded.")

    if summaries:
        rows = []
        for m, s in summaries.items():
            b = s["budget"]
            rows.append([m, s["status"], f"{b.get('epochs_completed')}/{b['epochs_planned']}", b["training_images"],
                         b["batches_per_epoch"], b.get("total_optimization_steps_completed"), b["batch_size"],
                         b["input_resolution"], b.get("total_training_samples_processed"), b["validation_frequency"],
                         b["hardware"], f"{b.get('training_time_sec', 0) / 60:.1f} min", s["interruptions_or_deviations"]])
        tables["budget"] = _table(
            ["Model", "Status", "Epochs", "Train images", "Batches/epoch", "Optimization steps", "Batch size", "Input",
             "Samples processed", "Validation", "Hardware", "Training time", "Interruptions / deviations"], rows)
    else:
        tables["budget"] = "_Training summaries not found in the training-log directory._"
    return tables


def write_report_tables(config: Mapping[str, Any]) -> Path:
    tables = build_tables(config)
    out = resolve_path(config["paths"]["results_dir"]) / "report_tables.md"
    out.write_text("# Auto-generated tables (from real result files)\n\n" + "\n\n".join(
        f"## {name}\n\n{text}" for name, text in tables.items()) + "\n", encoding="utf-8")
    return out


def update_report(config: Mapping[str, Any], report_path: Path) -> int:
    """Replace the content between <!-- AUTO:name:START --> and <!-- AUTO:name:END --> markers."""
    tables = build_tables(config)
    text = report_path.read_text(encoding="utf-8")
    count = 0
    for name, table in tables.items():
        pattern = re.compile(rf"(<!-- AUTO:{name}:START -->).*?(<!-- AUTO:{name}:END -->)", re.DOTALL)
        text, n = pattern.subn(lambda m: f"{m.group(1)}\n\n{table}\n\n{m.group(2)}", text)
        count += n
    report_path.write_text(text, encoding="utf-8")
    return count


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Markdown result tables from real result files.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--update-report", action="store_true", help="Fill AUTO blocks in reports/experiment_report.md.")
    parser.add_argument("--report", default="reports/experiment_report.md")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        print(f"Wrote {write_report_tables(config)}")
        if args.update_report:
            n = update_report(config, resolve_path(args.report))
            print(f"Updated {n} AUTO block(s) in {args.report}")
    except (ConfigError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
