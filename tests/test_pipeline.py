"""End-to-end smoke test on SYNTHETIC data: train both models, evaluate, build figures/tables.

This exercises the code paths only; the numbers it produces are meaningless and are never reported.
"""
import csv
from pathlib import Path

import pytest
import torch

from src.evaluate import run_evaluation
from src.report_tables import build_tables, update_report
from src.train import run_training
from src.utils import load_json, resolve_path


def read_log(cfg, name):
    with open(resolve_path(cfg["paths"]["training_logs_dir"]) / f"{name}_training_log.csv", newline="") as h:
        return list(csv.DictReader(h))


def test_full_pipeline(tiny_config):
    cfg = tiny_config
    cpu = torch.device("cpu")
    for name in ("baseline", "dilated"):
        summary = run_training(cfg, name, device=cpu, show_progress=False)
        assert summary["status"] == "completed"
        assert summary["budget"]["epochs_completed"] == 2
        assert summary["budget"]["total_optimization_steps_completed"] == 2 * 3  # 6 images / batch 2 = 3 per epoch
        rows = read_log(cfg, name)
        assert len(rows) == 2 and set(rows[0]) >= {"epoch", "train_loss", "val_loss", "val_miou", "learning_rate"}
        ckpt = torch.load(resolve_path(cfg["paths"]["checkpoints_dir"]) / f"{name}_best.pt", weights_only=True)
        assert set(ckpt) >= {"model_state_dict", "epoch", "optimizer_state_dict", "config", "seed", "val_miou", "model_name"}
        assert ckpt["model_name"] == name and ckpt["seed"] == 42
        assert (resolve_path(cfg["paths"]["checkpoints_dir"]) / f"{name}_last.pt").is_file()

    # existing outputs are never silently overwritten
    with pytest.raises(FileExistsError):
        run_training(cfg, "baseline", device=cpu, show_progress=False)

    out = run_evaluation(cfg, device=cpu, show_progress=False)
    rdir = resolve_path(cfg["paths"]["results_dir"])
    for fname in ("metrics.csv", "metrics.json", "comparison.csv", "per_class_iou.csv", "per_image_miou.csv",
                  "environment.json", "failure_cases.md", "failure_cases.json"):
        assert (rdir / fname).is_file(), fname
    metrics = load_json(rdir / "metrics.json")
    assert metrics["models"]["baseline"]["trainable_parameters"] == metrics["models"]["dilated"]["trainable_parameters"]
    assert metrics["models"]["baseline"]["latency"]["batch_size"] == 1
    fdir = resolve_path(cfg["paths"]["figures_dir"])
    assert (fdir / "qualitative_success.png").is_file() and (fdir / "qualitative_failure.png").is_file()
    assert out["figures"]

    # re-evaluated best-checkpoint mIoU must equal the mIoU logged at the best epoch
    for name in ("baseline", "dilated"):
        best = [r for r in read_log(cfg, name) if r["is_best"] == "1"][-1]
        assert abs(float(best["val_miou"]) - metrics["models"][name]["miou"]) < 1e-5

    # report-table injection
    tables = build_tables(cfg)
    assert {"per_class_iou", "miou", "params", "latency", "budget"} <= set(tables)
    report = Path(cfg["paths"]["results_dir"]) / "report.md"
    report.write_text("<!-- AUTO:miou:START -->\nplaceholder\n<!-- AUTO:miou:END -->\n", encoding="utf-8")
    assert update_report(cfg, report) == 1 and "placeholder" not in report.read_text(encoding="utf-8")


def test_training_is_deterministic_on_cpu(tiny_config, tmp_path):
    import copy
    cfg_a, cfg_b = copy.deepcopy(tiny_config), copy.deepcopy(tiny_config)
    for cfg, tag in ((cfg_a, "a"), (cfg_b, "b")):
        for key in ("checkpoints_dir", "training_logs_dir"):
            cfg["paths"][key] = str(tmp_path / f"{key}_{tag}")
        run_training(cfg, "dilated", device=torch.device("cpu"), show_progress=False)
    a, b = read_log(cfg_a, "dilated"), read_log(cfg_b, "dilated")
    assert [r["train_loss"] for r in a] == [r["train_loss"] for r in b]
    assert [r["val_miou"] for r in a] == [r["val_miou"] for r in b]


def test_evaluate_without_checkpoints_fails_clearly(tiny_config):
    with pytest.raises(FileNotFoundError, match="python -m src.train --model baseline"):
        run_evaluation(tiny_config, device=torch.device("cpu"), show_progress=False)
