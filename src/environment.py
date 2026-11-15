"""Environment reporting: software versions, hardware and experiment settings.

Run ``python -m src.environment`` to print and save the environment to
``<results_dir>/environment.json``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import platform
import sys
from typing import Any, Dict, Mapping, Optional

import numpy as np
import PIL
import torch
import yaml

from .utils import ensure_dir, load_config, resolve_path, save_json, select_device

REPRODUCIBILITY_NOTE = (
    "Seeds and deterministic settings are applied, but bitwise-identical results are NOT guaranteed "
    "across different hardware, drivers, CUDA/cuDNN versions, PyTorch versions or operating systems. "
    "Some CUDA kernels have no deterministic implementation (PyTorch only warns for them)."
)


def collect_environment(device: Optional[torch.device] = None) -> Dict[str, Any]:
    device = device or select_device()
    cuda = torch.cuda.is_available()
    try:
        import matplotlib
        matplotlib_version = matplotlib.__version__
    except ImportError:  # pragma: no cover
        matplotlib_version = None
    return {
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "python_version": sys.version.replace("\n", " "),
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pillow_version": PIL.__version__,
        "pyyaml_version": yaml.__version__,
        "matplotlib_version": matplotlib_version,
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda if cuda else None,
        "cudnn_version": torch.backends.cudnn.version() if cuda else None,
        "device": str(device),
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda"
        else f"CPU ({platform.processor() or platform.machine()}, {os.cpu_count()} logical cores)",
        "torch_num_threads": torch.get_num_threads(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "reproducibility_note": REPRODUCIBILITY_NOTE,
    }


def build_experiment_record(
    config: Mapping[str, Any], model_name: str, model: torch.nn.Module, device: torch.device
) -> Dict[str, Any]:
    """Everything the assignment asks to record about one training run (except timings)."""
    ds, tr = config["dataset"], config["training"]
    return {
        "model_name": model_name,
        "random_seed": config["experiment"]["seed"],
        "bottleneck_dilations": list(getattr(model, "bottleneck_dilations", [])),
        "input_resolution": f"{ds['image_size']}x{ds['image_size']}",
        "batch_size": tr["batch_size"],
        "optimizer": tr["optimizer"],
        "learning_rate": tr["learning_rate"],
        "weight_decay": tr["weight_decay"],
        "loss": tr["loss"],
        "ignore_index": ds["ignore_index"],
        "data_augmentation": {
            "train": f"random horizontal flip, p={ds['horizontal_flip_probability']}",
            "validation": "none",
        },
        "training_epochs": tr["epochs"],
        "stopping_rule": tr["stopping_rule"],
        "learning_rate_schedule": "constant (no scheduler)",
        "environment": collect_environment(device),
    }


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Report and save the software/hardware environment.")
    parser.add_argument("--config", default=None, help="Path to experiment YAML (default: configs/experiment.yaml)")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    env = collect_environment()
    env["seed"] = config["experiment"]["seed"]
    for key, value in env.items():
        print(f"{key:>22}: {value}")
    out = resolve_path(config["paths"]["results_dir"]) / "environment.json"
    ensure_dir(out.parent)
    save_json(env, out)
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
