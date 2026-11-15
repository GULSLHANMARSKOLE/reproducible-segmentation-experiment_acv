"""Shared utilities: configuration loading/validation, seeding, device selection, JSON helpers."""
from __future__ import annotations

import copy
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np
import torch
import yaml

# cuBLAS needs this variable to be set *before* its first use for deterministic behaviour on CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "experiment.yaml"
MODEL_NAMES = ("baseline", "dilated")

PathLike = Union[str, os.PathLike]


class ConfigError(ValueError):
    """Raised when the experiment configuration is missing keys or contains invalid values."""


# --------------------------------------------------------------------------------------
# Paths and configuration
# --------------------------------------------------------------------------------------
def resolve_path(path: PathLike, root: Path = PROJECT_ROOT) -> Path:
    """Return an absolute path; relative paths are interpreted relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else (root / p).resolve()


_NUM = (int, float)
_REQUIRED: dict[str, dict[str, Any]] = {
    "experiment": {"name": str, "seed": int},
    "dataset": {
        "name": str, "root": str, "image_size": int, "num_classes": int, "ignore_index": int,
        "train_split": str, "val_split": str, "mean": list, "std": list,
        "horizontal_flip_probability": _NUM,
    },
    "training": {
        "batch_size": int, "epochs": int, "learning_rate": _NUM, "weight_decay": _NUM,
        "optimizer": str, "loss": str, "num_workers": int, "deterministic": bool,
        "stopping_rule": str,
    },
    "model": {"base_channels": int, "pretrained": bool},
    "evaluation": {
        "batch_size": int, "warmup_iterations": int, "timed_iterations": int,
        "validation_batch_size": int, "num_success_examples": int, "num_failure_examples": int,
    },
    "paths": {
        "checkpoints_dir": str, "predictions_dir": str, "training_logs_dir": str,
        "results_dir": str, "figures_dir": str,
    },
}


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate structure, types and value ranges. Raises ConfigError with a precise message."""
    for section, keys in _REQUIRED.items():
        if section not in config or not isinstance(config[section], Mapping):
            raise ConfigError(f"Config is missing section '{section}'.")
        for key, expected in keys.items():
            if key not in config[section]:
                raise ConfigError(f"Config is missing '{section}.{key}'.")
            if not isinstance(config[section][key], expected):
                raise ConfigError(
                    f"Config value '{section}.{key}'={config[section][key]!r} has wrong type "
                    f"(expected {expected})."
                )

    ds, tr, ev, md = config["dataset"], config["training"], config["evaluation"], config["model"]
    if ds["image_size"] <= 0 or ds["image_size"] % 8 != 0:
        raise ConfigError("dataset.image_size must be a positive multiple of 8 (three 2x poolings).")
    if ds["num_classes"] < 2:
        raise ConfigError("dataset.num_classes must be >= 2.")
    if 0 <= ds["ignore_index"] < ds["num_classes"]:
        raise ConfigError("dataset.ignore_index must not collide with a valid class id.")
    if len(ds["mean"]) != 3 or len(ds["std"]) != 3 or any(s <= 0 for s in ds["std"]):
        raise ConfigError("dataset.mean / dataset.std must each contain 3 values (std > 0).")
    if not 0.0 <= ds["horizontal_flip_probability"] <= 1.0:
        raise ConfigError("dataset.horizontal_flip_probability must be in [0, 1].")
    if ds["train_split"] == ds["val_split"]:
        raise ConfigError("dataset.train_split and dataset.val_split must differ.")
    if tr["batch_size"] < 1 or tr["epochs"] < 1 or tr["num_workers"] < 0:
        raise ConfigError("training.batch_size/epochs must be >= 1 and num_workers >= 0.")
    if tr["learning_rate"] <= 0 or tr["weight_decay"] < 0:
        raise ConfigError("training.learning_rate must be > 0 and weight_decay >= 0.")
    if tr["optimizer"] != "AdamW":
        raise ConfigError("Only training.optimizer='AdamW' is supported by this experiment.")
    if tr["loss"] != "CrossEntropyLoss":
        raise ConfigError("Only training.loss='CrossEntropyLoss' is supported by this experiment.")
    if md["base_channels"] < 1:
        raise ConfigError("model.base_channels must be >= 1.")
    if md["pretrained"]:
        raise ConfigError("model.pretrained must be false: the networks are trained from scratch.")
    for key in ("batch_size", "validation_batch_size", "warmup_iterations", "timed_iterations"):
        if ev[key] < 1:
            raise ConfigError(f"evaluation.{key} must be >= 1.")


def load_config(path: Optional[PathLike] = None, data_root: Optional[PathLike] = None) -> dict:
    """Load and validate the YAML configuration.

    ``data_root`` (e.g. from ``--data-root``) overrides ``dataset.root`` and is recorded in the
    returned config, so checkpoints and logs reflect the dataset location that was really used.
    """
    cfg_path = resolve_path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise ConfigError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigError(f"Config file {cfg_path} is empty or not a mapping.")
    config = copy.deepcopy(config)
    if data_root is not None:
        config["dataset"]["root"] = str(data_root)
    validate_config(config)
    return config


# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA) and configure deterministic behaviour."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic)
    if deterministic:
        # warn_only: some CUDA kernels (e.g. nll_loss2d backward) have no deterministic variant.
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: derive Python/NumPy seeds from the worker's torch seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """A dedicated generator for DataLoader shuffling."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def select_device() -> torch.device:
    """CUDA if available, otherwise CPU (never changes any experimental setting)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------------------
def ensure_dir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/torch scalars/arrays; NaN/Inf become None (valid JSON)."""
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, torch.Tensor):
        return to_jsonable(obj.detach().cpu().tolist())
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(data: Any, path: PathLike) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(data), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path


def load_json(path: PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
