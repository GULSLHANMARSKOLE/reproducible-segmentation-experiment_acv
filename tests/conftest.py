import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.utils import load_config  # noqa: E402
from synthetic_voc import create_synthetic_voc  # noqa: E402


@pytest.fixture()
def synthetic_root(tmp_path):
    root = tmp_path / "VOC2012"
    masks = create_synthetic_voc(root)
    return root, masks


@pytest.fixture()
def tiny_config(tmp_path, synthetic_root):
    """The real experiment config, redirected to synthetic data/tmp dirs and shrunk for speed.

    Only used by tests; real experiments always use configs/experiment.yaml unchanged.
    """
    root, _ = synthetic_root
    cfg = copy.deepcopy(load_config())
    cfg["dataset"]["root"] = str(root)
    cfg["dataset"]["image_size"] = 32
    cfg["model"]["base_channels"] = 4
    cfg["training"]["epochs"] = 2
    cfg["training"]["batch_size"] = 2
    cfg["evaluation"].update({"validation_batch_size": 2, "warmup_iterations": 1, "timed_iterations": 3,
                              "num_success_examples": 2, "num_failure_examples": 2})
    for key, name in (("checkpoints_dir", "ckpt"), ("predictions_dir", "pred"), ("training_logs_dir", "logs"),
                      ("results_dir", "results"), ("figures_dir", "figs")):
        cfg["paths"][key] = str(tmp_path / name)
    return cfg
