import numpy as np
import pytest
import torch

from src.metrics import (
    ConfusionMatrix, confusion_matrix_from_batch, count_trainable_parameters, iou_from_confusion,
    mean_iou, measure_latency, per_sample_miou, pixel_accuracy,
)

IGN = 255


def test_iou_and_miou_hand_computed():
    # 3 classes. gt:   0 0 1 1 2 2     pred: 0 1 1 1 2 0
    gt = torch.tensor([[0, 0, 1, 1, 2, 2]])
    pred = torch.tensor([[0, 1, 1, 1, 2, 0]])
    conf = confusion_matrix_from_batch(pred, gt, 3, IGN).numpy()
    assert conf.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    iou = iou_from_confusion(conf)
    # class0: TP1, FP1 (gt2->pred0), FN1 -> 1/3 ; class1: TP2, FP1, FN0 -> 2/3 ; class2: TP1, FP0, FN1 -> 1/2
    assert np.allclose(iou, [1 / 3, 2 / 3, 1 / 2])
    assert mean_iou(iou) == pytest.approx((1 / 3 + 2 / 3 + 1 / 2) / 3)
    assert pixel_accuracy(conf) == pytest.approx(4 / 6)


def test_ignored_pixels_are_excluded():
    gt = torch.tensor([[0, 255, 1, 255]])
    pred = torch.tensor([[1, 0, 1, 1]])  # predictions on ignored pixels must not matter
    conf = confusion_matrix_from_batch(pred, gt, 2, IGN).numpy()
    assert conf.sum() == 2 and conf.tolist() == [[0, 1], [0, 1]]
    only_ignored = confusion_matrix_from_batch(torch.zeros(1, 4, dtype=torch.long), torch.full((1, 4), 255), 2, IGN)
    assert only_ignored.sum() == 0


def test_undefined_classes_are_nan_and_excluded_from_mean():
    conf = np.array([[4, 0, 0], [0, 0, 0], [0, 0, 0]])  # classes 1, 2 never occur
    iou = iou_from_confusion(conf)
    assert iou[0] == 1.0 and np.isnan(iou[1]) and np.isnan(iou[2])
    assert mean_iou(iou) == 1.0                              # NaN classes not counted as perfect
    assert np.isnan(mean_iou(np.array([np.nan, np.nan])))


def test_false_positive_on_absent_class_gives_zero_not_nan():
    gt = torch.tensor([[0, 0, 0, 0]])
    pred = torch.tensor([[0, 0, 1, 0]])
    iou = iou_from_confusion(confusion_matrix_from_batch(pred, gt, 2, IGN).numpy())
    assert iou[1] == 0.0 and iou[0] == pytest.approx(3 / 4)


def test_batch_accumulation_equals_single_pass():
    g = torch.randint(0, 5, (4, 8, 8))
    g[0, 0, :] = 255
    p = torch.randint(0, 5, (4, 8, 8))
    whole = ConfusionMatrix(5, IGN)
    whole.update(p, g)
    parts = ConfusionMatrix(5, IGN)
    for i in range(4):
        parts.update(p[i:i + 1], g[i:i + 1])
    assert np.array_equal(whole.matrix, parts.matrix)


def test_invalid_ids_raise():
    with pytest.raises(ValueError):
        confusion_matrix_from_batch(torch.zeros(1, 2, dtype=torch.long), torch.tensor([[0, 7]]), 3, IGN)


def test_per_sample_miou():
    gt = torch.tensor([[[0, 0], [1, 1]], [[0, 0], [0, 0]]])
    pred = torch.tensor([[[0, 0], [1, 1]], [[0, 0], [0, 1]]])
    res = per_sample_miou(pred, gt, 2, IGN)
    assert res[0]["miou"] == 1.0 and res[0]["has_foreground"] == 1.0
    assert res[1]["has_foreground"] == 0.0 and res[1]["miou"] == pytest.approx((3 / 4 + 0) / 2)


def test_parameter_count_and_latency_statistics():
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.ReLU())
    assert count_trainable_parameters(model) == 3 * 4 * 9 + 4
    stats = measure_latency(model, torch.device("cpu"), (1, 3, 16, 16), warmup_iterations=2, timed_iterations=5)
    assert stats["timed_iterations"] == 5 and stats["warmup_iterations"] == 2 and stats["batch_size"] == 1
    assert 0 < stats["min_ms"] <= stats["median_ms"] <= stats["p95_ms"] <= stats["max_ms"]
    assert not model.training  # evaluation mode
