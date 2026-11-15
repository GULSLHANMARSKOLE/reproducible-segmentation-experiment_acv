import numpy as np
import pytest
import torch
from PIL import Image

from src.dataset import VOCSegmentationDataset, voc_colormap


def make(root, split="train", train=False, flip=0.0, size=32):
    return VOCSegmentationDataset(root, split, size, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
                                  train=train, horizontal_flip_probability=flip)


def test_loading_shapes_and_dtypes(synthetic_root):
    root, _ = synthetic_root
    ds = make(root)
    assert len(ds) == 6 and len(make(root, "val")) == 4
    image, mask = ds[0]
    assert image.shape == (3, 32, 32) and image.dtype == torch.float32
    assert mask.shape == (32, 32) and mask.dtype == torch.int64


def test_class_ids_and_ignore_label_preserved(synthetic_root):
    root, masks = synthetic_root
    ds = make(root)
    for i, image_id in enumerate(ds.image_ids):
        _, mask = ds[i]
        raw = masks[image_id]
        expected = np.array(Image.fromarray(raw).resize((32, 32), Image.Resampling.NEAREST))
        assert np.array_equal(mask.numpy(), expected)           # exactly nearest-neighbour
        assert set(np.unique(mask.numpy())) <= set(np.unique(raw))  # no new (interpolated) labels
        assert 255 in mask.numpy()                                  # void pixels kept as 255


def test_masks_read_as_indices_not_rgb(synthetic_root):
    root, masks = synthetic_root
    image_id = sorted(masks)[0]
    with Image.open(root / "SegmentationClass" / f"{image_id}.png") as m:
        assert m.mode == "P"
    _, raw = make(root).load_raw(0)
    assert raw.ndim == 2 and np.array_equal(raw, masks[image_id])


def test_flip_is_synchronized_and_validation_never_augments(synthetic_root):
    root, _ = synthetic_root
    plain_img, plain_mask = make(root, train=False)[0]
    flip_img, flip_mask = make(root, train=True, flip=1.0)[0]
    assert torch.allclose(flip_img, plain_img.flip(-1)) and torch.equal(flip_mask, plain_mask.flip(-1))
    val_flip_img, val_flip_mask = make(root, train=False, flip=1.0)[0]  # train=False => no flip
    assert torch.equal(val_flip_mask, plain_mask) and torch.allclose(val_flip_img, plain_img)


def test_missing_mask_raises(synthetic_root):
    root, masks = synthetic_root
    (root / "SegmentationClass" / f"{sorted(masks)[0]}.png").unlink()
    with pytest.raises(FileNotFoundError, match="missing mask"):
        make(root)


def test_rgb_mask_is_rejected_not_converted(synthetic_root):
    root, masks = synthetic_root
    image_id = sorted(masks)[0]
    Image.fromarray(np.zeros((*masks[image_id].shape, 3), np.uint8)).save(root / "SegmentationClass" / f"{image_id}.png")
    with pytest.raises(ValueError, match="indexed PNG"):
        make(root)[0]


def test_unexpected_label_is_reported(synthetic_root):
    root, masks = synthetic_root
    image_id = sorted(masks)[0]
    bad = masks[image_id].copy()
    bad[5, 5] = 30
    Image.fromarray(bad).save(root / "SegmentationClass" / f"{image_id}.png")
    with pytest.raises(ValueError, match=r"unexpected label values \[30\]"):
        make(root)[0]


def test_size_mismatch_is_reported(synthetic_root):
    root, masks = synthetic_root
    image_id = sorted(masks)[0]
    Image.fromarray(np.zeros((10, 10), np.uint8)).save(root / "SegmentationClass" / f"{image_id}.png")
    with pytest.raises(ValueError, match="size mismatch"):
        make(root)[0]


def test_train_val_split_disjoint(synthetic_root):
    root, _ = synthetic_root
    assert not set(make(root).image_ids) & set(make(root, "val").image_ids)


def test_colormap_standard_entries():
    cmap = voc_colormap()
    assert tuple(cmap[0]) == (0, 0, 0) and tuple(cmap[1]) == (128, 0, 0) and tuple(cmap[15]) == (192, 128, 128)
    assert tuple(cmap[255]) == (224, 224, 192)


def test_check_dataset_on_synthetic(synthetic_root):
    from src.check_dataset import verify_dataset
    root, _ = synthetic_root
    report = verify_dataset(root, expected_counts={"train": 6, "val": 4})
    assert report["passed"], report["errors"]
    assert report["splits"]["train"]["invalid_label_values"] == []
    (root / "SegmentationClass" / f"{sorted(report['splits']['val']['missing_masks'] or ['2099_001000'])[0]}.png").unlink()
    bad = verify_dataset(root, expected_counts={"train": 6, "val": 4})
    assert not bad["passed"] and bad["splits"]["val"]["num_missing_masks"] == 1
