"""PASCAL VOC 2012 semantic-segmentation dataset with synchronized image/mask transforms.

Mask handling rules (see reports/data_contract.md):
  * masks are indexed PNGs; the raw palette *indices* are read (never converted to RGB),
  * masks are never normalised and are resized with NEAREST-neighbour interpolation only,
  * unexpected label values raise an error (they are reported, never silently remapped).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .utils import resolve_path

IGNORE_INDEX = 255
VOC_CLASS_NAMES: List[str] = [
    "Background", "Aeroplane", "Bicycle", "Bird", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair",
    "Cow", "Dining Table", "Dog", "Horse", "Motorbike", "Person", "Potted Plant", "Sheep", "Sofa",
    "Train", "TV/Monitor",
]
_BILINEAR = Image.Resampling.BILINEAR
_NEAREST = Image.Resampling.NEAREST
_FLIP = Image.Transpose.FLIP_LEFT_RIGHT
_ALLOWED_MASK_MODES = ("P", "L")  # both store raw class indices; RGB would be a colour render


def voc_colormap(size: int = 256) -> np.ndarray:
    """The standard PASCAL VOC colour map (uint8, [size, 3]); index 255 is set to the VOC void colour."""
    cmap = np.zeros((size, 3), dtype=np.uint8)
    for i in range(size):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        cmap[i] = (r, g, b)
    if size > IGNORE_INDEX:
        cmap[IGNORE_INDEX] = (224, 224, 192)
    return cmap


def split_file_path(root: Path, split: str) -> Path:
    return Path(root) / "ImageSets" / "Segmentation" / f"{split}.txt"


def load_split_ids(root: Path, split: str) -> List[str]:
    """Read image IDs from the official split file (one ID per line)."""
    path = split_file_path(root, split)
    if not path.is_file():
        raise FileNotFoundError(
            f"Split file not found: {path}\n"
            "Expected PASCAL VOC 2012 at '<root>/ImageSets/Segmentation/{train,val}.txt'. "
            "Run `python -m src.check_dataset` for diagnostics."
        )
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def denormalize_image(tensor: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """Invert ImageNet normalisation: [3,H,W] float tensor -> [H,W,3] uint8 array (for display)."""
    m = torch.tensor(mean, dtype=tensor.dtype).view(3, 1, 1)
    s = torch.tensor(std, dtype=tensor.dtype).view(3, 1, 1)
    img = (tensor.detach().cpu() * s + m).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


class VOCSegmentationDataset(Dataset):
    """Official VOC2012 train/val split. Returns ``(image[3,S,S] float32, mask[S,S] int64)``."""

    def __init__(
        self,
        root: Path,
        split: str,
        image_size: int,
        mean: Sequence[float],
        std: Sequence[float],
        train: bool,
        horizontal_flip_probability: float = 0.0,
        num_classes: int = 21,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = int(image_size)
        self.train = bool(train)
        # Validation never augments, regardless of the configured probability.
        self.flip_probability = float(horizontal_flip_probability) if self.train else 0.0
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.mean = torch.tensor(list(mean), dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(list(std), dtype=torch.float32).view(3, 1, 1)
        self.image_dir = self.root / "JPEGImages"
        self.mask_dir = self.root / "SegmentationClass"  # SegmentationObject is intentionally unused

        for directory in (self.image_dir, self.mask_dir):
            if not directory.is_dir():
                raise FileNotFoundError(
                    f"Dataset directory not found: {directory}\n"
                    "Check `dataset.root` in configs/experiment.yaml (or pass --data-root)."
                )
        self.image_ids = load_split_ids(self.root, split)
        if not self.image_ids:
            raise ValueError(f"Split '{split}' is empty.")
        if len(set(self.image_ids)) != len(self.image_ids):
            raise ValueError(f"Split '{split}' contains duplicate image IDs.")
        self._verify_files_exist()

    # ----------------------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: Mapping[str, Any], split: str, train: bool) -> "VOCSegmentationDataset":
        ds = config["dataset"]
        split_name = ds["train_split"] if split == "train" else ds["val_split"]
        return cls(
            root=resolve_path(ds["root"]),
            split=split_name,
            image_size=ds["image_size"],
            mean=ds["mean"],
            std=ds["std"],
            train=train,
            horizontal_flip_probability=ds["horizontal_flip_probability"],
            num_classes=ds["num_classes"],
            ignore_index=ds["ignore_index"],
        )

    def _verify_files_exist(self) -> None:
        missing_images = [i for i in self.image_ids if not (self.image_dir / f"{i}.jpg").is_file()]
        missing_masks = [i for i in self.image_ids if not (self.mask_dir / f"{i}.png").is_file()]
        if missing_images or missing_masks:
            raise FileNotFoundError(
                f"Split '{self.split}': {len(missing_images)} missing JPEG(s) "
                f"(e.g. {missing_images[:3]}), {len(missing_masks)} missing mask(s) "
                f"(e.g. {missing_masks[:3]}). Run `python -m src.check_dataset`."
            )

    def __len__(self) -> int:
        return len(self.image_ids)

    def index_of(self, image_id: str) -> int:
        try:
            return self.image_ids.index(image_id)
        except ValueError:
            raise KeyError(f"Image ID '{image_id}' is not in split '{self.split}'.") from None

    # ----------------------------------------------------------------------------------
    def load_raw(self, index: int) -> Tuple[Image.Image, np.ndarray]:
        """Load the RGB image and the raw uint8 index mask (no resizing, no augmentation)."""
        image_id = self.image_ids[index]
        with Image.open(self.image_dir / f"{image_id}.jpg") as img:
            image = img.convert("RGB")
        with Image.open(self.mask_dir / f"{image_id}.png") as msk:
            if msk.mode not in _ALLOWED_MASK_MODES:
                raise ValueError(
                    f"Mask '{image_id}.png' has mode '{msk.mode}', expected an indexed PNG "
                    f"('P'). Masks must not be converted to RGB."
                )
            mask = np.array(msk, dtype=np.uint8)  # palette INDICES, not colours
        if image.size != (mask.shape[1], mask.shape[0]):
            raise ValueError(
                f"Image/mask size mismatch for '{image_id}': image {image.size} (WxH) vs "
                f"mask {(mask.shape[1], mask.shape[0])}."
            )
        invalid = (mask >= self.num_classes) & (mask != self.ignore_index)
        if invalid.any():
            bad = np.unique(mask[invalid]).tolist()
            raise ValueError(f"Mask '{image_id}.png' contains unexpected label values {bad}.")
        return image, mask

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask_array = self.load_raw(index)
        size = (self.image_size, self.image_size)
        image = image.resize(size, _BILINEAR)
        mask = Image.fromarray(mask_array).resize(size, _NEAREST)  # NEAREST keeps class IDs intact

        # Synchronized geometric augmentation: the SAME flip decision for image and mask.
        if self.flip_probability > 0.0 and random.random() < self.flip_probability:
            image = image.transpose(_FLIP)
            mask = mask.transpose(_FLIP)

        image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
        image_tensor = (image_tensor - self.mean) / self.std
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image_tensor, mask_tensor
