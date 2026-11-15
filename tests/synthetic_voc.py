"""Tiny synthetic VOC-2012-shaped dataset used ONLY by the tests (never by experiments)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

from src.dataset import voc_colormap


def create_synthetic_voc(root: Path, n_train: int = 6, n_val: int = 4, seed: int = 0) -> Dict[str, np.ndarray]:
    """Create <root>/{JPEGImages,SegmentationClass,ImageSets/Segmentation}. Returns raw masks by ID.

    Masks are indexed ('P') PNGs with the VOC palette, contain classes 0..20 plus 255 (void borders),
    and images/masks have varying sizes so the resize path is exercised.
    """
    rng = np.random.default_rng(seed)
    (root / "JPEGImages").mkdir(parents=True, exist_ok=True)
    (root / "SegmentationClass").mkdir(parents=True, exist_ok=True)
    (root / "ImageSets" / "Segmentation").mkdir(parents=True, exist_ok=True)
    palette = voc_colormap(256).reshape(-1).tolist()
    masks: Dict[str, np.ndarray] = {}
    for split, count, offset in (("train", n_train, 0), ("val", n_val, 1000)):
        ids = []
        for k in range(count):
            image_id = f"2099_{offset + k:06d}"
            h, w = int(rng.integers(60, 100)), int(rng.integers(60, 100))
            mask = np.zeros((h, w), dtype=np.uint8)
            image = rng.integers(0, 60, size=(h, w, 3), dtype=np.uint8)
            for _ in range(2):  # two random rectangles of random foreground classes
                cls = int(rng.integers(1, 21))
                y0, x0 = int(rng.integers(0, h // 2)), int(rng.integers(0, w // 2))
                y1, x1 = y0 + int(rng.integers(10, h // 2)), x0 + int(rng.integers(10, w // 2))
                mask[y0:y1, x0:x1] = cls
                image[y0:y1, x0:x1] = np.clip(image[y0:y1, x0:x1] + 20 * cls % 200, 0, 255)
            mask[:8, :] = 255  # void border band (thick enough to survive nearest downsampling)
            Image.fromarray(image, "RGB").save(root / "JPEGImages" / f"{image_id}.jpg", quality=95)
            pm = Image.fromarray(mask)  # mode 'L' -> becomes palettised below
            pm.putpalette(palette)
            pm.save(root / "SegmentationClass" / f"{image_id}.png")
            masks[image_id] = mask
            ids.append(image_id)
        (root / "ImageSets" / "Segmentation" / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    return masks
