"""Dataset verification script - RUN THIS LOCALLY before training.

    python -m src.check_dataset
    python -m src.check_dataset --data-root "D:/datasets/VOCdevkit/VOC2012"

Checks the official PASCAL VOC 2012 segmentation data: split sizes, image/mask pairing, image and
mask dimensions, mask mode (indexed PNG), unique label values, invalid class ids and train/val
consistency. Exit code 0 only if every check passes. Writes <results_dir>/dataset_verification.json.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

from .dataset import VOC_CLASS_NAMES, load_split_ids
from .utils import ConfigError, load_config, resolve_path, save_json, PROJECT_ROOT

EXPECTED_COUNTS = {"train": 1464, "val": 1449}


def find_voc_candidates(search_root: Path, max_depth: int = 5) -> List[Path]:
    """Find directories that look like a VOC2012 root (contain ImageSets/Segmentation)."""
    found: List[Path] = []
    search_root = Path(search_root)
    if not search_root.is_dir():
        return found
    for path in search_root.rglob("Segmentation"):
        try:
            depth = len(path.relative_to(search_root).parts)
        except ValueError:
            continue
        if depth <= max_depth and path.parent.name == "ImageSets" and path.is_dir():
            found.append(path.parent.parent)
    return sorted(set(found))


def inspect_split(root: Path, split: str, num_classes: int, ignore_index: int, expected: Optional[int]) -> Dict[str, Any]:
    ids = load_split_ids(root, split)
    image_dir, mask_dir = root / "JPEGImages", root / "SegmentationClass"
    missing_images = [i for i in ids if not (image_dir / f"{i}.jpg").is_file()]
    missing_masks = [i for i in ids if not (mask_dir / f"{i}.png").is_file()]
    size_mismatch, bad_mode, bad_image_mode, unreadable = [], [], [], []
    invalid_examples: Dict[int, List[str]] = {}
    pixel_counts = np.zeros(256, dtype=np.int64)
    image_sizes, mask_sizes = Counter(), Counter()
    present = set(ids) - set(missing_images) - set(missing_masks)

    for image_id in tqdm(ids, desc=f"inspecting {split}", leave=False):
        if image_id not in present:
            continue
        try:
            with Image.open(image_dir / f"{image_id}.jpg") as img:
                image_size, image_mode = img.size, img.mode
            with Image.open(mask_dir / f"{image_id}.png") as msk:
                mask_size, mask_mode = msk.size, msk.mode
                if mask_mode in ("P", "L"):
                    arr = np.array(msk, dtype=np.uint8)
                else:
                    arr = None
        except Exception as exc:  # corrupt file
            unreadable.append(f"{image_id}: {exc!r}")
            continue
        image_sizes[f"{image_size[0]}x{image_size[1]}"] += 1
        mask_sizes[f"{mask_size[0]}x{mask_size[1]}"] += 1
        if image_size != mask_size:
            size_mismatch.append(f"{image_id}: image {image_size} vs mask {mask_size}")
        if image_mode not in ("RGB",):
            bad_image_mode.append(f"{image_id}: {image_mode}")
        if arr is None:
            bad_mode.append(f"{image_id}: {mask_mode}")
            continue
        if mask_mode != "P":
            bad_mode.append(f"{image_id}: {mask_mode} (expected indexed 'P')")
        pixel_counts += np.bincount(arr.ravel(), minlength=256)
        for label in np.unique(arr):
            if label >= num_classes and label != ignore_index:
                invalid_examples.setdefault(int(label), []).append(image_id)

    labels_present = [int(i) for i in np.nonzero(pixel_counts)[0]]
    invalid_labels = [l for l in labels_present if l >= num_classes and l != ignore_index]
    # Non-RGB JPEGs (e.g. grayscale) are converted to RGB on load, so they are reported but not fatal.
    ok = not (missing_images or missing_masks or size_mismatch or bad_mode or unreadable or invalid_labels)
    count_matches = expected is None or len(ids) == expected
    return {
        "split": split,
        "num_ids": len(ids),
        "expected_num_ids": expected,
        "count_matches_expected": count_matches,
        "duplicate_ids": len(ids) - len(set(ids)),
        "num_images_found": len(ids) - len(missing_images),
        "num_masks_found": len(ids) - len(missing_masks),
        "missing_images": missing_images[:20],
        "missing_masks": missing_masks[:20],
        "num_missing_images": len(missing_images),
        "num_missing_masks": len(missing_masks),
        "image_mask_size_mismatches": size_mismatch[:20],
        "num_size_mismatches": len(size_mismatch),
        "non_indexed_masks": bad_mode[:20],
        "non_rgb_images": bad_image_mode[:20],
        "unreadable_files": unreadable[:20],
        "image_size_distribution_top10": dict(image_sizes.most_common(10)),
        "mask_size_distribution_top10": dict(mask_sizes.most_common(10)),
        "unique_label_values": labels_present,
        "invalid_label_values": invalid_labels,
        "invalid_label_examples": {str(k): v[:5] for k, v in invalid_examples.items()},
        "pixel_counts_per_label": {str(l): int(pixel_counts[l]) for l in labels_present},
        "pixel_counts_per_class_name": {
            (VOC_CLASS_NAMES[l] if l < len(VOC_CLASS_NAMES) else ("Ignored / Void" if l == ignore_index else f"INVALID {l}")):
            int(pixel_counts[l]) for l in labels_present
        },
        "passed": bool(ok and count_matches and len(ids) == len(set(ids))),
    }


def verify_dataset(root: Path, num_classes: int = 21, ignore_index: int = 255,
                   expected_counts: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    expected_counts = EXPECTED_COUNTS if expected_counts is None else expected_counts
    report: Dict[str, Any] = {"root": str(root), "passed": False, "errors": []}
    required = [root / "JPEGImages", root / "SegmentationClass", root / "ImageSets" / "Segmentation"]
    for path in required:
        if not path.is_dir():
            report["errors"].append(f"Missing directory: {path}")
    if report["errors"]:
        return report
    report["total_jpeg_files"] = len(list((root / "JPEGImages").glob("*.jpg")))
    report["total_segmentation_class_pngs"] = len(list((root / "SegmentationClass").glob("*.png")))
    report["segmentation_object_dir_present_but_unused"] = (root / "SegmentationObject").is_dir()
    report["splits"] = {}
    for split in ("train", "val"):
        try:
            report["splits"][split] = inspect_split(root, split, num_classes, ignore_index, expected_counts.get(split))
        except FileNotFoundError as exc:
            report["errors"].append(str(exc))
    if len(report["splits"]) == 2:
        overlap = set(load_split_ids(root, "train")) & set(load_split_ids(root, "val"))
        report["train_val_overlap_count"] = len(overlap)
        report["train_val_overlap_examples"] = sorted(overlap)[:10]
        if overlap:
            report["errors"].append(f"Train/val overlap: {len(overlap)} IDs.")
    for split, info in report["splits"].items():
        reasons = []
        if not info["count_matches_expected"]:
            reasons.append(f"{info['num_ids']} IDs but {info['expected_num_ids']} expected")
        for label, key in (("missing images", "num_missing_images"), ("missing masks", "num_missing_masks"),
                           ("image/mask size mismatches", "num_size_mismatches"), ("duplicate IDs", "duplicate_ids")):
            if info[key]:
                reasons.append(f"{info[key]} {label}")
        if info["non_indexed_masks"]:
            reasons.append(f"{len(info['non_indexed_masks'])}+ non-indexed masks (e.g. {info['non_indexed_masks'][:2]})")
        if info["unreadable_files"]:
            reasons.append(f"{len(info['unreadable_files'])}+ unreadable files (e.g. {info['unreadable_files'][:2]})")
        if info["invalid_label_values"]:
            reasons.append(f"invalid label values {info['invalid_label_values']} "
                           f"(e.g. in {info['invalid_label_examples']})")
        if reasons:
            report["errors"].append(f"Split '{split}': " + "; ".join(reasons) + ".")
    report["passed"] = not report["errors"]
    return report


def print_report(report: Dict[str, Any]) -> None:
    print("\n" + "=" * 70 + f"\nDataset root: {report['root']}\n" + "=" * 70)
    for key in ("total_jpeg_files", "total_segmentation_class_pngs", "segmentation_object_dir_present_but_unused"):
        if key in report:
            print(f"{key}: {report[key]}")
    for split, info in report.get("splits", {}).items():
        print(f"\n[{split}] ids={info['num_ids']} (expected {info['expected_num_ids']}) "
              f"images found={info['num_images_found']} masks found={info['num_masks_found']}")
        print(f"  missing images={info['num_missing_images']} missing masks={info['num_missing_masks']} "
              f"size mismatches={info['num_size_mismatches']} duplicate ids={info['duplicate_ids']}")
        print(f"  non-indexed masks={len(info['non_indexed_masks'])} unreadable={len(info['unreadable_files'])}")
        print(f"  unique label values: {info['unique_label_values']}")
        print(f"  invalid label values: {info['invalid_label_values']}")
        print(f"  most common image sizes (WxH): {info['image_size_distribution_top10']}")
        print(f"  -> {'PASS' if info['passed'] else 'FAIL'}")
    if "train_val_overlap_count" in report:
        print(f"\ntrain/val overlap: {report['train_val_overlap_count']}")
    print("\nERRORS:" if report["errors"] else "\nNo errors found.")
    for err in report["errors"]:
        print(f"  - {err}")
    print(f"\nOVERALL: {'PASS' if report['passed'] else 'FAIL'}")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the PASCAL VOC 2012 segmentation dataset.")
    parser.add_argument("--config", default=None, help="Experiment YAML (default: configs/experiment.yaml).")
    parser.add_argument("--data-root", default=None, help="Override dataset.root (path to VOCdevkit/VOC2012).")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, data_root=args.data_root)
    except ConfigError as exc:
        print(f"ERROR: {exc}")
        return 1
    root = resolve_path(config["dataset"]["root"])
    if not (root / "ImageSets" / "Segmentation").is_dir():
        print(f"ERROR: VOC2012 not found at: {root}")
        candidates = find_voc_candidates(PROJECT_ROOT)
        if candidates:
            print("Possible VOC2012 locations found inside the project:")
            for cand in candidates:
                print(f"  - {cand}")
            print("Re-run with e.g.:  python -m src.check_dataset --data-root \"<path above>\"\n"
                  "and/or set dataset.root in configs/experiment.yaml.")
        else:
            print("No VOC2012 directory found inside the project. Extract VOCtrainval_11-May-2012.tar so that\n"
                  "  data/VOCdevkit/VOC2012/{JPEGImages,SegmentationClass,ImageSets/Segmentation}\n"
                  "exists, or point --data-root / dataset.root to your extracted VOC2012 folder.")
        return 1
    report = verify_dataset(root, config["dataset"]["num_classes"], config["dataset"]["ignore_index"])
    print_report(report)
    out = resolve_path(config["paths"]["results_dir"]) / "dataset_verification.json"
    save_json(report, out)
    print(f"Saved: {out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
