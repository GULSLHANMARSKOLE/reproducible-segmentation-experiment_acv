# Data Contract — PASCAL VOC 2012 Semantic Segmentation

This document is the binding specification of how data enter the experiment. The code in
`src/dataset.py` implements it; `src/check_dataset.py` verifies it on the real files.

> **Verification status:** the dataset has **not** been verified by the author of this file. Run
> `python -m src.check_dataset` locally; it writes `reports/results/dataset_verification.json`.
> Record the measured numbers in the table in §9 after you run it.

## 1. Dataset and provenance

| Item | Value |
|---|---|
| Dataset | PASCAL VOC 2012 (Visual Object Classes Challenge 2012), Segmentation task |
| Archive | `VOCtrainval_11-May-2012.tar` (official host: <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/>) |
| Task | Semantic segmentation (one class label per pixel) |
| Image source | Flickr photographs collected by the VOC organisers (use is subject to the VOC/Flickr terms) |
| Expected local root | `data/VOCdevkit/VOC2012` (configurable: `dataset.root` or `--data-root`) |

Folders used: `JPEGImages` (inputs), `SegmentationClass` (semantic masks), `ImageSets/Segmentation/{train,val}.txt` (splits).

Folders **not** used: `SegmentationObject` (instance masks — out of scope), and any externally augmented
“trainaug/SBD” annotations. `Annotations/` (detection XML) and `ImageSets/Main|Layout|Action` are also unused.

## 2. Splits

| Split | Source file | Expected images | Use |
|---|---|---|---|
| Training | `ImageSets/Segmentation/train.txt` | 1,464 | Fitting model weights |
| Validation | `ImageSets/Segmentation/val.txt` | 1,449 | Per-epoch monitoring, best-checkpoint selection, final reported evaluation |

* The official splits are used unchanged. No artificial test set is created.
* VOC2012 **test** segmentation annotations are not public; **no test-set result is claimed anywhere**.
* Train and validation IDs must be disjoint (checked by `check_dataset` and again at training start).
* Because the same validation split is used for checkpoint selection *and* for the reported numbers, the reported
  best-checkpoint mIoU is optimistically biased. The final-epoch validation mIoU (from the training log) is
  reported next to it for transparency.

## 3. Classes and labels

21 semantic classes + one ignore label. Masks store these integer values directly.

| ID | Class | | ID | Class |
|---|---|---|---|---|
| 0 | Background | | 11 | Dining Table |
| 1 | Aeroplane | | 12 | Dog |
| 2 | Bicycle | | 13 | Horse |
| 3 | Bird | | 14 | Motorbike |
| 4 | Boat | | 15 | Person |
| 5 | Bottle | | 16 | Potted Plant |
| 6 | Bus | | 17 | Sheep |
| 7 | Car | | 18 | Sofa |
| 8 | Cat | | 19 | Train |
| 9 | Chair | | 20 | TV/Monitor |
| 10 | Cow | | **255** | **Ignored / Void** |

* **Ignore index = 255.** In VOC these pixels mark object boundaries and objects flagged as “difficult”.
  They are excluded from the training loss (`CrossEntropyLoss(ignore_index=255)`) and from every metric.
* Any other value (22–254, or 21) is **invalid**. The loader raises an error naming the offending values and file;
  `check_dataset` lists them. Values are never silently remapped.

## 4. Mask handling rules (mandatory)

1. Masks are **indexed (palette, mode `P`) PNGs**. The raw pixel *indices* are read with `np.array(PIL.Image)`.
2. Masks are **never converted to RGB** (a mask with an RGB mode is rejected with an error).
3. Masks are **never normalised** (values stay integers 0–20 and 255).
4. Masks are resized with **nearest-neighbour** interpolation only — bilinear/bicubic would invent invalid in-between labels.
5. Mask tensors are `torch.int64` of shape `[H, W]`.

## 5. Image transformations

| Step | Setting |
|---|---|
| Colour mode | `convert("RGB")` |
| Resize | to **256 × 256** (`dataset.image_size`), **bilinear** (Pillow) |
| Aspect ratio | **Not preserved** — images are resized directly to a square (no padding/cropping), which stretches non-square photos. This is a deliberate simplicity choice applied identically to image and mask and to both models |
| Scaling | pixel / 255 → `[0, 1]` float32 |
| Normalisation | ImageNet statistics, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225) |
| Tensor layout | `[3, 256, 256]` float32 |

## 6. Mask transformations

| Step | Setting |
|---|---|
| Resize | to **256 × 256**, **nearest-neighbour** |
| Values | integer class IDs 0–20 and 255 preserved exactly (no new values can appear) |
| Type | `torch.int64` |

## 7. Augmentation

| Split | Augmentation |
|---|---|
| Training | Random horizontal flip, p = 0.5 — the **same** flip decision is applied to image and mask (applied after resizing) |
| Validation | None |

Nothing else (no crops, scale jitter, colour jitter) is used.

## 8. Integrity checks performed by the code

* Every ID in a split file must have a `JPEGImages/<id>.jpg` and a `SegmentationClass/<id>.png`; otherwise loading stops with an error listing examples.
* Image and mask must have identical width × height before resizing; otherwise an error is raised.
* Split files must have no duplicate IDs; train and validation must not overlap.
* `check_dataset` additionally reports: split sizes vs. expected (1,464 / 1,449), size distribution, mask modes, unique label values, invalid labels, per-class pixel counts, and the number of files in the folders.

## 9. Measured dataset facts (to be filled in after running `python -m src.check_dataset`)

| Fact | Expected | Measured (fill in) |
|---|---|---|
| Training IDs | 1,464 | _TODO_ |
| Validation IDs | 1,449 | _TODO_ |
| Missing images / masks | 0 / 0 | _TODO_ |
| Image–mask size mismatches | 0 | _TODO_ |
| Non-indexed masks | 0 | _TODO_ |
| Unique label values | 0–20, 255 | _TODO_ |
| Invalid label values | none | _TODO_ |
| Train ∩ validation | 0 | _TODO_ |

## 10. Known limitations of the contract

* Square resizing distorts object shapes; thin or small objects lose resolution at 256 × 256.
* Class frequencies are strongly imbalanced (background dominates); no class re-weighting is applied.
* VOC images come from Flickr and reflect a specific set of scenes and photographic styles.
