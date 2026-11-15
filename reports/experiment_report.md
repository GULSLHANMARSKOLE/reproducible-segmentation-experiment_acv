# Build a Reproducible Segmentation Experiment — Experiment Report

> **STATUS: TEMPLATE — NO EXPERIMENT HAS BEEN RUN YET.**
> Every value that depends on training or evaluation is a clearly marked placeholder
> (`PLACEHOLDER` or an `AUTO` block). After running the experiment, execute
> `python -m src.report_tables --update-report` to fill the `AUTO` blocks **from the real result files**, then
> complete the remaining `TODO` items by hand. Do not write numbers that are not in `reports/results/`.

## 1. Introduction

Semantic segmentation assigns a class label to every pixel of an image. Encoder–decoder networks such as U-Net
(Ronneberger et al., 2015) are a standard family of architectures for it. This report documents a small, controlled
experiment on PASCAL VOC 2012: two otherwise identical networks are trained under an identical budget, and the effect of
one architectural change is measured. The aim is a clean, reproducible comparison — not the highest possible accuracy.

## 2. Project objective

**Question:** does changing the dilation rate of one convolution in the bottleneck of a compact U-Net from 1 to 2
improve semantic-segmentation performance on PASCAL VOC 2012 at 256 × 256 resolution?

The experiment (i) prepares and verifies the data, (ii) trains a baseline and a one-change variant from scratch with
the same budget, (iii) evaluates both with identical procedures (per-class IoU, mIoU, parameter count, inference
latency), and (iv) documents visual evidence, failures and reproducibility settings.

## 3. Dataset description

PASCAL VOC 2012 Segmentation (Everingham et al., 2010; VOC2012 challenge). 21 classes (background + 20 object
categories) with an additional void label (255) for object boundaries and difficult objects. Official splits:
training = 1,464 images, validation = 1,449 images (counts to be confirmed by `src.check_dataset`; see `data_contract.md` §9).
No `trainaug`/SBD annotations and no `SegmentationObject` masks are used. VOC2012 test segmentation annotations are not
public, so there is **no test-set evaluation**; all reported numbers are on the official validation split.

## 4. Data contract

The full contract is in [`data_contract.md`](data_contract.md). Summary: class IDs 0–20 + ignore index 255; masks are
read as raw palette indices (never RGB, never normalised) and resized with nearest-neighbour; unexpected labels raise
errors; official `train.txt` / `val.txt` are used unchanged; train/validation are disjoint.

**Dataset verification result:** PLACEHOLDER — run `python -m src.check_dataset` and copy the outcome here
(`reports/results/dataset_verification.json`).

## 5. Preprocessing and transformations

| | Image | Mask |
|---|---|---|
| Read as | RGB | indexed PNG indices |
| Resize | 256 × 256, bilinear | 256 × 256, nearest-neighbour |
| Scaling / normalisation | /255, ImageNet mean/std | none |
| Training augmentation | random horizontal flip p = 0.5 (identical flip for image and mask) | same flip |
| Validation augmentation | none | none |
| Output type | float32 `[3,256,256]` | int64 `[256,256]` |

Aspect ratio is not preserved (direct square resize).

## 6. Baseline architecture

Compact U-Net implemented from scratch in PyTorch (no pretrained weights); source in `src/models/`.

| Stage | Operation | Channels | Resolution (for 256 × 256 input) |
|---|---|---|---|
| Encoder 1 | 2 × (3×3 conv, BN, ReLU) → 2×2 max-pool | 32 | 256 → 128 |
| Encoder 2 | same | 64 | 128 → 64 |
| Encoder 3 | same | 128 | 64 → 32 |
| Bottleneck | 2 × (3×3 conv, BN, ReLU), **dilation 1, 1** | 256 | 32 |
| Decoder 1 | 2×2 transposed conv (×2) → concat skip (enc 3) → 2 × (3×3 conv, BN, ReLU) | 128 | 32 → 64 |
| Decoder 2 | same with skip enc 2 | 64 | 64 → 128 |
| Decoder 3 | same with skip enc 1 | 32 | 128 → 256 |
| Head | 1×1 conv | 21 | 256 |

Output: raw logits `[B, 21, 256, 256]` (softmax is applied only implicitly inside the loss / by argmax at prediction time).
Convolutions have no bias (followed by BatchNorm). Weights use PyTorch default initialisation.

## 7. Controlled dilation modification

**The one and only difference (Model B, `dilated`):** the **second** 3×3 convolution of the bottleneck uses
**dilation = 2** (padding = 2) instead of dilation = 1.

```
baseline:  bottleneck.conv1 (3x3, d=1)   bottleneck.conv2 (3x3, d=1)
dilated :  bottleneck.conv1 (3x3, d=1)   bottleneck.conv2 (3x3, d=2)   <-- only change
```

* Defined in code by `BOTTLENECK_DILATIONS = (1, 1)` (`src/models/baseline.py`) vs. `(1, 2)` (`src/models/dilated.py`).
* Dilation adds **no parameters**: both models have identical layer shapes and parameter count (asserted by `tests/test_models.py`,
  which also asserts that `bottleneck.conv2.conv` is the only convolution whose dilation/padding differs).
* The effective extent of that kernel grows from 3 to 5 feature-map cells (k + (k−1)(d−1)). The bottleneck operates at
  1/8 of the input resolution, so this is 8 input pixels per cell, i.e. the receptive field at the bottleneck output grows
  by 16 input pixels per axis relative to the baseline (jump 8 × (5 − 3)). This is a statement about the architecture,
  not about measured performance.
* Unchanged: encoder, decoder, skip connections, channel widths, number of classes, loss, optimizer, no pooling/attention/boundary branch.
* With the same seed both models start from identical initial weights (asserted by a test).

## 8. Experimental configuration

All settings live in [`configs/experiment.yaml`](../configs/experiment.yaml) and are read by every script.

| Setting | Value |
|---|---|
| Seed | 42 |
| Input resolution | 256 × 256 |
| Batch size (training) | 8 |
| Epochs | 30 (fixed) |
| Optimizer | AdamW, lr = 0.001 (constant), weight decay = 0.0001 |
| Loss | CrossEntropyLoss, `ignore_index=255` |
| Workers | 0 |
| Base channels | 32 |
| Pretrained | false |
| Validation batch size | 8 (per-epoch validation and final evaluation; identical for both models) |
| Latency measurement | batch 1, 20 warm-up + 100 timed iterations |
| Device | selected automatically (CUDA if available, else CPU); printed and recorded; no setting depends on it |

## 9. Reproducibility settings

* Central `set_seed` (`src/utils.py`): Python `random`, NumPy, PyTorch CPU and CUDA; `PYTHONHASHSEED`;
  `torch.use_deterministic_algorithms(True, warn_only=True)`; `cudnn.deterministic = True`; `cudnn.benchmark = False`;
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
* DataLoader: dedicated seeded `torch.Generator` for shuffling; `worker_init_fn` seeding Python/NumPy.
* Both models: same seed, same initial weights, same data order and augmentation stream (data RNGs are re-seeded right after model creation).
* Recorded per run (`reports/training_logs/<model>_training_summary.json`, `reports/results/environment.json`): seed, Python / PyTorch / NumPy / Pillow versions,
  OS, CUDA version, GPU or CPU name, architecture (`<model>_architecture.txt`), resolution, batch size, optimizer, lr, weight decay, augmentation, epochs, stopping rule.
* **Limitations:** bitwise-identical results are *not* guaranteed across different hardware, drivers, CUDA/cuDNN or PyTorch versions or operating
  systems; some CUDA kernels have no deterministic implementation (PyTorch only warns). Reproducibility is claimed only for the same environment.

Environment used for the experiment: PLACEHOLDER — copy from `reports/results/environment.json` after running.

## 10. Training procedure

`python -m src.train --model baseline` and `python -m src.train --model dilated`. Each run: seed → build datasets → build model → AdamW +
cross-entropy → for each of 30 epochs: train over the whole training split (shuffled), then evaluate on the whole validation split
(loss, per-class IoU, mIoU). Logged per epoch: epoch, training loss, validation loss, validation mIoU, learning rate, epoch time and
cumulative time (`reports/training_logs/<model>_training_log.csv`). Checkpoints (`outputs/checkpoints/<model>_best.pt` — best validation
mIoU; `<model>_last.pt` — final epoch) hold weights, epoch, optimizer state, configuration, seed, validation metric and model name.
Stopping rule: exactly 30 completed epochs for both models; no early stopping. Training curves: `reports/figures/training_curves.png` (after evaluation).

## 11. Comparison budget

Expected from the configuration (to be **confirmed by the measured table below**): 1,464 training images ÷ batch size 8 = 183 batches per epoch;
× 30 epochs = 5,490 optimization steps per model; 1,464 × 30 = 43,920 processed training samples per model; validation after every epoch;
256 × 256 input; batch size 8; same seed, optimizer, loss and augmentation. Both models must have equal step budgets.

Measured budget (from the training summaries — includes hardware, training time, and any interruptions/deviations):

<!-- AUTO:budget:START -->
PLACEHOLDER — run training and evaluation, then `python -m src.report_tables --update-report`.
<!-- AUTO:budget:END -->

Interruptions or deviations: PLACEHOLDER (the `Interruptions / deviations` column above is `none recorded` only if the run was uninterrupted;
add any manual observations, e.g. restarts or hardware changes, here). If the budgets of the two runs differ, this comparison must not be presented as controlled without explaining why.

## 12. Evaluation methodology

`python -m src.evaluate` loads the **best** checkpoints and evaluates **only on the official validation split** (never on training images) with the
documented validation preprocessing, identical for both models.

* Confusion matrix over all validation pixels, ground-truth `255` pixels excluded; IoU_c = TP / (TP + FP + FN) for each of the 21 classes.
* mIoU = mean of the classes with a defined IoU. A class with TP + FP + FN = 0 is *undefined* (NaN) and excluded from the mean (never counted as perfect).
* Parameter count = trainable parameters.
* Latency: batch 1, 256 × 256, eval mode, `torch.inference_mode()`, 20 warm-up and 100 timed iterations, per-iteration CUDA synchronisation, mean / median / 95th percentile.
  Only the model forward pass on an already prepared tensor is timed; **image loading and preprocessing are excluded**.
* Checkpoint selection used the same validation split that is reported (an optimistic bias); the final-epoch validation mIoU is shown alongside.

## 13. Per-class IoU results

<!-- AUTO:per_class_iou:START -->
PLACEHOLDER — not yet evaluated.
<!-- AUTO:per_class_iou:END -->

Figure: `reports/figures/per_class_iou.png` (generated by `src.evaluate`). Raw data: `reports/results/per_class_iou.csv`, `metrics.csv`, `metrics.json`.

## 14. mIoU comparison

<!-- AUTO:miou:START -->
PLACEHOLDER — not yet evaluated.
<!-- AUTO:miou:END -->

## 15. Parameter count comparison

<!-- AUTO:params:START -->
PLACEHOLDER — not yet evaluated.
<!-- AUTO:params:END -->

## 16. Inference latency comparison

<!-- AUTO:latency:START -->
PLACEHOLDER — not yet evaluated.
<!-- AUTO:latency:END -->

## 17. Qualitative results

PLACEHOLDER — after evaluation, include `reports/figures/qualitative_success.png` (best-segmented validation images) and
`reports/figures/qualitative_failure.png`. Each row shows source image, ground truth, baseline prediction and dilated prediction, with the same colour palette
(VOC palette; void = cream). Examples are selected automatically by per-image mIoU (only images whose ground truth contains foreground), not hand-picked.

![Successful examples](figures/qualitative_success.png)

## 18. Failure analysis

PLACEHOLDER — after evaluation see `reports/figures/qualitative_failure.png` and `reports/results/failure_cases.md` (VOC image IDs, per-image mIoU and
error statistics computed directly from the masks, e.g. which class replaced which). Write a short visual explanation per failure **after looking at the figure**; do not
copy statements you cannot see in the images.

![Failure examples](figures/qualitative_failure.png)

| VOC image ID | Observed error (author's visual interpretation) |
|---|---|
| TODO | TODO |

## 19. Discussion

PLACEHOLDER — to be written only after results exist. Suggested questions: Is the mIoU difference larger than what you would expect from seed noise
(you have one seed; consider running further seeds)? Which classes changed most? Did the parameter count change (it should not)? Was there a latency cost? Did the
best epoch differ? Does the final-epoch mIoU tell the same story as the best-checkpoint mIoU?

## 20. Limitations

* One dataset, one resolution, one seed, one architectural change, one training budget.
* Validation split used for both checkpoint selection and reporting.
* Direct square resizing distorts aspect ratio; 256 × 256 loses small-object detail.
* Small from-scratch network without pretraining, trained for 30 epochs: absolute scores are expected to be far below state-of-the-art pretrained models.
* Latency depends on hardware, drivers, thread settings and system load; it excludes data loading and preprocessing.
* Determinism is limited to the same software/hardware environment.

### What this experiment does **not** establish

* Results on PASCAL VOC do not establish performance on other image domains.
* Results at 256 × 256 do not establish performance at higher resolutions.
* A single seed does not establish statistical significance.
* A validation result is not an official hidden-test result.
* An improvement from dilation in this architecture does not establish that dilation always improves segmentation.
* This modest experiment does not establish state-of-the-art performance.

## 21. Conclusion

PLACEHOLDER — write after the results are in; restrict claims to what Sections 13–18 support and repeat the limitations above.

## 22. References

1. Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., & Zisserman, A. (2010). The PASCAL Visual Object Classes (VOC) Challenge. *International Journal of Computer Vision*, 88(2), 303–338.
2. Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., & Zisserman, A. *The PASCAL Visual Object Classes Challenge 2012 (VOC2012)*. <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/> (dataset; images from Flickr, subject to the dataset's terms of use).
3. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI 2015*, LNCS 9351, 234–241.
4. Yu, F., & Koltun, V. (2016). Multi-Scale Context Aggregation by Dilated Convolutions. *ICLR 2016*.
5. Long, J., Shelhamer, E., & Darrell, T. (2015). Fully Convolutional Networks for Semantic Segmentation. *CVPR 2015*.
6. Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. *ICML 2015*.
7. Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR 2015*.
8. Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR 2019*.
9. Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS 2019*.
