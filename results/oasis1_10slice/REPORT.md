# OASIS-1, 10 axial slices: Fixed vs Canny, CNN Scratch vs ResNet18

Notebook: `notebooks/OASIS1_10slice_fixed_vs_canny.ipynb` (built by `scripts/build_10slice_notebook.py`; data by `scripts/build_oasis1_10slice.py`).
All numbers below are recomputed from the saved subject-level predictions (`predictions.csv`, local only).

## Method (short)

* **Data:** OASIS-1 cross-sectional, T88 masked gain-field-corrected volumes, age >= 60, CDR known: **198 subjects** (CDR=0: 98, CDR>0: 100; CDR 0.5 / 1 / 2: 70 / 28 / 2). Label: CDR=0 vs CDR>0. No OASIS-2.
* **Slices (0-based axial index, RAS):** Fixed = z 75–84. Canny = centre c in 50–110 with the most Canny edge pixels (ties -> lowest z), slices c-5 … c+4. Per slice: 1/99 percentile scaling of the brain foreground, zero-pad to square, 128 x 128. Labels are not used for slice selection. For 169 of 198 subjects the Canny window shares no slice with the fixed block.
* **Models:** one shared encoder applied to each of the 10 slices separately -> 10 feature vectors concatenated in slice order -> Dropout(0.3) -> Linear -> 2 classes (feature-level fusion by concatenation).
  * CNN Scratch: course 06-2 SimpleCNN (six 3x3 conv, 32-32-64-64-128-128, two max-pools, global average pool) **with BatchNorm after every convolution**, random init, 128-d per slice (289,890 parameters).
  * ResNet18: ImageNet weights, all layers fine-tuned, grey slice copied to 3 channels, 512-d per slice (11,186,754 parameters).
* **Validation:** stratified 5-fold CV at subject level (seed 31000), one repeat; inside each training portion 15 % validation for early stopping only. The four models use the same folds and the same model seed per fold.
* **Training:** AdamW, batch 16, weight decay 1e-4; CNN lr 1e-3 (max 35 epochs, patience 7), ResNet lr 1e-4 (max 20 epochs, patience 5); ±5° rotation of the training stacks only (same angle for all 10 slices). Hyper-parameters were fixed in advance and not tuned. Threshold 0.5, fixed in advance.

## Results

| Model | (A) mean fold AUC (sd) | (B) pooled out-of-fold AUC [95% CI] | (C) subject-averaged AUC | Acc / Sens / Spec at 0.5 |
|---|---|---|---|---|
| CNN Scratch, Fixed | 0.720 (0.074) | 0.704 [0.634, 0.772] | 0.704 | 0.662 / 0.77 / 0.55 |
| CNN Scratch, Canny | 0.716 (0.033) | 0.707 [0.636, 0.780] | 0.707 | 0.641 / 0.78 / 0.50 |
| ResNet18, Fixed | 0.719 (0.051) | 0.703 [0.628, 0.777] | 0.703 | 0.631 / 0.51 / 0.76 |
| ResNet18, Canny | 0.667 (0.078) | 0.666 [0.593, 0.736] | 0.666 | 0.616 / 0.60 / 0.63 |

(B) and (C) are identical here because with one repeat each subject has exactly one prediction.

**Paired comparisons (same folds, same subjects):**

| Comparison | Mean fold ΔAUC [t 95% CI] | Wilcoxon p | Folds first better | Pooled ΔAUC [paired bootstrap 95% CI] |
|---|---|---|---|---|
| CNN Scratch: Canny − Fixed | −0.004 [−0.062, +0.054] | 0.81 | 2/5 | +0.002 [−0.046, +0.059] |
| ResNet18: Canny − Fixed | −0.052 [−0.161, +0.056] | 0.25 | 1/5 | −0.037 [−0.116, +0.032] |
| Fixed: ResNet18 − CNN Scratch | −0.001 [−0.082, +0.081] | 1.00 | 3/5 | −0.002 [−0.065, +0.059] |
| Canny: ResNet18 − CNN Scratch | −0.049 [−0.122, +0.024] | 0.19 | 1/5 | −0.040 [−0.116, +0.027] |

**Reading:**

* All four models are above chance (AUC about 0.67–0.72) and none is reliably better than another: every interval includes 0.
* Canny selection gives no benefit: equal for the CNN, a non-significant drop for ResNet18 (−0.04 to −0.05).
* ImageNet pretraining gives no measurable gain over the small CNN at this sample size.
* At the fixed 0.5 threshold the CNN leans towards CDR>0 (sensitivity 0.77–0.78) and ResNet18 Fixed towards CDR=0 (sensitivity 0.51): the probabilities are not calibrated; AUC is the main metric.
* ResNet18 reaches its lowest validation loss after 1–5 epochs (median 2), i.e. it overfits quickly with about 134 training subjects.

**CNN without BatchNorm (first run, kept in `archive_cnn_noBN/`):** the course SimpleCNN without BatchNorm did not train on these slices (training loss stayed at 0.693 = ln 2, all outputs about 0.50, AUC 0.52–0.59). A 32-subject overfitting check showed the same with a lower learning rate or Kaiming initialisation; with BatchNorm it reached 100 % training accuracy. BatchNorm was therefore added; nothing else was changed.

## Figures (captions)

| File | Caption |
|---|---|
| `figures/method_flow.png` | Method flow: cohort -> preprocessing -> label-free slice selection -> subject-level 5-fold CV -> four models on identical folds -> subject-level evaluation; Grad-CAM on unseen test subjects. |
| `figures/model_structure.png` | Model structure drawn from the actual PyTorch modules (tensor shapes recorded with forward hooks): shared per-slice encoder, concatenation of the 10 slice features, dropout and linear classifier. |
| `figures/roc.png` | Pooled out-of-fold ROC curves of the four models (198 subjects, each tested once). |
| `figures/confusion.png` | Confusion matrices at the pre-specified threshold 0.5 (198 subjects). |
| `figures/training_curves.png` | Training/validation loss (top) and accuracy / validation AUC (bottom) per epoch, mean over the folds still training at that epoch (shaded = sd). |
| `figures/canny_centre_hist.png` | Distribution of the Canny centre c and its overlap with the fixed block 75–84. |
| `figures/gradcam_*.png` (local only) | Grad-CAM of ResNet18 on unseen test subjects of fold 0, one colour scale per subject; the per-slice mapping is checked by hook order. Descriptive only. |
| `figures/example_slices.png` (local only) | Fixed and Canny slices of one subject. |

Tables: `auc_summary.csv`, `fold_metrics.csv`, `paired_comparison.csv`, `confusion_subject_averaged.csv`, `histories.csv`, `checkpoint_index.csv`.
Validation and reproduction: `VALIDATION.md`.

---

## 中文摘要

**方法：** 只用 OASIS-1，198 人（CDR=0 98 人，CDR>0 100 人）。每人取 10 张轴位切片：固定层 75–84，或 Canny 选层（50–110 里边缘最多的一层为中心 c，取 c−5 到 c+4；不用标签）。每张切片分别经过同一个编码器（CNN Scratch 或 ImageNet ResNet18），10 个特征按顺序拼接后接 Dropout 和线性层（特征拼接融合）。受试者级分层 5 折交叉验证，跑 1 次；四个模型用完全相同的折和种子；参数事先固定、没有调；阈值固定 0.5。

**结果：** 四个模型 AUC 都在 0.67–0.72 之间，都高于随机，但两两之间的差别置信区间都包含 0，没有哪个模型可靠地更好。Canny 选层没有带来好处（CNN 持平，ResNet18 略低但不显著）；ImageNet 预训练在这个样本量下没有可测量的优势。0.5 阈值下 CNN 偏向判 CDR>0，ResNet18 固定层偏向判 CDR=0，概率没有校准，所以以 AUC 为主。

**CNN 的 BatchNorm：** 课程原版 SimpleCNN（没有 BatchNorm）在这些切片上完全训练不动（损失停在 0.693），加上 BatchNorm 后正常训练；其他设置不变。失败的第一次结果保留在 `archive_cnn_noBN/`。
