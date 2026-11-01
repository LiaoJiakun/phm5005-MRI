# Square-padded axial slices and MRI data augmentation — best model (ResNet18, late fusion by concatenation, fixed slices 78–82)

**Setup.** 236 subjects (OASIS-1 + non-overlapping OASIS-2, first visit), repeated stratified 5-fold CV (2 repeats), same folds and seeds throughout, so all comparisons are paired.
Preprocessing change: the axial slice (208 × 176) used to be resized straight to 128 × 128 (squashed by ~18%); it is now zero-padded to 208 × 208 first (brain width/height 0.94 → 0.81, close to real anatomy).
Augmentation (training images only, one random parameter set per subject shared by the 5 slices): baseline = rotation ±5°; **G** = rotation ±10°, shift ±5%, scale ±5%; **I** = baseline rotation + MRI-specific intensity changes (smooth bias field, gamma, noise inside the brain, mild blur); **GI** = G + I.

## 1) Effect of fixing the aspect ratio (pooled out-of-fold AUC, 95% CI)

| Slices | squashed (old) | square-padded | paired AUC difference |
|---|---|---|---|
| Fixed 78–82 | 0.740 (0.67–0.80) | 0.726 (0.66–0.79) | −0.009 (−0.05 to +0.03), p = 0.82 |
| Canny c±2 | 0.657 (0.59–0.72) | 0.688 (0.62–0.76) | +0.027 (−0.04 to +0.10), p = 0.43 |

Padding to a square **did not change performance in any reliable way**. Canny − fixed is −0.052 (CI −0.11 to +0.01, p = 0.064) with square slices vs −0.088 (p = 0.064) before: same direction, smaller gap. The Canny-selected slice changed for 56% of subjects after the preprocessing change (some jumped from cerebellum to ventricle level), showing how sensitive that rule is.

## 2) Augmentation (AUC, paired difference vs the rotation-only baseline)

| Arm | squashed (old) AUC | Δ vs baseline | square-padded AUC | Δ vs baseline |
|---|---|---|---|---|
| Baseline (rotation ±5°) | 0.740 | – | 0.726 | – |
| G geometric | 0.714 | −0.034 (p = 0.28) | 0.717 | −0.020 (p = 0.28) |
| I intensity | 0.698 | −0.040 (p = 0.064) | 0.682 | −0.048 (−0.09 to −0.01, p = 0.037) |
| GI both | 0.747 | +0.001 (p = 1.0) | 0.688 | −0.045 (p = 0.051) |

**Take-aways.** No augmentation improved the model under either preprocessing. Geometric augmentation was neutral (−0.02 to −0.03); intensity augmentation was *slightly worse* in both runs (−0.04 to −0.05; p = 0.037 is the only nominal significance, with six comparisons and correlated folds this is not conclusive). Plausible reason (not tested): the images are already atlas-registered and intensity-normalised, so extra intensity variation mostly perturbs the subtle CSF/tissue contrast the model relies on, while the models still overfit within ~7 epochs.

**Caveats.** n = 236 → AUC CI half-width ≈ 0.06; effects below ~0.05 cannot be resolved. The baseline is the best of many settings tried earlier, so it is optimistic. Labels are CDR-based dementia status, not pathology-confirmed Alzheimer's disease. The other earlier experiments (single slice, 2.5D, mean pooling, Youden) were **not** re-run on the square-padded slices.
