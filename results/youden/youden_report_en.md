# Youden's J threshold — mean-pooling trial (no-overlap cohort)

**Question.** Does choosing the decision threshold with Youden's J statistic (J = sensitivity + specificity − 1) improve accuracy compared with the default threshold 0.5?

**Setup.** 236 subjects (OASIS-1 198 + 38 OASIS-2 not present in OASIS-1; OASIS-2 first visit only). Repeated stratified 5-fold CV (2 repeats), identical folds/seeds to the mean-pooling trial. **Leak-free threshold selection:** for every fold the Youden threshold is computed on that fold's inner *validation* split (~30 subjects) and then applied to the untouched *test* fold, so the test data never influence the threshold. Pooled accuracy / confusion matrices use the out-of-fold score margin (probability − fold threshold) averaged over the 2 repeats. (Re-running reproduced the earlier AUCs exactly.)

## Results (out-of-fold, n = 236)

| Backbone | Slices | Method | Mean threshold (SD; range) | Accuracy @0.5 | Accuracy @Youden | Sens. / Spec. @Youden | Δ fold accuracy (95% CI) |
|---|---|---|---|---|---|---|---|
| Small CNN | Fixed 78–82 | 2.5D | 0.475 (0.03; 0.39–0.50) | 0.597 | 0.631 | 0.75 / 0.51 | +0.047 (−0.016, +0.109) |
| Small CNN | Fixed 78–82 | Late fusion, mean pooling | 0.484 (0.03; 0.42–0.54) | 0.610 | 0.653 | 0.71 / 0.60 | +0.058 (−0.034, +0.150) |
| Small CNN | Canny c±2 | 2.5D | 0.481 (0.02; 0.44–0.50) | 0.551 | 0.589 | 0.69 / 0.49 | +0.015 (−0.035, +0.065) |
| Small CNN | Canny c±2 | Late fusion, mean pooling | 0.494 (0.02; 0.47–0.53) | 0.572 | 0.576 | 0.51 / 0.64 | −0.008 (−0.067, +0.051) |
| ResNet18 | Fixed 78–82 | 2.5D | 0.413 (0.25; 0.15–0.91) | 0.623 | 0.627 | 0.73 / 0.53 | −0.008 (−0.041, +0.024) |
| ResNet18 | Fixed 78–82 | Late fusion, mean pooling | 0.386 (0.23; 0.15–0.87) | 0.661 | 0.636 | 0.65 / 0.62 | −0.005 (−0.061, +0.052) |
| ResNet18 | Canny c±2 | 2.5D | 0.386 (0.22; 0.07–0.83) | 0.555 | 0.572 | 0.69 / 0.45 | −0.025 (−0.068, +0.018) |
| ResNet18 | Canny c±2 | Late fusion, mean pooling | 0.293 (0.14; 0.12–0.55) | 0.661 | 0.636 | 0.76 / 0.51 | −0.015 (−0.053, +0.023) |

(Δ = per-fold accuracy at the validation-selected Youden threshold minus accuracy at 0.5, mean of 10 folds.)

## Take-aways
1. **No reliable gain in accuracy.** All 95% intervals for the change include 0. The small CNN improves slightly in 3 of 4 settings (+0.015 to +0.058), ResNet18 is slightly *worse* in all 4 (−0.005 to −0.025).
2. **Youden mainly shifts the balance between sensitivity and specificity**, it does not improve discrimination (AUC is unchanged). E.g. small CNN, Canny 2.5D: sensitivity 0.21 → 0.69, specificity 0.89 → 0.49.
3. **The Youden threshold is unstable for ResNet18** (fold thresholds range from 0.07 to 0.91, SD 0.14–0.25): with ~30 validation subjects the optimum of J is very noisy. The small CNN's thresholds stay near 0.5 (0.39–0.54) because its probabilities are weakly separated.
4. **Optimistic version overstates the benefit.** Choosing the threshold on the same out-of-fold predictions that are then scored gave, e.g., ResNet18 fixed + mean pooling 0.669 (vs 0.661 at 0.5); with a leak-free threshold the same setting falls to 0.636.
5. **Practical conclusion.** With n ≈ 236 and AUC of 0.6–0.7, threshold tuning is not a remedy: it trades sensitivity for specificity but accuracy stays ≈ 0.55–0.65. If sensitivity matters, choose the operating point on purpose (e.g. fixed sensitivity) and report it, rather than relying on Youden's J.

**Caveats.** Validation splits are small (~30 subjects), so thresholds are noisy by construction; 10 folds are not independent; several comparisons. Labels are CDR-based dementia status, not pathology-confirmed Alzheimer's disease.

Figures: `youden_confusion.png` (confusion matrices at the Youden threshold, 0.5 in the titles for comparison), `youden_thresholds.png` (spread of fold thresholds). Table: `youden_summary.csv`.
