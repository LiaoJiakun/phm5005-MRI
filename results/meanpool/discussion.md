## Discussion

**Results** (236 subjects, repeated stratified 5-fold CV, out-of-fold accuracy at threshold 0.5)

| Backbone | Slices | 2.5D | **Late fusion, mean pooling** | Late fusion, concat (ref.) | Single slice (ref.) |
|---|---|---|---|---|---|
| ResNet18 | Fixed 78-82 | 0.623 | **0.661** | 0.669 | 0.627 |
| ResNet18 | Canny c±2 | 0.555 | **0.661** | 0.619 | 0.589 |
| Small CNN | Fixed 78-82 | 0.597 | **0.610** | 0.581 | 0.576 |
| Small CNN | Canny c±2 | 0.551 | **0.572** | 0.534 | 0.564 |

- **Mean pooling was at least as good as 2.5D in all four requested cells** (pooled accuracy +0.013 to +0.106; paired fold-AUC difference +0.025 to +0.078). Only one paired AUC difference is clearly away from zero: ResNet18 on Canny slices (+0.078, 95% CI 0.014 to 0.142, better in 9 of 10 folds). The other intervals include 0, so for the fixed slices the evidence is suggestive, not conclusive.
- **Mean pooling vs concatenation is a tie.** For ResNet18 it is slightly worse on fixed slices (AUC -0.034) and slightly better on Canny slices (+0.047), both intervals include 0; accuracy differs by at most 0.05. The parameter-free average therefore loses nothing measurable - consistent with the five slices being neighbouring and highly redundant, so "which slice" carries little information.
- **Mean pooling removes the fixed-vs-Canny gap for ResNet18** (accuracy 0.661 for both slice sets, AUC 0.713 vs 0.703), whereas 2.5D and concatenation were clearly worse on the Canny slices.
- **Confusion matrices.** ResNet18 with mean pooling is fairly balanced (sensitivity 0.55-0.56, specificity 0.77); 2.5D on Canny slices detects almost nothing in the small CNN (sensitivity 0.21). The small CNN is generally biased toward CDR=0 at the 0.5 threshold, so compare models by AUC as well as accuracy.

**Limitations.** 236 subjects: accuracy 95% CIs are about +-0.06, so differences of a few percentage points cannot be resolved; the 10 folds are not independent and several comparisons were made. Accuracy depends on the 0.5 threshold (probabilities are not calibrated). Accuracies of 0.57-0.67 and AUCs of 0.60-0.74 mean none of the settings is a usable diagnostic tool. Labels are CDR-based dementia status, not pathology-confirmed Alzheimer's disease.
