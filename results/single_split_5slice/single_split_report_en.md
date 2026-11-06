# Mean pooling vs 2.5D with Youden-J threshold — single split, no cross-validation

**Setup.** 236 subjects (no OASIS-1/2 overlap; OASIS-2 first visit only). **One** subject-level split stratified by dataset × label: train 165 / validation 35 / **test 36** (18 CDR>0, 6 from OASIS-2); no folds. Settings: {fixed 78–82, Canny c±2} × {2.5D, feature fusion with mean pooling}; small CNN and ResNet18; 3 model seeds each (seeds only change initialisation, not the split). The Youden-J threshold is chosen on the **validation** set and applied once to the test set; 0.5 is shown for comparison. Mean ± SD over the 3 seeds.

| Backbone | Setting | Test AUC | Accuracy @0.5 | Accuracy @Youden | Sens. / Spec. @Youden | Youden threshold |
|---|---|---|---|---|---|---|
| Small CNN | 2.5D, fixed | 0.547 ± 0.008 | 0.528 ± 0.028 | 0.537 ± 0.042 | 0.44 / 0.63 | 0.49 ± 0.01 |
| Small CNN | Mean pooling, fixed | 0.590 ± 0.045 | 0.509 ± 0.016 | 0.565 ± 0.042 | 0.52 / 0.61 | 0.50 ± 0.03 |
| Small CNN | 2.5D, Canny | 0.604 ± 0.175 | 0.528 ± 0.048 | 0.546 ± 0.058 | 0.67 / 0.43 | 0.50 ± 0.01 |
| Small CNN | Mean pooling, Canny | 0.488 ± 0.111 | 0.509 ± 0.016 | 0.472 ± 0.048 | 0.17 / 0.78 | 0.50 ± 0.01 |
| ResNet18 | 2.5D, fixed | 0.594 ± 0.073 | 0.565 ± 0.042 | 0.537 ± 0.016 | 0.78 / 0.30 | 0.23 ± 0.09 |
| ResNet18 | Mean pooling, fixed | 0.695 ± 0.124 | 0.630 ± 0.112 | 0.630 ± 0.105 | 0.89 / 0.37 | 0.31 ± 0.21 |
| ResNet18 | 2.5D, Canny | 0.691 ± 0.101 | 0.639 ± 0.111 | 0.565 ± 0.042 | 0.59 / 0.54 | 0.60 ± 0.17 |
| ResNet18 | Mean pooling, Canny | 0.671 ± 0.010 | 0.694 ± 0.000 | 0.630 ± 0.032 | 0.93 / 0.33 | 0.28 ± 0.23 |

Confusion matrices (3-seed ensemble, 36 test subjects): `single_split_confusion.png`.

**Take-aways**
- With a 36-subject test set one subject = 2.8 percentage points of accuracy; the 95% interval of an accuracy near 0.65 is about ±0.16, so **none of the differences between settings can be established** (seed-to-seed SDs of 0.1–0.17 in AUC confirm this).
- The small CNN is at chance level (AUC 0.49–0.60, accuracy 0.47–0.57). ResNet18 is better (AUC 0.59–0.70, accuracy 0.57–0.69 at 0.5); mean pooling is ahead of 2.5D for fixed slices and level for Canny slices, but within noise.
- **Youden's J did not help on the test set.** For ResNet18 accuracy fell in all 4 settings (−0.00 to −0.07); for the small CNN it changed by −0.04 to +0.06. Youden mostly pushed sensitivity up (to 0.78–0.93) at the cost of specificity (0.30–0.54), and the ResNet18 thresholds are unstable across seeds (0.23–0.60, SD up to 0.23) because the validation set has only 35 subjects.
- This agrees with the earlier cross-validated analysis (236 subjects, 10 folds), which is more reliable: mean pooling ≥ 2.5D, Youden gives no reliable gain.

**Caveats.** One split, 36 test subjects; the result depends on which subjects fall in the test set. Labels are CDR-based dementia status, not pathology-confirmed Alzheimer's disease.
