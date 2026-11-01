# Method 1 (2.5D) vs Method 2 (late fusion) on 5 axial slices — no-overlap cohort

**Setup.** 236 subjects (OASIS-1 198 + 38 OASIS-2 not present in OASIS-1; OASIS-2 first visit only). Repeated stratified 5-fold CV (2 repeats, subject level, same folds/seeds as before).
Slice sets: fixed 78–82, and Canny group = best Canny slice c (50–110) ± 2. Method 1 = 5 slices stacked as input channels of one network; Method 2 = each slice through the same encoder, features concatenated. Single-slice references use the same pipeline.

**Pooled out-of-fold AUC (95% CI)**

| | Small CNN | ResNet18 |
|---|---|---|
| single slice, fixed 80 | 0.648 (0.58–0.72) | 0.679 (0.61–0.74) |
| M1 2.5D, fixed 78–82 | 0.636 (0.57–0.71) | 0.673 (0.60–0.74) |
| M2 late fusion, fixed 78–82 | 0.621 (0.55–0.69) | **0.740 (0.68–0.80)** |
| single slice, Canny c | 0.599 (0.53–0.67) | 0.624 (0.55–0.69) |
| M1 2.5D, Canny c±2 | 0.597 (0.53–0.67) | 0.613 (0.54–0.68) |
| M2 late fusion, Canny c±2 | 0.586 (0.52–0.66) | 0.657 (0.59–0.73) |

**Take-aways**
- Late fusion (M2) is the best setting with ResNet18 (fixed: M1 − M2 = −0.058, CI −0.105 to −0.011, M2 better in 8/10 folds; Canny: −0.031, CI includes 0). With the small CNN there is no difference between the two methods.
- Multiple slices do not clearly beat a single slice (ResNet18 M2 vs single: +0.047 fixed, +0.032 Canny, CIs include 0; 2.5D ≈ single slice).
- Fixed 78–82 is better than the Canny group in every setting (ResNet18 M1 −0.061, M2 −0.088; CNN M2 −0.056).
- Small CNN is weak (AUC 0.59–0.65, mostly predicts CDR=0 at 0.5).

**Caveats.** n = 236 → CI half-width ≈ 0.06; 10 folds are not independent and many comparisons were made, so the M2 advantage is suggestive, not confirmed. The 5 slices are adjacent (highly redundant). Labels are CDR-based dementia status, not pathology-confirmed AD.
