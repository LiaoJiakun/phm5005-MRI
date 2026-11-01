# OASIS-1 + OASIS-2: fixed vs Canny slices, axial / coronal / dual-view (team note)

**Question.** Can a 2D network detect dementia (CDR>0 vs CDR=0) from single MRI slices, and does the slice choice matter?

**Experiments (same subjects, folds, seeds, hyper-parameters)**

| | Axial | Coronal | Network |
|---|---|---|---|
| A | fixed (80) | – | single-view |
| B | – | fixed (105) | single-view |
| C | fixed (80) | fixed (105) | dual-view |
| D | Canny | Canny | dual-view |

Models: small 2D CNN (from scratch) and ImageNet-pretrained ResNet18 (full fine-tune). Evaluation: repeated stratified 5-fold CV (2 repeats), subject level, stratified by dataset × label.

## Data
- **OASIS-1** (age ≥ 60): 198 subjects (98 CDR=0 / 100 CDR>0).
- **OASIS-2** (PART1 + PART2, **first visit only**): 150 subjects → 149 registered to atlas space (1 failed QC) → **111 are the same people as OASIS-1 participants and were removed** → **38 unique** (21 / 17).
- **Final cohort: 236 subjects.**

> **Data-overlap warning.** Different IDs ≠ different people. 75% of the OASIS-2 first-visit subjects have a near-identical scan in OASIS-1 (image similarity 0.95–0.997 vs ≤ 0.92 between different OASIS-1 people; same sex/age/eTIV, same CDR in 104/111 pairs). Keeping them gave a spurious AUC of 0.85–0.90. All results from the OASIS-1 + OASIS-2 runs *before* de-duplication are in folders named `*_LEAKED_overlap` and **must not be used**. See `results/oasis_overlap_check.png`.

## Final results (236 subjects, pooled out-of-fold AUC, 95% bootstrap CI)

| | Small CNN | ResNet18 |
|---|---|---|
| A fixed axial 80 | 0.673 (0.60–0.74) | 0.659 (0.59–0.72) |
| B fixed coronal 105 | 0.582 (0.51–0.65) | 0.705 (0.64–0.77) |
| C fixed axial + coronal | 0.607 (0.53–0.68) | 0.685 (0.62–0.75) |
| D Canny axial + coronal | 0.563 (0.49–0.64) | 0.609 (0.53–0.68) |

- Modest discrimination overall (AUC ≈ 0.6–0.7); most differences are within noise (CI half-width ≈ 0.06).
- Canny-selected slices are lowest for both models (they sit in the cerebellum/posterior fossa).
- A second view does not clearly beat the best single view (ResNet18: C − A = +0.03, CI −0.02…+0.08).
- Only nominal positive paired difference: ResNet18 B − A = +0.054 (CI 0.006–0.102) – treat as a hypothesis.
- The small CNN mostly predicts CDR=0 at threshold 0.5 (poor calibration) – compare by AUC.

## 5-slice comparison (Method 1: 2.5D vs Method 2: late fusion; fixed 78–82 vs Canny c±2)
Same 236-subject cohort and folds. ResNet18: late fusion on fixed slices is best (AUC 0.740, CI 0.68–0.80) vs 0.673 (2.5D) and 0.679 (single slice); small CNN: no difference (0.59–0.65). Fixed beats Canny everywhere. Details: `results/five_slice/summary_en.md`, `five_slice_results.png`.

## Mean-pooling trial (late fusion with mean pooling vs 2.5D; fixed 78–82 vs Canny c±2)
No-overlap cohort (236), repeated stratified 5-fold CV. ResNet18 accuracy: mean pooling 0.661 (fixed) / 0.661 (Canny) vs 2.5D 0.623 / 0.555 vs concat 0.669 / 0.619. Mean pooling ≥ 2.5D in all cells (only ResNet18-Canny AUC clearly > 0: +0.078, CI 0.014–0.142); ≈ concatenation. Notebook: `notebooks/OASIS_5slice_meanpool_trial.ipynb`; tables/figures: `results/meanpool/`.

## Best model vs the same model with Canny slices (teacher-style notebook)
`notebooks/OASIS_best_model_vs_canny.ipynb`: ResNet18 late fusion (concat) on fixed 78–82 (AUC 0.740, acc 0.669) vs identical model on Canny c±2 (AUC 0.657, acc 0.619); paired AUC difference −0.088 (95% CI −0.177…+0.001, p = 0.064): suggestive, not conclusive. Code follows the course notebooks (explicit loop, checkpoint file, early stopping, torchmetrics, Grad-CAM); fold-0 results reproduce the cross-validation exactly.

## MRI data augmentation and square-padded axial slices (best model: ResNet18, late fusion by concatenation, fixed 78–82)
- **Aspect ratio:** axial slices (208 × 176) used to be resized straight to 128 × 128 (squashed by ~18%); now zero-padded to 208 × 208 first. No reliable effect: fixed 0.740 → 0.726 (paired Δ −0.009, p = 0.82), Canny 0.657 → 0.688 (Δ +0.027, p = 0.43). The Canny-selected slice changed for 56% of subjects.
- **Augmentation** (training only, one parameter set per subject for all 5 slices; same folds/seeds): rotation ±5° baseline vs G (rotation ±10°, shift ±5%, scale ±5%), I (+ bias field, gamma, noise, blur) and GI. None helped (Δ AUC vs baseline: G −0.02…−0.03, I −0.04…−0.05, GI +0.00…−0.05); intensity augmentation was slightly worse in both runs (p = 0.037 once, not conclusive with six comparisons).
- Details: `results/augmentation_square/summary_en.md`. The other experiments were **not** re-run on the square-padded slices. Slice axis / orientation / index checks: `scripts/verify_slices.py`.

## Layout
```
data/raw/            OASIS-1 disc archives (symlinks), oasis_cross-sectional.xlsx, OASIS-2 RAW tars + demographics
data/extracted/      OASIS-1 T88 volumes;  data/oasis2/extracted|registered/  OASIS-2 first-visit scans (raw / atlas space)
data/combined_full/  final cohort (236): combined_cohort.csv, combined_slices.npz, oasis2_overlap_report.csv
notebooks/           OASIS1_OASIS2_classification.ipynb  (executed, course-template format)
scripts/             oasis_lib.py (shared code), register_oasis2.py, build_combined.py, final_cv.py, final_analyze.py, ...
results/final_cv/    summary.csv, paired.csv, confusion.csv, final_results.png, final_confusion.png, discussion.md
results/*_LEAKED_overlap/  invalid (duplicated people) – do not use
```

## Reproduce
```bash
cd scripts
python register_oasis2.py                       # OASIS-2 first visit -> atlas space (+ --robust --only <ID> if QC fails)
python build_combined.py                        # cohort + image-level de-duplication + slices
python final_cv.py cnn & python final_cv.py resnet   # ~1 h on CPU
python final_analyze.py && python overlap_figure.py
python build_notebook.py                        # then execute the notebook
```
Requires Python 3.10 with `torch torchvision scikit-learn nibabel SimpleITK opencv-python-headless(4.11) pandas scipy matplotlib openpyxl grad-cam` and `numpy<2`.

**Note on the commit history.** The history was assembled after the work was finished, in the order in which the work was actually done. Commit timestamps therefore do not reflect when each step was performed.
