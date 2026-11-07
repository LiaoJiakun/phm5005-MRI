# OASIS-1 10-slice: nested hyper-parameter tuning

Branch `experiment/oasis1-10slice-tuned`. Code: `scripts/tune_10slice.py` (run), `scripts/analyze_tuning.py` (tables, figures). Protocol and rationale (Chinese, for the defence): `docs/tuning_explained_zh.md`. Settings of every candidate: `protocol.json`.

## Protocol

* Outer folds identical to the main notebook (stratified 5-fold, seed 31000, 198 subjects, each tested once).
* Candidates fixed before the run: ResNet18 R1–R6, CNN C1–C4 (R1 / C1 = main-notebook settings).
* Inner selection: 3-fold CV on the outer training portion only; the configuration with the highest mean inner AUC is chosen separately for each outer fold, backbone and slice selection.
* Final model per outer fold: chosen configuration retrained with 3 seeds (33000 / 43000 / 53000 + fold); test probabilities averaged. The test fold is used once.
* Run on Apple MPS (not bit-for-bit reproducible); the main notebook stays on CPU.

## Results (threshold 0.5, same subjects and folds as the main notebook)

| Model | Untuned pooled AUC [95% CI] | Tuned (3-seed) pooled AUC [95% CI] | Tuned − untuned [paired bootstrap 95% CI] | Tuned single seed, mean fold AUC | Acc / Sens / Spec (tuned) |
|---|---|---|---|---|---|
| ResNet18, Fixed | 0.703 [0.627, 0.769] | 0.738 [0.666, 0.806] | +0.035 [−0.013, +0.089] | 0.700 | 0.646 / 0.63 / 0.66 |
| ResNet18, Canny | 0.666 [0.587, 0.742] | 0.710 [0.634, 0.784] | +0.044 [−0.018, +0.111] | 0.618 | 0.657 / 0.69 / 0.62 |
| CNN Scratch, Fixed | 0.704 [0.627, 0.775] | 0.705 [0.629, 0.776] | +0.001 [−0.040, +0.045] | 0.713 | 0.646 / 0.77 / 0.52 |
| CNN Scratch, Canny | 0.707 [0.634, 0.779] | 0.691 [0.618, 0.765] | −0.015 [−0.059, +0.028] | 0.675 | 0.636 / 0.74 / 0.53 |

Tuned Canny − Fixed: ResNet18 −0.028 [−0.092, +0.037], CNN −0.014 [−0.058, +0.031].

* Tuning gives no reliable improvement: every interval includes 0.
* The ResNet18 gain comes from the 3-seed ensemble, not from the selected settings: the tuned single-seed models are not better than the untuned ones.
* Inner CV mostly chose per-slice probability averaging (R4 / C4), but on the outer folds it was not better than the base settings (`all_configs_outer.csv`); differences between candidates are within noise.
* The main (untuned) results remain the primary result; tuning is reported as a sensitivity analysis.

The transparency stage (every candidate on the outer folds, single seed) is complete for ResNet18 and covers only fold 0 for the CNN (stopped manually); it is descriptive and is not used for any selection.

Files: `tuned_vs_untuned.csv`, `fold_metrics_tuned.csv`, `chosen_configs.csv`, `inner_scores.csv`, `inner_scores_mean.csv`, `paired_fixed_vs_canny_tuned.csv`, `all_configs_outer.csv`, `figures/roc_tuned_vs_untuned.png`, `figures/fold_auc_tuned_vs_untuned.png`, `figures/confusion_tuned.png`. Subject-level predictions and the run log stay local.
