# Validation report and reproduction guide — OASIS-1 10-slice experiment

## 1. Versions

| Item | Value |
|---|---|
| Code base | branch `experiment/oasis1-10slice`, started from `main` at `af2dc93` |
| Data | OASIS-1 cross-sectional, `*t88_masked_gfc.img/.hdr` (Analyze), clinical spreadsheet `oasis_cross-sectional.xlsx`; one scan per subject (MR1 when a reliability rescan exists); not redistributable, kept local |
| Python / packages | Python 3.10.19; see `requirements.txt` (torch 2.5.1, torchvision 0.20.1, scikit-learn 1.7.2, numpy 1.26.4, pandas 2.3.3, opencv 4.11, nibabel 5.4.2, torchmetrics 1.9.0) |
| Hardware | macOS, CPU only (`DEVICE = cuda if available else cpu`; MPS deliberately not used so the run is reproducible) |
| Seeds | folds 31000, inner validation 32000 + fold, model 33000 + fold (identical for the four models of a fold) |

## 2. How to reproduce

```bash
pip install -r requirements.txt
```

1. Put the OASIS-1 data under `data/` (paths are relative; see the top of `scripts/build_oasis1_10slice.py`).
2. Build the cohort and the slices (writes `data/oasis1_10slice/oasis1_10slice.npz`, `results/oasis1_10slice/cohort_audit.json`, `cohort.csv`):

```bash
python scripts/build_oasis1_10slice.py
```

3. Build and run the notebook (clean kernel; about 3–4 h on CPU; resumable: finished folds are read from `predictions.csv`):

```bash
python scripts/build_10slice_notebook.py
```

```bash
jupyter nbconvert --to notebook --execute notebooks/OASIS1_10slice_fixed_vs_canny.ipynb --output OASIS1_10slice_fixed_vs_canny_executed.ipynb --ExecutePreprocessor.timeout=-1
```

`OASIS_SMOKE=1` runs one fold for two epochs into `results/oasis1_10slice_smoke/` (pipeline test only).

4. Figures `method_flow.png` and `model_structure.png`:

```bash
python scripts/make_10slice_diagrams.py
```

Saved for every run: `predictions.csv` (subject-level: model, backbone, selection, repeat, fold, subject, label, probability, prediction, threshold), `histories.csv`, `checkpoint_index.csv` (model, repeat, fold, seed, epochs run, best epoch, checkpoint file, n train/val/test), `split_table.csv`. All tables and figures are recomputed from these files. Checkpoints (`checkpoints/*.pt`), subject-level tables and brain-slice images stay local (data-use agreement); everything else is in the repository.

## 3. Checks that were run

| Check | Result |
|---|---|
| Cohort | 436 clinical rows (20 are reliability rescans of the same person); 238 rows excluded for age < 60; 198 subjects with CDR and image; same subject set as the earlier cohort file; no duplicate IDs |
| Orientation | LAS -> RAS (`nib.as_closest_canonical`), volume 176 x 208 x 176, axial slice = `v[:, :, z].T` (anterior at the top), checked visually in the notebook |
| Slices | no empty, constant or non-finite slice (asserted); Canny centre within 50–110 (asserted) |
| Partition | subject-level; train / validation / test disjoint in every fold; every subject tested exactly once (asserted); fold sizes 40/40/40/39/39 with 20 CDR>0 each; identical folds for all four models (one table) |
| Leakage | normalisation uses constants only; slice selection uses no labels; validation split used only for early stopping; threshold fixed at 0.5 before the run; test fold used once per model |
| Execution | the executed notebook has no error cell; the CNN run was redone after adding BatchNorm and the ResNet18 folds were reused unchanged |
| Grad-CAM | per-slice activations captured by a hook on `encoder.layer4`; a pre-hook asserts that the encoder calls follow the slice order 0…9 |
| Model sanity | CNN with BatchNorm fits 32 subjects to 100 % training accuracy; output probabilities range 0.05–0.97 (CNN) and 0.01–1.00 (ResNet18) |

## 4. What was run and what was not

* Run: four models (CNN Scratch, ResNet18) x (Fixed, Canny), one repeat of 5-fold CV, Grad-CAM for ResNet18 on fold 0.
* Run and replaced: CNN Scratch without BatchNorm (did not train; archived in `archive_cnn_noBN/`).
* Not run: the second CV repeat (set `N_REPEATS = 2`; repeat-0 folds are reused); hyper-parameter tuning (done separately on branch `experiment/oasis1-10slice-tuned`); OASIS-2, coronal or other views (out of scope).

## 5. Limitations

* 198 subjects and one repeat of 5-fold CV; the five folds are not independent, so intervals and p-values are optimistic. No external test set.
* CDR is a clinical rating, not a confirmed diagnosis; participants aged 60+ only.
* Hyper-parameters were taken from the earlier project and not tuned; CNN and ResNet18 use different learning rates and epoch limits, so the backbone comparison also contains this difference.
* `ReduceLROnPlateau` runs with its defaults (patience 10), longer than early stopping (patience 5 / 7), so it never lowered the learning rate.
* BatchNorm was added to the CNN after the first run had failed; the change only made the model trainable, but it was made after seeing results.
* Metrics other than AUC depend on the 0.5 threshold, and the probabilities are not calibrated.
* Grad-CAM is shown for one fold and is descriptive, not evidence of a biomarker.

## 6. Points to confirm

* AdamW instead of the course's Adam (kept from the earlier best model).
* Hyper-parameters reused from the earlier project rather than chosen for this study.
* CNN Scratch = course SimpleCNN encoder + BatchNorm, shared per slice with feature concatenation.
* One repeat of 5-fold CV for now; whether to add the second repeat.
