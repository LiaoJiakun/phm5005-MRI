"""notebooks/OASIS_5slice_meanpool_trial.ipynb - template-format notebook (06_2 DL for Image Data Classification) for the mean-pooling trial."""
from pathlib import Path
import nbformat as nbf
ROOT = Path(__file__).resolve().parent.parent
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r'''
# Deep Learning for Brain MRI - Multi-slice Fusion with Mean Pooling (OASIS-1 + OASIS-2, no overlap)

**Question.** If we feed several neighbouring axial slices to a classifier, how should the slices be combined, and does **mean pooling** of per-slice features work better than stacking the slices as channels (**2.5D**)?

| | Fixed slices (78, 79, 80, 81, 82) | Canny slices (best slice c +- 2) |
|---|---|---|
| **2.5D CNN** (early fusion: 5 slices = 5 input channels) | yes | yes |
| **Late fusion, mean pooling** (shared encoder per slice, features averaged) | yes | yes |

Reference points: late fusion with **concatenation** and a **single slice**. Backbones: a small 2D CNN (from scratch) and an ImageNet-pretrained ResNet18 (fully fine-tuned). Reported: **accuracy and confusion matrices** (plus AUC).

> The label is dementia status from the clinical dementia rating (CDR > 0 vs CDR = 0), not a pathologically confirmed Alzheimer's diagnosis.
''')

md(r'''
## Workflow

Same workflow as the lesson: setup -> data & splitting -> architecture -> training -> evaluation.

**Data-leakage safeguards**
- **No sample overlap.** OASIS-2 shares many participants with OASIS-1 (different IDs, same brain). 111 of the 149 OASIS-2 first-visit subjects were detected by image similarity and **removed**; the final cohort has **236 subjects** (198 OASIS-1 + 38 unique OASIS-2, first visit only), asserted to contain no near-duplicate pair.
- Subject-level **repeated stratified 5-fold cross-validation** (stratified by dataset x label); inner validation split (15% of each training portion) only for early stopping.
- Fixed normalisation constants, training-only augmentation (+-5 degree rotation, the same rotation for all slices of a subject), label-free slice selection, hyper-parameters fixed in advance.
''')

md(r'''
## Dataset

Each subject is a stack of **5 consecutive axial slices** (128 x 128, atlas-registered, brain-masked):
- **Fixed**: slices 78-82.
- **Canny**: the slice `c` with the most Canny edge pixels among slices 50-110 (label-independent), plus `c-2 ... c+2`.
''')

code(r'''
import sys, inspect
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch
from IPython.display import display, Image

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
import oasis_lib as L          # folds, seeds, hyper-parameters
import oasis_5slice as S       # 5-slice datasets and models
L.set_seed(42)
print("Device:", L.DEVICE, "| hyper-parameters:", L.HPARAMS)

cohort = pd.read_csv(ROOT / "data" / "combined_full" / "combined_cohort.csv")
rep = pd.read_csv(ROOT / "data" / "combined_full" / "oasis2_overlap_report.csv")
print(f"OASIS-2 first-visit subjects checked against OASIS-1: {len(rep)} | same person (removed): {int(rep.removed.sum())} | unique (kept): {int((~rep.removed).sum())}")
summary = cohort.groupby(["dataset", "label"]).size().unstack().rename(columns={0: "CDR=0", 1: "CDR>0"}); summary["total"] = summary.sum(axis=1); summary.loc["All"] = summary.sum()
display(summary)
Z = S.Z; print({k: Z[k].shape for k in ("fixed5", "canny5")}, "| Canny centre slice: mean %.1f, range %d-%d" % (Z["canny_center"].mean(), Z["canny_center"].min(), Z["canny_center"].max()))
''')

md(r'''
### Inspecting Images

One OASIS-1 and one OASIS-2 subject: the 5 fixed slices (top) and the 5 Canny slices (bottom).
''')

code(r'''
fig, axes = plt.subplots(4, 5, figsize=(11, 9.4))
for r, (d, key) in enumerate([("OASIS1", "fixed5"), ("OASIS1", "canny5"), ("OASIS2", "fixed5"), ("OASIS2", "canny5")]):
    i = np.where(Z["dataset"] == d)[0][3]
    c = int(Z["canny_center"][i]); idx = list(range(78, 83)) if key == "fixed5" else list(range(c - 2, c + 3))
    for k in range(5):
        axes[r, k].imshow(Z[key][i, k], cmap="gray"); axes[r, k].axis("off"); axes[r, k].set_title(("%s %s\n" % (d, "fixed" if key == "fixed5" else "Canny") if k == 0 else "") + f"slice {idx[k]}", fontsize=8.5)
plt.tight_layout(); plt.show()
''')

md(r'''
## Architecture

Three ways to use the 5 slices (`SliceNet` in `scripts/oasis_5slice.py`):

**1) 2.5D (early fusion, `method="stack"`).** The slices are the 5 input channels of one 2D network. The first convolution sees all slices at once. (For ResNet18 the pretrained RGB filters are averaged and replicated to 5 channels.)

**2) Late fusion with concatenation (`method="late"`, reference).** Each slice goes through the *same* encoder; the 5 feature vectors are concatenated, the head sees `5 x d` numbers.

**3) Late fusion with mean pooling (`method="late_mean"`, this trial).** Each slice goes through the same encoder and the 5 feature vectors are **averaged element-wise** to one `d`-dimensional vector before the classifier head.

Mean pooling is **parameter-free and independent of slice order**: it tells the head *what* appears in the slab but not *which slice* showed it.
''')

code(r'''
print(inspect.getsource(S.pool_slice_features))
print(inspect.getsource(S.SliceNet))
for method in ("stack", "late", "late_mean"):
    row = {bb: f"{sum(p.numel() for p in S.SliceNet(bb, 5, method).parameters()):,}" for bb in ("cnn", "resnet")}
    print(f"{method:10s} parameters:", row)
x = torch.randn(4, 5, 128, 128); m = S.SliceNet("cnn", 5, "late_mean").eval()
assert m(x).shape == (4, 2) and torch.allclose(m(x), m(x[:, [4, 3, 2, 1, 0]]), atol=1e-5)
print("mean pooling check passed: output (4, 2); slice order does not change the prediction")
''')

md(r'''
### Training

- **Loss**: cross-entropy on 2 logits. **Optimiser**: AdamW (CNN lr 1e-3, ResNet18 lr 1e-4).
- **Checkpoint / early stopping**: lowest validation loss on the inner validation split.
- **Example**: first fold of the first repeat (the same split used in the cross-validation below), fixed slices, small CNN, 2.5D vs mean pooling.
''')

code(r'''
data = {"subject_id": S.SID, "dataset": S.DS, "label": S.Y, "strata": np.array([f"{a}_{b}" for a, b in zip(S.DS, S.Y)])}
tr, va, te = L.cv_folds(data, repeat=0)[0]
print(f"train {len(tr)} | validation {len(va)} | test {len(te)} subjects (test fold untouched until evaluation)")
arr = S.Z["fixed5"]
models = {}
for method in ("stack", "late_mean"):
    models[method], epochs = S.train(arr, method, "cnn", tr, va, seed=L.SEEDS["model"])
    print(f"small CNN, {method}: stopped after {epochs} epochs")
''')

md(r'''
### Evaluation

Restore the best checkpoint, run the forward pass on the **held-out test fold** (`eval`, `no_grad`) and compute accuracy, AUC and the confusion matrix. A single fold has ~47 subjects, so these numbers are noisy - the cross-validated results below use all 236 subjects.
''')

code(r'''
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
for a, (method, model) in zip(ax, models.items()):
    p, y = S.predict(model, S.loader(arr, te, "cnn", train=False)); pred = (p >= .5).astype(int)
    ConfusionMatrixDisplay(confusion_matrix(y, pred, labels=[0, 1]), display_labels=["CDR=0", "CDR>0"]).plot(ax=a, cmap="Blues", colorbar=False)
    a.set_title(f"{method}: acc {accuracy_score(y, pred):.2f}, AUC {roc_auc_score(y, p):.2f}")
plt.tight_layout(); plt.show()
''')

md(r'''
### ResNet18 (pretrained, fine-tuned)

Same example with the pretrained ResNet18 (`late_mean`, fixed slices): all layers are updated with a small learning rate (1e-4), each slice replicated to 3 channels and normalised with ImageNet-equivalent constants.
''')

code(r'''
res_model, ep = S.train(arr, "late_mean", "resnet", tr, va, seed=L.SEEDS["model"])
p, y = S.predict(res_model, S.loader(arr, te, "resnet", train=False)); pred = (p >= .5).astype(int)
print(f"ResNet18 late fusion (mean pooling), fixed slices, fold 0: stopped after {ep} epochs | test accuracy {accuracy_score(y, pred):.3f} | AUC {roc_auc_score(y, p):.3f}")
ConfusionMatrixDisplay(confusion_matrix(y, pred, labels=[0, 1]), display_labels=["CDR=0", "CDR>0"]).plot(cmap="Blues"); plt.show()
''')

md(r'''
## Cross-validated results (accuracy and confusion matrices)

Produced by `scripts/five_slice_cv.py` (2.5D, concatenation, single slice) and `scripts/meanpool_cv.py` (mean pooling), identical folds and seeds: 5 folds x 2 repeats on all **236 subjects**. Accuracy / confusion matrices use the out-of-fold prediction of every subject (probability averaged over the two repeats, threshold 0.5); `fold_acc` is the mean of the 10 per-fold accuracies; the 95% CI is a bootstrap over subjects.
''')

code(r'''
RES = ROOT / "results" / "meanpool"
summ = pd.read_csv(RES / "meanpool_summary.csv")
main = summ[~summ.method.str.contains("ref")]
fmt = main.assign(**{"accuracy [95% CI]": main.apply(lambda r: f"{r.accuracy:.3f} [{r.acc_ci_lo:.2f}, {r.acc_ci_hi:.2f}]", axis=1), "AUC [95% CI]": main.apply(lambda r: f"{r.auc:.3f} [{r.auc_ci_lo:.2f}, {r.auc_ci_hi:.2f}]", axis=1)})
display(fmt[["backbone", "slices", "method", "accuracy [95% CI]", "fold_acc", "sensitivity", "specificity", "AUC [95% CI]", "params"]].round(3).set_index(["backbone", "slices", "method"]))
display(Image(str(RES / "meanpool_confusion.png")))
''')

md(r'''
**Reference settings** (same folds): late fusion with concatenation and a single slice.
''')

code(r'''
ref = summ[summ.method.str.contains("ref")]
display(ref[["backbone", "slices", "method", "accuracy", "sensitivity", "specificity", "auc", "params"]].round(3).set_index(["backbone", "slices", "method"]))
''')

md(r'''
**Paired differences** (same folds): mean pooling minus 2.5D, and mean pooling minus concatenation, for AUC and accuracy (mean of the 10 fold differences, t-interval, Wilcoxon p; descriptive only - folds are not independent).
''')

code(r'''
display(pd.read_csv(RES / "meanpool_paired.csv").round(3).set_index(["backbone", "comparison", "metric"]))
''')

disc = ROOT / "results" / "meanpool" / "discussion.md"
md(disc.read_text() if disc.exists() else "## Discussion\n\n_(to be completed after the run)_")

nb = nbf.v4.new_notebook(cells=cells); nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = ROOT / "notebooks" / "OASIS_5slice_meanpool_trial.ipynb"; nbf.write(nb, out); print("wrote", out, len(cells), "cells")
