"""Generate notebooks/OASIS1_OASIS2_classification.ipynb in the format of the course template
(06_2_DL_for_Image_Data_Classification): Dataset -> Loading/Transforming -> Inspecting -> Architecture (basic, modern)
-> Training -> Evaluation -> Explainability (Grad-CAM). All model/training code comes from scripts/oasis_lib.py.
The interpretation cell is read from results/final_cv/discussion.md (written after the run)."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r'''
# Deep Learning for Brain MRI Slices - Dementia Classification (OASIS-1 + OASIS-2)

**Question.** Can a 2D deep network detect dementia (**CDR > 0**) versus no dementia (**CDR = 0**) from a *single* MRI slice per view, and does the way we choose the slice matter?

We compare four experiments with identical subjects, folds, seeds and hyper-parameters:

| Experiment | Axial slice | Coronal slice | Network |
|---|---|---|---|
| **A** | fixed (index 80) | - | single-view |
| **B** | - | fixed (index 105) | single-view |
| **C** | fixed (index 80) | fixed (index 105) | dual-view (one branch per view) |
| **D** | Canny-selected | Canny-selected | dual-view |

Two model families are used: a **small 2D CNN trained from scratch** and an **ImageNet-pretrained ResNet18 (fully fine-tuned)**.

> **Note.** The label is *dementia status from the clinical dementia rating (CDR)*, not a pathologically confirmed Alzheimer's diagnosis.
''')

md(r'''
## Workflow

This notebook follows the same workflow as the lesson, with a few task-specific points:

**1) Setup & reproducibility** - fixed seeds, device, all hyper-parameters centralised in `scripts/oasis_lib.py`.

**2) Data import & splitting** - OASIS-1 + OASIS-2 (first visit only), atlas-registered, one slice per view. Splitting is done **at the subject level** with **repeated stratified 5-fold cross-validation**; an inner validation split (taken from each training portion) is used for early stopping and checkpoint selection. The test fold is untouched until the end.

**3) Architecture** - small CNN encoder per view; ResNet18 encoder per view; features are concatenated for dual-view experiments.

**4) Training** - AdamW + cross-entropy, checkpoint on the lowest validation loss.

**5) Evaluation** - ROC-AUC (primary), accuracy, sensitivity/specificity, confusion matrix; paired comparisons across experiments.

**6) Explainability** - Grad-CAM on held-out subjects.

**Data-leakage safeguards used in this run**
- Folds are split by **subject** (asserted: no subject appears in two partitions); OASIS-2 contributes **only its first visit** (one scan per person).
- **Image-level overlap check between OASIS-1 and OASIS-2.** Different IDs do *not* guarantee different people: 111 of the 149 OASIS-2 first-visit subjects have a near-identical scan in OASIS-1 (see below). They are removed, and the final cohort is asserted to contain no remaining near-duplicate pair.
- Normalisation uses **fixed constants** (0.5 / ImageNet statistics) - nothing is estimated from validation/test images.
- Augmentation (+-5 degree rotation) is applied to **training images only**.
- The slice-selection rules (fixed indices, Canny edge count) **never use labels**.
- Early stopping / checkpointing uses the inner validation split only. Hyper-parameters were fixed in advance and **not tuned on test folds**.
- This is a **fresh run**: new fold seeds, models trained from scratch, no weights, splits or selections re-used from earlier exploratory runs.
''')

md(r'''
## Dataset

| Dataset | Source | Used |
|---|---|---|
| **OASIS-1** (cross-sectional) | 416 subjects aged 18-96 (T88 atlas-registered, brain-masked volumes provided) | age >= 60 -> **198** subjects |
| **OASIS-2** (longitudinal) | 150 subjects aged 60-96, 373 sessions (RAW MPRAGE scans, `PART1` + `PART2`) | **first visit only** (`Visit == 1`, `MR Delay == 0`) -> 149 registered; **111 of them are the same people as OASIS-1 participants and are removed** -> **38** unique subjects |

**Labels.** `label = 1` if CDR > 0 (very mild / mild / moderate dementia), else `0`. For OASIS-2 the CDR of the **first visit** is used (so "converted" subjects are labelled by their status at the first scan).

**Preprocessing (done once in `scripts/`, not in this notebook).**
- OASIS-1: provided atlas-registered, gain-field-corrected, brain-masked volumes (`*_t88_masked_gfc`).
- OASIS-2: RAW scans of the first visit -> co-register and average the repeat scans -> N4 bias-field correction -> affine registration (mutual information) to a skull-on OASIS-1 T88 template -> OASIS-1 brain mask. Registration quality was checked with the correlation to the template (OASIS-1 reference range 0.50-0.78); failures were re-run with a multi-start search and one subject that still failed (`OAS2_0046`) was excluded.
- Both datasets then live in the same 176 x 208 x 176 (1 mm) RAS grid, so **axial slice 80** and **coronal slice 105** cover the same anatomy in every subject.
- Each slice is intensity-normalised (1st-99th percentile of non-zero voxels -> 0-255) and resized to **128 x 128**.

**Slice selection**
- *Fixed*: axial 80, coronal 105 (given by the project design).
- *Canny*: among the central axial slices 50-110 (coronal 66-140), pick the slice with the most Canny edge pixels. This is label-independent.
''')

code(r'''
import sys, random, inspect
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from IPython.display import display, Image

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
import oasis_lib as L          # shared dataset / model / training code (single source of truth)

L.set_seed(42)
print("Device:", L.DEVICE, "| torch", torch.__version__)
print("Hyper-parameters:", L.HPARAMS)
print("CV:", L.N_FOLDS, "folds x", L.N_REPEATS, "repeats | validation fraction:", L.VAL_FRACTION, "| seeds:", L.SEEDS)
''')

md(r'''
### Cohort, first-visit check and OASIS-1 / OASIS-2 overlap check
''')

code(r'''
cohort = pd.read_csv(ROOT / "data" / "combined_full" / "combined_cohort.csv")
dem = pd.read_excel(ROOT / "data" / "oasis2" / "raw" / "oasis_longitudinal_demographics.xlsx")

print("OASIS-2 demographics file: %d sessions from %d subjects; sessions per visit number:" % (len(dem), dem["Subject ID"].nunique()))
print(dem.Visit.value_counts().sort_index().to_dict())

o2 = cohort[cohort.dataset == "OASIS2"]
first = dem[(dem.Visit == 1) & (dem["MR Delay"] == 0)]
print("\nFirst-visit rows:", len(first), "| used in cohort:", len(o2))
assert o2.subject_id.is_unique and (o2.visit == 1).all()
assert o2.mri_path.str.contains("_MR1_").all(), "OASIS-2 must only use MR1 (first visit) scans"
assert set(o2.subject_id) <= set(first["Subject ID"])
assert set(cohort[cohort.dataset == "OASIS1"].subject_id).isdisjoint(o2.subject_id) and cohort.subject_id.is_unique
print("Checks passed: one scan per OASIS-2 person (first visit), unique subjects, OASIS-1 / OASIS-2 IDs disjoint.")

summary = cohort.groupby(["dataset", "label"]).size().unstack().rename(columns={0: "CDR=0", 1: "CDR>0"})
summary["total"] = summary.sum(axis=1); summary.loc["All"] = summary.sum()
display(summary)

# ---- image-level overlap between OASIS-2 and OASIS-1 (done in scripts/build_combined.py) ----
rep = pd.read_csv(ROOT / "data" / "combined_full" / "oasis2_overlap_report.csv")
print(f"OASIS-2 first-visit subjects checked: {len(rep)} | same person as an OASIS-1 subject (removed): {int(rep.removed.sum())} | unique (kept): {int((~rep.removed).sum())}")
print("Median image similarity of removed pairs: %.3f | identical CDR in %d of %d removed pairs" % (rep[rep.removed].image_similarity.median(), (rep[rep.removed].cdr_oasis1 == rep[rep.removed].cdr_oasis2).sum(), int(rep.removed.sum())))
assert not set(rep[rep.removed].oasis2_id) & set(cohort.subject_id), "removed duplicates must not be in the final cohort"
display(Image(str(ROOT / "results" / "oasis_overlap_check.png")))
display(cohort.groupby("dataset").agg(age_mean=("age", "mean"), age_sd=("age", "std"), female=("sex", lambda s: (s == "F").mean()), nWBV=("nwbv", "mean")).round(2))
''')

md(r'''
### Loading/Transforming

The slices of all subjects are stored in one array per view (`uint8`, 128 x 128). The transforms are defined in `oasis_lib.make_transform`:
- **Small CNN**: `ToTensor` -> `Normalize(0.5, 0.5)` (grayscale, 1 channel).
- **ResNet18**: slice replicated to 3 channels -> ImageNet mean/std.
- **Training only**: random rotation of +-5 degrees.

These are **constants**, so no statistics are computed from the data (no leakage from validation/test images).
''')

code(r'''
data = L.load_data()
print({k: (v.shape, v.dtype) for k, v in data.items() if isinstance(v, np.ndarray) and v.ndim >= 1 and k not in ("strata",)})
print("\nExperiments:", list(data["experiments"]))
print(inspect.getsource(L.make_transform))
''')

md(r'''
### Inspecting Images

One subject per dataset x class, showing the four slice types used by the experiments. Brains from both datasets should look alike (same atlas space, same masking).
''')

code(r'''
rng = np.random.default_rng(0)
fig, axes = plt.subplots(4, 4, figsize=(11, 11))
keys = [("fixed_axial", "fixed axial 80"), ("fixed_coronal", "fixed coronal 105"), ("canny_axial", "Canny axial"), ("canny_coronal", "Canny coronal")]
rows = [(d, l) for d in ("OASIS1", "OASIS2") for l in (0, 1)]
for r, (d, l) in enumerate(rows):
    i = rng.choice(np.where((data["dataset"] == d) & (data["label"] == l))[0])
    for c, (k, title) in enumerate(keys):
        axes[r, c].imshow(data[k][i], cmap="gray"); axes[r, c].axis("off")
        extra = f" ({data['canny_axial_idx' if 'axial' in k else 'canny_coronal_idx'][i]})" if "canny" in k else ""
        axes[r, c].set_title((f"{d}, {'CDR>0' if l else 'CDR=0'}\n" if c == 0 else "\n") + title + extra, fontsize=9)
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(1, 2, figsize=(10, 3))
for d in ("OASIS1", "OASIS2"):
    m = data["dataset"] == d
    ax[0].hist(data["canny_axial_idx"][m], bins=range(50, 112, 2), alpha=.6, label=d); ax[1].hist(data["canny_coronal_idx"][m], bins=range(66, 142, 2), alpha=.6, label=d)
ax[0].set_title("Canny-selected axial index"); ax[1].set_title("Canny-selected coronal index"); ax[0].legend(); plt.tight_layout(); plt.show()
''')

md(r'''
The Canny rule chooses slices through the **posterior fossa / cerebellum** (low axial, posterior coronal indices) rather than the hippocampus or the ventricles - edge density is highest where folia are fine. This is a property of the rule (it is not anatomy-aware) and is worth keeping in mind when interpreting experiment D.

### Setting `DataLoader` and checking shapes

A batch is a list of view tensors (one per view) plus the labels.
''')

code(r'''
tr, va, te = L.cv_folds(data, repeat=0)[0]
for name in ("A: fixed axial 80", "C: fixed axial+coronal"):
    loader = L.make_loader(data, data["experiments"][name], tr, "cnn", train=True, seed=0)
    views, labels = next(iter(loader))
    print(f"{name:26s} -> {len(views)} view(s), each {tuple(views[0].shape)} | labels {tuple(labels.shape)} {labels.dtype}")
loader = L.make_loader(data, data["experiments"]["A: fixed axial 80"], tr, "resnet", train=True, seed=0)
print("ResNet input:", tuple(next(iter(loader))[0][0].shape), "(3 channels, ImageNet normalisation)")
''')

md(r'''
## Cross-validation design

We use **repeated stratified 5-fold cross-validation at the subject level** (`StratifiedKFold(5, shuffle=True)` repeated twice with different seeds). Stratification is by **dataset x label**, so every fold keeps the same CDR>0 rate *and* the same OASIS-1 / OASIS-2 mix. From each training portion, 15% of the subjects form an inner validation set for early stopping.
''')

code(r'''
rows = []
for rep in range(L.N_REPEATS):
    for k, (tr_, va_, te_) in enumerate(L.cv_folds(data, rep)):
        for part, idx in (("train", tr_), ("val", va_), ("test", te_)):
            rows.append(dict(repeat=rep, fold=k, part=part, n=len(idx), cdr_pos=data["label"][idx].mean(), oasis2=(data["dataset"][idx] == "OASIS2").mean()))
folds = pd.DataFrame(rows)
print("Overall: n = %d, CDR>0 rate = %.3f, OASIS-2 share = %.3f" % (len(data["label"]), data["label"].mean(), (data["dataset"] == "OASIS2").mean()))
display(folds[folds.repeat == 0].pivot(index="fold", columns="part", values=["n", "cdr_pos", "oasis2"]).round(3))
print("Range of CDR>0 rate in test folds (both repeats): %.3f - %.3f" % (folds[folds.part == "test"].cdr_pos.min(), folds[folds.part == "test"].cdr_pos.max()))
''')

md(r'''
## Architecture - Basic

Each view is processed by its own **encoder** and the resulting feature vectors are concatenated before a small classifier head (`ViewNet`). Single-view experiments (A, B) have one encoder, dual-view experiments (C, D) two encoders (the weights are *not* shared, because axial and coronal images look different).

The small CNN encoder is three `Conv-ReLU` blocks (16 -> 32 -> 64 filters, two max-pools) followed by **global average pooling**, so it works for any image size.
''')

code(r'''
print(inspect.getsource(L.SimpleCNN))
print(inspect.getsource(L.ViewNet))
for views in (1, 2):
    m = L.ViewNet("cnn", views)
    print(f"small CNN, {views} view(s): {sum(p.numel() for p in m.parameters()):,} parameters")
''')

md(r'''
### Training

**Set-up**
- **Loss**: `CrossEntropyLoss` on 2 logits (CDR=0 / CDR>0).
- **Optimiser**: AdamW (lr 1e-3, weight decay 1e-4 for the CNN).
- **Metrics**: validation loss (for early stopping/checkpoint), validation ROC-AUC and accuracy; test metrics only at the end.
- **Checkpoint**: weights of the epoch with the lowest validation loss; training stops after 7 epochs without improvement (CNN).

**Example** - experiment **A** (fixed axial 80) on the first fold of the first repeat.
''')

code(r'''
tr, va, te = L.cv_folds(data, repeat=0)[0]
views_A = data["experiments"]["A: fixed axial 80"]
print(f"train {len(tr)} | val {len(va)} | test {len(te)} subjects")
model_cnn, hist_cnn = L.train_model(data, views_A, "cnn", tr, va, seed=L.SEEDS["model"], verbose=True)
''')

code(r'''
def plot_history(hist, title):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    ax[0].plot(hist.epoch, hist.train_loss, label="train"); ax[0].plot(hist.epoch, hist.val_loss, label="validation")
    ax[0].set_title(title + " - loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(hist.epoch, hist.val_auc, color="C2"); ax[1].axhline(.5, ls="--", c="gray")
    ax[1].set_title("validation ROC-AUC"); ax[1].set_xlabel("epoch"); plt.tight_layout(); plt.show()
plot_history(hist_cnn, "Small CNN, experiment A")
''')

md(r'''
### Evaluation

We restore the best checkpoint (already done in `train_model`), run the forward pass in `eval` mode under `no_grad` on the **held-out test fold** and compute the metrics once.
''')

code(r'''
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

def evaluate(model, views, backbone, idx):
    p, y = L.predict_proba(model, L.make_loader(data, views, idx, backbone, train=False))
    return p, y

p, y = evaluate(model_cnn, views_A, "cnn", te)
print(f"Test fold AUC = {roc_auc_score(y, p):.3f}")
print(classification_report(y, (p >= .5).astype(int), target_names=["CDR=0", "CDR>0"], zero_division=0))
ConfusionMatrixDisplay(confusion_matrix(y, (p >= .5).astype(int)), display_labels=["CDR=0", "CDR>0"]).plot(cmap="Blues"); plt.show()
''')

md(r'''
With only ~70 test subjects per fold the numbers of a single fold are noisy - this is why all conclusions below come from the **repeated cross-validation** over all 236 subjects.

We can also inspect individual predictions: one test subject with its true label, predicted probability and the slice the model saw.
''')

code(r'''
j = 5
i = te[j]
print(f"Subject {data['subject_id'][i]} ({data['dataset'][i]}): true = {'CDR>0' if y[j] else 'CDR=0'}, P(CDR>0) = {p[j]:.2f}")
plt.imshow(views_A[0][i], cmap="gray"); plt.axis("off"); plt.show()
''')

md(r'''
## Architecture - Modern

### ResNet18 with pretrained weights

**ResNet** (He et al., 2015) uses *residual (skip) connections*, which make deeper networks trainable. We use `torchvision`'s `resnet18` with **ImageNet weights** and replace the classifier by our head. In the lesson this is "fine-tuning": all layers are updated with a **small learning rate** (1e-4).

- Input: the grayscale slice is replicated to 3 channels and normalised with ImageNet statistics (128 x 128, i.e. native resolution).
- Dual-view experiments use **two separate ResNet18 encoders**.
''')

code(r'''
print(inspect.getsource(L.resnet_encoder))
for views in (1, 2):
    m = L.ViewNet("resnet", views)
    print(f"ResNet18, {views} view(s): {sum(p.numel() for p in m.parameters()):,} parameters (all trainable)")
''')

md(r'''
**Example** - experiment **C** (fixed axial + fixed coronal) on the same first fold. AdamW, lr 1e-4, up to 20 epochs, early stopping after 5 epochs without improvement.
''')

code(r'''
views_C = data["experiments"]["C: fixed axial+coronal"]
model_res, hist_res = L.train_model(data, views_C, "resnet", tr, va, seed=L.SEEDS["model"], verbose=True)
plot_history(hist_res, "ResNet18, experiment C")
p2, y2 = evaluate(model_res, views_C, "resnet", te)
print(f"Test fold AUC = {roc_auc_score(y2, p2):.3f}")
ConfusionMatrixDisplay(confusion_matrix(y2, (p2 >= .5).astype(int)), display_labels=["CDR=0", "CDR>0"]).plot(cmap="Blues"); plt.show()
''')

md(r'''
## Cross-validated results

The experiments were run for both model families with `scripts/final_cv.py` (the same functions as above, 5 folds x 2 repeats x 4 experiments). The table pools the **out-of-fold predictions of all 236 subjects** (probabilities averaged over the two repeats) and gives a bootstrap 95% CI; `fold_auc` is the mean +- SD of the 10 per-fold AUCs. Per-dataset AUCs show whether the OASIS-2 images behave differently from OASIS-1.
''')

code(r'''
RES = ROOT / "results" / "final_cv"
if (RES / "summary.csv").exists():
    summ = pd.read_csv(RES / "summary.csv")
    show = summ.assign(**{"pooled AUC [95% CI]": summ.apply(lambda r: f"{r.pooled_auc:.3f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]", axis=1),
                          "fold AUC (mean±SD)": summ.apply(lambda r: f"{r.fold_auc:.3f} ± {r.fold_sd:.3f}", axis=1)})
    display(show[["backbone", "exp", "pooled AUC [95% CI]", "fold AUC (mean±SD)", "auc_OASIS1", "auc_OASIS2", "acc"]].round(3).set_index(["backbone", "exp"]))
    display(Image(str(RES / "final_results.png")))
else:
    print("Run scripts/final_cv.py cnn / resnet and scripts/final_analyze.py first.")
''')

md(r'''
**Paired differences** (same folds, so differences are paired): mean difference of the fold AUCs with a 95% t-interval and a Wilcoxon signed-rank p-value (10 folds; treat p-values as descriptive - the folds of one repeat are not independent).
''')

code(r'''
if (RES / "paired.csv").exists():
    display(pd.read_csv(RES / "paired.csv").round(3).set_index(["backbone", "comparison"]))
''')

md(r'''
**Confusion matrices** at the default 0.5 threshold (out-of-fold, subject level).
''')

code(r'''
if (RES / "confusion.csv").exists():
    display(pd.read_csv(RES / "confusion.csv").round(3).set_index(["backbone", "exp"]))
    display(Image(str(RES / "final_confusion.png")))
''')

md(r'''
## Explainability

### Grad-CAM

**Grad-CAM** (Gradient-weighted Class Activation Mapping) shows which image regions increased the score of a class: gradients of the class score with respect to the feature maps of a chosen convolutional layer are averaged to give channel weights, the weighted sum of the feature maps is passed through a ReLU and up-sampled to the image size.

**Example** - we take the ResNet18 experiment with the highest cross-validated AUC and the model that was trained on the **first fold of the first repeat**. Grad-CAM is computed only on the **held-out test subjects of that fold** (the model never saw them). The target layer is the last residual block (`layer4[-1]`) of each view's encoder, and the class is CDR>0.

> Grad-CAM explains what the *model* used, not what is anatomically *causal*. A heat-map on a plausible region does not prove the model is right; a heat-map outside the brain or on the boundary would be a red flag for shortcuts.
''')

code(r'''
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

best = summ[summ.backbone == "resnet"].sort_values("pooled_auc").iloc[-1].exp
print("Best ResNet18 experiment (by cross-validated pooled AUC):", best)
views = data["experiments"][best]
view_names = {"A": ["axial 80"], "B": ["coronal 105"], "C": ["axial 80", "coronal 105"], "D": ["Canny axial", "Canny coronal"]}[best[0]]

model = L.ViewNet("resnet", len(views)).to(L.DEVICE)
model.load_state_dict(torch.load(RES / "resnet" / f"fold0_{best[0]}.pt", map_location=L.DEVICE)); model.eval()
cam_model = L.StackedInput(model, channels=3).to(L.DEVICE).eval()      # views stacked along channels for Grad-CAM
tf = L.make_transform("resnet", augment=False)

# held-out subjects of fold 0 / repeat 0, with the predictions of this model
p3, y3 = L.predict_proba(model, L.make_loader(data, views, te, "resnet", train=False))
assert set(te).isdisjoint(tr) and set(te).isdisjoint(va)          # these subjects were never used for training or checkpointing
pred3 = (p3 >= .5).astype(int)

def pick(mask, order):                                              # most confident examples of a group
    ids = np.where(mask)[0]
    return ids[np.argsort(order[ids])][:2] if len(ids) else []
groups = [("true CDR>0, predicted CDR>0", pick((y3 == 1) & (pred3 == 1), -p3)), ("true CDR=0, predicted CDR=0", pick((y3 == 0) & (pred3 == 0), p3)),
          ("true CDR>0, predicted CDR=0 (miss)", pick((y3 == 1) & (pred3 == 0), p3)), ("true CDR=0, predicted CDR>0 (false alarm)", pick((y3 == 0) & (pred3 == 1), -p3))]
examples = [(t, j) for t, js in groups for j in js]

fig, axes = plt.subplots(len(examples), 2 * len(views), figsize=(3.1 * 2 * len(views), 3.1 * len(examples)))
axes = np.atleast_2d(axes)
for r, (title, j) in enumerate(examples):
    i = te[j]
    x = torch.cat([tf(v[i]) for v in views], dim=0).unsqueeze(0).to(L.DEVICE)
    for k in range(len(views)):
        with GradCAM(model=cam_model, target_layers=[model.encoders[k].layer4[-1]]) as cam:
            heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(1)])[0]
        rgb = np.repeat(views[k][i][..., None] / 255., 3, axis=2).astype(np.float32)
        axes[r, 2 * k].imshow(views[k][i], cmap="gray"); axes[r, 2 * k + 1].imshow(show_cam_on_image(rgb, heat, use_rgb=True))
        axes[r, 2 * k].set_title(f"{data['subject_id'][i]} {view_names[k]}\n{title}" if k == 0 else view_names[k], fontsize=7.5)
        axes[r, 2 * k + 1].set_title(f"Grad-CAM (P(CDR>0)={p3[j]:.2f})" if k == 0 else "Grad-CAM", fontsize=7.5)
for a in axes.ravel(): a.axis("off")
plt.tight_layout(); plt.show()
''')

md(r'''
**Reading the maps.** For subjects the model scores as CDR=0 the CDR>0 map is (correctly) almost empty, because Grad-CAM keeps only evidence *for* the target class. For subjects scored as CDR>0 the heat is concentrated around the ventricles and the medial/inferior part of the slice in some cases (atrophy- and ventricle-related regions are plausible), but in others it is diffuse or spills over the brain boundary (e.g. the left edge in the last example). The false alarms look similar to the true positives, so the maps do **not** show that the model has learned a specific, reliable disease pattern; with an AUC of about 0.7 they should be read as a sanity check only.
''')

# interpretation written after the run
disc = ROOT / "results" / "final_cv" / "discussion.md"
md(disc.read_text() if disc.exists() else "## Discussion\n\n_(to be completed after the final run)_")

md(r'''
## Reproducing this run

```bash
cd scripts
python register_oasis2.py            # OASIS-2 first-visit RAW scans -> atlas space (+ --robust --only <ID> for failures)
python build_combined.py             # OASIS-1 + OASIS-2 cohort, image-level de-duplication, slices (fixed / Canny, axial / coronal)
python final_cv.py cnn & python final_cv.py resnet     # repeated stratified 5-fold CV (about 2 h on CPU)
python final_analyze.py              # tables and figures used above
```
''')

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = ROOT / "notebooks" / "OASIS1_OASIS2_classification.ipynb"
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
