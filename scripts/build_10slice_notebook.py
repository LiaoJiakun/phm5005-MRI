"""Build notebooks/OASIS1_10slice_fixed_vs_canny.ipynb (OASIS-1 only, 10 axial slices; fixed 75-84 vs Canny c-5..c+4;
CNN Scratch vs shared ResNet18 with feature concatenation). Course-notebook style (05-4 / 06-2 / 10-2): the datasets, models,
training loop, evaluation and Grad-CAM are written out in the notebook cells. The notebook is executed afterwards from a clean kernel."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r'''
# Deep Learning for Brain MRI (OASIS-1) - Fixed vs Canny-selected Slices, CNN Scratch vs ResNet18

**Question.** Using **OASIS-1 only**, classify dementia status (CDR>0 vs CDR=0, subjects aged 60+) from **10 consecutive axial slices** per subject, and compare

| | Slice selection | Model |
|---|---|---|
| 1 | **Fixed** axial slices 75-84 | CNN Scratch (shared encoder per slice, features concatenated) |
| 2 | **Canny** window `c-5 ... c+4` | CNN Scratch |
| 3 | **Fixed** axial slices 75-84 | ResNet18 (ImageNet weights, shared encoder per slice, features concatenated, all layers fine-tuned) |
| 4 | **Canny** window `c-5 ... c+4` | ResNet18 |

Models 1 vs 2 and 3 vs 4 use the same subjects, the same cross-validation folds and the same training settings, so the comparison of slice selection is **paired**.
The code follows the structure of the course notebooks (05-4 workflow, 06-2 classification, 10-2 fusion): set-up -> data -> architecture -> training loop with checkpointing and early stopping -> evaluation. The model definitions, the training loop, the evaluation and Grad-CAM are written out in this notebook.

> The label is the clinical dementia rating (CDR>0 vs CDR=0), a screening label, **not** a pathologically confirmed Alzheimer's diagnosis. Nothing here is a clinical tool.
> This notebook does not assume that any model or slice-selection method is better; whatever the numbers show is reported.
''')

md(r'''
## Workflow

1. **Setup & Reproducibility** - seeds, device, hyper-parameters, paths
2. **Data** - cohort audit (written when the slices were built), loading, slice and orientation checks, example slices
3. **Splitting** - repeated stratified 5-fold CV at subject level, plus an inner validation split; leakage checks
4. **Dataset / DataLoader** - subject = 10 slices
5. **Architecture** - CNN Scratch and ResNet18, both: shared encoder applied to each slice, 10 feature vectors concatenated, dropout, linear classifier
6. **Training with validation** - course-style loop (checkpoint, early stopping, scheduler)
7. **Cross-validation run** - all four models on the same folds (resumable; saves predictions, histories and checkpoints)
8. **Evaluation** - per-fold, per-repeat pooled and subject-averaged AUC; accuracy, F1, sensitivity, specificity; ROC; confusion matrices; training curves
9. **Fixed vs Canny** - paired comparison
10. **Explainability** - Grad-CAM per slice through the shared encoder
11. **Limitations**

**What the 10 slices are.** Volumes are 176 x 208 x 176 (x = left-right, y = anterior-posterior, z = inferior-superior after `as_closest_canonical`). An axial slice is `volume[:, :, z].T`, `z` is the **0-based** index along the 3rd axis.
Fixed: `z = 75, 76, ..., 84`. Canny: the centre `c` is the slice with most Canny edge pixels among `z = 50 ... 110` (ties -> lowest index, as in the earlier project); slices `c-5 ... c+4`.
Every slice is brightness-normalised (1st/99th percentile of the foreground), zero-padded left/right to a square (208 x 208) and resized to 128 x 128 *before* Canny is evaluated, so fixed and Canny slices share exactly the same preprocessing.
''')

md(r'''
## Setup and Reproducibility
''')

code(r'''
import os, sys, json, random, time, copy
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score, ConfusionMatrixDisplay

try:
    from torchmetrics.classification import Accuracy
except ImportError:                      # as in the course notebooks (Colab): install on demand
    !pip install torchmetrics
    from torchmetrics.classification import Accuracy

print("torch", torch.__version__, "| numpy", np.__version__, "| pandas", pd.__version__)
''')

code(r'''
# ---- Set-up ----

# Fix seeds for the different RNGs
SEED = 42
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_seed(SEED)

# Select device (MPS is deliberately not used: its results are not reproducible across runs)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths (this notebook lives in <project>/notebooks)
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
SMOKE = os.environ.get("OASIS_SMOKE") == "1"          # 1 fold x 2 epochs for a quick pipeline test; must be False for the reported run
DATA_NPZ = ROOT / "data" / "oasis1_10slice" / "oasis1_10slice.npz"
OUT = ROOT / "results" / ("oasis1_10slice_smoke" if SMOKE else "oasis1_10slice")
CKPT_DIR = OUT / "checkpoints"; FIG_DIR = OUT / "figures"
for d in (OUT, CKPT_DIR, FIG_DIR): d.mkdir(parents=True, exist_ok=True)

# Cross-validation: stratified 5-fold at subject level (repeat 0 only for now); NEW seeds (not used by any earlier experiment)
N_FOLDS, N_REPEATS = 5, 1                                # first run: ONE repeat of 5-fold CV (raise N_REPEATS to 2 to add the second repeat; finished folds are reused)
FOLD_SEED, VAL_SEED, MODEL_SEED = 31000, 32000, 33000
VAL_FRACTION = 0.15                                   # inner validation split taken from each training portion (early stopping only)
THRESHOLD = 0.5                                       # pre-specified decision threshold (never tuned on test data)

# Hyper-parameters, fixed before any run (taken from the earlier project, not tuned here)
BATCH_SIZE = 16
MIN_DELTA = 1e-4                                      # minimum validation-loss change that counts as an improvement
HP = {"cnn":    dict(lr=1e-3, weight_decay=1e-4, epochs=35, patience=7),
      "resnet": dict(lr=1e-4, weight_decay=1e-4, epochs=20, patience=5)}
if SMOKE:
    N_REPEATS = 1
    HP = {k: dict(v, epochs=2) for k, v in HP.items()}
INPUT_NORM = {"cnn": (0.5, 0.5), "resnet": (0.449, 0.226)}   # constants (ResNet: grey-level equivalent of the ImageNet statistics); nothing is estimated from data

print("DEVICE:", DEVICE, "| SMOKE:", SMOKE, "| output folder:", OUT.relative_to(ROOT))
print("CV:", N_FOLDS, "folds x", N_REPEATS, "repeats | seeds fold/val/model:", FOLD_SEED, VAL_SEED, MODEL_SEED)
print("Hyper-parameters:", json.dumps(HP))
''')

md(r'''
## Data Loading and Splitting

The slices were built by `scripts/build_oasis1_10slice.py` from the OASIS-1 T88 masked volumes (local only, not in the repository). The cell below shows the audit written at that time:
which clinical rows and scans were found, which were excluded and why, and the orientation check.
''')

code(r'''
audit = json.load(open(ROOT / "results" / "oasis1_10slice" / "cohort_audit.json"))
for k, v in audit.items():
    print(f"{k:45s} {v}")
''')

code(r'''
npz = np.load(DATA_NPZ, allow_pickle=True)
subject_id, labels, cdr = npz["subject_id"], npz["label"].astype(int), npz["cdr"]
fixed10, canny10, canny_center = npz["fixed10"], npz["canny10"], npz["canny_center"]

print("Subjects:", len(subject_id), "| unique:", len(set(subject_id)), "| all OASIS-1 ids:", all(s.startswith("OAS1_") for s in subject_id))
print("Labels: CDR=0 ->", int((labels == 0).sum()), "| CDR>0 ->", int((labels == 1).sum()), "| CDR values:", dict(zip(*np.unique(cdr, return_counts=True))))
print("fixed10:", fixed10.shape, fixed10.dtype, "| canny10:", canny10.shape, canny10.dtype)
print("Canny centre c: mean %.1f, sd %.1f, range %d-%d (window c-5..c+4 = %d-%d)" % (canny_center.mean(), canny_center.std(), canny_center.min(), canny_center.max(), canny_center.min() - 5, canny_center.max() + 4))

# Sanity checks on the inputs
assert fixed10.shape == canny10.shape == (len(subject_id), 10, 128, 128)
assert len(set(subject_id)) == len(subject_id)
assert canny_center.min() >= 50 and canny_center.max() <= 110
assert (fixed10.reshape(len(subject_id), 10, -1).max(2) > 0).all() and (canny10.reshape(len(subject_id), 10, -1).max(2) > 0).all()   # no empty slice
''')

md(r'''
### Inspecting Images

Check that both slice sets look like axial brain slices in the same orientation (anterior at the top). Left: the 10 fixed slices of one subject (z = 75 ... 84); right: the 10 Canny slices of the same subject.
''')

code(r'''
i = 0
fig, axes = plt.subplots(2, 10, figsize=(20, 4.6))
for j in range(10):
    axes[0, j].imshow(fixed10[i, j], cmap="gray"); axes[0, j].set_title(f"fixed z={75 + j}", fontsize=9); axes[0, j].axis("off")
    axes[1, j].imshow(canny10[i, j], cmap="gray"); axes[1, j].set_title(f"Canny z={canny_center[i] - 5 + j}", fontsize=9); axes[1, j].axis("off")
plt.suptitle(f"{subject_id[i]} (label {labels[i]}), Canny centre c = {canny_center[i]}")
plt.tight_layout(); plt.savefig(FIG_DIR / "example_slices.png", dpi=110); plt.show()
''')

code(r'''
# Where does Canny look, and how far is it from the fixed block?
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].hist(canny_center, bins=range(canny_center.min(), canny_center.max() + 2), color="tab:orange")
ax[0].axvspan(75, 84, color="tab:blue", alpha=.25, label="fixed 75-84"); ax[0].set_xlabel("Canny centre c (0-based axial index)"); ax[0].set_ylabel("subjects"); ax[0].legend()
overlap = [len(set(range(c - 5, c + 5)) & set(range(75, 85))) for c in canny_center]
ax[1].hist(overlap, bins=range(0, 12), color="gray", align="left"); ax[1].set_xlabel("number of shared slice indices (of 10)"); ax[1].set_ylabel("subjects")
plt.tight_layout(); plt.savefig(FIG_DIR / "canny_centre_hist.png", dpi=110); plt.show()
print("Subjects whose Canny window shares NO slice with the fixed block: %d of %d" % (sum(o == 0 for o in overlap), len(overlap)))
''')

md(r'''
### Splitting

Stratified 5-fold cross-validation **at subject level** (one repeat for now; a second repeat can be added later with `N_REPEATS = 2`, the first repeat's folds do not change) (every subject has exactly one 10-slice stack, so slices of one person can never fall in different partitions).
Each training portion is split again (85% / 15%) into *train* and *validation*; the validation part is used only for early stopping, the test fold is never touched during training.
All four models use exactly the same folds.
''')

code(r'''
def cv_splits(repeat):
    """Returns a list of (train_idx, val_idx, test_idx) for one repeat; identical for all four models."""
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED + repeat)
    out = []
    for fold, (trainval, test) in enumerate(skf.split(subject_id, labels)):
        train, val = train_test_split(trainval, test_size=VAL_FRACTION, stratify=labels[trainval], random_state=VAL_SEED + repeat * 10 + fold)
        out.append((np.sort(train), np.sort(val), np.sort(test)))
    return out

SPLITS = {rep: cv_splits(rep) for rep in range(N_REPEATS)}

# Leakage checks: partitions are disjoint, every subject is tested exactly once per repeat, folds are the same for every model (they come from this one table)
rows = []
for rep, folds in SPLITS.items():
    seen = []
    for fold, (tr, va, te) in enumerate(folds):
        assert set(subject_id[tr]).isdisjoint(subject_id[va]) and set(subject_id[tr]).isdisjoint(subject_id[te]) and set(subject_id[va]).isdisjoint(subject_id[te])
        seen += list(te)
        for part, idx in (("train", tr), ("val", va), ("test", te)):
            rows += [dict(repeat=rep, fold=fold, split=part, subject_id=s, label=int(l)) for s, l in zip(subject_id[idx], labels[idx])]
    assert sorted(seen) == list(range(len(subject_id))), "every subject must be a test subject exactly once per repeat"
split_table = pd.DataFrame(rows); split_table.to_csv(OUT / "split_table.csv", index=False)
print(split_table.groupby(["repeat", "fold", "split"]).agg(n=("label", "size"), cdr_pos=("label", "sum")).unstack("split").to_string())
print("Leakage checks passed. Split table saved to", (OUT / "split_table.csv").relative_to(ROOT))
''')

md(r'''
### `Dataset`

One sample = one subject = a tensor `[10, 128, 128]` (10 grey slices) and a label. Training samples receive a random rotation of +-5 degrees (the same angle for all 10 slices of a subject); validation and test samples are not augmented.
Normalisation uses constants only.
''')

code(r'''
class SliceStackDataset(Dataset):
    def __init__(self, stacks, labels, indices, backbone, augment=False):
        self.stacks, self.labels, self.indices, self.augment = stacks, labels, indices, augment
        self.mean, self.std = INPUT_NORM[backbone]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = self.indices[k]
        x = torch.from_numpy(self.stacks[i]).float() / 255.            # [10, H, W] in [0, 1]
        if self.augment:                                               # same rotation for every slice of this subject
            x = TF.rotate(x, random.uniform(-5, 5), interpolation=TF.InterpolationMode.BILINEAR)
        return (x - self.mean) / self.std, torch.tensor(self.labels[i])
''')

md(r'''
### `DataLoader`
''')

code(r'''
def make_loader(stacks, indices, backbone, train, seed=0):
    g = torch.Generator(); g.manual_seed(seed)                         # reproducible shuffling
    ds = SliceStackDataset(stacks, labels, indices, backbone, augment=train)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=train, generator=g if train else None)

# Look at one batch
tr0, va0, te0 = SPLITS[0][0]
xb, yb = next(iter(make_loader(fixed10, tr0, "resnet", train=True)))
print("One batch:", tuple(xb.shape), tuple(yb.shape), "(batch, slices, height, width) | labels:", yb.tolist())
print("Input range after normalisation: %.2f ... %.2f" % (xb.min(), xb.max()))
''')

md(r'''
## Architecture

Both models apply **one shared encoder to every slice separately**, concatenate the 10 feature vectors in slice order, apply dropout and a linear layer:

```
slice_j [B,1,128,128]  ->  encoder (shared weights)  ->  f_j [B,d]        j = 0..9
[f_0 ; f_1 ; ... ; f_9]  ->  [B, 10*d]  ->  Dropout(0.3)  ->  Linear(10*d, 2)
```

* **CNN Scratch**: the course `SimpleCNN` feature extractor (06-2, six 3x3 convolutions, channels 32-32-64-64-128-128, two max-pools) with `in_channels = 1`, **plus `BatchNorm2d` after every convolution**, global average pooling -> `d = 128`, random initialisation, 1280-d concatenated vector.
  *Why BatchNorm was added:* the course version (no BatchNorm) did not train on these 128 x 128 MRI slices. In the first full run both CNN Scratch models stayed at training loss 0.693 (= ln 2) with every output about 0.50, and in a separate check it could not even fit 32 training subjects in 60 epochs (lower learning rate and Kaiming initialisation did not help either). With BatchNorm the same check reached 100 % training accuracy. Adding BatchNorm fixes the problem; it was not tested whether other changes would also fix it. The failed no-BatchNorm results are kept in `results/oasis1_10slice/archive_cnn_noBN/`.
* **ResNet18**: ImageNet weights, `fc` removed -> `d = 512`; each grey slice is copied to 3 channels; 5120-d concatenated vector. **All layers are fine-tuned.**

This is feature-level fusion by concatenation (the course, 10-2, calls feature-level combination *early fusion*).
It is **not** a 10-channel 2.5D network and predictions are **not** averaged over slices.

Each slice is passed through the encoder in its own forward call (a loop), so BatchNorm statistics (ResNet18 and the CNN) are computed per slice position over the batch, exactly as in the earlier best model. Batching all 10 slices into one pass would change those statistics, so it is not done.
''')

code(r'''
class SimpleCNN(nn.Module):
    """Course 06-2 SimpleCNN feature extractor, 1 input channel, with BatchNorm2d after every convolution (added: see the note above);
    returns a 128-d vector per slice (the course classifier head is replaced by the fusion head)."""
    out_dim = 128

    def __init__(self, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: keep spatial size, then downsample by 2
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 2: keep spatial size, then downsample by 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 3: keep spatial size
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1))   # [B,128,1,1] -> [B,128]

    def forward(self, x):                                    # x: [B, 1, H, W]
        return self.pool(self.features(x))                   # [B, 128]


class SliceFusionNet(nn.Module):
    """Shared encoder per slice -> concatenate the n_slices feature vectors -> dropout -> linear classifier."""
    def __init__(self, backbone, n_slices=10, num_classes=2):
        super().__init__()
        self.backbone, self.n_slices = backbone, n_slices
        if backbone == "resnet":
            self.encoder = models.resnet18(weights="DEFAULT")    # ImageNet weights, shared by all slices
            self.encoder.fc = nn.Identity()                       # remove the classifier -> 512-d features
            d = 512
        else:
            self.encoder = SimpleCNN(in_channels=1); d = SimpleCNN.out_dim
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(d * n_slices, num_classes))   # output for CrossEntropyLoss

    def forward(self, x):                                         # x: [B, 10, H, W]
        features = []
        for j in range(self.n_slices):
            xj = x[:, j:j + 1]                                    # [B, 1, H, W]
            if self.backbone == "resnet":
                xj = xj.repeat(1, 3, 1, 1)                        # grey -> 3 channels
            features.append(self.encoder(xj))                     # [B, d]
        return self.classifier(torch.cat(features, dim=1))        # [B, 10*d] -> [B, 2]
''')

code(r'''
# Instantiate both models and check shapes / parameter counts
for name in ("cnn", "resnet"):
    set_seed(SEED)
    m = SliceFusionNet(name).to(DEVICE)
    out = m(xb.to(DEVICE))
    n_par = sum(p.numel() for p in m.parameters()); n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"{name:7s} output {tuple(out.shape)} | parameters {n_par:,} (trainable {n_train:,}) | classifier: {m.classifier}")
del m
''')

md(r'''
## Training with Validation

The loop follows the course (06-2): cross-entropy loss, `ReduceLROnPlateau`, per-epoch training/validation loss and accuracy, a checkpoint dictionary saved when the validation loss improves, early stopping with `min_delta`.
Differences from the course: AdamW (with weight decay) as in the earlier best model instead of Adam; validation ROC-AUC is logged next to accuracy; `ReduceLROnPlateau` is used with its defaults and, because its patience (10) is longer than the early-stopping patience, it does not actually reduce the learning rate here.
''')

code(r'''
def train_with_validation(model, train_loader, val_loader, checkpoint_file, hp, verbose=False):
    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

    # Set up metrics
    train_metric = Accuracy(task="multiclass", num_classes=2).to(DEVICE)
    val_metric = Accuracy(task="multiclass", num_classes=2).to(DEVICE)

    # Set up scheduler
    scheduler = ReduceLROnPlateau(optimizer)

    # Set-up checkpointing and early stopping
    best_val_loss = float("inf")
    no_improve_count = 0

    # Set-up tracking of loss/metric per epoch
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auc": []}

    # Epoch loop
    for epoch in range(hp["epochs"]):

        # ---- Training ----
        model.train()
        train_loss_sum = 0.0
        train_metric.reset()

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            # Forward + loss
            output = model(xb)
            loss = criterion(output, yb)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate training loss + metric
            train_loss_sum += loss.item() * xb.size(0)
            train_metric.update(output, yb)

        avg_train_loss = train_loss_sum / len(train_loader.dataset)
        avg_train_metric = train_metric.compute().item()

        # ---- Validation ----
        model.eval()
        val_loss_sum = 0.0
        val_metric.reset()
        val_prob, val_true = [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                output = model(xb)
                val_loss_sum += criterion(output, yb).item() * xb.size(0)
                val_metric.update(output, yb)
                val_prob += output.softmax(1)[:, 1].cpu().tolist(); val_true += yb.cpu().tolist()

        avg_val_loss = val_loss_sum / len(val_loader.dataset)
        avg_val_metric = val_metric.compute().item()
        val_auc = roc_auc_score(val_true, val_prob)

        for key, value in zip(history, (epoch + 1, avg_train_loss, avg_train_metric, avg_val_loss, avg_val_metric, val_auc)):
            history[key].append(value)

        # ---- Scheduler step (after validation) ----
        scheduler.step(avg_val_loss)

        # ---- Checkpointing + Early Stopping ----
        if avg_val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = avg_val_loss
            no_improve_count = 0
            torch.save({"epoch": epoch, "best_val_loss": best_val_loss,
                        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}, checkpoint_file)
        else:
            no_improve_count += 1
            if no_improve_count >= hp["patience"]:
                break

        # ---- Report ----
        if verbose:
            print(f"Epoch {epoch+1}/{hp['epochs']} | Train loss {avg_train_loss:.4f} Acc {avg_train_metric:.3f} | "
                  f"Val loss {avg_val_loss:.4f} Acc {avg_val_metric:.3f} AUC {val_auc:.3f}")

    return pd.DataFrame(history)
''')

md(r'''
### One model, verbose

Before the full cross-validation, train a single model (ResNet18, fixed slices, repeat 0 / fold 0) with a printed report per epoch.
This model is **not** reused in the results; the cross-validation below retrains it with the seed given by the fold.
''')

code(r'''
tr0, va0, te0 = SPLITS[0][0]
set_seed(MODEL_SEED)
demo = SliceFusionNet("resnet").to(DEVICE)
demo_hist = train_with_validation(demo, make_loader(fixed10, tr0, "resnet", True, MODEL_SEED), make_loader(fixed10, va0, "resnet", False),
                                  CKPT_DIR / "demo_resnet_fixed.pt", HP["resnet"], verbose=True)
del demo
''')

md(r'''
## Cross-validation run: four models on the same folds

For every repeat and fold, and for each of the four models, we: set the seed, train with validation, restore the best checkpoint, then predict the test fold.
Everything is saved as it is produced (subject-level predictions, training histories, checkpoint index), so an interrupted run resumes and all tables and figures below are recomputed from the saved files.
The decision threshold is fixed at 0.5 in advance and is never tuned.
''')

code(r'''
MODELS = {  # name: (backbone, slice array)
    "CNN Scratch | Fixed":  ("cnn",    fixed10),
    "CNN Scratch | Canny":  ("cnn",    canny10),
    "ResNet18 | Fixed":     ("resnet", fixed10),
    "ResNet18 | Canny":     ("resnet", canny10),
}
PRED_FILE, HIST_FILE, CKPT_INDEX_FILE = OUT / "predictions.csv", OUT / "histories.csv", OUT / "checkpoint_index.csv"

def predict(model, loader):
    model.eval(); prob = []
    with torch.no_grad():
        for xb, _ in loader:
            prob += model(xb.to(DEVICE)).softmax(1)[:, 1].cpu().tolist()
    return np.array(prob)

def done_keys():
    if not PRED_FILE.exists(): return set()
    d = pd.read_csv(PRED_FILE, usecols=["model", "repeat", "fold"]).drop_duplicates()
    return set(map(tuple, d.values.tolist()))
''')

code(r'''
t0 = time.time(); finished = done_keys()
pred_rows = pd.read_csv(PRED_FILE).to_dict("records") if PRED_FILE.exists() else []
hist_rows = pd.read_csv(HIST_FILE).to_dict("records") if HIST_FILE.exists() else []
ckpt_rows = pd.read_csv(CKPT_INDEX_FILE).to_dict("records") if CKPT_INDEX_FILE.exists() else []

for rep in range(N_REPEATS):
    for fold, (tr, va, te) in enumerate(SPLITS[rep]):
        if SMOKE and fold > 0: break
        for name, (backbone, stacks) in MODELS.items():
            if (name, rep, fold) in finished: continue
            seed = MODEL_SEED + rep * 100 + fold                      # identical seed for the four models of one fold
            ckpt = CKPT_DIR / f"{name.replace(' | ', '_').replace(' ', '')}_rep{rep}_fold{fold}.pt"
            set_seed(seed); model = SliceFusionNet(backbone).to(DEVICE)
            hist = train_with_validation(model, make_loader(stacks, tr, backbone, True, seed), make_loader(stacks, va, backbone, False), ckpt, HP[backbone])
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE)["model_state"])      # restore best checkpoint
            p_test = predict(model, make_loader(stacks, te, backbone, False))
            for s, y, p in zip(subject_id[te], labels[te], p_test):
                pred_rows.append(dict(model=name, backbone=backbone, selection=name.split(" | ")[1], repeat=rep, fold=fold, subject_id=s, label=int(y), prob=float(p),
                                      pred=int(p >= THRESHOLD), threshold=THRESHOLD))
            hist_rows += [dict(model=name, repeat=rep, fold=fold, **r) for r in hist.to_dict("records")]
            ckpt_rows.append(dict(model=name, repeat=rep, fold=fold, seed=seed, epochs_run=len(hist), best_epoch=int(hist.val_loss.idxmin()) + 1, checkpoint=str(ckpt.relative_to(ROOT)),
                                  n_train=len(tr), n_val=len(va), n_test=len(te)))
            pd.DataFrame(pred_rows).to_csv(PRED_FILE, index=False); pd.DataFrame(hist_rows).to_csv(HIST_FILE, index=False); pd.DataFrame(ckpt_rows).to_csv(CKPT_INDEX_FILE, index=False)
            print(f"[{(time.time() - t0) / 60:6.1f} min] rep{rep} fold{fold} {name:22s} test AUC = {roc_auc_score(labels[te], p_test):.3f} | epochs {len(hist)}", flush=True)

print("Finished. Predictions:", PRED_FILE.relative_to(ROOT))
''')

md(r'''
## Evaluation on Test

All numbers below are recomputed from the saved subject-level predictions (`predictions.csv`: model, selection, repeat, fold, subject, label, probability, predicted class, threshold).
Three different AUC definitions are kept apart:

* **(A) Mean of per-fold AUC** - the AUC of each test fold (5 folds x `N_REPEATS`), then averaged (reported with its standard deviation over folds).
* **(B) Pooled out-of-fold AUC per repeat** - within one repeat every subject is predicted exactly once; the AUC of those 198 predictions. Reported for each repeat and as their mean.
* **(C) Subject-averaged AUC** - each subject's probability is averaged over the repeats (one value per subject), AUC over the 198 subjects. With a single repeat (B) and (C) are the same number.

Subjects are never counted twice: (B) and (C) contain every subject once. Metrics other than AUC use the fixed 0.5 threshold.
''')

code(r'''
P = pd.read_csv(PRED_FILE)
H = pd.read_csv(HIST_FILE)
order = list(MODELS)
assert set(P.model) == set(order)
# every subject once per repeat per model
assert (P.groupby(["model", "repeat"]).subject_id.nunique() == len(subject_id)).all() if not SMOKE else True

def binary_metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(acc=(tp + tn) / len(y), f1=f1_score(y, pred, zero_division=0), sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1))

# (A) per-fold table
fold_rows = []
for (m, rep, fold), g in P.groupby(["model", "repeat", "fold"]):
    y, p = g.label.to_numpy(), g.prob.to_numpy()
    fold_rows.append(dict(model=m, repeat=rep, fold=fold, n_test=len(g), auc=roc_auc_score(y, p), **binary_metrics(y, p, THRESHOLD)))
F = pd.DataFrame(fold_rows); F.to_csv(OUT / "fold_metrics.csv", index=False)

summary = (F.groupby("model")[["auc", "acc", "f1", "sens", "spec"]].agg(["mean", "std"]).loc[order]).round(3)
print("Per-fold metrics at threshold 0.5 (mean and sd over the", len(F) // len(order), "test folds):")
summary
''')

code(r'''
# (B) pooled OOF AUC per repeat and (C) subject-averaged AUC, with subject-level bootstrap 95% intervals
def pooled_auc(frame):
    return [roc_auc_score(g.label, g.prob) for _, g in frame.sort_values("subject_id").groupby("repeat")]

def subject_avg(frame):
    g = frame.groupby("subject_id").agg(label=("label", "first"), prob=("prob", "mean"))
    return g

rng = np.random.default_rng(0); sids = np.array(sorted(P.subject_id.unique())); N_BOOT = 1000
by_model = {m: P[P.model == m].set_index("subject_id") for m in order}
boot_idx = [rng.choice(len(sids), len(sids), replace=True) for _ in range(N_BOOT)]

def boot_auc(m, kind):
    d = by_model[m]; out = []
    avg = d.groupby(level=0).agg(label=("label", "first"), prob=("prob", "mean")).loc[sids]
    for b in boot_idx:
        pick = sids[b]
        if kind == "C":
            y, p = avg.label.loc[pick].to_numpy(), avg.prob.loc[pick].to_numpy()
            if len(set(y)) > 1: out.append(roc_auc_score(y, p))
        else:
            vals = []
            for rep in sorted(d.repeat.unique()):
                r = d[d.repeat == rep].loc[pick]; vals.append(roc_auc_score(r.label, r.prob))
            out.append(np.mean(vals))
    return np.percentile(out, [2.5, 97.5])

rows = []
for m in order:
    b = pooled_auc(P[P.model == m]); c = subject_avg(P[P.model == m]); cA = F[F.model == m].auc
    lo_b, hi_b = boot_auc(m, "B"); lo_c, hi_c = boot_auc(m, "C")
    rows.append({"model": m, "(A) mean fold AUC": f"{cA.mean():.3f} (sd {cA.std():.3f})",
                 "(B) pooled OOF AUC, repeat 0": round(b[0], 3), "(B) repeat 1": round(b[1], 3) if len(b) > 1 else np.nan, "(B) mean of repeats [95% CI]": f"{np.mean(b):.3f} [{lo_b:.3f}, {hi_b:.3f}]",
                 "(C) subject-averaged AUC [95% CI]": f"{roc_auc_score(c.label, c.prob):.3f} [{lo_c:.3f}, {hi_c:.3f}]"})
auc_table = pd.DataFrame(rows).set_index("model"); auc_table.to_csv(OUT / "auc_summary.csv")
auc_table
''')

md(r'''
### Inspecting Predictions

ROC curves and confusion matrices use the **subject-averaged** probabilities (one value per subject, 198 subjects), so no subject is counted twice. The confusion matrix uses the pre-specified 0.5 threshold.
''')

code(r'''
avg = {m: subject_avg(P[P.model == m]) for m in order}
colors = {"CNN Scratch | Fixed": "tab:blue", "CNN Scratch | Canny": "tab:orange", "ResNet18 | Fixed": "tab:green", "ResNet18 | Canny": "tab:red"}
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, bb in zip(axes, ("CNN Scratch", "ResNet18")):
    for m in order:
        if not m.startswith(bb): continue
        g = avg[m]; fpr, tpr, _ = roc_curve(g.label, g.prob)
        ax.plot(fpr, tpr, lw=2, color=colors[m], label=f"{m.split(' | ')[1]}  AUC {roc_auc_score(g.label, g.prob):.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_title(f"{bb}: ROC (subject-averaged out-of-fold probabilities)"); ax.set_xlabel("1 - specificity"); ax.set_ylabel("sensitivity"); ax.legend(loc="lower right")
plt.tight_layout(); plt.savefig(FIG_DIR / "roc.png", dpi=120); plt.show()

fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
cm_rows = []
for ax, m in zip(axes, order):
    g = avg[m]; pred = (g.prob >= THRESHOLD).astype(int); cm = confusion_matrix(g.label, pred, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["CDR=0", "CDR>0"]).plot(ax=ax, cmap="Blues", colorbar=False)
    tn, fp, fn, tp = cm.ravel(); ax.set_title(f"{m}\nacc {(tp + tn) / cm.sum():.3f} sens {tp / (tp + fn):.2f} spec {tn / (tn + fp):.2f}", fontsize=10)
    cm_rows.append(dict(model=m, tn=tn, fp=fp, fn=fn, tp=tp))
plt.suptitle("Confusion matrices (subject-averaged probabilities, threshold 0.5, 198 subjects)"); plt.tight_layout(); plt.savefig(FIG_DIR / "confusion.png", dpi=120); plt.show()
pd.DataFrame(cm_rows).to_csv(OUT / "confusion_subject_averaged.csv", index=False)
''')

md(r'''
### Training curves

Validation loss and ROC-AUC per epoch (mean over the folds that were still training at that epoch; shaded area = standard deviation). Because of early stopping, later epochs contain fewer folds.
''')

code(r'''
fig, axes = plt.subplots(2, 4, figsize=(18, 7))
for k, m in enumerate(order):
    h = H[H.model == m]
    g = h.groupby("epoch")
    for col, c in (("train_loss", "tab:blue"), ("val_loss", "tab:orange")):
        mu, sd = g[col].mean(), g[col].std().fillna(0); axes[0, k].plot(mu.index, mu, color=c, label=col.replace("_", " ")); axes[0, k].fill_between(mu.index, mu - sd, mu + sd, color=c, alpha=.15)
    axes[0, k].set_title(m); axes[0, k].set_xlabel("epoch"); axes[0, k].set_ylabel("loss"); axes[0, k].legend(); axes[0, k].grid(True)
    for col, c in (("train_acc", "tab:blue"), ("val_acc", "tab:orange"), ("val_auc", "tab:green")):
        mu = g[col].mean(); axes[1, k].plot(mu.index, mu, color=c, label=col.replace("_", " "))
    axes[1, k].set_xlabel("epoch"); axes[1, k].set_ylabel("accuracy / AUC"); axes[1, k].legend(); axes[1, k].grid(True)
plt.suptitle("Training and validation curves (mean over folds)"); plt.tight_layout(); plt.savefig(FIG_DIR / "training_curves.png", dpi=110); plt.show()
print(pd.read_csv(CKPT_INDEX_FILE).groupby("model").agg(epochs_run_mean=("epochs_run", "mean"), best_epoch_mean=("best_epoch", "mean")).round(1))
''')

md(r'''
## Fixed vs Canny: paired comparison

For each backbone the fixed and the Canny model were trained on the same subjects, the same folds and with the same seeds, so the comparison is paired.
Two paired summaries are given:

* **Per-fold AUC difference** (Canny - Fixed) over the test folds: mean, 95% t-interval and the Wilcoxon signed-rank p-value. Folds share training data, so the per-fold differences are **not independent**; with only 5 folds the interval is wide and optimistic.
* **Subject-level paired bootstrap** of the difference in subject-averaged AUC (the same resampled subjects are used for both models).

No hypothesis is assumed in advance about which selection is better; p-values are not corrected for multiple comparisons and the study is small.
''')

code(r'''
from scipy import stats

def paired_table(a, b, label):
    fa = F[F.model == a].sort_values(["repeat", "fold"]).auc.to_numpy(); fb = F[F.model == b].sort_values(["repeat", "fold"]).auc.to_numpy(); d = fa - fb
    se = d.std(ddof=1) / np.sqrt(len(d)); t = stats.t.ppf(.975, len(d) - 1)
    try: w = stats.wilcoxon(d).pvalue
    except Exception: w = np.nan
    ca, cb = avg[a].loc[sids], avg[b].loc[sids]; diffs = []
    for bidx in boot_idx:
        pick = sids[bidx]; y = ca.label.loc[pick].to_numpy()
        if len(set(y)) > 1: diffs.append(roc_auc_score(y, ca.prob.loc[pick]) - roc_auc_score(y, cb.prob.loc[pick]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return dict(comparison=label, mean_fold_diff=d.mean(), fold_ci_lo=d.mean() - t * se, fold_ci_hi=d.mean() + t * se, wilcoxon_p=w, folds_first_better=int((d > 0).sum()), n_folds=len(d),
                subj_avg_auc_diff=roc_auc_score(ca.label, ca.prob) - roc_auc_score(cb.label, cb.prob), boot_ci_lo=lo, boot_ci_hi=hi)

paired = pd.DataFrame([paired_table("CNN Scratch | Canny", "CNN Scratch | Fixed", "CNN Scratch: Canny - Fixed"),
                       paired_table("ResNet18 | Canny", "ResNet18 | Fixed", "ResNet18: Canny - Fixed"),
                       paired_table("ResNet18 | Fixed", "CNN Scratch | Fixed", "Fixed slices: ResNet18 - CNN Scratch"),
                       paired_table("ResNet18 | Canny", "CNN Scratch | Canny", "Canny slices: ResNet18 - CNN Scratch")]).set_index("comparison").round(3)
paired.to_csv(OUT / "paired_comparison.csv"); paired
''')

code(r'''
# Plain-language summary generated from the numbers above (no hand-written performance claims)
for cmp_, r in paired.iterrows():
    direction = "higher" if r.mean_fold_diff > 0 else "lower" if r.mean_fold_diff < 0 else "equal"
    ok = not np.isnan([r.fold_ci_lo, r.fold_ci_hi, r.boot_ci_lo, r.boot_ci_hi]).any()
    contains0 = ok and (r.fold_ci_lo <= 0 <= r.fold_ci_hi) and (r.boot_ci_lo <= 0 <= r.boot_ci_hi)
    print(f"{cmp_}: mean per-fold AUC difference {r.mean_fold_diff:+.3f} (95% t-interval {r.fold_ci_lo:+.3f} to {r.fold_ci_hi:+.3f}, Wilcoxon p = {r.wilcoxon_p:.3f}); "
          f"subject-averaged AUC difference {r.subj_avg_auc_diff:+.3f} (bootstrap {r.boot_ci_lo:+.3f} to {r.boot_ci_hi:+.3f}) -> first model {direction}; "
          f"{'interval not available' if not ok else 'both intervals include 0: no reliable difference' if contains0 else 'at least one interval excludes 0'}.")
''')

md(r'''
## Explainability: Grad-CAM through the shared encoder

For the ResNet18 models we compute Grad-CAM for the CDR>0 score on the last convolutional block (`encoder.layer4`) for **each of the 10 slices**. Because the same encoder is called 10 times, a forward hook records the activation of every call, and the gradients of the class score with respect to each recorded activation are obtained with `torch.autograd.grad`. The recorded call order is checked against the slice order, so map *j* belongs to slice *j*.

Maps are computed with the fold-0 / repeat-0 checkpoints on subjects of **that fold's test set** (not used to train or validate that checkpoint). The same subjects are used for the fixed and the Canny model.
Grad-CAM shows where the network's CDR>0 score is sensitive; it is not a segmentation of disease and does not prove that the model uses a given anatomical region.
''')

code(r'''
def per_slice_gradcam(model, x, target_class=1):
    """x: [1, 10, H, W]. Returns (cams [10, H, W] in [0,1] per slice before global scaling, logits, hook_inputs_match)."""
    acts, in_sums = [], []
    h1 = model.encoder.layer4.register_forward_hook(lambda mod, inp, out: acts.append(out))
    h2 = model.encoder.register_forward_pre_hook(lambda mod, inp: in_sums.append(inp[0].detach().sum().item()))
    model.eval(); model.zero_grad()
    logits = model(x)
    h1.remove(); h2.remove()
    # order check: the j-th encoder call must have received slice j (3 identical channels -> sum = 3 * sum of the slice)
    expected = [3 * x[:, j].sum().item() for j in range(x.shape[1])]
    assert len(acts) == x.shape[1] and np.allclose(in_sums, expected, rtol=1e-4, atol=1e-3), "hook order does not match slice order"
    grads = torch.autograd.grad(logits[0, target_class], acts)           # one gradient per recorded activation
    cams = []
    for a, g in zip(acts, grads):
        w = g.mean(dim=(2, 3), keepdim=True)                              # [1, 512, 1, 1]
        cam = torch.relu((w * a).sum(dim=1, keepdim=True))                # [1, 1, h, w]
        cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cams.append(cam.detach().cpu())
    return torch.stack(cams).numpy(), logits.detach().cpu(), True

def load_fold0(name):
    backbone, stacks = MODELS[name]; ckpt = CKPT_DIR / f"{name.replace(' | ', '_').replace(' ', '')}_rep0_fold0.pt"
    m = SliceFusionNet(backbone).to(DEVICE); m.load_state_dict(torch.load(ckpt, map_location=DEVICE)["model_state"]); return m.eval(), stacks
''')

code(r'''
# Choose test subjects of fold 0 / repeat 0: the most confidently correct CDR>0 and CDR=0 case and one error of each kind (if the fixed model makes one)
tr0, va0, te0 = SPLITS[0][0]
pf = P[(P.model == "ResNet18 | Fixed") & (P.repeat == 0) & (P.fold == 0)].set_index("subject_id")
assert set(pf.index) == set(subject_id[te0]) and set(subject_id[te0]).isdisjoint(subject_id[tr0]) and set(subject_id[te0]).isdisjoint(subject_id[va0])
picks = {}
for tag, mask, asc in (("true CDR>0 (correct)", (pf.label == 1) & (pf.pred == 1), False), ("true CDR=0 (correct)", (pf.label == 0) & (pf.pred == 0), True),
                       ("missed CDR>0", (pf.label == 1) & (pf.pred == 0), False), ("false alarm", (pf.label == 0) & (pf.pred == 1), True)):
    sub = pf[mask].sort_values("prob", ascending=asc)
    if len(sub): picks[tag] = sub.index[0]
print({k: (v, round(pf.loc[v, "prob"], 3)) for k, v in picks.items()})
''')

code(r'''
def show_cams(name, sid, fname):
    model, stacks = load_fold0(name); i = int(np.where(subject_id == sid)[0][0])
    x = torch.from_numpy(stacks[i]).float().div(255.).sub(INPUT_NORM["resnet"][0]).div(INPUT_NORM["resnet"][1]).unsqueeze(0).to(DEVICE)
    cams, logits, _ = per_slice_gradcam(model, x)
    cams = cams / (cams.max() + 1e-8)                                      # one common scale per subject, so slices stay comparable
    z0 = 75 if "Fixed" in name else int(canny_center[i]) - 5
    fig, axes = plt.subplots(2, 10, figsize=(20, 4.6))
    for j in range(10):
        axes[0, j].imshow(stacks[i, j], cmap="gray"); axes[0, j].set_title(f"z={z0 + j}", fontsize=9); axes[0, j].axis("off")
        axes[1, j].imshow(stacks[i, j], cmap="gray"); axes[1, j].imshow(cams[j], cmap="jet", alpha=.45, vmin=0, vmax=1); axes[1, j].axis("off")
    plt.suptitle(f"{name} | {sid} | label {labels[i]} | P(CDR>0) = {logits.softmax(1)[0, 1].item():.2f} (test subject of fold 0, repeat 0)")
    plt.tight_layout(); plt.savefig(FIG_DIR / fname, dpi=110); plt.show()

for tag, sid in picks.items():
    print("==", tag, sid)
    for name in ("ResNet18 | Fixed", "ResNet18 | Canny"):
        show_cams(name, sid, f"gradcam_{name.split(' | ')[1].lower()}_{sid}.png")
''')

md(r'''
## Limitations

* **One cohort, 198 subjects, one repeat of 5-fold CV.** The folds are not independent samples and a single partition of the subjects can be lucky or unlucky; intervals and p-values are optimistic. There is no independent test set.
* **CDR is a screening label**, not a confirmed diagnosis; subjects aged 60+ only; OASIS-1 only (no external validation).
* **Hyper-parameters were not tuned here.** They were fixed in advance from the earlier project; the CNN and ResNet18 settings differ (as in the earlier work), so the backbone comparison also reflects different training settings.
* **Selection of earlier design choices.** The fixed block 75-84 and the Canny rule come from the brief and the earlier project; this notebook does not search for better slices.
* **Preprocessing:** slices are zero-padded to a square and resized to 128 x 128 (the earlier project squashed the 208 x 176 slice instead), so absolute numbers are not directly comparable with the earlier results.
* **CNN Scratch deviates from the course SimpleCNN:** BatchNorm was added after the first run showed the course version did not train (see Architecture). This change was made after seeing results, but only to make the model trainable; no other setting was changed.
* **Grad-CAM** is shown for the ResNet18 models of one fold only and is descriptive.
* **Threshold:** metrics other than AUC depend on the threshold; 0.5 is pre-specified and was not tuned, so sensitivity and specificity may be unbalanced.
''')

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
out = ROOT / "notebooks" / "OASIS1_10slice_fixed_vs_canny.ipynb"
nbf.write(nb, out); print("written", out, len(cells), "cells")
