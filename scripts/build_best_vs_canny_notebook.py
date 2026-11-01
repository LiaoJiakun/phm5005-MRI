"""Build notebooks/OASIS_best_model_vs_canny.ipynb in the style of the course notebooks (05_4 Workflow, 06_2 Classification):
explicit training loop (checkpoint file, early stopping, ReduceLROnPlateau, torchmetrics), nn.Module model definition, restore-best-then-test,
prediction DataFrame, confusion matrix, Grad-CAM. Model A = ResNet18 late fusion (concat) on FIXED slices 78-82 (best model),
Model B = identical model on CANNY slices (c-2 .. c+2)."""
from pathlib import Path
import pandas as pd
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results" / "five_slice"
S = pd.read_csv(R / "summary.csv"); P = pd.read_csv(R / "paired.csv"); CM = pd.read_csv(R / "confusion.csv")
sa = S[(S.backbone == "resnet") & (S.exp == "M2: late fusion fixed 78-82")].iloc[0]; sb = S[(S.backbone == "resnet") & (S.exp == "M2: late fusion Canny c±2")].iloc[0]
pr = P[(P.backbone == "resnet") & (P.comparison == "M2 Canny - M2 fixed")].iloc[0]
ca = CM[(CM.backbone == "resnet") & (CM.exp == "M2: late fusion fixed 78-82")].iloc[0]; cb = CM[(CM.backbone == "resnet") & (CM.exp == "M2: late fusion Canny c±2")].iloc[0]

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r'''
# Deep Learning for Brain MRI - Best Model vs the Same Model with Canny-selected Slices

**Question.** In our dementia classification project (CDR>0 vs CDR=0, OASIS-1 + OASIS-2 without overlapping subjects) the best-performing model was a **ResNet18 with late fusion (concatenation) of 5 fixed axial slices (78-82)**. Does the result change if the *same* model is given the 5 slices chosen by a **Canny edge rule** instead?

| | Model A (best model) | Model B (comparison) |
|---|---|---|
| Input | 5 **fixed** axial slices: 78, 79, 80, 81, 82 | 5 **Canny** slices: best slice `c` +- 2 |
| Architecture | ResNet18 (ImageNet), shared encoder per slice, features **concatenated** | identical |
| Training, split, seed, hyper-parameters | identical | identical |

Only the choice of slices differs, so any difference in performance can be attributed to it. The code follows the structure of the course notebooks (setup -> data -> model -> training loop with checkpointing and early stopping -> evaluation -> Grad-CAM).

> The label is dementia status from the clinical dementia rating (CDR), not a pathologically confirmed Alzheimer's diagnosis.
''')

md(r'''
## Workflow

**1) Setup & Reproducibility** - fix seeds, device, hyper-parameters, checkpoint paths.

**2) Data Loading and Splitting** - 236 subjects, **subject-level** splits. The split used in this notebook is *fold 0 of repeat 0* of our repeated stratified 5-fold cross-validation, so the numbers can be checked against the cross-validation results.

**3) Model Definition** - `nn.Module` with a shared ResNet18 encoder and a classifier on the concatenated slice features.

**4) Training with Validation** - training loop with checkpointing, early stopping and a learning-rate scheduler (as in the course workflow notebook).

**5) Evaluation on Test** - restore the best checkpoint, test metrics, predictions, confusion matrix.

**6) Cross-validated comparison** - a single fold is noisy (about 48 test subjects), so the conclusion is based on 5 folds x 2 repeats over all 236 subjects.

**7) Explainability** - Grad-CAM on held-out subjects.

**Data-leakage safeguards.** No subject appears in more than one of train/val/test; OASIS-2 contributes first-visit scans only, and the 111 OASIS-2 subjects that are the same people as OASIS-1 participants were removed; normalisation uses fixed constants; the slice rules never use labels; the test fold is only used after training.
''')

md(r'''
## Setup and Reproducibility
''')

code(r'''
import sys, random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)     # torch.load FutureWarning (checkpoint files are our own)

# Project folder (this notebook lives in <project>/notebooks)
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
from oasis_lib import cv_folds, SEEDS      # only used to get the SAME folds as the cross-validation

# Fix seeds for the different RNGs
SEED = SEEDS["model"]          # 9000 = the seed of fold 0 / repeat 0 in the cross-validation
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(SEED)

# Select device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4        # small learning rate: we fine-tune a pretrained network
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 20
PATIENCE = 5                # early stopping
MIN_DELTA = 1e-4

# Paths
OUT_DIR = ROOT / "results" / "best_vs_canny"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_FIXED = OUT_DIR / "checkpoint_fixed.pt"
CHECKPOINT_CANNY = OUT_DIR / "checkpoint_canny.pt"

print("Device:", DEVICE)
''')

code(r'''
# Install torchmetrics and grad-cam if they are missing (as in the course notebooks)
try:
    import torchmetrics, pytorch_grad_cam
except ImportError:
    %pip install -q torchmetrics grad-cam
''')

md(r'''
## Data Loading and Splitting

### Loading

Each subject is a stack of **5 consecutive axial slices** (128 x 128, uint8, atlas-registered and brain-masked), stored in one array per slice rule:
- `fixed5`: slices 78-82
- `canny5`: the slice `c` with the most Canny edge pixels among slices 50-110 (the rule does not use labels), plus `c-2 ... c+2`

The tensor shape for one subject is `[5, 128, 128]`, for a batch `[B, 5, 128, 128]` (5 slices play the role of "channels" in the storage, the model treats them as 5 separate images).
''')

code(r'''
npz = np.load(ROOT / "data" / "combined_full" / "axial5_slices.npz", allow_pickle=True)
subject_id, dataset, labels = npz["subject_id"], npz["dataset"], npz["label"].astype(int)
fixed5, canny5 = npz["fixed5"], npz["canny5"]
canny_center = npz["canny_center"]

print("Subjects:", len(subject_id), "| CDR>0:", labels.sum(), "| CDR=0:", (labels == 0).sum())
print(pd.crosstab(dataset, labels, rownames=["dataset"], colnames=["CDR>0"]))
print("fixed5:", fixed5.shape, fixed5.dtype, "| canny5:", canny5.shape)
print("Canny centre slice: mean %.1f, range %d-%d" % (canny_center.mean(), canny_center.min(), canny_center.max()))
''')

md(r'''
### Splitting

We use **fold 0 of repeat 0** of the repeated stratified 5-fold cross-validation (stratified by dataset x label, subject level). The training portion is further split into train and validation (15%) - the validation set is only used for early stopping and checkpointing, the test fold is untouched until evaluation.
''')

code(r'''
data = {"subject_id": subject_id, "strata": np.array([f"{d}_{l}" for d, l in zip(dataset, labels)])}
train_idx, val_idx, test_idx = cv_folds(data, repeat=0)[0]

# Subject-level split: no subject may appear in two partitions
assert set(subject_id[train_idx]).isdisjoint(subject_id[val_idx])
assert set(subject_id[train_idx]).isdisjoint(subject_id[test_idx])
assert set(subject_id[val_idx]).isdisjoint(subject_id[test_idx])

for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
    print(f"{name:5s}: {len(idx):3d} subjects | CDR>0 rate {labels[idx].mean():.2f} | OASIS-2 share {(dataset[idx] == 'OASIS2').mean():.2f}")
''')

md(r'''
### `Dataset`

A custom `Dataset` returns one subject: the 5 slices as a float tensor `[5, 128, 128]` and the label.
- Pixel values are scaled to `[0, 1]` and normalised with **fixed constants** (grey-level equivalent of the ImageNet statistics) - nothing is computed from the data, so nothing can leak from validation/test subjects.
- **Augmentation** (training only): a random rotation of +-5 degrees, the same rotation for all 5 slices of a subject.
''')

code(r'''
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

MEAN, STD = 0.449, 0.226      # constants (grey-level equivalent of the ImageNet mean/std)

class SliceStackDataset(Dataset):
    def __init__(self, stacks, labels, indices, augment=False):
        self.stacks, self.labels, self.indices, self.augment = stacks, labels, indices, augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = self.indices[k]
        x = torch.from_numpy(self.stacks[i]).float() / 255.                  # [5, 128, 128] in [0, 1]
        if self.augment:                                                      # training images only
            x = TF.rotate(x, random.uniform(-5, 5), interpolation=TF.InterpolationMode.BILINEAR)
        x = (x - MEAN) / STD
        return x, torch.tensor(self.labels[i])
''')

md(r'''
### `DataLoader`
''')

code(r'''
def make_loaders(stacks):
    g = torch.Generator()
    g.manual_seed(SEED)                                   # same shuffling order for both models
    train_loader = DataLoader(SliceStackDataset(stacks, labels, train_idx, augment=True), batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val_loader   = DataLoader(SliceStackDataset(stacks, labels, val_idx),  batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(SliceStackDataset(stacks, labels, test_idx), batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader

# Model A: fixed slices | Model B: Canny slices
set_seed(SEED); train_loader_A, val_loader_A, test_loader_A = make_loaders(fixed5)
set_seed(SEED); train_loader_B, val_loader_B, test_loader_B = make_loaders(canny5)

xb, yb = next(iter(train_loader_A))
print("One batch:", xb.shape, yb.shape, "(batch, slices, height, width)")
''')

md(r'''
### Inspecting Images

One test subject: the 5 fixed slices (top) and the 5 Canny slices (bottom). The Canny slices come from the **posterior fossa / cerebellum** level in almost all subjects - the edge-count rule prefers the finely folded cerebellar cortex - whereas slices 78-82 pass through the lateral ventricles.
''')

code(r'''
i = test_idx[0]
fig, axes = plt.subplots(2, 5, figsize=(12, 5.2))
for k in range(5):
    axes[0, k].imshow(fixed5[i, k], cmap="gray"); axes[0, k].set_title(f"fixed, slice {78 + k}", fontsize=9)
    axes[1, k].imshow(canny5[i, k], cmap="gray"); axes[1, k].set_title(f"Canny, slice {canny_center[i] - 2 + k}", fontsize=9)
for a in axes.ravel(): a.axis("off")
plt.suptitle(f"Subject {subject_id[i]} ({dataset[i]}, {'CDR>0' if labels[i] else 'CDR=0'})")
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(6, 2.8))
ax.hist(canny_center, bins=range(55, 95, 2), color="#d1495b"); ax.axvspan(78, 82, color="#2a6f97", alpha=.4, label="fixed slices 78-82")
ax.set_xlabel("Canny-selected centre slice c"); ax.set_ylabel("subjects"); ax.legend(); plt.show()
''')

md(r'''
## Model Definition

**ResNet18** (He et al., 2015) is built from residual blocks with skip connections. We use `torchvision`'s `resnet18` with **ImageNet weights** and fine-tune all layers.

**Late fusion by concatenation.** The 5 slices are processed *separately* by the *same* encoder (shared weights). Each slice is repeated to 3 channels (the pretrained network expects RGB) and gives a 512-dimensional feature vector. The 5 vectors are **concatenated** (`5 x 512 = 2560`) and a linear classifier on top makes the decision. Features are merged *late* - after each slice has been encoded - and the order of the slices is preserved by the concatenation.

```
x [B,5,H,W] -> slice j [B,1,H,W] -> repeat to [B,3,H,W] -> ResNet18 -> [B,512]   (j = 1..5, shared weights)
concat([B,512] x 5) -> [B,2560] -> Dropout -> Linear -> [B,2]
```
''')

code(r'''
import torch.nn as nn
import torchvision.models as models

# Define late-fusion ResNet18
class LateFusionResNet(nn.Module):
    def __init__(self, n_slices=5, num_classes=2):
        super().__init__()
        self.n_slices = n_slices
        self.encoder = models.resnet18(weights="DEFAULT")        # ImageNet weights, shared by all slices
        self.encoder.fc = nn.Identity()                           # remove the classifier -> 512-d features
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512 * n_slices, num_classes)                # output for CrossEntropyLoss
        )

    def forward(self, x):                                         # x: [B, 5, H, W]
        features = []
        for j in range(self.n_slices):
            xj = x[:, j:j + 1].repeat(1, 3, 1, 1)                 # [B, 1, H, W] -> [B, 3, H, W]
            features.append(self.encoder(xj))                     # [B, 512]
        return self.classifier(torch.cat(features, dim=1))        # [B, 2560] -> [B, 2]

# Instantiate the two models: identical architecture, only the input slices will differ
set_seed(SEED); model_A = LateFusionResNet().to(DEVICE)           # fixed slices 78-82
set_seed(SEED); model_B = LateFusionResNet().to(DEVICE)           # Canny slices

n_params = sum(p.numel() for p in model_A.parameters())
assert n_params == sum(p.numel() for p in model_B.parameters())
print(f"Parameters per model: {n_params:,} (identical for A and B)")
print(model_A.classifier)
''')

md(r'''
## Training with Validation

### Training Loop Set-up

- **Loss**: `CrossEntropyLoss` (2 logits: CDR=0 / CDR>0).
- **Optimizer**: `AdamW` (lr 1e-4, weight decay 1e-4).
- **Metrics** (`torchmetrics`): accuracy and ROC-AUC.
- **Scheduler**: `ReduceLROnPlateau` (default settings; with our early-stopping patience of 5 it does not trigger, but it keeps the loop identical to the course workflow).
- **Checkpointing + early stopping**: the weights with the lowest validation loss are saved to `CHECKPOINT_FILE`; training stops after `PATIENCE` epochs without improvement.

The loop below is the course loop. Two small differences: (1) losses are averaged over *subjects* instead of over *batches* (the last batch is smaller), (2) the validation ROC-AUC is tracked in addition to accuracy.
''')

code(r'''
import torch.optim as optim
from torchmetrics.classification import Accuracy, AUROC
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_with_validation(model, train_loader, val_loader, checkpoint_file):
    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Set up metrics
    train_metric = Accuracy(task="multiclass", num_classes=2).to(DEVICE)
    val_metric = Accuracy(task="multiclass", num_classes=2).to(DEVICE)
    val_auc = AUROC(task="binary").to(DEVICE)

    # Set up scheduler
    scheduler = ReduceLROnPlateau(optimizer)

    # Set-up checkpointing and early stopping
    best_val_loss = float("inf")
    no_improve_count = 0

    # Set-up tracking of loss/metric per epoch
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auc": []}

    # Epoch loop
    for epoch in range(NUM_EPOCHS):

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
        val_auc.reset()

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                output = model(xb)
                loss = criterion(output, yb)
                val_loss_sum += loss.item() * xb.size(0)
                val_metric.update(output, yb)
                val_auc.update(output.softmax(dim=1)[:, 1], yb)

        avg_val_loss = val_loss_sum / len(val_loader.dataset)
        avg_val_metric = val_metric.compute().item()

        history["train_loss"].append(avg_train_loss); history["train_acc"].append(avg_train_metric)
        history["val_loss"].append(avg_val_loss); history["val_acc"].append(avg_val_metric); history["val_auc"].append(val_auc.compute().item())

        # ---- Scheduler step (after validation) ----
        scheduler.step(avg_val_loss)

        # ---- Checkpointing + Early Stopping ----
        if avg_val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = avg_val_loss
            no_improve_count = 0

            # Save checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                checkpoint_file
            )
            status = "saved new best"
        else:
            no_improve_count += 1
            status = f"no improvement ({no_improve_count}/{PATIENCE})"

        # ---- Report ----
        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | Train loss {avg_train_loss:.4f} Acc {avg_train_metric:.3f} | "
              f"Val loss {avg_val_loss:.4f} Acc {avg_val_metric:.3f} AUC {history['val_auc'][-1]:.3f} | {status}")

        if no_improve_count >= PATIENCE:
            print(f"Epoch {epoch+1}: Early stopping triggered.")
            break

    return history
''')

md(r'''
### Model A: fixed slices 78-82 (best model)
''')

code(r'''
set_seed(SEED); train_loader_A, val_loader_A, test_loader_A = make_loaders(fixed5)
set_seed(SEED); model_A = LateFusionResNet().to(DEVICE)
history_A = train_with_validation(model_A, train_loader_A, val_loader_A, CHECKPOINT_FIXED)
''')

md(r'''
### Model B: Canny slices (same model, same seed, same hyper-parameters)
''')

code(r'''
set_seed(SEED); train_loader_B, val_loader_B, test_loader_B = make_loaders(canny5)
set_seed(SEED); model_B = LateFusionResNet().to(DEVICE)
history_B = train_with_validation(model_B, train_loader_B, val_loader_B, CHECKPOINT_CANNY)
''')

md(r'''
### Inspecting Performance

Loss and metric curves for the two models (training and validation).
''')

code(r'''
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for row, (name, h) in enumerate([("Model A: fixed 78-82", history_A), ("Model B: Canny c±2", history_B)]):
    ep = range(1, len(h["train_loss"]) + 1)
    axes[row, 0].plot(ep, h["train_loss"], label="train"); axes[row, 0].plot(ep, h["val_loss"], label="validation"); axes[row, 0].set_title(f"{name} - loss"); axes[row, 0].legend()
    axes[row, 1].plot(ep, h["train_acc"], label="train"); axes[row, 1].plot(ep, h["val_acc"], label="validation"); axes[row, 1].set_title("accuracy"); axes[row, 1].legend()
    axes[row, 2].plot(ep, h["val_auc"], color="C2"); axes[row, 2].axhline(.5, ls="--", c="gray"); axes[row, 2].set_title("validation ROC-AUC")
    for a in axes[row]: a.set_xlabel("epoch")
plt.tight_layout(); plt.show()
''')

md(r'''
## Evaluation on Test

To evaluate on the **held-out test fold** we
- restore the best checkpoint from the checkpoint file,
- set up the test metrics,
- run the forward pass in `eval` mode under `no_grad`,
- collect the predictions of every test subject.
''')

code(r'''
def evaluate_on_test(model, test_loader, checkpoint_file):
    # Restore best model for evaluation
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Restored epoch {checkpoint['epoch'] + 1} (validation loss {checkpoint['best_val_loss']:.4f})")

    # Set up test metrics
    criterion = nn.CrossEntropyLoss()
    test_metric = Accuracy(task="multiclass", num_classes=2).to(DEVICE)
    test_auc = AUROC(task="binary").to(DEVICE)
    test_loss_sum = 0.0

    # Evaluate on test set
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            output = model(xb)                                   # logits
            test_loss_sum += criterion(output, yb).item() * xb.size(0)
            test_metric.update(output, yb)
            prob = output.softmax(dim=1)[:, 1]
            test_auc.update(prob, yb)

            # Collect labels and predictions
            all_probs.extend(prob.cpu().numpy())
            all_preds.extend(output.argmax(dim=1).cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    print(f"Test loss: {test_loss_sum / len(test_loader.dataset):.4f} | Accuracy: {test_metric.compute().item():.3f} | ROC-AUC: {test_auc.compute().item():.3f}")
    return np.array(all_labels), np.array(all_preds), np.array(all_probs), test_auc.compute().item(), test_metric.compute().item()

print("Model A (fixed 78-82):"); y_A, pred_A, prob_A, auc_A, acc_A = evaluate_on_test(model_A, test_loader_A, CHECKPOINT_FIXED)
print("Model B (Canny c±2):  "); y_B, pred_B, prob_B, auc_B, acc_B = evaluate_on_test(model_B, test_loader_B, CHECKPOINT_CANNY)
''')

md(r'''
### Inspecting Predictions

A table of the true labels and predictions per test subject (with a flag for misclassified subjects), and the confusion matrices of the two models.
''')

code(r'''
# Create test dataframe with true/predicted labels and misclassified flag
test_df = pd.DataFrame({"subject_id": subject_id[test_idx], "dataset": dataset[test_idx], "label": y_A,
                        "pred_A_fixed": pred_A, "prob_A": prob_A.round(3), "pred_B_canny": pred_B, "prob_B": prob_B.round(3)})
test_df["misclassified_A"] = test_df["label"] != test_df["pred_A_fixed"]
test_df["misclassified_B"] = test_df["label"] != test_df["pred_B_canny"]
display(test_df.head(10))
print("Misclassified: A =", int(test_df.misclassified_A.sum()), "| B =", int(test_df.misclassified_B.sum()), "of", len(test_df))
''')

code(r'''
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, y, pred, acc, auc) in zip(axes, [("Model A: fixed 78-82", y_A, pred_A, acc_A, auc_A), ("Model B: Canny c±2", y_B, pred_B, acc_B, auc_B)]):
    ConfusionMatrixDisplay(confusion_matrix(y, pred, labels=[0, 1]), display_labels=["CDR=0", "CDR>0"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name}\naccuracy {acc:.3f} | AUC {auc:.3f}")
plt.tight_layout(); plt.show()
''')

md(r'''
### Does this notebook reproduce the cross-validation?

The cross-validation (`scripts/five_slice_cv.py`) trained the same two models on the same fold with the same seed. If the code above is equivalent, fold 0 / repeat 0 must give the same test AUC.
''')

code(r'''
cv_fold = pd.read_csv(ROOT / "results" / "five_slice" / "resnet" / "fold_results.csv")
cv_fold = cv_fold[(cv_fold["repeat"] == 0) & (cv_fold["fold"] == 0)].set_index("exp")
check = pd.DataFrame({"this notebook AUC": [auc_A, auc_B], "cross-validation AUC": [cv_fold.loc["M2: late fusion fixed 78-82", "auc"], cv_fold.loc["M2: late fusion Canny c±2", "auc"]],
                      "this notebook acc": [acc_A, acc_B], "cross-validation acc": [cv_fold.loc["M2: late fusion fixed 78-82", "acc"], cv_fold.loc["M2: late fusion Canny c±2", "acc"]]},
                     index=["Model A (fixed)", "Model B (Canny)"]).round(4)
display(check)
print("AUC matches:", np.allclose(check["this notebook AUC"], check["cross-validation AUC"], atol=1e-3))
''')

md(r'''
**A warning from this single fold.** On this fold the Canny model (B) looks *much better* than the fixed-slice model (A) - the opposite of what the cross-validation below shows. With ~48 test subjects one fold can easily point in the wrong direction (one subject is 2 percentage points). This is why we do **not** draw conclusions from a single split.
''')

md(r'''
## Cross-validated comparison (all 236 subjects)

The same two models were trained and tested with **repeated stratified 5-fold cross-validation (2 repeats, subject level)**; every subject is tested once per repeat. Below: pooled out-of-fold AUC (bootstrap 95% CI), accuracy, sensitivity, specificity and the **paired** per-fold difference between the two models (same folds).
''')

code(r'''
RES = ROOT / "results" / "five_slice"
summ = pd.read_csv(RES / "summary.csv"); summ = summ[(summ.backbone == "resnet") & summ.exp.isin(["M2: late fusion fixed 78-82", "M2: late fusion Canny c±2"])]
conf = pd.read_csv(RES / "confusion.csv"); conf = conf[(conf.backbone == "resnet") & conf.exp.isin(summ.exp)]
table = summ.set_index("exp")[["pooled_auc", "ci_lo", "ci_hi", "fold_auc", "fold_sd"]].join(conf.set_index("exp")[["accuracy", "sensitivity", "specificity", "TN", "FP", "FN", "TP"]])
table.index = ["Model A: fixed 78-82", "Model B: Canny c±2"]
display(table.round(3))

paired = pd.read_csv(RES / "paired.csv"); paired = paired[(paired.backbone == "resnet") & (paired.comparison == "M2 Canny - M2 fixed")]
display(paired[["comparison", "diff", "ci_lo", "ci_hi", "wilcoxon_p", "better_folds", "n"]].round(3).rename(columns={"diff": "AUC diff (B - A)", "better_folds": "folds where B is better"}))
''')

code(r'''
from sklearn.metrics import roc_curve, roc_auc_score
oof = pd.read_csv(RES / "resnet" / "oof_predictions.csv")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for name, exp, col in [("Model A: fixed 78-82", "M2: late fusion fixed 78-82", "#2a6f97"), ("Model B: Canny c±2", "M2: late fusion Canny c±2", "#d1495b")]:
    o = oof[oof.exp == exp].groupby("subject_id").agg(true=("true", "first"), prob=("prob", "mean"))        # probability averaged over the 2 repeats
    fpr, tpr, _ = roc_curve(o.true, o.prob)
    ax[0].plot(fpr, tpr, color=col, lw=2.2, label=f"{name}: AUC {roc_auc_score(o.true, o.prob):.3f}")
ax[0].plot([0, 1], [0, 1], ":", c="gray"); ax[0].set_title("Out-of-fold ROC (n = 236)"); ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate"); ax[0].legend(loc="lower right")

for a, (name, row) in zip(ax[1:], zip(["Model A: fixed 78-82", "Model B: Canny c±2"], [conf.iloc[0], conf.iloc[1]])):
    m = np.array([[row.TN, row.FP], [row.FN, row.TP]]).astype(int)
    a.imshow(m / m.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(m): a.text(j, i, f"{v}\n({v / m[i].sum():.0%})", ha="center", va="center", fontsize=12)
    a.set_xticks([0, 1]); a.set_xticklabels(["CDR=0", "CDR>0"]); a.set_yticks([0, 1]); a.set_yticklabels(["CDR=0", "CDR>0"]); a.set_xlabel("Predicted"); a.set_ylabel("Actual")
    a.set_title(f"{name}\naccuracy {row.accuracy:.3f} | sens {row.sensitivity:.2f} | spec {row.specificity:.2f}")
plt.tight_layout(); plt.show()
''')

md(r'''
## Explainability

### Grad-CAM

**Grad-CAM** highlights the regions of an image that increased the score of a class: the gradients of the class score with respect to the feature maps of a chosen convolutional layer are averaged to give channel weights, the weighted sum of the feature maps is passed through a ReLU and up-sampled to the image size. We use the `grad-cam` library with the last residual block (`encoder.layer4[-1]`) as target layer and the class **CDR>0**.

**Shared encoder.** The encoder is applied to the 5 slices one after the other, so the hooked layer is called 5 times per forward pass. The library stores one activation/gradient per call; `slice_cams` turns each into the heat-map of the corresponding slice (and checks the first against the library output).

Grad-CAM is only computed for **held-out test subjects** of this fold.
''')

code(r'''
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def slice_cams(model, x, target_class=1):
    """Grad-CAM heat-map (128x128, in [0,1]) for each of the 5 slices of one subject; x: [1, 5, H, W]."""
    model.eval()
    with GradCAM(model=model, target_layers=[model.encoder.layer4[-1]]) as cam:
        first = cam(input_tensor=x, targets=[ClassifierOutputTarget(target_class)])[0]            # forward + backward through the 5 calls
        acts = [a.numpy() for a in cam.activations_and_grads.activations]                         # one per slice
        grads = [g.numpy() for g in cam.activations_and_grads.gradients]
    def scale(m):                                                                                 # min-max normalisation (as in the library)
        m = m - m.min()
        return m / (1e-7 + m.max())
    maps = []
    for a, g in zip(acts, grads):
        w = g.mean(axis=(2, 3), keepdims=True)[0]                                                 # channel weights
        m = np.maximum((w * a[0]).sum(axis=0), 0)                                                 # weighted sum + ReLU
        m = scale(cv2.resize(scale(m).astype(np.float32), (128, 128)))                            # normalise, up-sample, normalise again
        maps.append(m)
    assert np.allclose(maps[0], first, atol=1e-4), "slice-0 map does not match the library output"
    return maps

# Reload the best checkpoints
for model, ckpt in [(model_A, CHECKPOINT_FIXED), (model_B, CHECKPOINT_CANNY)]:
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE)["model_state"]); model.eval()

# Example subjects: true positive, true negative, miss and false alarm of Model A (most confident ones)
pos = np.arange(len(test_idx))
def pick(mask, order):
    ids = pos[mask]; return ids[np.argsort(order[ids])][:1]
examples = [("true CDR>0, A predicts CDR>0", pick((y_A == 1) & (pred_A == 1), -prob_A)), ("true CDR=0, A predicts CDR=0", pick((y_A == 0) & (pred_A == 0), prob_A)),
            ("true CDR>0, A predicts CDR=0 (miss)", pick((y_A == 1) & (pred_A == 0), prob_A)), ("true CDR=0, A predicts CDR>0 (false alarm)", pick((y_A == 0) & (pred_A == 1), -prob_A))]
examples = [(t, j[0]) for t, j in examples if len(j)]

def to_input(stacks, i):
    x = torch.from_numpy(stacks[i]).float().unsqueeze(0) / 255.
    return ((x - MEAN) / STD).to(DEVICE)

fig, axes = plt.subplots(len(examples), 4, figsize=(12.5, 3.2 * len(examples)))
axes = np.atleast_2d(axes)
for r, (title, j) in enumerate(examples):
    i = test_idx[j]
    for c, (stacks, model, tag, prob) in enumerate([(fixed5, model_A, "A fixed", prob_A[j]), (canny5, model_B, "B Canny", prob_B[j])]):
        maps = slice_cams(model, to_input(stacks, i))
        rgb = np.repeat(stacks[i, 2][..., None] / 255., 3, axis=2).astype(np.float32)                 # middle slice
        axes[r, 2 * c].imshow(stacks[i, 2], cmap="gray"); axes[r, 2 * c + 1].imshow(show_cam_on_image(rgb, maps[2], use_rgb=True))
        axes[r, 2 * c].set_title(f"{subject_id[i]} - {tag}\n{title}" if c == 0 else f"{tag} (middle slice)", fontsize=8)
        axes[r, 2 * c + 1].set_title(f"Grad-CAM, P(CDR>0)={prob:.2f}", fontsize=8)
for a in axes.ravel(): a.axis("off")
plt.tight_layout(); plt.show()
''')

md(r'''
**Sanity check of the per-slice maps.** The gradient of the CDR>0 score with respect to the output of the shared layer is computed exactly with `torch.autograd.grad` for each of the 5 calls and compared with the maps obtained from the library hooks (they must be identical).
''')

code(r'''
j = examples[0][1]; i = test_idx[j]
x = to_input(fixed5, i)
store = []
handle = model_A.encoder.layer4[-1].register_forward_hook(lambda module, inp, out: store.append(out))    # keep the 5 outputs of the shared layer
out = model_A(x)
handle.remove()
exact_grads = torch.autograd.grad(out[0, 1], store)                                                        # d score(CDR>0) / d output of call k

def scale(m):
    m = m - m.min()
    return m / (1e-7 + m.max())
exact_maps = []
for a, g in zip(store, exact_grads):
    w = g.detach().numpy().mean(axis=(2, 3), keepdims=True)[0]
    m = np.maximum((w * a.detach().numpy()[0]).sum(axis=0), 0)
    exact_maps.append(scale(cv2.resize(scale(m).astype(np.float32), (128, 128))))
hook_maps = slice_cams(model_A, x)
print("max difference between library-hook maps and exact autograd maps per slice:", [f"{np.abs(a - b).max():.1e}" for a, b in zip(hook_maps, exact_maps)])
assert all(np.allclose(a, b, atol=1e-4) for a, b in zip(hook_maps, exact_maps))
''')

md(r'''
One subject, all five slices of each model:
''')

code(r'''
j = examples[0][1]; i = test_idx[j]
fig, axes = plt.subplots(2, 5, figsize=(13, 5.6))
for row, (stacks, model, tag) in enumerate([(fixed5, model_A, "A fixed"), (canny5, model_B, "B Canny")]):
    maps = slice_cams(model, to_input(stacks, i))
    for k in range(5):
        rgb = np.repeat(stacks[i, k][..., None] / 255., 3, axis=2).astype(np.float32)
        axes[row, k].imshow(show_cam_on_image(rgb, maps[k], use_rgb=True)); axes[row, k].axis("off")
        axes[row, k].set_title(f"{tag}, slice {78 + k if row == 0 else canny_center[i] - 2 + k}", fontsize=8.5)
plt.suptitle(f"Grad-CAM (class CDR>0), subject {subject_id[i]}"); plt.tight_layout(); plt.show()
''')

md(r'''
**How to read the maps.** For subjects scored as CDR=0 the CDR>0 map is almost empty (Grad-CAM keeps only evidence *for* the target class). For subjects scored as CDR>0 the picture is **not** a clean 'ventricle' pattern: in several examples the strongest heat sits in the **corners of the image, outside the brain** (empty background), and in the Canny model it is often over the cerebellum. A brain-size / silhouette shortcut is *not* the explanation: the brain-mask outline is almost identical in all subjects (its area varies by only about 0.1-0.2%, see the preprocessing check). More likely reasons are (1) **zero-padding at the image border**, which gives convolutional networks a position signal, (2) the last layer has only **4 x 4** cells for a 128 x 128 slice, so the maps are very coarse, and (3) tiny dataset-specific differences at the mask border (the mask outline alone can tell OASIS-1 from OASIS-2, AUC 0.89). The false alarms look similar to the true positives. The maps therefore do **not** show that either model has learned a specific, reliable disease pattern; they should be read as a sanity check only, not as evidence of pathology.
''')

md(f'''
## Discussion

**Result (all 236 subjects, cross-validated).**

| | Model A: fixed 78-82 | Model B: Canny c±2 |
|---|---|---|
| ROC-AUC (95% CI) | **{sa.pooled_auc:.3f}** ({sa.ci_lo:.2f}-{sa.ci_hi:.2f}) | {sb.pooled_auc:.3f} ({sb.ci_lo:.2f}-{sb.ci_hi:.2f}) |
| Accuracy @0.5 | {ca.accuracy:.3f} | {cb.accuracy:.3f} |
| Sensitivity / specificity | {ca.sensitivity:.2f} / {ca.specificity:.2f} | {cb.sensitivity:.2f} / {cb.specificity:.2f} |

Paired difference in AUC (B - A) over the 10 folds: **{pr["diff"]:.3f}** (95% CI {pr.ci_lo:.3f} to {pr.ci_hi:.3f}; Wilcoxon p = {pr.wilcoxon_p:.3f}); the Canny model is better in {int(pr.better_folds)} of 10 folds.

**What we can and cannot say**
- The Canny slices gave a *lower* AUC (about {abs(pr["diff"]):.2f} lower) and lower accuracy, and the same direction appeared in all six model/fusion settings we tried (-0.01 to -0.09 AUC). But for this particular comparison the interval just includes 0 and p = {pr.wilcoxon_p:.2f}, so it is **suggestive, not conclusive** at the 5% level.
- A plausible reason is the rule itself: it counts edge pixels, which are highest in the finely folded **cerebellum**, so it selects posterior-fossa slices that carry little information about cortical atrophy and ventricle size. The comparison therefore tells us more about this *rule* than about automatic slice selection in general.
- **The "best model" was selected** among many settings on the same cross-validation, so its AUC of {sa.pooled_auc:.2f} is optimistic, and the intervals of the top models overlap (others reach 0.70-0.71).
- Even the best model is **not a diagnostic tool** (AUC about 0.74, accuracy about 0.67, sensitivity about 0.63). Labels are CDR-based dementia status, not biomarker-confirmed Alzheimer's disease; the sex ratio differs between classes; only 236 subjects (198 OASIS-1 + 38 OASIS-2 not present in OASIS-1).
- The single fold shown above favours Canny - a reminder that small test sets can mislead.

**Course concepts used:** `Dataset`/`DataLoader`, custom `nn.Module`, pretrained ResNet18 (fine-tuning), `CrossEntropyLoss`, `AdamW`, `ReduceLROnPlateau`, checkpointing, early stopping, `torchmetrics`, confusion matrix, Grad-CAM, and a feature-level (late) fusion of several inputs by concatenation.
''')

nb = nbf.v4.new_notebook(cells=cells); nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = ROOT / "notebooks" / "OASIS_best_model_vs_canny.ipynb"; nbf.write(nb, out); print("wrote", out, len(cells), "cells")
