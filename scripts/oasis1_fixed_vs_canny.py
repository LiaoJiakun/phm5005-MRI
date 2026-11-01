"""
OASIS-1: fixed slice vs Canny-selected slice -- controlled 2D CNN comparison (VS Code version).

Same process as the Colab notebook, no Colab/Drive dependence:
  1. extract disc archives (if not already extracted)   2. find T88 .img volumes
  3. clinical labels (CDR>0 vs CDR=0, Age>=60)           4. ONE subject-level split for both arms
  5. fixed slice vs label-independent Canny slice        6. identical CNN / hyperparameters / seed
  7. evaluate the test set once, save tables + figures + weights

Run as a script:        python oasis1_fixed_vs_canny.py
or cell-by-cell in VS Code (the `# %%` markers give "Run Cell" buttons).

Requires: pip install nibabel openpyxl scikit-learn opencv-python-headless torch torchvision pandas matplotlib pillow
Data (put in DATA_DIR): the OASIS-1 disc archives (file names containing "disc") and
                        oasis_cross-sectional.xlsx (or .csv).
"""
# %% 1. Configuration ---------------------------------------------------------
import copy
import random
import re
import tarfile
import zipfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ---- paths (edit if your data lives elsewhere) ----
PROJECT_DIR = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd().parent
DATA_DIR = PROJECT_DIR / "data" / "raw"              # archives + metadata
MRI_ROOT = PROJECT_DIR / "data" / "extracted"        # extraction target
OUT = PROJECT_DIR / "results" / "oasis1_comparison"  # tables, figures, weights
for d in (DATA_DIR, MRI_ROOT, OUT):
    d.mkdir(parents=True, exist_ok=True)

# ---- experiment settings (identical for both arms) ----
SEED = 42
MIN_AGE = 60
FIXED_INDEX = 80
CANDIDATES = range(50, 111)   # axial slice indices searched by the Canny rule
SIZE = 128
BATCH = 16
EPOCHS = 35
PATIENCE = 7

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")   # MPS deliberately not used: keeps results reproducible
print("Device:", DEVICE, "| data dir:", DATA_DIR)


# %% 2. Extract archives (skipped if volumes already present) -------------------
NEEDED = re.compile(r"t88_masked_gfc\.(img|hdr)$")   # only these files are used downstream


def safe_tar_extract(t, dest):
    """Extract only the T88 masked volumes (saves ~80% disk vs. extracting RAW/SEGM/FSL too)."""
    members = [m for m in t if m.isfile() and NEEDED.search(m.name) and ".." not in Path(m.name).parts
               and not m.name.startswith("/") and not Path(m.name).name.startswith("._")]
    try:
        t.extractall(dest, members=members, filter="data")
    except TypeError:               # Python without the `filter` argument
        t.extractall(dest, members=members)


def extract_archives():
    if any(MRI_ROOT.rglob("*t88_masked_gfc.img")):
        print("T88 volumes already extracted - skipping.")
        return
    archives = sorted(p for p in DATA_DIR.glob("*disc*")
                      if p.name.lower().endswith((".zip", ".tar.gz", ".tgz", ".tar")))
    if not archives:
        raise FileNotFoundError(
            f"No disc archives (*disc*.zip / .tar.gz) and no extracted volumes found.\n"
            f"Put the OASIS-1 archives in: {DATA_DIR}")
    for path in archives:
        print("Extracting:", path.name)
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as z:
                members = [m for m in z.infolist()
                           if "__MACOSX" not in Path(m.filename).parts
                           and not Path(m.filename).name.startswith("._")
                           and (m.is_dir() or NEEDED.search(m.filename))]
                z.extractall(MRI_ROOT, members=members)
        else:
            with tarfile.open(path, "r:*") as t:
                safe_tar_extract(t, MRI_ROOT)
    print("Extraction complete.")


extract_archives()


# %% 3. Locate T88 volumes and metadata -----------------------------------------
paths = sorted(p for p in MRI_ROOT.rglob("*t88_masked_gfc.img")
               if "__MACOSX" not in p.parts and not p.name.startswith("._")
               and p.with_suffix(".hdr").exists())
rows = []
for p in paths:
    m = re.search(r"OAS1_\d+", str(p.relative_to(MRI_ROOT)))
    if m:
        rows.append({"subject_id": m.group(), "mri_path": str(p)})
files_df = pd.DataFrame(rows)
assert len(files_df) > 0, "No valid T88 files found; inspect your extraction paths."
print("Volumes:", len(files_df), "| unique subjects:", files_df.subject_id.nunique(),
      "| repeated scans:", files_df.subject_id.duplicated().sum())


def find_metadata():
    for folder in (DATA_DIR, PROJECT_DIR, MRI_ROOT):
        for pat in ("oasis_cross-sectional.xlsx", "oasis_cross-sectional.csv"):
            hits = sorted(folder.glob(pat)) or sorted(folder.glob("**/" + pat))
            if hits:
                return hits[0]
    raise FileNotFoundError(f"oasis_cross-sectional.xlsx/.csv not found; put it in {DATA_DIR}")


METADATA_PATH = find_metadata()
print("Metadata file:", METADATA_PATH)


# %% 4. Clinical labels, eligibility, merge ------------------------------------
clinical = (pd.read_csv(METADATA_PATH) if METADATA_PATH.suffix.lower() == ".csv"
            else pd.read_excel(METADATA_PATH))
clinical.columns = clinical.columns.astype(str).str.strip()
print("Metadata columns:", clinical.columns.tolist())
assert {"ID", "Age", "CDR"}.issubset(clinical.columns)
clinical["subject_id"] = clinical["ID"].astype(str).str.extract(r"(OAS1_\d+)", expand=False)
clinical["Age"] = pd.to_numeric(clinical["Age"], errors="coerce")
clinical["CDR"] = pd.to_numeric(clinical["CDR"], errors="coerce")
clinical = clinical.dropna(subset=["subject_id", "Age", "CDR"]).copy()
clinical = clinical.loc[clinical.Age >= MIN_AGE].copy()
clinical["label"] = (clinical.CDR > 0).astype(int)
assert clinical.groupby("subject_id")["label"].nunique().max() == 1, \
    "Conflicting labels: inspect repeated sessions"

# deterministic selection of one scan / one record per subject
files_df = files_df.sort_values(["subject_id", "mri_path"]).drop_duplicates("subject_id")
subjects = clinical.sort_values(["subject_id", "ID"]).drop_duplicates("subject_id")
data = subjects[["subject_id", "Age", "CDR", "label"]].merge(
    files_df, on="subject_id", validate="one_to_one")
print("Matched eligible subjects:", len(data))
print("Class counts (0=CDR0, 1=CDR>0):", data.label.value_counts().to_dict())
assert set(data.label) == {0, 1}, "Need both classes; obtain additional MRI archives if required."
assert data.label.value_counts().min() >= 8, "Too few subjects in one class for stable stratification."


# %% 5. One subject-level split for BOTH methods --------------------------------
train_df, tmp_df = train_test_split(data, test_size=.30, random_state=SEED, stratify=data.label)
val_df, test_df = train_test_split(tmp_df, test_size=.50, random_state=SEED, stratify=tmp_df.label)
sets = [set(x.subject_id) for x in (train_df, val_df, test_df)]
assert all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i + 1, 3))
for name, frame in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(name, len(frame), frame.label.value_counts().to_dict())
    frame.to_csv(OUT / f"oasis1_{name}_subjects.csv", index=False)


# %% 6. Slice preprocessing and label-independent Canny selection ----------------
def load_volume(path):
    img = nib.as_closest_canonical(nib.load(path))
    vol = np.squeeze(np.asarray(img.dataobj, dtype=np.float32))
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {vol.shape}: {path}")
    return vol


def normalize_uint8(s):
    s = np.nan_to_num(np.asarray(s, dtype=np.float32), nan=0, posinf=0, neginf=0)
    fg = s[s > 0]
    if fg.size < 100:
        return np.zeros(s.shape, dtype=np.uint8)
    lo, hi = np.percentile(fg, [1, 99])
    z = np.clip((s - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.uint8(z * 255)


def to_cnn_slice(vol, idx):
    return cv2.resize(normalize_uint8(vol[:, :, idx].T), (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def select_canny(vol):
    scores = {}
    for idx in CANDIDATES:
        if idx >= vol.shape[2]:
            continue
        s = to_cnn_slice(vol, idx)
        edges = cv2.Canny(cv2.GaussianBlur(s, (5, 5), 1), 50, 150)
        scores[idx] = int(np.count_nonzero(edges))
    if not scores:
        raise ValueError("No available slices in CANDIDATES")
    best = max(scores, key=lambda i: (scores[i], -i))   # ties -> smallest index
    return best, scores


# sanity check on one volume (look at fixed_vs_canny_example.png to confirm anatomy/orientation)
v = load_volume(data.iloc[0].mri_path)
print("Volume shape:", v.shape)
assert v.shape[2] > FIXED_INDEX
idx, _ = select_canny(v)
print("Fixed slice:", FIXED_INDEX, "| Canny slice:", idx)
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
for a, j, title in zip(ax, [FIXED_INDEX, idx], ["Fixed", "Canny"]):
    a.imshow(to_cnn_slice(v, j), cmap="gray"); a.set_title(f"{title} slice {j}"); a.axis("off")
plt.tight_layout(); fig.savefig(OUT / "fixed_vs_canny_example.png", dpi=150); plt.show()


# %% 7. Precompute one-slice images for both arms -------------------------------
fixed_cache, canny_cache, canny_indices, failures = {}, {}, {}, []
for k, row in enumerate(data.itertuples(index=False), 1):
    try:
        v = load_volume(row.mri_path)
        if FIXED_INDEX >= v.shape[2]:
            raise ValueError("Fixed slice out of range")
        ci, _ = select_canny(v)
        fixed_cache[row.subject_id] = to_cnn_slice(v, FIXED_INDEX)
        canny_cache[row.subject_id] = to_cnn_slice(v, ci)
        canny_indices[row.subject_id] = ci
    except Exception as exc:
        failures.append((row.subject_id, str(exc)))
    if k % 50 == 0:
        print(f"  processed {k}/{len(data)}")
print("Usable:", len(fixed_cache), "| failed:", len(failures), failures[:10])

# subjects that failed are removed from BOTH arms, keeping the pre-assigned partitions
usable = set(fixed_cache)
train_df = train_df[train_df.subject_id.isin(usable)].copy()
val_df = val_df[val_df.subject_id.isin(usable)].copy()
test_df = test_df[test_df.subject_id.isin(usable)].copy()
assert all(set(x.label) == {0, 1} for x in (train_df, val_df, test_df)), \
    "After exclusions one split lost a class."
print("Final N (train/val/test):", [len(x) for x in (train_df, val_df, test_df)])
plt.figure(figsize=(7, 3)); plt.hist(list(canny_indices.values()), bins=25)
plt.xlabel("Canny-selected axial index"); plt.ylabel("Number of subjects")
plt.title("Slice selection distribution"); plt.tight_layout()
plt.savefig(OUT / "canny_index_hist.png", dpi=150); plt.show()


# %% 8. Dataset / DataLoaders (identical transforms and splits) ------------------
class SlicesDataset(Dataset):
    def __init__(self, frame, cache, augment=False):
        self.df = frame.reset_index(drop=True); self.cache = cache
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            *([transforms.RandomRotation(5)] if augment else []),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        return self.tf(self.cache[row.subject_id]), torch.tensor(int(row.label), dtype=torch.long)


def loaders(cache):
    g = torch.Generator(); g.manual_seed(SEED)   # same shuffle order in both arms
    train = DataLoader(SlicesDataset(train_df, cache, True), batch_size=BATCH, shuffle=True,
                       num_workers=0, generator=g)
    val = DataLoader(SlicesDataset(val_df, cache), batch_size=BATCH, shuffle=False, num_workers=0)
    test = DataLoader(SlicesDataset(test_df, cache), batch_size=BATCH, shuffle=False, num_workers=0)
    return train, val, test


print("Batch shape:", next(iter(loaders(fixed_cache)[0]))[0].shape)


# %% 9. Same compact CNN, trained in both arms ----------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(.3), nn.Linear(32, 2))

    def forward(self, x):
        return self.net(x)


def train_one(name, cache):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)   # same init for both arms
    model = SimpleCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    dl_train, dl_val, dl_test = loaders(cache)
    best_loss, best_state, wait, history = float("inf"), None, 0, []
    for epoch in range(EPOCHS):
        model.train(); total = 0.0
        for x, y in dl_train:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y); loss.backward(); optimizer.step()
            total += loss.item() * len(y)
        train_loss = total / len(dl_train.dataset)
        model.eval(); loss_sum = 0.0; correct = 0
        with torch.no_grad():
            for x, y in dl_val:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                loss_sum += criterion(logits, y).item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
        val_loss = loss_sum / len(dl_val.dataset)
        history.append((train_loss, val_loss, correct / len(dl_val.dataset)))
        print(f"{name}: epoch {epoch + 1:02d} train_loss={train_loss:.3f} "
              f"val_loss={val_loss:.3f} val_acc={history[-1][2]:.3f}")
        if val_loss < best_loss - 1e-4:
            best_loss, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, np.array(history), dl_test


fixed_model, fixed_hist, fixed_test = train_one("Fixed", fixed_cache)
canny_model, canny_hist, canny_test = train_one("Canny", canny_cache)


# %% 10. Evaluate ONCE on the untouched test set --------------------------------
def evaluate(model, dl):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for x, y in dl:
            ps.extend(model(x.to(DEVICE)).softmax(dim=1)[:, 1].cpu().numpy()); ys.extend(y.numpy())
    y, p = np.array(ys), np.array(ps); pred = (p >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    metrics = {"Accuracy": float(np.mean(pred == y)),
               "Balanced accuracy": balanced_accuracy_score(y, pred),
               "F1": f1_score(y, pred, zero_division=0),
               "Sensitivity": tp / (tp + fn) if tp + fn else np.nan,
               "Specificity": tn / (tn + fp) if tn + fp else np.nan,
               "ROC-AUC": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan}
    return metrics, y, p, pred


f_metrics, fy, fprob, fpred = evaluate(fixed_model, fixed_test)
c_metrics, cy, cprob, cpred = evaluate(canny_model, canny_test)
assert np.array_equal(fy, cy), "Test sample alignment differs!"
summary = pd.DataFrame([f_metrics, c_metrics], index=[f"Fixed slice {FIXED_INDEX}", "Canny selected"]).T
print(summary.round(3).to_string())
print("\nFixed slice classification report:\n", classification_report(fy, fpred, zero_division=0))
print("Canny classification report:\n", classification_report(cy, cpred, zero_division=0))

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].plot(fixed_hist[:, 0], label="Fixed train"); ax[0].plot(fixed_hist[:, 1], label="Fixed val")
ax[0].plot(canny_hist[:, 0], label="Canny train"); ax[0].plot(canny_hist[:, 1], label="Canny val")
ax[0].set_title("Learning curves"); ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss"); ax[0].legend()
for a, y, pred, title in [(ax[1], fy, fpred, "Fixed"), (ax[2], cy, cpred, "Canny")]:
    mat = confusion_matrix(y, pred, labels=[0, 1]); a.imshow(mat, cmap="Blues")
    a.set_title(title + " confusion matrix"); a.set_xlabel("Predicted"); a.set_ylabel("Actual")
    a.set_xticks([0, 1]); a.set_yticks([0, 1])
    for (i, j), n in np.ndenumerate(mat):
        a.text(j, i, str(n), ha="center", va="center")
plt.tight_layout(); fig.savefig(OUT / "curves_and_confusion.png", dpi=150); plt.show()

plt.figure(figsize=(6, 5))
for label, y, p in [("Fixed", fy, fprob), ("Canny", cy, cprob)]:
    fpr, tpr, _ = roc_curve(y, p); plt.plot(fpr, tpr, label=f"{label}: AUC={roc_auc_score(y, p):.3f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.legend()
plt.title("Subject-level ROC curves"); plt.tight_layout()
plt.savefig(OUT / "roc_curves.png", dpi=150); plt.show()


# %% 11. Save results and weights -----------------------------------------------
summary.to_csv(OUT / "test_metrics.csv")
pd.DataFrame({"subject_id": test_df.subject_id.to_numpy(),
              "true_label": fy, "fixed_probability": fprob, "canny_probability": cprob,
              "canny_slice_index": [canny_indices[s] for s in test_df.subject_id]}
             ).to_csv(OUT / "test_predictions.csv", index=False)
torch.save(fixed_model.state_dict(), OUT / "fixed_cnn.pt")
torch.save(canny_model.state_dict(), OUT / "canny_cnn.pt")
print("Saved to:", OUT)

for name, frame in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(f"\n{name}: {len(frame)} records, {frame.subject_id.nunique()} unique subjects")
    print(frame.label.value_counts().to_string())
