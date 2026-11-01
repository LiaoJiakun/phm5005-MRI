"""Shared code for the final OASIS-1 + OASIS-2 (first visit only) experiments A-D.
Single source of truth: used by final_cv.py / final_analyze.py and shown inside the notebook.

Experiments (slices are 128x128, T88-like atlas space, brain-masked):
  A  fixed axial 80                           single-view network
  B  fixed coronal 105                        single-view network
  C  fixed axial 80 + fixed coronal 105       dual-view network (one branch per view)
  D  Canny-selected axial + Canny coronal     dual-view network

All hyper-parameters below are fixed a priori (no tuning on test folds).
"""
import copy
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

ROOT = Path(__file__).resolve().parent.parent
DATA_NPZ = ROOT / "data" / "combined_full" / "combined_slices.npz"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # MPS deliberately not used (reproducibility)

# ---------------------------------------------------------------- hyper-parameters (pre-specified)
HPARAMS = {
    "cnn":    dict(lr=1e-3, weight_decay=1e-4, epochs=35, patience=7, batch=16),
    "resnet": dict(lr=1e-4, weight_decay=1e-4, epochs=20, patience=5, batch=16),   # ImageNet weights, ALL layers fine-tuned
}
N_FOLDS, N_REPEATS = 5, 2
VAL_FRACTION = 0.15          # inner validation split taken from each training portion (early stopping / checkpoint only)
SEEDS = dict(fold=7000, val=8000, model=9000)   # NEW seeds: none of the earlier exploratory runs used these


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- data
def load_data(path=DATA_NPZ):
    z = np.load(path, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    d["label"] = d["label"].astype(int)
    d["strata"] = np.array([f"{a}_{b}" for a, b in zip(d["dataset"], d["label"])])
    d["experiments"] = {
        "A: fixed axial 80":      [d["fixed_axial"]],
        "B: fixed coronal 105":   [d["fixed_coronal"]],
        "C: fixed axial+coronal": [d["fixed_axial"], d["fixed_coronal"]],
        "D: Canny axial+coronal": [d["canny_axial"], d["canny_coronal"]],
    }
    return d


def make_transform(backbone, augment):
    """Slices are uint8 128x128. CNN: normalise to [-1, 1]. ResNet: replicate to 3 channels + ImageNet statistics.
    Constants only - nothing is estimated from the data, so no information can leak from validation/test subjects."""
    ops = [transforms.ToPILImage()]
    if augment:
        ops.append(transforms.RandomRotation(5))      # training images only
    ops.append(transforms.ToTensor())
    if backbone == "cnn":
        ops.append(transforms.Normalize((0.5,), (0.5,)))
    else:
        ops += [transforms.Lambda(lambda t: t.repeat(3, 1, 1)),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return transforms.Compose(ops)


class MultiViewDataset(Dataset):
    """Returns ([view_1, view_2, ...], label) for one subject; `views` is a list of arrays [N, 128, 128]."""
    def __init__(self, views, labels, indices, transform):
        self.views, self.labels, self.indices, self.transform = views, labels, indices, transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = self.indices[k]
        return [self.transform(v[i]) for v in self.views], torch.tensor(self.labels[i])


def collate(batch):
    return [torch.stack(x) for x in zip(*[b[0] for b in batch])], torch.stack([b[1] for b in batch])


def make_loader(data, views, indices, backbone, train, seed=0):
    g = torch.Generator(); g.manual_seed(seed)
    ds = MultiViewDataset(views, data["label"], indices, make_transform(backbone, augment=train))
    return DataLoader(ds, batch_size=HPARAMS[backbone]["batch"], shuffle=train, generator=g if train else None, collate_fn=collate)


# ---------------------------------------------------------------- architectures
class SimpleCNN(nn.Module):
    """Small 3-block CNN encoder (one per view). Output: 64-d feature vector."""
    out_dim = 64

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        return self.net(x)


def resnet_encoder():
    """ImageNet-pretrained ResNet18 without its classifier: 512-d feature vector."""
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Identity()
    m.out_dim = 512
    return m


class ViewNet(nn.Module):
    """One encoder per view -> concatenated features -> classifier head. n_views = 1 (A, B) or 2 (C, D)."""
    def __init__(self, backbone, n_views):
        super().__init__()
        self.backbone, self.n_views = backbone, n_views
        make = SimpleCNN if backbone == "cnn" else resnet_encoder
        self.encoders = nn.ModuleList([make() for _ in range(n_views)])
        d = self.encoders[0].out_dim * n_views
        self.head = (nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 2)) if backbone == "cnn"
                     else nn.Sequential(nn.Dropout(0.3), nn.Linear(d, 2)))

    def forward(self, views):
        return self.head(torch.cat([enc(v) for enc, v in zip(self.encoders, views)], dim=1))


class StackedInput(nn.Module):
    """Adapter for Grad-CAM: takes the views concatenated along channels [B, n_views*C, H, W]."""
    def __init__(self, model, channels):
        super().__init__(); self.model, self.channels = model, channels

    def forward(self, x):
        return self.model(list(torch.split(x, self.channels, dim=1)))


# ---------------------------------------------------------------- training / evaluation
def predict_proba(model, loader):
    model.eval(); probs, labels = [], []
    with torch.no_grad():
        for views, y in loader:
            probs += model([v.to(DEVICE) for v in views]).softmax(1)[:, 1].cpu().tolist(); labels += y.tolist()
    return np.array(probs), np.array(labels)


def train_model(data, views, backbone, train_idx, val_idx, seed, verbose=False):
    """Train with AdamW; after every epoch compute validation loss/AUC; keep the checkpoint with the lowest validation loss
    (early stopping on validation loss only - the test fold is never touched here). Returns (model, history DataFrame)."""
    hp = HPARAMS[backbone]; set_seed(seed)
    loader_tr = make_loader(data, views, train_idx, backbone, train=True, seed=seed)
    loader_va = make_loader(data, views, val_idx, backbone, train=False)
    model = ViewNet(backbone, len(views)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    best_loss, best_state, wait, hist = float("inf"), None, 0, []
    for epoch in range(hp["epochs"]):
        model.train(); tl = 0.0
        for xs, y in loader_tr:
            optimizer.zero_grad()
            loss = criterion(model([x.to(DEVICE) for x in xs]), y.to(DEVICE)); loss.backward(); optimizer.step()
            tl += loss.item() * len(y)
        model.eval(); vl = 0.0; pv, yv = [], []
        with torch.no_grad():
            for xs, y in loader_va:
                out = model([x.to(DEVICE) for x in xs]); vl += criterion(out, y.to(DEVICE)).item() * len(y)
                pv += out.softmax(1)[:, 1].cpu().tolist(); yv += y.tolist()
        tl /= len(loader_tr.dataset); vl /= len(loader_va.dataset)
        hist.append(dict(epoch=epoch + 1, train_loss=tl, val_loss=vl, val_auc=roc_auc_score(yv, pv),
                         val_acc=float(((np.array(pv) >= .5) == np.array(yv)).mean())))
        if verbose:
            print(f"epoch {epoch + 1:02d}  train_loss={tl:.3f}  val_loss={vl:.3f}  val_auc={hist[-1]['val_auc']:.3f}")
        if vl < best_loss - 1e-4:
            best_loss, best_state, wait = vl, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= hp["patience"]:
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(hist)


def cv_folds(data, repeat):
    """Subject-level, stratified by dataset x label. Returns list of (train_idx, val_idx, test_idx)."""
    ids, strata = data["subject_id"], data["strata"]
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEEDS["fold"] + repeat)
    out = []
    for fold, (trv, te) in enumerate(skf.split(ids, strata)):
        tr, va = train_test_split(trv, test_size=VAL_FRACTION, stratify=strata[trv], random_state=SEEDS["val"] + repeat * 10 + fold)
        assert set(ids[tr]).isdisjoint(ids[va]) and set(ids[tr]).isdisjoint(ids[te]) and set(ids[va]).isdisjoint(ids[te])   # no subject in two partitions
        out.append((tr, va, te))
    return out


def run_cv(data, backbone, out_dir, experiments=None, repeats=N_REPEATS):
    """Repeated stratified 5-fold CV for all experiments. Resumable (fold_results.csv / oof_predictions.csv).
    The model of repeat 0 / fold 0 is saved for every experiment (used for Grad-CAM on that fold's held-out subjects)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    f_res, f_oof = out_dir / "fold_results.csv", out_dir / "oof_predictions.csv"
    done = pd.read_csv(f_res) if f_res.exists() else pd.DataFrame(columns=["repeat", "fold", "exp"])
    oof = pd.read_csv(f_oof) if f_oof.exists() else pd.DataFrame()
    experiments = experiments or data["experiments"]; t0 = time.time()
    for rep in range(repeats):
        for fold, (tr, va, te) in enumerate(cv_folds(data, rep)):
            for name, views in experiments.items():
                if ((done["repeat"] == rep) & (done["fold"] == fold) & (done["exp"] == name)).any():
                    continue
                model, hist = train_model(data, views, backbone, tr, va, seed=SEEDS["model"] + rep * 100 + fold)
                p, y = predict_proba(model, make_loader(data, views, te, backbone, train=False))
                pred = (p >= .5).astype(int)
                row = dict(repeat=rep, fold=fold, exp=name, auc=roc_auc_score(y, p), acc=float((pred == y).mean()),
                           bal_acc=balanced_accuracy_score(y, pred), epochs=len(hist), n_test=len(te))
                done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
                oof = pd.concat([oof, pd.DataFrame(dict(repeat=rep, fold=fold, exp=name, subject_id=data["subject_id"][te],
                                                        dataset=data["dataset"][te], true=y, prob=p))], ignore_index=True)
                done.to_csv(f_res, index=False); oof.to_csv(f_oof, index=False)
                if rep == 0 and fold == 0:
                    torch.save(model.state_dict(), out_dir / f"fold0_{name.split(':')[0]}.pt"); hist.to_csv(out_dir / f"fold0_{name.split(':')[0]}_history.csv", index=False)
                print(f"[{backbone} {(time.time() - t0) / 60:6.1f} min] rep{rep} fold{fold} {name:24s} AUC={row['auc']:.3f} epochs={row['epochs']}", flush=True)
    return done, oof
