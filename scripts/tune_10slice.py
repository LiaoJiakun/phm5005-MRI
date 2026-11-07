"""Nested hyper-parameter tuning for the OASIS-1 10-slice models (CNN Scratch with BatchNorm, ResNet18), Fixed 75-84 vs Canny c-5..c+4.

Protocol (no test-fold information is used for any choice):
  * Outer folds = exactly the folds of the main notebook (StratifiedKFold seed 31000, inner early-stopping split seed 32000).
  * Inner selection: for each outer fold, 3-fold stratified CV on the outer training portion (train + val of the main notebook).
    Every candidate configuration is trained on 2/3 and scored (ROC-AUC) on the held-out 1/3; score = mean of the 3 inner AUCs.
    The configuration with the highest inner score is chosen separately for every outer fold, backbone and slice selection.
  * Final model: the chosen configuration is trained on the outer training portion with the same early-stopping split as the main
    notebook, with 3 model seeds; the test-fold prediction is the mean probability of the 3 seeds (seed ensemble).
  * Optional transparency stage: every configuration is also evaluated on the outer test folds; choosing the best of these
    afterwards would be optimistic and is reported only as such.
Runs on Apple MPS (about 15x faster than CPU here); MPS is not bit-for-bit deterministic, so reruns differ slightly.
Resumable: every finished fit is appended to a CSV and skipped on restart.

Usage:  python scripts/tune_10slice.py [--stage inner|final|all_configs|everything] [--backbone resnet|cnn|both] [--smoke]
"""
import argparse, json, math, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms.functional as TF
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_NPZ = ROOT / "data" / "oasis1_10slice" / "oasis1_10slice.npz"
OUT = ROOT / "results" / "oasis1_10slice_tuned"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

# Same outer folds / early-stopping split as the main notebook
N_FOLDS, FOLD_SEED, VAL_SEED, VAL_FRACTION = 5, 31000, 32000, 0.15
INNER_FOLDS, INNER_SEED = 3, 34000
FINAL_SEEDS = (33000, 43000, 53000)          # + fold; 33000 + fold is the seed of the main notebook
BATCH_SIZE, MIN_DELTA = 16, 1e-4
INPUT_NORM = {"cnn": (0.5, 0.5), "resnet": (0.449, 0.226)}

# Candidate configurations, fixed before any tuning run. "base" = settings of the main notebook.
COMMON = dict(lr=1e-4, wd=1e-4, epochs=20, patience=5, dropout=0.3, fusion="concat", smoothing=0.0, sched=False, freeze=False)
CONFIGS = {
    "resnet": {
        "R1_base":          dict(COMMON),
        "R2_lr3e-5":        dict(COMMON, lr=3e-5, epochs=30, patience=7, sched=True),
        "R3_lr1e-5":        dict(COMMON, lr=1e-5, epochs=30, patience=7, sched=True),
        "R4_probmean":      dict(COMMON, lr=3e-5, epochs=30, patience=7, sched=True, fusion="probmean"),
        "R5_regularised":   dict(COMMON, lr=3e-5, epochs=30, patience=7, sched=True, dropout=0.5, wd=1e-2, smoothing=0.1),
        "R6_freeze_early":  dict(COMMON, lr=3e-5, epochs=30, patience=7, sched=True, freeze=True),
    },
    "cnn": {
        "C1_base":          dict(COMMON, lr=1e-3, epochs=35, patience=7),
        "C2_lr3e-4":        dict(COMMON, lr=3e-4, epochs=50, patience=10, sched=True),
        "C3_regularised":   dict(COMMON, lr=1e-3, epochs=50, patience=10, sched=True, dropout=0.5, wd=1e-3, smoothing=0.1),
        "C4_probmean":      dict(COMMON, lr=3e-4, epochs=50, patience=10, sched=True, fusion="probmean"),
    },
}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


# ------------------------------------------------------------------ data
npz = np.load(DATA_NPZ, allow_pickle=True)
SUBJECT, LABELS = npz["subject_id"], npz["label"].astype(int)
STACKS = {"Fixed": npz["fixed10"], "Canny": npz["canny10"]}


def outer_splits():
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    out = []
    for fold, (trainval, test) in enumerate(skf.split(SUBJECT, LABELS)):
        tr, va = train_test_split(trainval, test_size=VAL_FRACTION, stratify=LABELS[trainval], random_state=VAL_SEED + fold)
        out.append((np.sort(tr), np.sort(va), np.sort(test)))
    return out


def inner_splits(trainval, fold):
    skf = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=INNER_SEED + fold)
    return [(trainval[a], trainval[b]) for a, b in skf.split(trainval, LABELS[trainval])]


def to_tensor(stacks, idx, backbone):
    m, s = INPUT_NORM[backbone]
    x = torch.from_numpy(stacks[idx]).float().div(255.)
    return x.to(DEVICE), torch.from_numpy(LABELS[idx]).to(DEVICE)


def augment(x):
    """+-5 degree rotation, same angle for all 10 slices of a subject (as in the main notebook)."""
    return torch.stack([TF.rotate(xi, random.uniform(-5, 5), interpolation=TF.InterpolationMode.BILINEAR) for xi in x])


def normalise(x, backbone):
    m, s = INPUT_NORM[backbone]
    return (x - m) / s


# ------------------------------------------------------------------ models
class SimpleCNN(nn.Module):
    """Course 06-2 SimpleCNN feature extractor with BatchNorm2d after every convolution (as in the main notebook)."""
    out_dim = 128

    def __init__(self):
        super().__init__()
        def block(i, o): return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU()]
        self.features = nn.Sequential(*block(1, 32), *block(32, 32), nn.MaxPool2d(2, 2), *block(32, 64), *block(64, 64), nn.MaxPool2d(2, 2),
                                      *block(64, 128), *block(128, 128))
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1))

    def forward(self, x):
        return self.pool(self.features(x))


class SliceFusionNet(nn.Module):
    """Shared encoder per slice (one forward call per slice position, as in the main notebook).
    fusion='concat'  : 10 feature vectors concatenated -> dropout -> linear (main notebook)
    fusion='probmean': dropout -> linear per slice -> softmax per slice -> mean of the 10 probabilities"""
    def __init__(self, backbone, cfg, n_slices=10):
        super().__init__()
        self.backbone, self.n, self.fusion = backbone, n_slices, cfg["fusion"]
        if backbone == "resnet":
            self.encoder = models.resnet18(weights="DEFAULT"); self.encoder.fc = nn.Identity(); d = 512
            if cfg["freeze"]:
                for name, p in self.encoder.named_parameters():
                    if name.startswith(("conv1", "bn1", "layer1")): p.requires_grad = False
        else:
            self.encoder = SimpleCNN(); d = SimpleCNN.out_dim
        self.drop = nn.Dropout(cfg["dropout"])
        self.head = nn.Linear(d * n_slices if self.fusion == "concat" else d, 2)

    def forward(self, x):                      # returns log-probabilities [B, 2]
        feats = []
        for j in range(self.n):
            xj = x[:, j:j + 1]
            if self.backbone == "resnet": xj = xj.repeat(1, 3, 1, 1)
            feats.append(self.encoder(xj))
        if self.fusion == "concat":
            return F.log_softmax(self.head(self.drop(torch.cat(feats, 1))), 1)
        logp = torch.stack([F.log_softmax(self.head(self.drop(f)), 1) for f in feats], 1)   # [B, 10, 2]
        return torch.logsumexp(logp, 1) - math.log(self.n)                                   # log of the mean probability


def loss_fn(logp, y, smoothing):
    nll = F.nll_loss(logp, y)
    return nll if smoothing == 0 else (1 - smoothing) * nll + smoothing * (-logp.mean(1)).mean()


@torch.no_grad()
def predict(model, x, backbone):
    model.eval(); out = []
    for i in range(0, len(x), 32):
        out.append(model(normalise(x[i:i + 32], backbone)).exp()[:, 1])
    return torch.cat(out).cpu().numpy()


def fit(backbone, cfg, stacks, tr, va, seed):
    """Train on tr, early stopping on va (validation loss, min_delta, patience); returns best model and info."""
    set_seed(seed)
    xtr, ytr = to_tensor(stacks, tr, backbone); xva, yva = to_tensor(stacks, va, backbone)
    model = SliceFusionNet(backbone, cfg).to(DEVICE)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2) if cfg["sched"] else None
    g = torch.Generator().manual_seed(seed)
    best, best_state, best_ep, wait = float("inf"), None, 0, 0
    for ep in range(cfg["epochs"]):
        model.train()
        for b in torch.randperm(len(tr), generator=g).split(BATCH_SIZE):
            b = b.to(DEVICE)
            loss = loss_fn(model(normalise(augment(xtr[b]), backbone)), ytr[b], cfg["smoothing"])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(F.nll_loss(torch.cat([model(normalise(xva[i:i + 32], backbone)) for i in range(0, len(va), 32)]), yva))
        if sched: sched.step(vl)
        if vl < best - MIN_DELTA:
            best, best_ep, wait = vl, ep + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg["patience"]: break
    model.load_state_dict(best_state)
    return model, dict(best_epoch=best_ep, epochs_run=ep + 1, best_val_loss=best)


# ------------------------------------------------------------------ bookkeeping
def load(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def append(path, row):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def run_inner(backbones, smoke):
    path = OUT / "inner_scores.csv"; done = load(path)
    keys = set(map(tuple, done[["backbone", "selection", "config", "fold", "inner"]].values.tolist())) if len(done) else set()
    for backbone in backbones:
        for fold, (tr, va, te) in enumerate(outer_splits()):
            trainval = np.sort(np.concatenate([tr, va]))
            for inner, (itr, iva) in enumerate(inner_splits(trainval, fold)):
                for sel, stacks in STACKS.items():
                    for name, cfg in CONFIGS[backbone].items():
                        if (backbone, sel, name, fold, inner) in keys: continue
                        if smoke: cfg = dict(cfg, epochs=1)
                        t = time.time()
                        model, info = fit(backbone, cfg, stacks, itr, iva, seed=35000 + fold * 10 + inner)
                        auc = roc_auc_score(LABELS[iva], predict(model, to_tensor(stacks, iva, backbone)[0], backbone))
                        append(path, dict(backbone=backbone, selection=sel, config=name, fold=fold, inner=inner, auc=auc, n_train=len(itr), n_val=len(iva),
                                          seconds=round(time.time() - t, 1), **info))
                        print(f"[inner] {backbone} {sel} {name:16s} fold{fold} inner{inner} AUC={auc:.3f} ep={info['epochs_run']} ({time.time() - t:.0f}s)", flush=True)
                        if smoke: return


def chosen_configs():
    s = load(OUT / "inner_scores.csv")
    m = s.groupby(["backbone", "selection", "fold", "config"]).auc.agg(["mean", "count"]).reset_index()
    m = m[m["count"] == INNER_FOLDS]
    order = {b: list(c) for b, c in CONFIGS.items()}
    m["rank_tiebreak"] = [order[b].index(c) for b, c in zip(m.backbone, m.config)]   # tie -> earlier (simpler) configuration
    m = m.sort_values(["backbone", "selection", "fold", "mean", "rank_tiebreak"], ascending=[True, True, True, False, True])
    return m.groupby(["backbone", "selection", "fold"]).head(1).rename(columns={"mean": "inner_auc"}).drop(columns=["count", "rank_tiebreak"])


def run_outer(backbones, which, smoke):
    """which='final': chosen config per fold; which='all_configs': every config (transparency only)."""
    path = OUT / ("final_predictions.csv" if which == "final" else "all_config_predictions.csv")
    done = load(path)
    keys = set(map(tuple, done[["backbone", "selection", "config", "fold", "seed"]].drop_duplicates().values.tolist())) if len(done) else set()
    choice = chosen_configs() if which == "final" else None
    for backbone in backbones:
        for fold, (tr, va, te) in enumerate(outer_splits()):
            for sel, stacks in STACKS.items():
                if which == "final":
                    row = choice[(choice.backbone == backbone) & (choice.selection == sel) & (choice.fold == fold)]
                    if not len(row): continue
                    names, seeds = [row.config.iloc[0]], FINAL_SEEDS
                else:
                    names, seeds = list(CONFIGS[backbone]), FINAL_SEEDS[:1]
                for name in names:
                    for s0 in seeds:
                        seed = s0 + fold
                        if (backbone, sel, name, fold, seed) in keys: continue
                        cfg = dict(CONFIGS[backbone][name], **({"epochs": 1} if smoke else {}))
                        t = time.time()
                        model, info = fit(backbone, cfg, stacks, tr, va, seed)
                        p = predict(model, to_tensor(stacks, te, backbone)[0], backbone)
                        rows = pd.DataFrame(dict(backbone=backbone, selection=sel, config=name, fold=fold, seed=seed, subject_id=SUBJECT[te], label=LABELS[te], prob=p,
                                                 best_epoch=info["best_epoch"], epochs_run=info["epochs_run"]))
                        rows.to_csv(path, mode="a", header=not path.exists(), index=False)
                        print(f"[{which}] {backbone} {sel} {name:16s} fold{fold} seed{seed} AUC={roc_auc_score(LABELS[te], p):.3f} ep={info['epochs_run']} ({time.time() - t:.0f}s)", flush=True)
                        if smoke: return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="everything", choices=["inner", "final", "all_configs", "everything"])
    ap.add_argument("--backbone", default="both", choices=["resnet", "cnn", "both"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke: OUT = ROOT / "results" / "oasis1_10slice_tuned_smoke"
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(dict(device=str(DEVICE), configs=CONFIGS, outer=dict(folds=N_FOLDS, fold_seed=FOLD_SEED, val_seed=VAL_SEED, val_fraction=VAL_FRACTION),
                   inner=dict(folds=INNER_FOLDS, seed=INNER_SEED), final_seeds=FINAL_SEEDS, torch=torch.__version__),
              open(OUT / "protocol.json", "w"), indent=1)
    bbs = ["resnet", "cnn"] if a.backbone == "both" else [a.backbone]
    print("device", DEVICE, "| output", OUT, flush=True)
    for bb in bbs:            # ResNet first (main model), then CNN
        if a.stage in ("inner", "everything"): run_inner([bb], a.smoke)
        if a.stage in ("final", "everything"): run_outer([bb], "final", a.smoke)
    if a.stage in ("all_configs", "everything"):
        for bb in bbs: run_outer([bb], "all_configs", a.smoke)
    print("DONE", flush=True)
