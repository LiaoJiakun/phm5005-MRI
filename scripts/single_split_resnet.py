# NOTE (kept for the record): this script ran on the 279-subject OASIS-1 + OASIS-2 cohort (data/combined), which was later found to contain
# people who are present in both datasets. All results from it are invalid and that cohort is no longer in the project.
# See build_combined.py (de-duplicated 236-subject cohort) and final_cv.py for the valid experiments.
"""ResNet18 (ImageNet-pretrained) on the combined OASIS-1 + OASIS-2 cohort, experiments A-D, ONE subject-level split (no CV).
Same split as single_split_experiments.py (stratified by dataset x label, random_state=42). Usage: python single_split_resnet.py [layer4|full]
  layer4 = freeze all but the last residual stage (+ new head);  full = fine-tune everything.  Separate (non-shared) ResNet18 per view."""
import copy, random, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

MODE = sys.argv[1] if len(sys.argv) > 1 else "layer4"; assert MODE in ("layer4", "full")
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT / "results" / "combined_single_split_resnet" / MODE; OUT.mkdir(parents=True, exist_ok=True)
torch.hub.set_dir(str(ROOT / "weights" / "torch_hub")); DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH, EPOCHS, PAT, LR, WD, IMG = 16, 20, 5, 1e-4, 1e-4, 224
Z = np.load(ROOT / "data" / "combined" / "combined_slices.npz", allow_pickle=True)
sid, y_all, ds = Z["subject_id"], Z["label"].astype(int), Z["dataset"]; strat = np.array([f"{d}_{l}" for d, l in zip(ds, y_all)])
EXPS = {"A: fixed axial 80": [Z["fixed_axial"]], "B: fixed coronal 105": [Z["fixed_coronal"]],
        "C: fixed axial+coronal": [Z["fixed_axial"], Z["fixed_coronal"]], "D: Canny axial+coronal": [Z["canny_axial"], Z["canny_coronal"]]}
idx = np.arange(len(sid)); tr, tmp = train_test_split(idx, test_size=.30, random_state=42, stratify=strat)
va, te = train_test_split(tmp, test_size=.50, random_state=42, stratify=strat[tmp])
ref = pd.read_csv(ROOT / "results" / "combined_single_split" / "split.csv"); assert (ref.split.to_numpy()[te] == "test").all() and (ref.split.to_numpy()[tr] == "train").all()  # same split as the small-CNN run

def trunk():
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1); m.fc = nn.Identity()
    if MODE == "layer4":
        for n, p in m.named_parameters(): p.requires_grad = n.startswith("layer4.")
    return m
class Net(nn.Module):
    def __init__(s, nb): super().__init__(); s.b = nn.ModuleList([trunk() for _ in range(nb)]); s.head = nn.Sequential(nn.Dropout(.3), nn.Linear(512 * nb, 2))
    def forward(s, xs): return s.head(torch.cat([b(x) for b, x in zip(s.b, xs)], 1))
    def set_train(s):
        s.train()
        if MODE == "layer4":
            for b in s.b:
                for n, m in b.named_modules():
                    if isinstance(m, nn.BatchNorm2d) and not n.startswith("layer4"): m.eval()
def tf(aug): return transforms.Compose([transforms.ToPILImage(), transforms.Resize(IMG), *([transforms.RandomRotation(5)] if aug else []), transforms.ToTensor(),
                                        transforms.Lambda(lambda t: t.repeat(3, 1, 1)), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
class DS(Dataset):
    def __init__(s, v, i, aug): s.v, s.i, s.t = v, i, tf(aug)
    def __len__(s): return len(s.i)
    def __getitem__(s, k): i = s.i[k]; return [s.t(v[i]) for v in s.v], torch.tensor(y_all[i])
def collate(b): return [torch.stack(x) for x in zip(*[s[0] for s in b])], torch.stack([s[1] for s in b])
def fit_predict(views, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); g = torch.Generator(); g.manual_seed(seed)
    mk = lambda i, aug, sh: DataLoader(DS(views, i, aug), batch_size=BATCH, shuffle=sh, generator=g if sh else None, collate_fn=collate)
    dtr, dva, dte = mk(tr, True, True), mk(va, False, False), mk(te, False, False)
    model = Net(len(views)).to(DEVICE); opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WD); crit = nn.CrossEntropyLoss()
    best, state, wait = 1e9, None, 0
    for ep in range(EPOCHS):
        model.set_train()
        for xs, y in dtr: opt.zero_grad(); crit(model([x.to(DEVICE) for x in xs]), y.to(DEVICE)).backward(); opt.step()
        model.eval(); vl = 0
        with torch.no_grad():
            for xs, y in dva: vl += crit(model([x.to(DEVICE) for x in xs]), y.to(DEVICE)).item() * len(y)
        vl /= len(va)
        if vl < best - 1e-4: best, state, wait = vl, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= PAT: break
    model.load_state_dict(state); model.eval(); p = []
    with torch.no_grad():
        for xs, _ in dte: p += model([x.to(DEVICE) for x in xs]).softmax(1)[:, 1].cpu().tolist()
    return np.array(p)

rows, preds = [], {}; y = y_all[te]
for name, views in EXPS.items():
    for seed in (42, 43, 44):
        p = fit_predict(views, seed); pred = (p >= .5).astype(int); tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        r = dict(mode=MODE, exp=name, seed=seed, auc=roc_auc_score(y, p), acc=float((pred == y).mean()), bal_acc=balanced_accuracy_score(y, pred), f1=f1_score(y, pred, zero_division=0),
                 sens=tp / (tp + fn), spec=tn / (tn + fp), TN=tn, FP=fp, FN=fn, TP=tp)
        for d in ("OASIS1", "OASIS2"): m = ds[te] == d; r[f"auc_{d}"] = roc_auc_score(y[m], p[m])
        rows.append(r); preds[(name, seed)] = p; print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
R = pd.DataFrame(rows); R.to_csv(OUT / "per_seed.csv", index=False)
S = R.groupby("exp")[["auc", "acc", "bal_acc", "sens", "spec", "auc_OASIS1", "auc_OASIS2"]].agg(["mean", "std"]).round(3); S.to_csv(OUT / "summary.csv"); print(S.to_string())
pd.DataFrame({"subject_id": sid[te], "dataset": ds[te], "true": y, **{f"{e[0]}_seed{s}": p for (e, s), p in preds.items()}}).to_csv(OUT / "test_predictions.csv", index=False)
EX = list(EXPS); COL = dict(zip(EX, ["#2a6f97", "#e09f3e", "#6a994e", "#d1495b"]))
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for e in EX:
    a_ = R[R.exp == e].auc
    for k, seed in enumerate((42, 43, 44)):
        f, t, _ = roc_curve(y, preds[(e, seed)]); ax[0].plot(f, t, color=COL[e], lw=2.2 if k == 0 else 1, alpha=1 if k == 0 else .4, label=f"{e}: AUC {a_.mean():.3f} ± {a_.std():.3f}" if k == 0 else None)
ax[0].plot([0, 1], [0, 1], "--", c="#aab4be"); ax[0].set_title(f"Test ROC, ResNet18 ({MODE}); n = {len(te)} (bold = seed 42)"); ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate"); ax[0].legend(loc="lower right", fontsize=8.5, frameon=False)
for k, e in enumerate(EX): ax[1].bar(np.arange(3) + (k - 1.5) * .2, R[R.exp == e].auc.to_numpy(), .18, color=COL[e], label=e.split(":")[0])
ax[1].axhline(.5, ls="--", c="#aab4be"); ax[1].set_xticks(range(3)); ax[1].set_xticklabels([f"seed {s}" for s in (42, 43, 44)]); ax[1].set_ylim(0, 1); ax[1].set_ylabel("Test AUC"); ax[1].set_title("AUC by seed"); ax[1].legend(ncol=4, frameon=False, loc="upper left")
for a_ in ax: a_.spines["top"].set_visible(False); a_.spines["right"].set_visible(False)
fig.suptitle(f"OASIS-1 + OASIS-2: ResNet18 ({MODE}), single subject-level split (no cross-validation)", x=.01, ha="left", fontsize=13, weight="bold")
plt.tight_layout(); fig.savefig(OUT / "single_split_resnet.png", dpi=150); print("DONE", MODE)
