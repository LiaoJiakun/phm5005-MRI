# NOTE (kept for the record): this script ran on the 279-subject OASIS-1 + OASIS-2 cohort (data/combined), which was later found to contain
# people who are present in both datasets. All results from it are invalid and that cohort is no longer in the project.
# See build_combined.py (de-duplicated 236-subject cohort) and final_cv.py for the valid experiments.
"""OASIS-1 + OASIS-2 (first visit) combined cohort, small 2D CNN only (no ResNet). Four experiments:
  A  fixed axial (80)                        single-branch CNN
  B  fixed coronal (105)                     single-branch CNN
  C  fixed axial (80) + fixed coronal (105)  dual-branch CNN
  D  Canny axial     + Canny coronal         dual-branch CNN
Repeated stratified 5-fold subject-level CV (3 repeats), stratified by dataset x label; identical folds/seeds for all experiments.
Resumable: results/combined_experiments/fold_results.csv"""
import copy, random, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT / "results" / "combined_experiments"; OUT.mkdir(parents=True, exist_ok=True)
REPEATS, FOLDS, BATCH, EPOCHS, PAT, LR, WD = 3, 5, 16, 35, 7, 1e-3, 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Z = np.load(ROOT / "data" / "combined" / "combined_slices.npz", allow_pickle=True)
sid, y_all, ds = Z["subject_id"], Z["label"].astype(int), Z["dataset"]
strat = np.array([f"{d}_{l}" for d, l in zip(ds, y_all)])
EXPS = {"A: fixed axial 80": [Z["fixed_axial"]], "B: fixed coronal 105": [Z["fixed_coronal"]],
        "C: fixed axial+coronal": [Z["fixed_axial"], Z["fixed_coronal"]], "D: Canny axial+coronal": [Z["canny_axial"], Z["canny_coronal"]]}

def branch(): return nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                   nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
class Net(nn.Module):
    def __init__(s, nb):
        super().__init__(); s.b = nn.ModuleList([branch() for _ in range(nb)])
        s.head = nn.Sequential(nn.Linear(64 * nb, 32), nn.ReLU(), nn.Dropout(.3), nn.Linear(32, 2))
    def forward(s, xs): return s.head(torch.cat([b(x) for b, x in zip(s.b, xs)], 1))
def tf(aug): return transforms.Compose([transforms.ToPILImage(), *([transforms.RandomRotation(5)] if aug else []), transforms.ToTensor(), transforms.Normalize((.5,), (.5,))])
class DS(Dataset):
    def __init__(s, views, idx, aug): s.v, s.i, s.t = views, idx, tf(aug)
    def __len__(s): return len(s.i)
    def __getitem__(s, k): i = s.i[k]; return [s.t(v[i]) for v in s.v], torch.tensor(y_all[i])
def collate(b): return [torch.stack(x) for x in zip(*[s[0] for s in b])], torch.stack([s[1] for s in b])

def fit_predict(views, tr, va, te, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); g = torch.Generator(); g.manual_seed(seed)
    mk = lambda idx, aug, sh: DataLoader(DS(views, idx, aug), batch_size=BATCH, shuffle=sh, generator=g if sh else None, collate_fn=collate)
    dtr, dva, dte = mk(tr, True, True), mk(va, False, False), mk(te, False, False)
    model = Net(len(views)).to(DEVICE); opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD); crit = nn.CrossEntropyLoss()
    best, state, wait = 1e9, None, 0
    for ep in range(EPOCHS):
        model.train()
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

if __name__ == "__main__":
    f_res, f_oof = OUT / "fold_results.csv", OUT / "oof_predictions.csv"
    done = pd.read_csv(f_res) if f_res.exists() else pd.DataFrame(columns=["repeat", "fold", "exp"])
    oof = pd.read_csv(f_oof) if f_oof.exists() else pd.DataFrame(); t0 = time.time()
    print("N =", len(sid), pd.Series(strat).value_counts().to_dict(), flush=True)
    for rep in range(REPEATS):
        for fold, (trv, te) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=300 + rep).split(sid, strat)):
            tr, va = train_test_split(trv, test_size=.15, stratify=strat[trv], random_state=400 + rep * 10 + fold)
            for name, views in EXPS.items():
                if ((done.repeat == rep) & (done.fold == fold) & (done.exp == name)).any(): continue
                p = fit_predict(views, tr, va, te, rep * 100 + fold); auc = roc_auc_score(y_all[te], p); pred = (p >= .5).astype(int)
                done = pd.concat([done, pd.DataFrame([dict(repeat=rep, fold=fold, exp=name, auc=auc, acc=float((pred == y_all[te]).mean()), bal_acc=balanced_accuracy_score(y_all[te], pred))])], ignore_index=True)
                oof = pd.concat([oof, pd.DataFrame(dict(repeat=rep, fold=fold, exp=name, subject_id=sid[te], dataset=ds[te], true=y_all[te], prob=p))], ignore_index=True)
                done.to_csv(f_res, index=False); oof.to_csv(f_oof, index=False)
                print(f"[{(time.time()-t0)/60:5.1f} min] rep{rep} fold{fold} {name:24s} AUC={auc:.3f}", flush=True)
    print("ALL DONE")
