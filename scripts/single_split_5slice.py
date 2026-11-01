"""NO cross-validation / NO folds. One subject-level stratified train/val/test split (70/15/15, stratified by dataset x label, random_state=2026)
of the 236-subject no-overlap cohort. Four settings: {fixed 78-82, Canny c±2} x {2.5D, late fusion with mean pooling}.
The Youden-J threshold is chosen on the VALIDATION set and applied once to the TEST set (0.5 is reported next to it). Model seeds 42/43/44.
Usage: python single_split_5slice.py cnn|resnet"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import oasis_lib as L, oasis_5slice as S

backbone = sys.argv[1]; assert backbone in ("cnn", "resnet"); torch.set_num_threads(6)
OUT = L.ROOT / "results" / "single_split_5slice"; OUT.mkdir(parents=True, exist_ok=True)
strata = np.array([f"{a}_{b}" for a, b in zip(S.DS, S.Y)]); idx = np.arange(len(S.SID))
tr, tmp = train_test_split(idx, test_size=.30, random_state=2026, stratify=strata); va, te = train_test_split(tmp, test_size=.50, random_state=2026, stratify=strata[tmp])
assert set(S.SID[tr]).isdisjoint(S.SID[va]) and set(S.SID[tr]).isdisjoint(S.SID[te]) and set(S.SID[va]).isdisjoint(S.SID[te])
pd.DataFrame(dict(subject_id=S.SID, dataset=S.DS, label=S.Y, split=np.select([np.isin(idx, tr), np.isin(idx, va)], ["train", "val"], "test"))).to_csv(OUT / "split.csv", index=False)
print(f"train {len(tr)} | val {len(va)} | test {len(te)} | test CDR>0 rate {S.Y[te].mean():.2f} | test OASIS-2: {(S.DS[te] == 'OASIS2').sum()}", flush=True)
ARMS = {"2.5D fixed 78-82": (S.Z["fixed5"], "stack"), "Late fusion (mean pool) fixed 78-82": (S.Z["fixed5"], "late_mean"),
        "2.5D Canny c±2": (S.Z["canny5"], "stack"), "Late fusion (mean pool) Canny c±2": (S.Z["canny5"], "late_mean")}

def youden(y, p):
    fpr, tpr, thr = roc_curve(y, p); j = tpr - fpr; best = np.flatnonzero(j >= j.max() - 1e-12); t = thr[best]; t = t[np.isfinite(t)] if np.isfinite(t).any() else np.array([.5])
    return float(t[np.argmin(np.abs(t - .5))]), float(j.max())
def cm(y, pred): return int(((pred == 0) & (y == 0)).sum()), int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum()), int(((pred == 1) & (y == 1)).sum())

rows, preds = [], []
for name, (arr, method) in ARMS.items():
    for seed in (42, 43, 44):
        model, ep = S.train(arr, method, backbone, tr, va, seed)
        pv, yv = S.predict(model, S.loader(arr, va, backbone, False)); pt, yt = S.predict(model, S.loader(arr, te, backbone, False)); thr, J = youden(yv, pv)
        r = dict(backbone=backbone, exp=name, seed=seed, epochs=ep, thr_youden=thr, J_val=J, val_auc=roc_auc_score(yv, pv), test_auc=roc_auc_score(yt, pt))
        for tag, t in (("05", .5), ("youden", thr)):
            tn, fp, fn, tp = cm(yt, (pt >= t).astype(int)); r.update({f"TN_{tag}": tn, f"FP_{tag}": fp, f"FN_{tag}": fn, f"TP_{tag}": tp, f"acc_{tag}": (tn + tp) / len(yt), f"sens_{tag}": tp / (tp + fn), f"spec_{tag}": tn / (tn + fp)})
        rows.append(r); preds.append(pd.DataFrame(dict(exp=name, seed=seed, subject_id=S.SID[te], dataset=S.DS[te], true=yt, prob=pt, thr_youden=thr)))
        print(f"[{backbone}] {name:38s} seed{seed} thr={thr:.3f} acc@0.5={r['acc_05']:.3f} acc@Youden={r['acc_youden']:.3f} AUC={r['test_auc']:.3f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT / f"{backbone}_per_seed.csv", index=False); pd.concat(preds).to_csv(OUT / f"{backbone}_test_predictions.csv", index=False)
print("ALL DONE", flush=True)
