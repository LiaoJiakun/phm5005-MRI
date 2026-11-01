"""Youden's J threshold, chosen WITHOUT touching the test fold.
For every (repeat, fold, arm): train as before (same folds/seeds), predict the inner VALIDATION split, take the threshold that maximises
J = sensitivity + specificity - 1 on those validation predictions, and apply it to the held-out TEST fold. The 0.5 threshold is reported next to it.
Arms = the four requested cells (fixed / Canny x 2.5D / late fusion with mean pooling). Usage: python youden_cv.py cnn|resnet"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, roc_curve
import oasis_lib as L, oasis_5slice as S

backbone = sys.argv[1]; assert backbone in ("cnn", "resnet"); torch.set_num_threads(6)
OUT = L.ROOT / "results" / "youden" / backbone; OUT.mkdir(parents=True, exist_ok=True)
ARMS = {"M1: 2.5D fixed 78-82": (S.Z["fixed5"], "stack"), "M2-mean: late fusion (mean pool) fixed 78-82": (S.Z["fixed5"], "late_mean"),
        "M1: 2.5D Canny c±2": (S.Z["canny5"], "stack"), "M2-mean: late fusion (mean pool) Canny c±2": (S.Z["canny5"], "late_mean")}

def youden_threshold(y, p):
    """Threshold maximising J = TPR - FPR (ties -> the one closest to 0.5). Falls back to 0.5 if y has a single class."""
    if len(set(y)) < 2: return 0.5
    fpr, tpr, thr = roc_curve(y, p); j = tpr - fpr; best = np.flatnonzero(j >= j.max() - 1e-12)
    t = thr[best]; t = t[np.isfinite(t)] if np.isfinite(t).any() else np.array([0.5]); return float(t[np.argmin(np.abs(t - 0.5))])

def metrics(y, p, t):
    pred = (p >= t).astype(int); tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    return dict(acc=(tp + tn) / len(y), sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1))

f_res, f_pred = OUT / "fold_results.csv", OUT / "predictions.csv"
done = pd.read_csv(f_res) if f_res.exists() else pd.DataFrame(columns=["repeat", "fold", "exp"]); preds = pd.read_csv(f_pred) if f_pred.exists() else pd.DataFrame(); t0 = time.time()
data = {"subject_id": S.SID, "dataset": S.DS, "label": S.Y, "strata": np.array([f"{a}_{b}" for a, b in zip(S.DS, S.Y)])}
for rep in range(L.N_REPEATS):
    for fold, (tr, va, te) in enumerate(L.cv_folds(data, rep)):
        for name, (arr, method) in ARMS.items():
            if ((done["repeat"] == rep) & (done["fold"] == fold) & (done["exp"] == name)).any(): continue
            model, ep = S.train(arr, method, backbone, tr, va, L.SEEDS["model"] + rep * 100 + fold)
            pv, yv = S.predict(model, S.loader(arr, va, backbone, False)); pt, yt = S.predict(model, S.loader(arr, te, backbone, False))
            thr = youden_threshold(yv, pv); m05, mY = metrics(yt, pt, 0.5), metrics(yt, pt, thr)
            row = dict(repeat=rep, fold=fold, exp=name, n_val=len(va), thr_youden=thr, val_auc=roc_auc_score(yv, pv), test_auc=roc_auc_score(yt, pt),
                       **{f"{k}_05": v for k, v in m05.items()}, **{f"{k}_youden": v for k, v in mY.items()})
            done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
            preds = pd.concat([preds, pd.DataFrame(dict(repeat=rep, fold=fold, exp=name, subject_id=S.SID[te], dataset=S.DS[te], true=yt, prob=pt, thr_youden=thr))], ignore_index=True)
            done.to_csv(f_res, index=False); preds.to_csv(f_pred, index=False)
            print(f"[{backbone} {(time.time() - t0) / 60:6.1f} min] rep{rep} fold{fold} {name:44s} thr={thr:.3f} acc@0.5={m05['acc']:.3f} acc@Youden={mY['acc']:.3f}", flush=True)
print("ALL DONE", flush=True)
