# NOTE (kept for the record): this script ran on the 279-subject OASIS-1 + OASIS-2 cohort (data/combined), which was later found to contain
# people who are present in both datasets. All results from it are invalid and that cohort is no longer in the project.
# See build_combined.py (de-duplicated 236-subject cohort) and final_cv.py for the valid experiments.
"""Same four experiments (A-D, small 2D CNN) on the combined OASIS-1 + OASIS-2 cohort, but with ONE subject-level
stratified 70/15/15 split (stratified by dataset x label, random_state=42) instead of repeated 5-fold CV.
Model seeds 42/43/44 change only initialisation/shuffling. Test set evaluated once per run."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from combined_experiments import EXPS, fit_predict, sid, y_all, ds, strat, ROOT
OUT = ROOT / "results" / "combined_single_split"; OUT.mkdir(parents=True, exist_ok=True)
idx = np.arange(len(sid)); tr, tmp = train_test_split(idx, test_size=.30, random_state=42, stratify=strat)
va, te = train_test_split(tmp, test_size=.50, random_state=42, stratify=strat[tmp])
assert set(sid[tr]).isdisjoint(sid[va]) and set(sid[tr]).isdisjoint(sid[te]) and set(sid[va]).isdisjoint(sid[te])
print("split sizes train/val/test:", len(tr), len(va), len(te), "| test by dataset x label:", pd.Series(strat[te]).value_counts().to_dict(), flush=True)
pd.DataFrame(dict(subject_id=sid, dataset=ds, label=y_all, split=np.select([np.isin(idx, tr), np.isin(idx, va)], ["train", "val"], "test"))).to_csv(OUT / "split.csv", index=False)
rows, preds = [], {}
for name, views in EXPS.items():
    for seed in (42, 43, 44):
        p = fit_predict(views, tr, va, te, seed); y = y_all[te]; pred = (p >= .5).astype(int); tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        r = dict(exp=name, seed=seed, auc=roc_auc_score(y, p), acc=float((pred == y).mean()), bal_acc=balanced_accuracy_score(y, pred), f1=f1_score(y, pred, zero_division=0),
                 sens=tp / (tp + fn), spec=tn / (tn + fp), TN=tn, FP=fp, FN=fn, TP=tp)
        for d in ("OASIS1", "OASIS2"): m = ds[te] == d; r[f"auc_{d}"] = roc_auc_score(y[m], p[m])
        rows.append(r); preds[(name, seed)] = p; print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
R = pd.DataFrame(rows); R.to_csv(OUT / "per_seed.csv", index=False)
S = R.groupby("exp")[["auc", "acc", "bal_acc", "sens", "spec", "auc_OASIS1", "auc_OASIS2"]].agg(["mean", "std"]).round(3); S.to_csv(OUT / "summary.csv"); print(S.to_string())
EX = list(EXPS); COL = dict(zip(EX, ["#2a6f97", "#e09f3e", "#6a994e", "#d1495b"])); y = y_all[te]
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for e in EX:
    a_ = R[R.exp == e].auc
    for k, seed in enumerate((42, 43, 44)):
        f, t, _ = roc_curve(y, preds[(e, seed)]); ax[0].plot(f, t, color=COL[e], lw=2.2 if k == 0 else 1, alpha=1 if k == 0 else .4, label=f"{e}: AUC {a_.mean():.3f} ± {a_.std():.3f}" if k == 0 else None)
ax[0].plot([0, 1], [0, 1], "--", c="#aab4be"); ax[0].set_title(f"Test-set ROC (n = {len(te)}; bold = seed 42, faint = seeds 43, 44)"); ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate"); ax[0].legend(loc="lower right", fontsize=8.5, frameon=False)
for k, e in enumerate(EX):
    v = R[R.exp == e].auc.to_numpy(); ax[1].bar(np.arange(3) + (k - 1.5) * .2, v, .18, color=COL[e], label=e.split(":")[0])
ax[1].axhline(.5, ls="--", c="#aab4be"); ax[1].set_xticks(range(3)); ax[1].set_xticklabels([f"seed {s}" for s in (42, 43, 44)]); ax[1].set_ylim(0, 1); ax[1].set_ylabel("Test AUC"); ax[1].set_title("AUC by seed"); ax[1].legend(ncol=4, frameon=False, loc="upper left")
for a_ in ax: a_.spines["top"].set_visible(False); a_.spines["right"].set_visible(False)
fig.suptitle("OASIS-1 + OASIS-2: single subject-level split (no cross-validation), small 2D CNN", x=.01, ha="left", fontsize=13, weight="bold")
plt.tight_layout(); fig.savefig(OUT / "single_split_results.png", dpi=150)
