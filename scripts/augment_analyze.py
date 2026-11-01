"""Paired comparison of augmentation arms against the baseline (rotation +-5 deg) on the best model.
Usage: python augment_analyze.py old|square   (old = squashed axial slices, square = square-padded)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
which = sys.argv[1]; R = Path(__file__).resolve().parent.parent / "results"
ARM_DIR = R / ("augmentation" if which == "old" else "augmentation_square")
BASE = (R / "five_slice" / "resnet") if which == "old" else (R / "square" / "fixed")
BASE_EXP = "M2: late fusion fixed 78-82"
LAB = {"baseline": "Baseline (rotation ±5°)", "G": "G: geometric (rot ±10°, shift ±5%, scale ±5%)", "I": "I: rotation ±5° + intensity (bias field, gamma, noise, blur)", "GI": "GI: geometric + intensity"}
COL = {"baseline": "#2a6f97", "G": "#e09f3e", "I": "#6a994e", "GI": "#d1495b"}
rng = np.random.default_rng(0); rows, preds, folds = [], {}, {}
fb = pd.read_csv(BASE / "fold_results.csv"); fb = fb[fb.exp == BASE_EXP]; ob = pd.read_csv(BASE / "oof_predictions.csv"); ob = ob[ob.exp == BASE_EXP]
folds["baseline"] = fb.sort_values(["repeat", "fold"]); preds["baseline"] = ob
for a in ("G", "I", "GI"):
    folds[a] = pd.read_csv(ARM_DIR / a / "fold_results.csv").sort_values(["repeat", "fold"]); preds[a] = pd.read_csv(ARM_DIR / a / "oof_predictions.csv")
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
for a in ("baseline", "G", "I", "GI"):
    o = preds[a].groupby("subject_id").agg(true=("true", "first"), prob=("prob", "mean")); f = folds[a]; bs = []
    for _ in range(2000):
        i = rng.integers(0, len(o), len(o)); y = o.true.to_numpy()[i]
        if len(set(y)) == 2: bs.append(roc_auc_score(y, o.prob.to_numpy()[i]))
    pred = (o.prob >= .5).astype(int); tn, fp, fn, tp = confusion_matrix(o.true, pred, labels=[0, 1]).ravel()
    rows.append(dict(arm=a, n_folds=len(f), pooled_auc=roc_auc_score(o.true, o.prob), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5), fold_auc=f.auc.mean(), fold_sd=f.auc.std(),
                     accuracy=(tn + tp) / len(o), sensitivity=tp / (tp + fn), specificity=tn / (tn + fp), epochs=f.epochs.mean()))
    fpr, tpr, _ = roc_curve(o.true, o.prob); ax[0].plot(fpr, tpr, color=COL[a], lw=2.2, label=f"{LAB[a].split(':')[0]}: AUC {rows[-1]['pooled_auc']:.3f} [{rows[-1]['ci_lo']:.2f}, {rows[-1]['ci_hi']:.2f}]")
S = pd.DataFrame(rows); P = []
for k, a in enumerate(("G", "I", "GI")):
    for met in ("auc", "acc"):
        d = folds[a][met].to_numpy() - folds["baseline"][met].to_numpy(); t = stats.t.interval(.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
        P.append(dict(arm=a, metric=met, diff=d.mean(), ci_lo=t[0], ci_hi=t[1], wilcoxon_p=stats.wilcoxon(d).pvalue, better_folds=int((d > 0).sum()), n=len(d)))
        if met == "auc": ax[1].errorbar(d.mean(), k, xerr=[[d.mean() - t[0]], [t[1] - d.mean()]], fmt="o", color=COL[a], capsize=5)
P = pd.DataFrame(P); ax[0].plot([0, 1], [0, 1], ":", c="gray"); ax[0].set_title(f"Augmentation arms ({which} preprocessing): out-of-fold ROC, n = 236"); ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate"); ax[0].legend(loc="lower right", fontsize=8.5, frameon=False)
ax[1].axvline(0, ls="--", c="gray"); ax[1].set_yticks(range(3)); ax[1].set_yticklabels(["G geometric", "I intensity", "GI both"]); ax[1].invert_yaxis(); ax[1].set_xlabel("Paired fold-AUC difference vs baseline (mean, 95% CI; 10 folds)"); ax[1].set_title("Effect of augmentation vs rotation-only baseline")
for a_ in ax: a_.spines["top"].set_visible(False); a_.spines["right"].set_visible(False)
plt.tight_layout(); fig.savefig(ARM_DIR / "augmentation_results.png", dpi=150); S.to_csv(ARM_DIR / "summary.csv", index=False); P.to_csv(ARM_DIR / "paired.csv", index=False)
pd.set_option("display.width", 220); print(S.round(3).to_string()); print(P.round(3).to_string())
