"""Accuracy / confusion matrices / AUC for the mean-pooling trial.
Requested cells: {fixed 78-82, Canny c±2} x {2.5D, late fusion with mean pooling}. References: concat late fusion, single slice.
2.5D / concat / single-slice results come from results/five_slice (same cohort, folds, seeds, code)."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score, confusion_matrix
ROOT = Path(__file__).resolve().parent.parent / "results"; OUT = ROOT / "meanpool"
BB = {"cnn": "Small 2D CNN", "resnet": "ResNet18"}
MAIN = [("Fixed 78-82", "2.5D", "M1: 2.5D fixed 78-82"), ("Fixed 78-82", "Late fusion, mean pooling", "M2-mean: late fusion (mean pool) fixed 78-82"),
        ("Canny c±2", "2.5D", "M1: 2.5D Canny c±2"), ("Canny c±2", "Late fusion, mean pooling", "M2-mean: late fusion (mean pool) Canny c±2")]
REF = [("Fixed 78-82", "Late fusion, concatenation (ref.)", "M2: late fusion fixed 78-82"), ("Canny c±2", "Late fusion, concatenation (ref.)", "M2: late fusion Canny c±2"),
       ("Fixed 80", "Single slice (ref.)", "S1: single slice fixed 80"), ("Canny c", "Single slice (ref.)", "S1: single slice Canny c")]
def load(b, e):
    p = (OUT if e.startswith("M2-mean") else ROOT / "five_slice") / b
    fr, oof = pd.read_csv(p / "fold_results.csv"), pd.read_csv(p / "oof_predictions.csv"); return fr[fr.exp == e], oof[oof.exp == e]
rng = np.random.default_rng(0); rows, cms = [], {}
for b in BB:
    for slices, method, e in MAIN + REF:
        fr, oof = load(b, e); o = oof.groupby("subject_id").agg(true=("true", "first"), prob=("prob", "mean"))
        pred = (o.prob >= .5).astype(int); tn, fp, fn, tp = confusion_matrix(o.true, pred, labels=[0, 1]).ravel(); acc = (tn + tp) / len(o)
        bs = []; 
        for _ in range(2000):
            i = rng.integers(0, len(o), len(o)); bs.append(((pred.to_numpy()[i] == o.true.to_numpy()[i]).mean()))
        bsa = []
        for _ in range(2000):
            i = rng.integers(0, len(o), len(o)); y = o.true.to_numpy()[i]
            if len(set(y)) == 2: bsa.append(roc_auc_score(y, o.prob.to_numpy()[i]))
        rows.append(dict(backbone=b, slices=slices, method=method, exp=e, accuracy=acc, acc_ci_lo=np.percentile(bs, 2.5), acc_ci_hi=np.percentile(bs, 97.5), fold_acc=fr.acc.mean(), fold_acc_sd=fr.acc.std(),
                         sensitivity=tp / (tp + fn), specificity=tn / (tn + fp), auc=roc_auc_score(o.true, o.prob), auc_ci_lo=np.percentile(bsa, 2.5), auc_ci_hi=np.percentile(bsa, 97.5), fold_auc=fr.auc.mean(), TN=tn, FP=fp, FN=fn, TP=tp, params=int(fr.params.iloc[0])))
R = pd.DataFrame(rows); R.to_csv(OUT / "meanpool_summary.csv", index=False)
P = []
for b in BB:
    for a, c, lab in [("M2-mean: late fusion (mean pool) fixed 78-82", "M1: 2.5D fixed 78-82", "mean-pool − 2.5D (fixed)"), ("M2-mean: late fusion (mean pool) Canny c±2", "M1: 2.5D Canny c±2", "mean-pool − 2.5D (Canny)"),
                      ("M2-mean: late fusion (mean pool) fixed 78-82", "M2: late fusion fixed 78-82", "mean-pool − concat (fixed)"), ("M2-mean: late fusion (mean pool) Canny c±2", "M2: late fusion Canny c±2", "mean-pool − concat (Canny)")]:
        fa, fc = load(b, a)[0].sort_values(["repeat", "fold"]), load(b, c)[0].sort_values(["repeat", "fold"])
        for met in ("auc", "acc"):
            d = fa[met].to_numpy() - fc[met].to_numpy(); t = stats.t.interval(.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d)) if d.std() > 0 else (d.mean(), d.mean())
            P.append(dict(backbone=b, comparison=lab, metric=met, diff=d.mean(), ci_lo=t[0], ci_hi=t[1], wilcoxon_p=stats.wilcoxon(d).pvalue if d.std() > 0 else 1.0, better_folds=int((d > 0).sum()), n=len(d)))
P = pd.DataFrame(P); P.to_csv(OUT / "meanpool_paired.csv", index=False)
pd.set_option("display.width", 250); print(R[["backbone", "slices", "method", "accuracy", "acc_ci_lo", "acc_ci_hi", "fold_acc", "sensitivity", "specificity", "auc", "params"]].round(3).to_string()); print(P.round(3).to_string())
# confusion matrices of the 4 requested cells x 2 backbones
fig, ax = plt.subplots(2, 4, figsize=(15.5, 7.8))
for i, b in enumerate(BB):
    for j, (slices, method, e) in enumerate(MAIN):
        r = R[(R.backbone == b) & (R.exp == e)].iloc[0]; m = np.array([[r.TN, r.FP], [r.FN, r.TP]]); a = ax[i, j]; a.imshow(m / m.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for (p_, q_), v in np.ndenumerate(m): a.text(q_, p_, f"{v}\n({v / m[p_].sum():.0%})", ha="center", va="center", fontsize=12, color="white" if m[p_, q_] / m[p_].sum() > .55 else "#1f2933")
        a.set_xticks([0, 1]); a.set_xticklabels(["CDR=0", "CDR>0"]); a.set_yticks([0, 1]); a.set_yticklabels(["CDR=0", "CDR>0"]); a.set_xlabel("Predicted")
        a.set_title(f"{BB[b]} · {slices}\n{method}\nacc {r.accuracy:.3f} · sens {r.sensitivity:.2f} · spec {r.specificity:.2f} · AUC {r.auc:.3f}", fontsize=9)
        if j == 0: a.set_ylabel("Actual")
fig.suptitle("Mean-pooling trial, 236 subjects (no OASIS-1/2 overlap): out-of-fold confusion matrices (threshold 0.5)", x=.01, ha="left", fontsize=12, weight="bold")
plt.tight_layout(); fig.savefig(OUT / "meanpool_confusion.png", dpi=150)
