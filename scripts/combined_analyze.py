# NOTE (kept for the record): this script ran on the 279-subject OASIS-1 + OASIS-2 cohort (data/combined), which was later found to contain
# people who are present in both datasets. All results from it are invalid and that cohort is no longer in the project.
# See build_combined.py (de-duplicated 236-subject cohort) and final_cv.py for the valid experiments.
from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
R = Path(__file__).resolve().parent.parent / "results" / "combined_experiments"
fr = pd.read_csv(R / "fold_results.csv"); oof = pd.read_csv(R / "oof_predictions.csv")
EX = list(dict.fromkeys(fr.exp)); COL = dict(zip(EX, ["#2a6f97", "#e09f3e", "#6a994e", "#d1495b"])); rng = np.random.default_rng(0)
S, CM = [], []
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
for e in EX:
    f = fr[fr.exp == e]; o = oof[oof.exp == e].groupby("subject_id").agg(true=("true", "first"), prob=("prob", "mean"), dataset=("dataset", "first"))
    def ci(oo):
        b = []
        for _ in range(2000):
            i = rng.integers(0, len(oo), len(oo)); y = oo.true.to_numpy()[i]
            if len(set(y)) == 2: b.append(roc_auc_score(y, oo.prob.to_numpy()[i]))
        return np.percentile(b, [2.5, 97.5])
    lo, hi = ci(o); row = dict(exp=e, n_folds=len(f), fold_auc=f.auc.mean(), fold_sd=f.auc.std(), pooled_auc=roc_auc_score(o.true, o.prob), ci_lo=lo, ci_hi=hi, acc=f.acc.mean())
    for d in ("OASIS1", "OASIS2"):
        od = o[o.dataset == d]; row[f"auc_{d}"] = roc_auc_score(od.true, od.prob)
    S.append(row); fp, tp, _ = roc_curve(o.true, o.prob); ax[0].plot(fp, tp, color=COL[e], lw=2.2, label=f"{e}: {row['pooled_auc']:.3f} [{lo:.2f}, {hi:.2f}]")
    pred = (o.prob >= .5).astype(int); tn, fp_, fn, tp_ = confusion_matrix(o.true, pred, labels=[0, 1]).ravel()
    CM.append(dict(exp=e, TN=tn, FP=fp_, FN=fn, TP=tp_, sensitivity=tp_ / (tp_ + fn), specificity=tn / (tn + fp_), accuracy=(tn + tp_) / len(o)))
ax[0].plot([0, 1], [0, 1], "--", c="#aab4be"); ax[0].set_title("Pooled out-of-fold ROC (n = 279)"); ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate"); ax[0].legend(loc="lower right", fontsize=8.5, frameon=False)
PAIRS = [("B: fixed coronal 105", "A: fixed axial 80"), ("C: fixed axial+coronal", "A: fixed axial 80"), ("C: fixed axial+coronal", "B: fixed coronal 105"), ("D: Canny axial+coronal", "C: fixed axial+coronal"), ("D: Canny axial+coronal", "A: fixed axial 80")]
P = []
for k, (a, c) in enumerate(PAIRS):
    x = fr[fr.exp == a].sort_values(["repeat", "fold"]).auc.to_numpy(); y = fr[fr.exp == c].sort_values(["repeat", "fold"]).auc.to_numpy(); d = x - y
    t = stats.t.interval(.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d)); P.append(dict(comparison=f"{a[0]} - {c[0]}", diff=d.mean(), ci_lo=t[0], ci_hi=t[1], wilcoxon_p=stats.wilcoxon(d).pvalue, better_folds=int((d > 0).sum()), n=len(d)))
    ax[1].errorbar(d.mean(), k, xerr=[[d.mean() - t[0]], [t[1] - d.mean()]], fmt="o", color="#1f2933", capsize=5)
ax[1].axvline(0, ls="--", c="#aab4be"); ax[1].set_yticks(range(len(PAIRS))); ax[1].set_yticklabels([p["comparison"].replace(" - ", "  −  ") for p in P]); ax[1].invert_yaxis()
ax[1].set_xlabel("Paired fold-AUC difference (mean, 95% CI; 15 folds)"); ax[1].set_title("Paired differences between experiments")
Sd = pd.DataFrame(S); w = .36
for k, d in enumerate(("OASIS1", "OASIS2")):
    ax[2].bar(np.arange(len(EX)) + (k - .5) * w, Sd[f"auc_{d}"], w, label=d, color=["#5b8db8", "#e6a1ab"][k])
    for i, v in enumerate(Sd[f"auc_{d}"]): ax[2].text(i + (k - .5) * w, v + .01, f"{v:.2f}", ha="center", fontsize=8.5)
ax[2].axhline(.5, ls="--", c="#aab4be"); ax[2].set_xticks(range(len(EX))); ax[2].set_xticklabels([e.split(":")[0] for e in EX]); ax[2].set_ylim(0, 1); ax[2].legend(frameon=False); ax[2].set_title("Out-of-fold AUC by dataset"); ax[2].set_ylabel("AUC")
for a_ in ax: a_.spines["top"].set_visible(False); a_.spines["right"].set_visible(False)
fig.suptitle("OASIS-1 + OASIS-2 (first visit): small 2D CNN, repeated 5-fold subject-level CV", x=.01, ha="left", fontsize=13, weight="bold")
plt.tight_layout(); fig.savefig(R / "combined_results.png", dpi=150)
S, P, CM = Sd, pd.DataFrame(P), pd.DataFrame(CM); S.to_csv(R / "summary.csv", index=False); P.to_csv(R / "paired.csv", index=False); CM.to_csv(R / "confusion.csv", index=False)
pd.set_option("display.width", 220); print(S.round(3).to_string()); print(P.round(3).to_string()); print(CM.round(3).to_string())
fig, ax = plt.subplots(1, 4, figsize=(16, 4.3))
for a, (e, r) in zip(ax, zip(EX, CM.itertuples())):
    m = np.array([[r.TN, r.FP], [r.FN, r.TP]]); a.imshow(m / m.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(m): a.text(j, i, f"{v}\n({v / m[i].sum():.0%})", ha="center", va="center", fontsize=12, color="white" if m[i, j] / m[i].sum() > .55 else "#1f2933")
    a.set_xticks([0, 1]); a.set_xticklabels(["CDR=0", "CDR>0"]); a.set_yticks([0, 1]); a.set_yticklabels(["CDR=0", "CDR>0"]); a.set_xlabel("Predicted"); a.set_title(f"{e}\nacc {r.accuracy:.2f} · sens {r.sensitivity:.2f} · spec {r.specificity:.2f}", fontsize=9.5)
ax[0].set_ylabel("Actual"); plt.tight_layout(); fig.savefig(R / "combined_confusion.png", dpi=150)
