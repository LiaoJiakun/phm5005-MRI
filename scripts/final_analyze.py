"""Summaries for the final run: pooled out-of-fold AUC (+bootstrap CI), fold-level AUC, paired differences,
per-dataset AUC, confusion matrices, ROC figure. Reads results/final_cv/{cnn,resnet}/ ; writes results/final_cv/."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
R = Path(__file__).resolve().parent.parent / "results" / "final_cv"
BB = {"cnn": "Small 2D CNN (from scratch)", "resnet": "ResNet18 (ImageNet, full fine-tune)"}
backs = [b for b in BB if (R / b / "fold_results.csv").exists()]
EX = ["A: fixed axial 80", "B: fixed coronal 105", "C: fixed axial+coronal", "D: Canny axial+coronal"]
COL = dict(zip(EX, ["#2a6f97", "#e09f3e", "#6a994e", "#d1495b"]))
PAIRS = [("B: fixed coronal 105", "A: fixed axial 80"), ("C: fixed axial+coronal", "A: fixed axial 80"), ("C: fixed axial+coronal", "B: fixed coronal 105"),
         ("D: Canny axial+coronal", "C: fixed axial+coronal"), ("D: Canny axial+coronal", "A: fixed axial 80")]
rng = np.random.default_rng(0); S, P, CM = [], [], []
fig, ax = plt.subplots(2, len(backs), figsize=(6.4 * len(backs), 9.4), squeeze=False)
for j, b in enumerate(backs):
    fr = pd.read_csv(R / b / "fold_results.csv"); oof = pd.read_csv(R / b / "oof_predictions.csv")
    for e in EX:
        f = fr[fr.exp == e]; o = oof[oof.exp == e].groupby("subject_id").agg(true=("true", "first"), prob=("prob", "mean"), dataset=("dataset", "first"))
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(o), len(o)); y = o.true.to_numpy()[i]
            if len(set(y)) == 2: bs.append(roc_auc_score(y, o.prob.to_numpy()[i]))
        row = dict(backbone=b, exp=e, n_folds=len(f), fold_auc=f.auc.mean(), fold_sd=f.auc.std(), pooled_auc=roc_auc_score(o.true, o.prob),
                   ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5), acc=f.acc.mean(), n_subjects=len(o))
        for d in ("OASIS1", "OASIS2"):
            od = o[o.dataset == d]; row[f"auc_{d}"] = roc_auc_score(od.true, od.prob)
        S.append(row); fp, tp, _ = roc_curve(o.true, o.prob)
        ax[0, j].plot(fp, tp, color=COL[e], lw=2.2, label=f"{e}: {row['pooled_auc']:.3f} [{row['ci_lo']:.2f}, {row['ci_hi']:.2f}]")
        pred = (o.prob >= .5).astype(int); tn, fp_, fn, tp_ = confusion_matrix(o.true, pred, labels=[0, 1]).ravel()
        CM.append(dict(backbone=b, exp=e, TN=tn, FP=fp_, FN=fn, TP=tp_, sensitivity=tp_ / (tp_ + fn), specificity=tn / (tn + fp_), accuracy=(tn + tp_) / len(o)))
    ax[0, j].plot([0, 1], [0, 1], "--", c="#aab4be"); ax[0, j].set_title(f"{BB[b]}\npooled out-of-fold ROC (n = {len(o)})", fontsize=10.5)
    ax[0, j].set_xlabel("False positive rate"); ax[0, j].set_ylabel("True positive rate"); ax[0, j].legend(loc="lower right", fontsize=8.3, frameon=False)
    for k, (a, c) in enumerate(PAIRS):
        x = fr[fr.exp == a].sort_values(["repeat", "fold"]).auc.to_numpy(); y_ = fr[fr.exp == c].sort_values(["repeat", "fold"]).auc.to_numpy(); d = x - y_
        t = stats.t.interval(.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
        P.append(dict(backbone=b, comparison=f"{a[0]} - {c[0]}", diff=d.mean(), ci_lo=t[0], ci_hi=t[1], wilcoxon_p=stats.wilcoxon(d).pvalue, better_folds=int((d > 0).sum()), n=len(d)))
        ax[1, j].errorbar(d.mean(), k, xerr=[[d.mean() - t[0]], [t[1] - d.mean()]], fmt="o", color="#1f2933", capsize=5)
    ax[1, j].axvline(0, ls="--", c="#aab4be"); ax[1, j].set_yticks(range(len(PAIRS))); ax[1, j].set_yticklabels([f"{a[0]} − {c[0]}" for a, c in PAIRS]); ax[1, j].invert_yaxis()
    ax[1, j].set_xlabel("Paired fold-AUC difference (mean, 95% CI; 10 folds)"); ax[1, j].set_title("Paired differences between experiments", fontsize=10.5)
    for a_ in (ax[0, j], ax[1, j]): a_.spines["top"].set_visible(False); a_.spines["right"].set_visible(False)
fig.suptitle("OASIS-1 + OASIS-2 (PART1+PART2, first visit): repeated stratified 5-fold subject-level CV", x=.01, ha="left", fontsize=13, weight="bold")
plt.tight_layout(); fig.savefig(R / "final_results.png", dpi=150)
S, P, CM = pd.DataFrame(S), pd.DataFrame(P), pd.DataFrame(CM); S.to_csv(R / "summary.csv", index=False); P.to_csv(R / "paired.csv", index=False); CM.to_csv(R / "confusion.csv", index=False)
pd.set_option("display.width", 230); print(S.round(3).to_string()); print(P.round(3).to_string()); print(CM.round(3).to_string())
fig, ax = plt.subplots(len(backs), 4, figsize=(15, 3.9 * len(backs) + .4), squeeze=False)
for i, b in enumerate(backs):
    for j, e in enumerate(EX):
        r = CM[(CM.backbone == b) & (CM.exp == e)].iloc[0]; m = np.array([[r.TN, r.FP], [r.FN, r.TP]]); a = ax[i, j]; a.imshow(m / m.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for (p_, q_), v in np.ndenumerate(m): a.text(q_, p_, f"{v}\n({v / m[p_].sum():.0%})", ha="center", va="center", fontsize=12, color="white" if m[p_, q_] / m[p_].sum() > .55 else "#1f2933")
        a.set_xticks([0, 1]); a.set_xticklabels(["CDR=0", "CDR>0"]); a.set_yticks([0, 1]); a.set_yticklabels(["CDR=0", "CDR>0"]); a.set_xlabel("Predicted")
        a.set_title(f"{e}\nacc {r.accuracy:.2f} · sens {r.sensitivity:.2f} · spec {r.specificity:.2f}", fontsize=9.5)
        if j == 0: a.set_ylabel(("CNN\n" if b == "cnn" else "ResNet18\n") + "Actual")
fig.suptitle("Out-of-fold confusion matrices (threshold 0.5; mean probability of the 2 repeats per subject)", x=.01, ha="left", fontsize=12, weight="bold")
plt.tight_layout(); fig.savefig(R / "final_confusion.png", dpi=150)
