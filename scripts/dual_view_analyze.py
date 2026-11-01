from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
R = Path(__file__).resolve().parent.parent / "results" / "dual_view"
ARMS = ["axial", "coronal", "dual", "dual_axax"]; LAB = {"axial": "Axial only", "coronal": "Coronal only", "dual": "Dual (axial+coronal)", "dual_axax": "Dual control (axial+axial)"}
COL = {"axial": "#2a6f97", "coronal": "#e09f3e", "dual": "#d1495b", "dual_axax": "#7a7a7a"}
PAIRS = [("coronal", "axial"), ("dual", "axial"), ("dual", "dual_axax")]
rng = np.random.default_rng(0); S, P, backs = [], [], [b for b in ("cnn", "resnet") if (R / f"{b}_fold_results.csv").exists()]
fig, axs = plt.subplots(2, len(backs), figsize=(6.2 * len(backs) + 0.5, 9.5), squeeze=False)
for j, b in enumerate(backs):
    fr = pd.read_csv(R / f"{b}_fold_results.csv"); oof = pd.read_csv(R / f"{b}_oof_predictions.csv")
    for arm in ARMS:
        f = fr[fr.arm == arm]; o = oof[oof.arm == arm].groupby("subject_id").agg(true=("true", "first"), prob=("prob", "mean"))
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(o), len(o)); yy = o.true.to_numpy()[i]
            if len(set(yy)) == 2: bs.append(roc_auc_score(yy, o.prob.to_numpy()[i]))
        S.append(dict(backbone=b, arm=arm, n_folds=len(f), fold_auc=f.auc.mean(), fold_sd=f.auc.std(), pooled_auc=roc_auc_score(o.true, o.prob),
                      ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5), acc=f.acc.mean(), params=int(f.trainable_params.iloc[0])))
        fp, tp, _ = roc_curve(o.true, o.prob); r = S[-1]
        axs[0, j].plot(fp, tp, color=COL[arm], lw=2.2, label=f"{LAB[arm]}: {r['pooled_auc']:.3f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]")
    axs[0, j].plot([0, 1], [0, 1], "--", c="#aab4be"); axs[0, j].set_title({"cnn": "Small CNN branches (scratch)", "resnet": "ResNet18 branches (layer4 + head trained)"}[b])
    axs[0, j].set_xlabel("False positive rate"); axs[0, j].set_ylabel("True positive rate"); axs[0, j].legend(loc="lower right", fontsize=8.5, frameon=False)
    for k, (a, c) in enumerate(PAIRS):
        x = fr[fr.arm == a].sort_values(["repeat", "fold"]).auc.to_numpy(); y = fr[fr.arm == c].sort_values(["repeat", "fold"]).auc.to_numpy(); d = x - y
        ci = stats.t.interval(.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
        P.append(dict(backbone=b, comparison=f"{a} - {c}", diff=d.mean(), ci_lo=ci[0], ci_hi=ci[1], wilcoxon_p=stats.wilcoxon(d).pvalue, better_folds=int((d > 0).sum()), n=len(d)))
        axs[1, j].errorbar(d.mean(), k, xerr=[[d.mean() - ci[0]], [ci[1] - d.mean()]], fmt="o", color="#1f2933", capsize=5)
    axs[1, j].axvline(0, ls="--", c="#aab4be"); axs[1, j].set_yticks(range(len(PAIRS))); axs[1, j].set_yticklabels([f"{LAB[a]}\n− {LAB[c]}" for a, c in PAIRS], fontsize=8.5)
    axs[1, j].invert_yaxis(); axs[1, j].set_xlabel("Paired fold-AUC difference (mean, 95% CI; 10 folds)")
    for a_ in (axs[0, j], axs[1, j]): a_.spines["top"].set_visible(False); a_.spines["right"].set_visible(False)
S, P = pd.DataFrame(S), pd.DataFrame(P); S.to_csv(R / "dual_view_summary.csv", index=False); P.to_csv(R / "dual_view_paired.csv", index=False)
pd.set_option("display.width", 220); print(S.round(3).to_string()); print(P.round(3).to_string())
fig.suptitle("Axial vs coronal vs dual-branch (Canny-selected slices) — repeated 5-fold subject-level CV, n = 198", x=.01, ha="left", fontsize=13, weight="bold")
plt.tight_layout(); fig.savefig(R / "dual_view_results.png", dpi=150)
