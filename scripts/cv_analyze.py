from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
R = Path(__file__).resolve().parent.parent / "results" / "cv"
oof = pd.read_csv(R / "oof_predictions.csv")
# rebuild per-fold metrics from out-of-fold predictions (fold split is deterministic: StratifiedKFold(5, shuffle, rs=100+repeat))
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
z = np.load(R.parent.parent / "data" / "cohort" / "cohort_slices.npz", allow_pickle=True)
sid, yy_all = z["subject_id"], z["label"].astype(int); fold_of = {}
for rep in range(2):
    for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=100 + rep).split(sid, yy_all)):
        for i in te: fold_of[(rep, sid[i])] = k
oof["fold"] = [fold_of[(r, s)] for r, s in zip(oof["repeat"], oof.subject_id)]
rows_ = []
for (rep, fold, m, arm), g in oof.groupby(["repeat", "fold", "model", "arm"]):
    pred = (g.prob >= .5).astype(int)
    rows_.append(dict(repeat=rep, fold=fold, model=m, arm=arm, auc=roc_auc_score(g.true, g.prob), acc=(pred == g.true).mean(), bal_acc=balanced_accuracy_score(g.true, pred)))
fr = pd.DataFrame(rows_); fr.to_csv(R / "fold_results.csv", index=False)
rng = np.random.default_rng(0); models = ["CNN scratch", "ResNet18 full", "ResNet18 layer4"]
C = {"Fixed": "#2a6f97", "Canny": "#d1495b"}
rows = []
for m in models:
    for arm in ("Fixed", "Canny"):
        f = fr[(fr.model == m) & (fr.arm == arm)]
        o = oof[(oof.model == m) & (oof.arm == arm)].groupby("subject_id").agg(true=("true", "first"), prob=("prob", "mean"))
        boots = []
        for _ in range(2000):
            i = rng.integers(0, len(o), len(o)); yy = o.true.to_numpy()[i]
            if len(set(yy)) == 2: boots.append(roc_auc_score(yy, o.prob.to_numpy()[i]))
        rows.append(dict(model=m, arm=arm, fold_auc_mean=f.auc.mean(), fold_auc_sd=f.auc.std(), pooled_auc=roc_auc_score(o.true, o.prob),
                         ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5), acc=f.acc.mean(), bal_acc=f.bal_acc.mean()))
S = pd.DataFrame(rows); S.to_csv(R / "cv_summary.csv", index=False)
P = []
for m in models:
    a = fr[(fr.model == m) & (fr.arm == "Canny")].sort_values(["repeat", "fold"]).auc.to_numpy()
    b = fr[(fr.model == m) & (fr.arm == "Fixed")].sort_values(["repeat", "fold"]).auc.to_numpy(); d = a - b
    ci = stats.t.interval(.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    P.append(dict(model=m, canny_minus_fixed=d.mean(), ci_lo=ci[0], ci_hi=ci[1], wilcoxon_p=stats.wilcoxon(d).pvalue, canny_better_folds=int((d > 0).sum()), n=len(d)))
P = pd.DataFrame(P); P.to_csv(R / "cv_paired_diff.csv", index=False)
pd.set_option("display.width", 200); print(S.round(3).to_string()); print(P.round(3).to_string())

fig, ax = plt.subplots(1, 4, figsize=(19, 4.8))
for a, m in zip(ax[:3], models):
    for arm in ("Fixed", "Canny"):
        o = oof[(oof.model == m) & (oof.arm == arm)].groupby("subject_id").agg(true=("true", "first"), prob=("prob", "mean"))
        f, t, _ = roc_curve(o.true, o.prob); r = S[(S.model == m) & (S.arm == arm)].iloc[0]
        a.plot(f, t, color=C[arm], lw=2.2, label=f"{arm}: AUC {r.pooled_auc:.3f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]")
    a.plot([0, 1], [0, 1], "--", c="#aab4be"); a.set_title(m); a.set_xlabel("False positive rate"); a.legend(loc="lower right", fontsize=9, frameon=False)
    for s in ("top", "right"): a.spines[s].set_visible(False)
ax[0].set_ylabel("True positive rate")
a = ax[3]
for k, m in enumerate(models):
    r = P[P.model == m].iloc[0]; a.errorbar(r.canny_minus_fixed, k, xerr=[[r.canny_minus_fixed - r.ci_lo], [r.ci_hi - r.canny_minus_fixed]], fmt="o", color="#1f2933", capsize=5)
a.axvline(0, ls="--", c="#aab4be"); a.set_yticks(range(3)); a.set_yticklabels(models); a.invert_yaxis()
a.set_xlabel("Canny − Fixed (fold AUC, mean and 95% CI)"); a.set_title("Paired difference over 10 folds")
for s in ("top", "right"): a.spines[s].set_visible(False)
fig.suptitle("Repeated 5-fold subject-level CV (2 repeats, n = 198): out-of-fold ROC", x=.01, ha="left", fontsize=14, weight="bold")
plt.tight_layout(); fig.savefig(R / "cv_roc.png", dpi=150)
