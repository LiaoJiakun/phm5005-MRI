from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
R = Path(__file__).resolve().parent.parent / "results" / "five_slice"
BB = {"cnn": "Small 2D CNN (from scratch)", "resnet": "ResNet18 (ImageNet, full fine-tune)"}
EX = ["S1: single slice fixed 80", "M1: 2.5D fixed 78-82", "M2: late fusion fixed 78-82", "S1: single slice Canny c", "M1: 2.5D Canny c±2", "M2: late fusion Canny c±2"]
COL = dict(zip(EX, ["#9db4c0", "#2a6f97", "#6a994e", "#e5b3b8", "#d1495b", "#e09f3e"]))
PAIRS = [("M1: 2.5D fixed 78-82", "M2: late fusion fixed 78-82"), ("M1: 2.5D Canny c±2", "M2: late fusion Canny c±2"),
         ("M1: 2.5D fixed 78-82", "S1: single slice fixed 80"), ("M2: late fusion fixed 78-82", "S1: single slice fixed 80"),
         ("M1: 2.5D Canny c±2", "S1: single slice Canny c"), ("M2: late fusion Canny c±2", "S1: single slice Canny c"),
         ("M1: 2.5D Canny c±2", "M1: 2.5D fixed 78-82"), ("M2: late fusion Canny c±2", "M2: late fusion fixed 78-82")]
short = lambda e: e.split(":")[0] + (" fixed" if "fixed" in e else " Canny")
rng = np.random.default_rng(0); S, P, CM = [], [], []
fig, ax = plt.subplots(2, 2, figsize=(14, 10.4))
for j, b in enumerate(BB):
    fr = pd.read_csv(R / b / "fold_results.csv"); oof = pd.read_csv(R / b / "oof_predictions.csv")
    for e in EX:
        f = fr[fr.exp == e]; o = oof[oof.exp == e].groupby("subject_id").agg(true=("true", "first"), prob=("prob", "mean"), dataset=("dataset", "first"))
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(o), len(o)); y = o.true.to_numpy()[i]
            if len(set(y)) == 2: bs.append(roc_auc_score(y, o.prob.to_numpy()[i]))
        row = dict(backbone=b, exp=e, fold_auc=f.auc.mean(), fold_sd=f.auc.std(), pooled_auc=roc_auc_score(o.true, o.prob), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5), acc=f.acc.mean(), params=int(f.params.iloc[0]), n=len(o))
        for d in ("OASIS1", "OASIS2"): od = o[o.dataset == d]; row[f"auc_{d}"] = roc_auc_score(od.true, od.prob)
        S.append(row); fp, tp, _ = roc_curve(o.true, o.prob)
        ax[0, j].plot(fp, tp, color=COL[e], lw=2.1, ls="--" if e.startswith("S1") else "-", label=f"{e}: {row['pooled_auc']:.3f} [{row['ci_lo']:.2f}, {row['ci_hi']:.2f}]")
        pred = (o.prob >= .5).astype(int); tn, fp_, fn, tp_ = confusion_matrix(o.true, pred, labels=[0, 1]).ravel()
        CM.append(dict(backbone=b, exp=e, TN=tn, FP=fp_, FN=fn, TP=tp_, sensitivity=tp_ / (tp_ + fn), specificity=tn / (tn + fp_), accuracy=(tn + tp_) / len(o)))
    ax[0, j].plot([0, 1], [0, 1], ":", c="#aab4be"); ax[0, j].set_title(f"{BB[b]}\npooled out-of-fold ROC (n = {len(o)})", fontsize=10.5); ax[0, j].set_xlabel("False positive rate"); ax[0, j].set_ylabel("True positive rate")
    ax[0, j].legend(loc="lower right", fontsize=7.6, frameon=False)
    for k, (a, c) in enumerate(PAIRS):
        x = fr[fr.exp == a].sort_values(["repeat", "fold"]).auc.to_numpy(); y_ = fr[fr.exp == c].sort_values(["repeat", "fold"]).auc.to_numpy(); d = x - y_
        t = stats.t.interval(.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
        P.append(dict(backbone=b, comparison=f"{short(a)} - {short(c)}", diff=d.mean(), ci_lo=t[0], ci_hi=t[1], wilcoxon_p=stats.wilcoxon(d).pvalue, better_folds=int((d > 0).sum()), n=len(d)))
        ax[1, j].errorbar(d.mean(), k, xerr=[[d.mean() - t[0]], [t[1] - d.mean()]], fmt="o", color="#1f2933", capsize=4)
    ax[1, j].axvline(0, ls="--", c="#aab4be"); ax[1, j].set_yticks(range(len(PAIRS))); ax[1, j].set_yticklabels([f"{short(a)} − {short(c)}" for a, c in PAIRS], fontsize=8.5); ax[1, j].invert_yaxis()
    ax[1, j].set_xlabel("Paired fold-AUC difference (mean, 95% CI; 10 folds)"); ax[1, j].set_title("Paired differences", fontsize=10.5)
    for a_ in (ax[0, j], ax[1, j]): a_.spines["top"].set_visible(False); a_.spines["right"].set_visible(False)
fig.suptitle("5 axial slices: Method 1 (2.5D) vs Method 2 (late fusion), fixed 78-82 vs Canny c±2 — repeated stratified 5-fold CV, n = 236 (no overlap)", x=.01, ha="left", fontsize=12, weight="bold")
plt.tight_layout(); fig.savefig(R / "five_slice_results.png", dpi=150)
S, P, CM = pd.DataFrame(S), pd.DataFrame(P), pd.DataFrame(CM); S.to_csv(R / "summary.csv", index=False); P.to_csv(R / "paired.csv", index=False); CM.to_csv(R / "confusion.csv", index=False)
pd.set_option("display.width", 240); print(S.round(3).to_string()); print(P.round(3).to_string()); print(CM.round(3).to_string())
