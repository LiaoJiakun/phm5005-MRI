"""Leak-free Youden-J results: threshold chosen on each fold's validation split, applied to the held-out test fold."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import confusion_matrix, roc_auc_score
R = Path(__file__).resolve().parent.parent / "results" / "youden"
BB = {"cnn": "Small CNN", "resnet": "ResNet18"}
CELLS = [("M1: 2.5D fixed 78-82", "Fixed", "2.5D"), ("M2-mean: late fusion (mean pool) fixed 78-82", "Fixed", "Late fusion, mean pooling"),
         ("M1: 2.5D Canny c±2", "Canny", "2.5D"), ("M2-mean: late fusion (mean pool) Canny c±2", "Canny", "Late fusion, mean pooling")]
rows, CM, thr_rows = [], {}, []
for b in BB:
    fr = pd.read_csv(R / b / "fold_results.csv"); pr = pd.read_csv(R / b / "predictions.csv")
    for e, sl, me in CELLS:
        f, p = fr[fr.exp == e], pr[pr.exp == e].copy(); assert len(f) == 10 and p.groupby("repeat").subject_id.nunique().eq(236).all(), (b, e, len(f))
        res = {}
        for tag, thr in (("05", np.full(len(p), .5)), ("youden", p.thr_youden.to_numpy())):
            p["margin"] = p.prob.to_numpy() - thr; o = p.groupby("subject_id").agg(true=("true", "first"), m=("margin", "mean"))     # margin averaged over the 2 repeats
            pred = (o.m >= 0).astype(int); tn, fp, fn, tp = confusion_matrix(o.true, pred, labels=[0, 1]).ravel(); res[tag] = (tn, fp, fn, tp)
        tn, fp, fn, tp = res["youden"]; tn5, fp5, fn5, tp5 = res["05"]
        d = f.acc_youden.to_numpy() - f.acc_05.to_numpy(); t = stats.t.interval(.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d)) if d.std() > 0 else (d.mean(),) * 2
        rows.append(dict(backbone=b, slices=sl, method=me, exp=e, thr_mean=f.thr_youden.mean(), thr_sd=f.thr_youden.std(), thr_min=f.thr_youden.min(), thr_max=f.thr_youden.max(),
                         acc_05=(tn5 + tp5) / 236, sens_05=tp5 / (tp5 + fn5), spec_05=tn5 / (tn5 + fp5), acc_youden=(tn + tp) / 236, sens_youden=tp / (tp + fn), spec_youden=tn / (tn + fp),
                         fold_acc_05=f.acc_05.mean(), fold_acc_youden=f.acc_youden.mean(), fold_acc_youden_sd=f.acc_youden.std(), diff_acc=d.mean(), diff_lo=t[0], diff_hi=t[1],
                         wilcoxon_p=stats.wilcoxon(d).pvalue if d.std() > 0 else 1.0, better_folds=int((d > 0).sum()), TN=tn, FP=fp, FN=fn, TP=tp, val_auc=f.val_auc.mean(), test_auc=f.test_auc.mean()))
        CM[(b, e)] = (tn, fp, fn, tp)
        for v in f.thr_youden: thr_rows.append(dict(backbone=b, exp=e, label=f"{sl}\n{me}", thr=v))
T = pd.DataFrame(rows); T.to_csv(R / "youden_summary.csv", index=False)
pd.set_option("display.width", 250); print(T[["backbone", "slices", "method", "thr_mean", "thr_sd", "acc_05", "acc_youden", "sens_youden", "spec_youden", "fold_acc_05", "fold_acc_youden", "diff_acc", "diff_lo", "diff_hi", "better_folds"]].round(3).to_string())
fig, ax = plt.subplots(2, 4, figsize=(15.5, 7.8))
for i, b in enumerate(BB):
    for j, (e, sl, me) in enumerate(CELLS):
        tn, fp, fn, tp = CM[(b, e)]; m = np.array([[tn, fp], [fn, tp]]); r = T[(T.backbone == b) & (T.exp == e)].iloc[0]; a = ax[i, j]; a.imshow(m / m.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for (p_, q_), v in np.ndenumerate(m): a.text(q_, p_, f"{v}\n({v / m[p_].sum():.0%})", ha="center", va="center", fontsize=12, color="white" if m[p_, q_] / m[p_].sum() > .55 else "#1f2933")
        a.set_xticks([0, 1]); a.set_xticklabels(["CDR=0", "CDR>0"]); a.set_yticks([0, 1]); a.set_yticklabels(["CDR=0", "CDR>0"]); a.set_xlabel("Predicted")
        a.set_title(f"{BB[b]} · {sl} · {me}\nYouden: acc {r.acc_youden:.3f} · sens {r.sens_youden:.2f} · spec {r.spec_youden:.2f}\n(0.5: acc {r.acc_05:.3f} · sens {r.sens_05:.2f} · spec {r.spec_05:.2f})", fontsize=8.6)
        if j == 0: a.set_ylabel("Actual")
fig.suptitle("Youden-J threshold chosen on each fold's validation split (test folds untouched) — out-of-fold confusion matrices, n = 236", x=.01, ha="left", fontsize=12, weight="bold")
plt.tight_layout(); fig.savefig(R / "youden_confusion.png", dpi=150)
TH = pd.DataFrame(thr_rows); fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
for a, b in zip(ax, BB):
    d = TH[TH.backbone == b]; labs = list(dict.fromkeys(d.label))
    for k, l in enumerate(labs): v = d[d.label == l].thr.to_numpy(); a.scatter(np.full(len(v), k) + np.random.default_rng(k).uniform(-.12, .12, len(v)), v, s=28, alpha=.7); a.hlines(v.mean(), k - .25, k + .25, color="k")
    a.axhline(.5, ls="--", c="#aab4be"); a.set_xticks(range(len(labs))); a.set_xticklabels(labs, fontsize=8); a.set_title(f"{BB[b]}: Youden thresholds of the 10 folds"); a.set_ylabel("threshold (probability of CDR>0)")
    for s in ("top", "right"): a.spines[s].set_visible(False)
plt.tight_layout(); fig.savefig(R / "youden_thresholds.png", dpi=150)
