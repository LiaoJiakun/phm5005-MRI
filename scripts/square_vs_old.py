"""Square-padded vs squashed axial slices: best model (ResNet18 late fusion concat), fixed and Canny, paired over the same 10 folds."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
R = Path(__file__).resolve().parent.parent / "results"; rng = np.random.default_rng(0)
SETS = {"Fixed 78-82": ("M2: late fusion fixed 78-82", R / "five_slice" / "resnet", R / "square" / "fixed"), "Canny c±2": ("M2: late fusion Canny c±2", R / "five_slice" / "resnet", R / "square" / "canny")}
rows, fig = [], plt.figure(figsize=(12, 4.8)); ax = [fig.add_subplot(1, 2, k + 1) for k in range(2)]
def load(p, e): f = pd.read_csv(p / "fold_results.csv"); o = pd.read_csv(p / "oof_predictions.csv"); return f[f.exp == e].sort_values(["repeat", "fold"]), o[o.exp == e]
for a, (name, (e, p_old, p_new)) in zip(ax, SETS.items()):
    res = {}
    for tag, p in (("old (squashed)", p_old), ("square-padded", p_new)):
        f, o = load(p, e); o = o.groupby("subject_id").agg(true=("true", "first"), prob=("prob", "mean")); res[tag] = (f, o); bs = []
        for _ in range(2000):
            i = rng.integers(0, len(o), len(o)); y = o.true.to_numpy()[i]
            if len(set(y)) == 2: bs.append(roc_auc_score(y, o.prob.to_numpy()[i]))
        pred = (o.prob >= .5).astype(int); tn, fp, fn, tp = confusion_matrix(o.true, pred, labels=[0, 1]).ravel()
        rows.append(dict(slices=name, preprocessing=tag, pooled_auc=roc_auc_score(o.true, o.prob), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5), fold_auc=f.auc.mean(), accuracy=(tn + tp) / len(o), sensitivity=tp / (tp + fn), specificity=tn / (tn + fp)))
        fpr, tpr, _ = roc_curve(o.true, o.prob); a.plot(fpr, tpr, lw=2.2, label=f"{tag}: AUC {rows[-1]['pooled_auc']:.3f} [{rows[-1]['ci_lo']:.2f}, {rows[-1]['ci_hi']:.2f}]")
    d = res["square-padded"][0].auc.to_numpy() - res["old (squashed)"][0].auc.to_numpy(); t = stats.t.interval(.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d)); da = res["square-padded"][0].acc.to_numpy() - res["old (squashed)"][0].acc.to_numpy()
    rows[-1].update(paired_auc_diff=d.mean(), diff_lo=t[0], diff_hi=t[1], wilcoxon_p=stats.wilcoxon(d).pvalue, better_folds=int((d > 0).sum()), paired_acc_diff=da.mean())
    a.plot([0, 1], [0, 1], ":", c="gray"); a.set_title(f"{name}: squashed vs square-padded axial slices"); a.set_xlabel("False positive rate"); a.legend(loc="lower right", fontsize=8.5, frameon=False)
    for s in ("top", "right"): a.spines[s].set_visible(False)
ax[0].set_ylabel("True positive rate"); plt.tight_layout(); fig.savefig(R / "square" / "square_vs_old.png", dpi=150)
S = pd.DataFrame(rows); S.to_csv(R / "square" / "square_vs_old.csv", index=False); pd.set_option("display.width", 220); print(S.round(3).to_string())
