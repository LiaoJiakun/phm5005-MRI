"""Out-of-fold ROC curves of the mean-pooling trial (2.5D vs late fusion with mean pooling; fixed vs Canny): results/meanpool/meanpool_roc.png."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
ROOT = Path(__file__).resolve().parent.parent / "results"; OUT = ROOT / "meanpool"
BB = {"cnn": "Small 2D CNN (from scratch)", "resnet": "ResNet18 (ImageNet, fine-tuned)"}
CELLS = [("M1: 2.5D fixed 78-82", "Fixed 78-82 · 2.5D", "#2a6f97", "-"), ("M2-mean: late fusion (mean pool) fixed 78-82", "Fixed 78-82 · late fusion, mean pooling", "#2a6f97", "--"),
         ("M1: 2.5D Canny c±2", "Canny c±2 · 2.5D", "#d1495b", "-"), ("M2-mean: late fusion (mean pool) Canny c±2", "Canny c±2 · late fusion, mean pooling", "#d1495b", "--")]
rng = np.random.default_rng(0); fig, ax = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
for a, (b, title) in zip(ax, BB.items()):
    for e, lab, col, ls in CELLS:
        p = (OUT if e.startswith("M2-mean") else ROOT / "five_slice") / b / "oof_predictions.csv"
        o = pd.read_csv(p); o = o[o.exp == e].groupby("subject_id").agg(true=("true", "first"), prob=("prob", "mean"))
        auc = roc_auc_score(o.true, o.prob); bs = []
        for _ in range(2000):
            i = rng.integers(0, len(o), len(o)); y = o.true.to_numpy()[i]
            if len(set(y)) == 2: bs.append(roc_auc_score(y, o.prob.to_numpy()[i]))
        fp, tp, _ = roc_curve(o.true, o.prob); a.plot(fp, tp, color=col, ls=ls, lw=2.3, label=f"{lab}: AUC {auc:.3f} [{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}]")
    a.plot([0, 1], [0, 1], ":", c="#aab4be"); a.set_title(f"{title}\nout-of-fold ROC, n = {len(o)} (no OASIS-1/2 overlap)", fontsize=10.5); a.set_xlabel("False positive rate"); a.legend(loc="lower right", fontsize=8, frameon=False)
    for s in ("top", "right"): a.spines[s].set_visible(False)
ax[0].set_ylabel("True positive rate"); plt.tight_layout(); fig.savefig(OUT / "meanpool_roc.png", dpi=150); print("saved")
