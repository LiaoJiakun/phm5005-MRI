from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
R = Path(__file__).resolve().parent.parent / "results" / "single_split_5slice"
BB = {"cnn": "Small 2D CNN (from scratch)", "resnet": "ResNet18 (ImageNet, fine-tuned)"}
CELLS = [("2.5D fixed 78-82", "Fixed 78-82 · 2.5D", "#2a6f97", "-"), ("Late fusion (mean pool) fixed 78-82", "Fixed 78-82 · late fusion, mean pooling", "#2a6f97", "--"),
         ("2.5D Canny c±2", "Canny c±2 · 2.5D", "#d1495b", "-"), ("Late fusion (mean pool) Canny c±2", "Canny c±2 · late fusion, mean pooling", "#d1495b", "--")]
rng = np.random.default_rng(0); fig, ax = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
for a, (b, title) in zip(ax, BB.items()):
    pr = pd.read_csv(R / f"{b}_test_predictions.csv"); ps = pd.read_csv(R / f"{b}_per_seed.csv")
    for e, lab, col, ls in CELLS:
        p = pr[pr.exp == e]
        for s, g in p.groupby("seed"): fp, tp, _ = roc_curve(g.true, g.prob); a.plot(fp, tp, color=col, ls=ls, lw=.8, alpha=.25)            # individual seeds (faint)
        o = p.groupby("subject_id").agg(true=("true", "first"), prob=("prob", "mean")); auc = roc_auc_score(o.true, o.prob); sd = ps[ps.exp == e].test_auc.std(); bs = []
        for _ in range(2000):
            i = rng.integers(0, len(o), len(o)); y = o.true.to_numpy()[i]
            if len(set(y)) == 2: bs.append(roc_auc_score(y, o.prob.to_numpy()[i]))
        fp, tp, _ = roc_curve(o.true, o.prob); a.plot(fp, tp, color=col, ls=ls, lw=2.3, label=f"{lab}: AUC {auc:.3f} [{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}] (seeds ±{sd:.2f})")
    a.plot([0, 1], [0, 1], ":", c="#aab4be"); a.set_title(f"{title}\ntest-set ROC, single split, n = {len(o)} (bold = 3-seed ensemble, faint = single seeds)", fontsize=9.8)
    a.set_xlabel("False positive rate"); a.legend(loc="lower right", fontsize=7.4, frameon=False)
    for s in ("top", "right"): a.spines[s].set_visible(False)
ax[0].set_ylabel("True positive rate"); plt.tight_layout(); fig.savefig(R / "single_split_roc.png", dpi=150); print("saved")
