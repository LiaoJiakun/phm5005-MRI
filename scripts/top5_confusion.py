from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = Path(__file__).resolve().parent.parent / "results"
fc, fs, mp = pd.read_csv(R / "final_cv/confusion.csv"), pd.read_csv(R / "five_slice/confusion.csv"), pd.read_csv(R / "meanpool/meanpool_summary.csv")
g = lambda df, e: df[(df.backbone == "resnet") & (df.exp == e)].iloc[0]
TOP = [("1  ResNet18 · late fusion (concat)\nfixed 78-82", g(fs, "M2: late fusion fixed 78-82")), ("2  ResNet18 · single view\ncoronal 105", g(fc, "B: fixed coronal 105")),
       ("3  ResNet18 · late fusion (mean pool)\nfixed 78-82", g(mp, "M2-mean: late fusion (mean pool) fixed 78-82")), ("4  ResNet18 · late fusion (mean pool)\nCanny c±2", g(mp, "M2-mean: late fusion (mean pool) Canny c±2")),
       ("5  ResNet18 · axial 80 + coronal 105\ndual-view (fixed)", g(fc, "C: fixed axial+coronal"))]
rows = []; fig, ax = plt.subplots(1, 5, figsize=(19, 4.7))
for a, (title, r) in zip(ax, TOP):
    tn, fp, fn, tp = int(r.TN), int(r.FP), int(r.FN), int(r.TP); m = np.array([[tn, fp], [fn, tp]]); n = m.sum()
    sens, spec, ppv, npv, acc = tp / (tp + fn), tn / (tn + fp), tp / (tp + fp), tn / (tn + fn), (tn + tp) / n
    rows.append(dict(model=title.replace("\n", " · "), TN=tn, FP=fp, FN=fn, TP=tp, accuracy=acc, sensitivity=sens, specificity=spec, PPV=ppv, NPV=npv))
    a.imshow(m / m.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(m): a.text(j, i, f"{v}\n({v / m[i].sum():.0%})", ha="center", va="center", fontsize=13, color="white" if m[i, j] / m[i].sum() > .55 else "#1f2933")
    a.set_xticks([0, 1]); a.set_xticklabels(["CDR=0", "CDR>0"]); a.set_yticks([0, 1]); a.set_yticklabels(["CDR=0", "CDR>0"]); a.set_xlabel("Predicted")
    a.set_title(f"{title}\nacc {acc:.3f} · sens {sens:.2f} · spec {spec:.2f}\nPPV {ppv:.2f} · NPV {npv:.2f}", fontsize=9.2)
ax[0].set_ylabel("Actual")
fig.suptitle("Top-5 ResNet18 settings: out-of-fold confusion matrices (236 subjects, no OASIS-1/2 overlap, threshold 0.5; cell % = row-normalised)", x=.01, ha="left", fontsize=11.5, weight="bold")
plt.tight_layout(); fig.savefig(R / "top5_confusion.png", dpi=150); pd.DataFrame(rows).to_csv(R / "top5_confusion.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string())
