from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
R = Path(__file__).resolve().parent.parent / "results" / "dual_view"
ARMS = ["axial", "coronal", "dual", "dual_axax"]; LAB = {"axial": "Axial only", "coronal": "Coronal only", "dual": "Dual (axial+coronal)", "dual_axax": "Dual control (axial+axial)"}
BB = {"cnn": "Small CNN", "resnet": "ResNet18 (layer4 + head)"}
fig, ax = plt.subplots(2, 4, figsize=(15, 7.6)); rows = []
for i, b in enumerate(BB):
    oof = pd.read_csv(R / f"{b}_oof_predictions.csv")
    for j, arm in enumerate(ARMS):
        o = oof[oof.arm == arm].groupby("subject_id").agg(true=("true", "first"), prob=("prob", "mean"))   # mean of the 2 repeats per subject
        pred = (o.prob >= .5).astype(int); tn, fp, fn, tp = confusion_matrix(o.true, pred, labels=[0, 1]).ravel()
        sens, spec, acc = tp / (tp + fn), tn / (tn + fp), (tp + tn) / len(o)
        rows.append(dict(backbone=b, arm=arm, TN=tn, FP=fp, FN=fn, TP=tp, sensitivity=sens, specificity=spec, accuracy=acc))
        m = np.array([[tn, fp], [fn, tp]]); a = ax[i, j]; a.imshow(m / m.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for (r, c), v in np.ndenumerate(m): a.text(c, r, f"{v}\n({v / m[r].sum():.0%})", ha="center", va="center", fontsize=12, color="white" if m[r, c] / m[r].sum() > .55 else "#1f2933")
        a.set_xticks([0, 1]); a.set_xticklabels(["CDR=0", "CDR>0"]); a.set_yticks([0, 1]); a.set_yticklabels(["CDR=0", "CDR>0"])
        a.set_xlabel("Predicted"); a.set_ylabel("Actual" if j == 0 else ""); a.set_title(f"{BB[b]}\n{LAB[arm]}\nacc {acc:.2f} · sens {sens:.2f} · spec {spec:.2f}", fontsize=9.5)
D = pd.DataFrame(rows); D.to_csv(R / "dual_view_confusion.csv", index=False); print(D.round(3).to_string())
fig.suptitle("Out-of-fold confusion matrices (threshold 0.5, all 198 subjects; cell % = row-normalised)", x=.01, ha="left", fontsize=13, weight="bold")
plt.tight_layout(); fig.savefig(R / "dual_view_confusion.png", dpi=150)
