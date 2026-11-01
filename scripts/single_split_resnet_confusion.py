# NOTE (kept for the record): this script ran on the 279-subject OASIS-1 + OASIS-2 cohort (data/combined), which was later found to contain
# people who are present in both datasets. All results from it are invalid and that cohort is no longer in the project.
# See build_combined.py (de-duplicated 236-subject cohort) and final_cv.py for the valid experiments.
from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = Path(__file__).resolve().parent.parent / "results" / "combined_single_split_resnet"
fig, ax = plt.subplots(2, 4, figsize=(15, 7.6)); out = []
for i, mode in enumerate(("full", "layer4")):
    d = pd.read_csv(R / mode / "per_seed.csv")
    for j, (e, g) in enumerate(d.groupby("exp")):
        s = g[["TN", "FP", "FN", "TP"]].sum(); m = np.array([[s.TN, s.FP], [s.FN, s.TP]]); a = ax[i, j]
        sens, spec, acc = s.TP / (s.TP + s.FN), s.TN / (s.TN + s.FP), (s.TN + s.TP) / m.sum(); out.append(dict(mode=mode, exp=e, TN=s.TN, FP=s.FP, FN=s.FN, TP=s.TP, sens=sens, spec=spec, acc=acc))
        a.imshow(m / m.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for (r, c), v in np.ndenumerate(m): a.text(c, r, f"{v}\n({v / m[r].sum():.0%})", ha="center", va="center", fontsize=12, color="white" if m[r, c] / m[r].sum() > .55 else "#1f2933")
        a.set_xticks([0, 1]); a.set_xticklabels(["CDR=0", "CDR>0"]); a.set_yticks([0, 1]); a.set_yticklabels(["CDR=0", "CDR>0"]); a.set_xlabel("Predicted"); a.set_ylabel(("Full fine-tune\n" if mode == "full" else "layer4 only\n") + "Actual" if j == 0 else "")
        a.set_title(f"{e}\nacc {acc:.2f} · sens {sens:.2f} · spec {spec:.2f}", fontsize=9.5)
fig.suptitle("ResNet18, single split (test n = 42): confusion matrices, threshold 0.5, counts summed over 3 seeds (126 predictions); % = row-normalised", x=.01, ha="left", fontsize=12, weight="bold")
plt.tight_layout(); fig.savefig(R / "resnet_single_split_confusion.png", dpi=150); pd.DataFrame(out).to_csv(R / "confusion_summary.csv", index=False); print(pd.DataFrame(out).round(3).to_string())
