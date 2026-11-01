from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve, roc_auc_score

R = Path(__file__).resolve().parent.parent / "results" / "resnet18"
p = pd.read_csv(R / "resnet18_test_predictions.csv"); y = p.true_label.to_numpy()
m = pd.read_csv(R / "resnet18_metrics_per_seed.csv")
C = {"Fixed slice 80": "#2a6f97", "Canny selected": "#d1495b"}
INK, MUTED, BG = "#1f2933", "#6b7785", "#ffffff"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})

fig = plt.figure(figsize=(14, 10.4), facecolor=BG)
fig.text(.04, .955, "Fixed vs Canny-selected MRI slice with ImageNet-pretrained ResNet18", fontsize=20, weight="bold", va="top")
fig.text(.04, .915, "OASIS-1 · dementia (CDR>0) vs non-demented (CDR=0) · age ≥ 60 · one 2D axial slice per subject", fontsize=11.5, color=MUTED, va="top")

# stat tiles
def tile(x, w, big, small, col):
    ax = fig.add_axes([x, .795, w, .09]); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=.08", fc="#f4f6f8", ec="none", transform=ax.transAxes))
    ax.text(.06, .56, big, fontsize=21, weight="bold", color=col, va="center", transform=ax.transAxes)
    ax.text(.06, .17, small, fontsize=9.5, color=MUTED, va="center", transform=ax.transAxes)
a = m.groupby("arm").test_auc.agg(["mean", "std"])
tile(.04, .21, "198", "subjects (98 CDR=0 / 100 CDR>0)", INK)
tile(.27, .21, "138 / 30 / 30", "train / val / test (subject-level)", INK)
for x, arm in [(.50, "Fixed slice 80"), (.73, "Canny selected")]:
    tile(x, .22, f"{a.loc[arm,'mean']:.3f} ± {a.loc[arm,'std']:.3f}", f"mean test AUC · {arm} (3 seeds)", C[arm])

# ROC panel
ax = fig.add_axes([.07, .25, .40, .47])
for arm, col in C.items():
    seeds = m[m.arm == arm].seed.tolist()
    for i, s in enumerate(seeds):
        pr = p[f"{arm}|{s}"].to_numpy(); f, t, _ = roc_curve(y, pr)
        ax.plot(f, t, color=col, lw=2.6 if i == 0 else 1.2, alpha=1 if i == 0 else .45, drawstyle="steps-post",
                label=f"{arm} (mean AUC {a.loc[arm,'mean']:.3f})" if i == 0 else None)
ax.plot([0, 1], [0, 1], "--", color="#aab4be", lw=1.2)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
ax.set_title("ROC curves on the test set (bold = seed 42, faint = seeds 43, 44)", fontsize=11, loc="left", color=INK)
ax.legend(loc="lower right", frameon=False, fontsize=10)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(alpha=.25)

# AUC per seed panel
ax2 = fig.add_axes([.56, .25, .40, .47])
for k, (arm, col) in enumerate(C.items()):
    v = m[m.arm == arm].test_auc.to_numpy()
    ax2.bar(np.arange(3) + k * .38 - .19, v, .34, color=col, label=arm)
    for i, x in enumerate(v): ax2.text(i + k * .38 - .19, x + .012, f"{x:.2f}", ha="center", fontsize=9.5)
ax2.axhline(.5, ls="--", color="#aab4be", lw=1.2, label="chance (AUC 0.5)")
ax2.set_xticks(range(3)); ax2.set_xticklabels([f"seed {s}" for s in (42, 43, 44)]); ax2.set_ylim(0, 1)
ax2.set_ylabel("Test AUC"); ax2.set_title("AUC by seed — differences are within seed-to-seed spread", fontsize=11, loc="left")
ax2.legend(loc="upper left", frameon=False, fontsize=10, ncol=3)
for s in ("top", "right"): ax2.spines[s].set_visible(False)

fig.text(.04, .165, "Takeaways", fontsize=12.5, weight="bold")
fig.text(.04, .135, "•  Both slice choices give only modest discrimination (AUC ≈ 0.7–0.75). Canny is nominally +0.03 higher, but seed SD ≈ 0.06 and n_test = 30:  no difference can be claimed.\n"
                    "•  Fine-tuning pretrained ResNet18 overfits within 2–4 epochs (138 training images) and did not clearly beat the small from-scratch CNN (AUC 0.70 fixed / 0.67 Canny, one run).\n"
                    "•  Needed next: repeated subject-level cross-validation; CDR is a clinical rating (not confirmed Alzheimer's); sex ratio differs between classes (72F/26M vs 59F/41M).",
         fontsize=10, color=INK, va="top", linespacing=1.6)
fig.text(.04, .018, "Fine-tuning: AdamW lr 1e-4, batch 16, ≤ 20 epochs, early stopping on validation loss (patience 5), ±5° rotation augmentation; same subjects, split and hyperparameters for both arms.",
         fontsize=8.5, color=MUTED)
fig.savefig(R / "report_resnet18.png", dpi=170, facecolor=BG); print("saved")
