"""Figure for the image-level overlap check between OASIS-2 (first visit) and OASIS-1."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent; C = ROOT / "data" / "combined_full"
rep = pd.read_csv(C / "oasis2_overlap_report.csv"); z = np.load(C / "combined_slices.npz", allow_pickle=True)
m = z["dataset"] == "OASIS1"; V = np.concatenate([z["fixed_axial"][m].reshape(m.sum(), -1), z["fixed_coronal"][m].reshape(m.sum(), -1)], 1).astype(np.float32)
V = (V - V.mean(1, keepdims=True)) / (V.std(1, keepdims=True) + 1e-6); V /= np.linalg.norm(V, axis=1, keepdims=True); S = V @ V.T; np.fill_diagonal(S, -1); w = S.max(1)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2)); bins = np.linspace(.8, 1, 41)
ax[0].hist(w, bins, color="#7a7a7a", alpha=.8, label=f"OASIS-1 vs other OASIS-1 people (n={len(w)})")
ax[0].hist(rep.image_similarity[rep.removed], bins, color="#d1495b", alpha=.85, label=f"OASIS-2 -> nearest OASIS-1: same person, removed (n={int(rep.removed.sum())})")
ax[0].hist(rep.image_similarity[~rep.removed], bins, color="#2a6f97", alpha=.85, label=f"OASIS-2 -> nearest OASIS-1: unique, kept (n={int((~rep.removed).sum())})")
ax[0].set_ylim(0, 46); ax[0].axvline(.93, ls="--", c="k"); ax[0].set_xlabel("Image similarity to the nearest OASIS-1 subject"); ax[0].set_ylabel("subjects"); ax[0].legend(fontsize=7.8, loc="upper left", frameon=False)
ax[0].set_title("OASIS-2 first-visit scans that duplicate OASIS-1 participants")
d = rep[rep.removed]; ax[1].scatter(d.cdr_oasis1, d.cdr_oasis2, s=60, alpha=.5, c="#d1495b"); ax[1].set_xlabel("CDR in OASIS-1 (matched person)"); ax[1].set_ylabel("CDR in OASIS-2 (first visit)")
ax[1].set_title(f"Matched pairs: identical CDR in {(d.cdr_oasis1 == d.cdr_oasis2).sum()} of {len(d)}")
for a in ax: a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
plt.tight_layout(); fig.savefig(ROOT / "results" / "oasis_overlap_check.png", dpi=150)
print(rep.removed.value_counts().to_dict(), "| max between different OASIS-1 people %.3f" % w.max(), "| median similarity of removed %.3f" % d.image_similarity.median())
