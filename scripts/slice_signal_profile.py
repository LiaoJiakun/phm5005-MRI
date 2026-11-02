"""Exploratory, descriptive: where in the brain is (a) Canny edge density highest and (b) a simple atrophy proxy most associated with CDR>0?
Atrophy proxy = fraction of brain-mask pixels that are dark (CSF-like, < 20% of the normalised intensity range) in that slice.
Uses labels for a DESCRIPTIVE univariate AUC only (not for selecting slices that are then evaluated)."""
from pathlib import Path
import numpy as np, pandas as pd, cv2, nibabel as nib, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parent.parent; C = pd.read_csv(ROOT / "data" / "combined_full" / "combined_cohort.csv"); y = C.label.to_numpy()
def norm(s):
    s = np.nan_to_num(s.astype(np.float32)); fg = s[s > 0]
    if fg.size < 100: return np.zeros(s.shape, np.uint8)
    lo, hi = np.percentile(fg, [1, 99]); return np.uint8(np.clip((s - lo) / max(hi - lo, 1e-6), 0, 1) * 255)
rs = lambda x: cv2.resize(norm(x), (128, 128), interpolation=cv2.INTER_AREA)
def load(p): return np.squeeze(np.asarray((nib.load(p) if p.endswith(".nii.gz") else nib.as_closest_canonical(nib.load(p))).dataobj, dtype=np.float32))
AX, COR = range(40, 121), range(50, 171)
def stats(s):
    brain = s > 0; edges = np.count_nonzero(cv2.Canny(cv2.GaussianBlur(s, (5, 5), 1), 50, 150)); n = max(brain.sum(), 1)
    return edges, ((s < 51) & brain).sum() / n, brain.sum()
E_ax, F_ax, A_ax = (np.zeros((len(C), len(AX))) for _ in range(3)); E_co, F_co, A_co = (np.zeros((len(C), len(COR))) for _ in range(3))
for n, p in enumerate(C.mri_path):
    v = load(p)
    for k, z in enumerate(AX): E_ax[n, k], F_ax[n, k], A_ax[n, k] = stats(rs(v[:, :, z].T))
    for k, j in enumerate(COR): E_co[n, k], F_co[n, k], A_co[n, k] = stats(rs(v[:, j, :].T[::-1]))
auc = lambda M: np.array([roc_auc_score(y, M[:, k]) if M[:, k].std() > 0 else .5 for k in range(M.shape[1])])
auc_ax, auc_co = auc(F_ax), auc(F_co)
np.savez(ROOT / "results" / "slice_signal_profile.npz", AX=list(AX), COR=list(COR), E_ax=E_ax.mean(0), E_co=E_co.mean(0), auc_ax=auc_ax, auc_co=auc_co)
cz = np.load(ROOT / "data" / "combined_full" / "axial5_slices.npz", allow_pickle=True)["canny_center"]; cy = np.load(ROOT / "data" / "combined_full" / "combined_slices.npz", allow_pickle=True)["canny_coronal_idx"]
fig, ax = plt.subplots(2, 2, figsize=(13, 7.6))
for j, (name, idx, E, auc_, sel, lo_hi, fixed) in enumerate([("Axial slice index z", list(AX), E_ax.mean(0), auc_ax, cz, (50, 110), (78, 82)), ("Coronal slice index y", list(COR), E_co.mean(0), auc_co, cy, (66, 140), (105, 105))]):
    a = ax[0, j]; a.plot(idx, E, color="#d1495b", lw=2); a.set_ylabel("mean Canny edge pixels per slice"); a.set_title(f"{name}: where the Canny rule 'sees' edges")
    a.axvspan(*lo_hi, color="gray", alpha=.12, label="Canny search range"); a.axvspan(fixed[0] - .5, fixed[1] + .5, color="#2a6f97", alpha=.5, label="fixed slice(s)")
    a2 = a.twinx(); a2.hist(sel, bins=range(idx[0], idx[-1] + 2, 2), color="#e09f3e", alpha=.45); a2.set_ylabel("subjects whose Canny pick is here", color="#b07a00"); a.legend(loc="upper right", fontsize=8)
    b = ax[1, j]; b.plot(idx, auc_, color="#2a6f97", lw=2); b.axhline(.5, ls="--", c="gray"); b.set_ylabel("AUC of CSF-fraction vs CDR>0 (n = 236)"); b.set_xlabel(name)
    b.axvspan(*lo_hi, color="gray", alpha=.12); b.axvspan(fixed[0] - .5, fixed[1] + .5, color="#2a6f97", alpha=.5)
    b.set_title("Where a simple atrophy proxy (dark-pixel fraction) separates the groups")
for a in ax.ravel(): a.spines["top"].set_visible(False)
plt.tight_layout(); fig.savefig(ROOT / "results" / "slice_signal_profile.png", dpi=150)
for nm, idx, E, A in (("axial", list(AX), E_ax.mean(0), auc_ax), ("coronal", list(COR), E_co.mean(0), auc_co)):
    print(f"{nm}: Canny edge peak at index {idx[int(np.argmax(E))]} | CSF-proxy AUC peak {A.max():.3f} at index {idx[int(np.argmax(A))]} | AUC>0.65 at indices {[i for i, a in zip(idx, A) if a > .65][:3]}...{[i for i, a in zip(idx, A) if a > .65][-3:]}")
print("axial: mean proxy AUC in Canny-chosen slices (c±2 per subject)", round(float(np.mean([auc_ax[max(0, min(len(AX) - 1, int(c) - 40))] for c in cz])), 3), "| in fixed 78-82", round(float(auc_ax[78 - 40:83 - 40].mean()), 3))
