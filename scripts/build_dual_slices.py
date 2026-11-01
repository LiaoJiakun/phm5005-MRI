"""Add a label-independent Canny-selected CORONAL slice to the Age>=60 cohort (axial Canny slice is reused).
Volumes are RAS after as_closest_canonical: axis0=x(L-R), axis1=y(A-P), axis2=z(I-S).
Coronal candidates = central ~45% of the mean brain extent along y (same proportion as the axial range 50-110)."""
from pathlib import Path
import numpy as np, pandas as pd, cv2, nibabel as nib

ROOT = Path(__file__).resolve().parent.parent
C = pd.read_csv(ROOT / "data" / "cohort" / "cohort_age60plus.csv")
Z = np.load(ROOT / "data" / "cohort" / "cohort_slices.npz", allow_pickle=True)
assert (Z["subject_id"] == C.subject_id.to_numpy()).all()
SIZE, COR_CANDIDATES = 128, range(66, 141)

def norm(s):
    s = np.nan_to_num(s.astype(np.float32)); fg = s[s > 0]
    if fg.size < 100: return np.zeros(s.shape, np.uint8)
    lo, hi = np.percentile(fg, [1, 99]); return np.uint8(np.clip((s - lo) / max(hi - lo, 1e-6), 0, 1) * 255)
def cor(v, j):   # coronal: rows = superior->inferior, cols = x
    return cv2.resize(norm(v[:, j, :].T[::-1]), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
def canny_cor(v):
    sc = {j: int(np.count_nonzero(cv2.Canny(cv2.GaussianBlur(cor(v, j), (5, 5), 1), 50, 150))) for j in COR_CANDIDATES if j < v.shape[1]}
    return max(sc, key=lambda j: (sc[j], -j))

cors, idx = [], []
for p in C.mri_path:
    v = np.squeeze(np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj, dtype=np.float32))
    j = canny_cor(v); cors.append(cor(v, j)); idx.append(j)
np.savez_compressed(ROOT / "data" / "cohort" / "cohort_slices_dual.npz", subject_id=Z["subject_id"], label=Z["label"], split=Z["split"],
                    axial=Z["canny"], axial_index=Z["canny_index"], coronal=np.stack(cors), coronal_index=np.array(idx), fixed_axial=Z["fixed"])
print("coronal index: mean %.1f sd %.1f min %d max %d" % (np.mean(idx), np.std(idx), min(idx), max(idx)),
      "| at range edge:", int(np.sum((np.array(idx) == COR_CANDIDATES[0]) | (np.array(idx) == COR_CANDIDATES[-1]))))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 4, figsize=(12, 6))
for k in range(4):
    ax[0, k].imshow(Z["canny"][k * 40], cmap="gray"); ax[0, k].set_title(f"axial Canny {Z['canny_index'][k*40]}"); ax[0, k].axis("off")
    ax[1, k].imshow(cors[k * 40], cmap="gray"); ax[1, k].set_title(f"coronal Canny {idx[k*40]}"); ax[1, k].axis("off")
plt.tight_layout(); plt.savefig(ROOT / "results" / "dual_view_examples.png", dpi=110)
plt.figure(figsize=(6, 3)); plt.hist(idx, bins=25); plt.xlabel("Canny coronal index"); plt.tight_layout(); plt.savefig(ROOT / "results" / "coronal_index_hist.png", dpi=110)
