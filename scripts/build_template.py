"""Build a skull-ON T88 template (mean of OASIS-1 unmasked T88_111 gfc volumes) and a brain mask
(voxels inside the OASIS-1 *masked* volumes in >=50% of the same subjects). Used to atlas-register OASIS-2 RAW scans."""
from pathlib import Path
import numpy as np, nibabel as nib
from scipy import ndimage as ndi
ROOT = Path(__file__).resolve().parent.parent
T = ROOT / "data" / "template"; U = T / "oasis1_unmasked"; MASKED = ROOT / "data" / "extracted"
def load(p): return np.squeeze(np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj, dtype=np.float32))
vols, masks = [], []
for f in sorted(U.rglob("*_t88_gfc.img")):
    sid = f.name.split("_mpr")[0]
    m = list(MASKED.rglob(f"{sid}_mpr*t88_masked_gfc.img"))
    if not m: continue
    v = load(f); hi = np.percentile(v[v > 0], 99); vols.append(np.clip(v / hi, 0, 1.5)); masks.append(load(m[0]) > 0)
tpl = np.mean(vols, 0).astype(np.float32); freq = np.mean(masks, 0)
mask = ndi.binary_fill_holes(ndi.binary_closing(freq >= .5, iterations=2))
print("subjects:", len(vols), "shape:", tpl.shape, "mask voxels:", int(mask.sum()))
np.save(T / "template_skull_T88.npy", tpl); np.save(T / "brain_mask_T88.npy", mask)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for a, s in zip(ax, [tpl[:, :, 80].T, tpl[:, 105, :].T[::-1], tpl[88].T[::-1]]): a.imshow(s, cmap="gray"); a.axis("off")
ax[0].contour(mask[:, :, 80].T, [.5], colors="r", linewidths=.8); ax[1].contour(mask[:, 105, :].T[::-1], [.5], colors="r", linewidths=.8)
plt.tight_layout(); plt.savefig(ROOT / "results" / "template_check.png", dpi=90)
