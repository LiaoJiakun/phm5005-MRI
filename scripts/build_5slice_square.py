"""Axial 5-slice stacks WITHOUT aspect-ratio distortion.
Old: the 208 (A-P) x 176 (L-R) axial slice was resized straight to 128x128 (squashed by about 18%).
New: the slice is zero-padded on the left/right to 208 x 208 (background is already 0), then resized to 128x128 (same scale in both directions).
Fixed = slices 78-82; Canny = best slice c in 50-110 (same rule, recomputed on the new images) and c-2..c+2.
Output: data/combined_full/axial5_slices_square.npz   (coronal slices are 176x176 already and need no change)"""
from pathlib import Path
import numpy as np, pandas as pd, cv2, nibabel as nib
ROOT = Path(__file__).resolve().parent.parent; C = ROOT / "data" / "combined_full"
cohort = pd.read_csv(C / "combined_cohort.csv"); old = np.load(C / "axial5_slices.npz", allow_pickle=True)
assert (old["subject_id"] == cohort.subject_id.to_numpy()).all()
def norm(s):
    s = np.nan_to_num(s.astype(np.float32)); fg = s[s > 0]
    if fg.size < 100: return np.zeros(s.shape, np.uint8)
    lo, hi = np.percentile(fg, [1, 99]); return np.uint8(np.clip((s - lo) / max(hi - lo, 1e-6), 0, 1) * 255)
def ax_sq(v, i):
    s = norm(v[:, :, i].T)                                   # [208 (y), 176 (x)]
    h, w = s.shape; pad = (h - w) // 2; assert h >= w and (h - w) % 2 == 0
    s = np.pad(s, ((0, 0), (pad, h - w - pad)))              # pad left/right with 0 -> square
    return cv2.resize(s, (128, 128), interpolation=cv2.INTER_AREA)
def load(p):
    img = nib.load(p) if p.endswith(".nii.gz") else nib.as_closest_canonical(nib.load(p))
    return np.squeeze(np.asarray(img.dataobj, dtype=np.float32))
fixed5, canny5, centers = [], [], []
for n, p in enumerate(cohort.mri_path):
    v = load(p); assert v.shape == (176, 208, 176)
    sc = {i: int(np.count_nonzero(cv2.Canny(cv2.GaussianBlur(ax_sq(v, i), (5, 5), 1), 50, 150))) for i in range(50, 111)}
    c = max(sc, key=lambda i: (sc[i], -i)); centers.append(c)
    fixed5.append([ax_sq(v, i) for i in range(78, 83)]); canny5.append([ax_sq(v, i) for i in range(c - 2, c + 3)])
    if (n + 1) % 60 == 0: print(f"  {n+1}/{len(cohort)}", flush=True)
fixed5, canny5, centers = np.array(fixed5), np.array(canny5), np.array(centers)
np.savez_compressed(C / "axial5_slices_square.npz", subject_id=old["subject_id"], dataset=old["dataset"], label=old["label"], fixed5=fixed5, canny5=canny5, canny_center=centers)
def aspect(a):                                                # brain bounding box width / height of the middle fixed slice
    r = []
    for s in a[:, 2]:
        ys, xs = np.where(s > 0); r.append((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1))
    return float(np.mean(r))
print(f"brain bbox width/height at slice 80:  old (squashed) {aspect(old['fixed5']):.3f}  ->  new (square-padded) {aspect(fixed5):.3f}")
oc = old["canny_center"]
print(f"Canny centre: old mean {oc.mean():.1f} (sd {oc.std():.1f}, range {oc.min()}-{oc.max()})  new mean {centers.mean():.1f} (sd {centers.std():.1f}, range {centers.min()}-{centers.max()}) | identical for {(oc == centers).mean():.0%} of subjects | mean |shift| {np.abs(oc - centers).mean():.2f} slices")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 4, figsize=(12, 6.2))
for r, n in enumerate((3, 150)):
    for k, (arr, t) in enumerate([(old["fixed5"][n, 2], "old, fixed 80"), (fixed5[n, 2], "square, fixed 80"), (old["canny5"][n, 2], f"old, Canny c={oc[n]}"), (canny5[n, 2], f"square, Canny c={centers[n]}")]):
        ax[r, k].imshow(arr, cmap="gray"); ax[r, k].set_title(f"{cohort.subject_id[n]}\n{t}", fontsize=8.5); ax[r, k].axis("off")
plt.tight_layout(); plt.savefig(ROOT / "results" / "square_padding_check.png", dpi=110)
