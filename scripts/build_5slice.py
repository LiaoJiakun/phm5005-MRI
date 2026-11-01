"""Five consecutive axial slices per subject for the 236-subject no-overlap cohort.
  fixed : slices 78, 79, 80, 81, 82
  canny : the slice c with the most Canny edge pixels in 50-110 (same label-independent rule as before), plus c-2 ... c+2.
Output: data/combined_full/axial5_slices.npz  (fixed5, canny5: [N, 5, 128, 128] uint8 ; canny_center [N])"""
from pathlib import Path
import numpy as np, pandas as pd, cv2, nibabel as nib
ROOT = Path(__file__).resolve().parent.parent; C = ROOT / "data" / "combined_full"
cohort = pd.read_csv(C / "combined_cohort.csv"); z = np.load(C / "combined_slices.npz", allow_pickle=True)
assert (z["subject_id"] == cohort.subject_id.to_numpy()).all()
SIZE, CAND = 128, range(50, 111)
def norm(s):
    s = np.nan_to_num(s.astype(np.float32)); fg = s[s > 0]
    if fg.size < 100: return np.zeros(s.shape, np.uint8)
    lo, hi = np.percentile(fg, [1, 99]); return np.uint8(np.clip((s - lo) / max(hi - lo, 1e-6), 0, 1) * 255)
ax = lambda v, i: cv2.resize(norm(v[:, :, i].T), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
def load(p):
    img = nib.load(p) if p.endswith(".nii.gz") else nib.as_closest_canonical(nib.load(p))
    return np.squeeze(np.asarray(img.dataobj, dtype=np.float32))
fixed5, canny5, centers = [], [], []
for k, p in enumerate(cohort.mri_path):
    v = load(p); assert v.shape == (176, 208, 176)
    sc = {i: int(np.count_nonzero(cv2.Canny(cv2.GaussianBlur(ax(v, i), (5, 5), 1), 50, 150))) for i in CAND}
    c = max(sc, key=lambda i: (sc[i], -i)); centers.append(c)
    fixed5.append([ax(v, i) for i in range(78, 83)]); canny5.append([ax(v, i) for i in range(c - 2, c + 3)])
    if (k + 1) % 60 == 0: print(f"  {k+1}/{len(cohort)}", flush=True)
centers = np.array(centers); assert (centers == z["canny_axial_idx"]).all(), "Canny centre must equal the previously selected axial slice"
fixed5, canny5 = np.array(fixed5), np.array(canny5); assert (fixed5[:, 2] == z["fixed_axial"]).all()
np.savez_compressed(C / "axial5_slices.npz", subject_id=z["subject_id"], dataset=z["dataset"], label=z["label"], fixed5=fixed5, canny5=canny5, canny_center=centers)
print("saved", fixed5.shape, canny5.shape, "| Canny centre mean %.1f sd %.1f range %d-%d" % (centers.mean(), centers.std(), centers.min(), centers.max()))
