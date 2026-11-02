"""Hippocampus-level axial slices for the 236-subject no-overlap cohort.
Level chosen BEFORE any model run, from anatomy only (no labels): on the 30-subject mean T88 image the hippocampal body/head appears next to the
temporal horn at axial z ~ 60-66 (coronal y ~ 100-110, which already contains the fixed coronal 105). Primary choice: z = 62, stack 60-64.
Output: data/combined_full/hippo_slices.npz  (hippo5: [N, 5, 128, 128] uint8 for z=60..64, same preprocessing as build_5slice.py)"""
from pathlib import Path
import numpy as np, pandas as pd, cv2, nibabel as nib
ROOT = Path(__file__).resolve().parent.parent; C = ROOT / "data" / "combined_full"
cohort = pd.read_csv(C / "combined_cohort.csv"); z = np.load(C / "combined_slices.npz", allow_pickle=True)
assert (z["subject_id"] == cohort.subject_id.to_numpy()).all()
import sys
Z0 = int(sys.argv[1]) if len(sys.argv) > 1 else 60      # first slice of the 5-slice stack (default 60-64; 59 -> 59-63)
SIZE, ZS = 128, range(Z0, Z0 + 5)
OUTNAME = "hippo_slices.npz" if Z0 == 60 else f"hippo_slices_z{Z0}-{Z0+4}.npz"
def norm(s):
    s = np.nan_to_num(s.astype(np.float32)); fg = s[s > 0]
    if fg.size < 100: return np.zeros(s.shape, np.uint8)
    lo, hi = np.percentile(fg, [1, 99]); return np.uint8(np.clip((s - lo) / max(hi - lo, 1e-6), 0, 1) * 255)
ax = lambda v, i: cv2.resize(norm(v[:, :, i].T), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
def load(p):
    img = nib.load(p) if p.endswith(".nii.gz") else nib.as_closest_canonical(nib.load(p))
    return np.squeeze(np.asarray(img.dataobj, dtype=np.float32))
hip = []
for k, p in enumerate(cohort.mri_path):
    v = load(p); assert v.shape == (176, 208, 176); hip.append([ax(v, i) for i in ZS])
    if (k + 1) % 60 == 0: print(f"  {k+1}/{len(cohort)}", flush=True)
hip = np.array(hip); print("hippo5", hip.shape, "empty slices:", int((hip.reshape(len(hip), 5, -1).max(2) == 0).sum()))
np.savez_compressed(C / OUTNAME, subject_id=z["subject_id"], dataset=z["dataset"], label=z["label"], hippo5=hip)
