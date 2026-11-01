"""Check preprocessing, slice orientation and slice indices of the 236-subject cohort.
A) axes/orientation of every volume; B) fixed and Canny slices are re-extracted from the SAME loaded volume along the SAME axis (axis 2) and compared
with the stored arrays; C) orthogonal-view figure with the fixed / Canny slice positions; D) alignment between OASIS-1 and OASIS-2 using the
111 people who are in both datasets (flip / slice-shift tests); E) preprocessing sanity (empty slices, intensity range)."""
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib, cv2, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
C = pd.read_csv(ROOT / "data" / "combined_full" / "combined_cohort.csv"); Z = np.load(ROOT / "data" / "combined_full" / "axial5_slices.npz", allow_pickle=True)
W = np.load(ROOT / "data" / "combined_full" / "combined_slices.npz", allow_pickle=True)
assert (Z["subject_id"] == C.subject_id.to_numpy()).all() and (W["subject_id"] == C.subject_id.to_numpy()).all()
def norm(s):
    s = np.nan_to_num(s.astype(np.float32)); fg = s[s > 0]
    if fg.size < 100: return np.zeros(s.shape, np.uint8)
    lo, hi = np.percentile(fg, [1, 99]); return np.uint8(np.clip((s - lo) / max(hi - lo, 1e-6), 0, 1) * 255)
rs = lambda x: cv2.resize(norm(x), (128, 128), interpolation=cv2.INTER_AREA)
AX = lambda v, i: rs(v[:, :, i].T)             # axial = index along axis 2
COR = lambda v, j: rs(v[:, j, :].T[::-1])      # coronal = index along axis 1
def load(p):
    img = nib.load(p); raw = nib.aff2axcodes(img.affine)
    if not p.endswith(".nii.gz"): img = nib.as_closest_canonical(img)
    return np.squeeze(np.asarray(img.dataobj, dtype=np.float32)), raw, nib.aff2axcodes(img.affine), img.affine
def canny_center(v):
    sc = {i: int(np.count_nonzero(cv2.Canny(cv2.GaussianBlur(AX(v, i), (5, 5), 1), 50, 150))) for i in range(50, 111)}
    return max(sc, key=lambda i: (sc[i], -i))

# ---------------- A + B + E (all 236 subjects)
codes, same_fixed, same_canny, same_center, empty, shapes, int_max = [], 0, 0, 0, 0, set(), []
for n, r in enumerate(C.itertuples()):
    v, raw, can, aff = load(r.mri_path); shapes.add(v.shape); codes.append((r.dataset, "".join(raw), "".join(can), bool(np.allclose(aff, np.eye(4)))))
    c = canny_center(v); same_center += int(c == Z["canny_center"][n])
    f = np.array([AX(v, i) for i in range(78, 83)]); k = np.array([AX(v, i) for i in range(c - 2, c + 3)])
    same_fixed += int(np.array_equal(f, Z["fixed5"][n])); same_canny += int(np.array_equal(k, Z["canny5"][n]))
    empty += int(any(s.max() == 0 for s in list(f) + list(k))); int_max.append(min(s.max() for s in list(f) + list(k)))
cd = pd.DataFrame(codes, columns=["dataset", "header_axcodes", "after_canonical", "affine_is_identity"])
print("== A) orientation / grid of the 236 volumes"); print("volume shapes:", shapes)
print(cd.groupby(["dataset", "header_axcodes", "after_canonical", "affine_is_identity"]).size().to_string())
print(f"== B) re-extracted from the same volume, same axis (axis 2): fixed 78-82 identical for {same_fixed}/236, Canny c-2..c+2 identical for {same_canny}/236, Canny centre identical for {same_center}/236")
print(f"== E) subjects with an all-zero slice among the 10 slices: {empty}/236 | min over subjects of the slice max intensity: {min(int_max)}")

# ---------------- C) figure
pick = [C[(C.dataset == d) & (C.label == l)].index[3] for d in ("OASIS1", "OASIS2") for l in (0, 1)]
fig, ax = plt.subplots(len(pick), 3, figsize=(12.5, 4.1 * len(pick)))
for r, n in enumerate(pick):
    v, _, _, _ = load(C.mri_path[n]); c = int(Z["canny_center"][n]); cy = int(W["canny_coronal_idx"][n])
    views = [("sagittal x=88 (horizontal = y, vertical = z)", v[88].T), ("coronal y=105 (horizontal = x, vertical = z)", v[:, 105, :].T), ("axial z=80 (horizontal = x, vertical = y)", v[:, :, 80].T)]
    for k, (title, im) in enumerate(views):
        a = ax[r, k]; a.imshow(im, cmap="gray", origin="lower"); a.set_title(f"{C.subject_id[n]} ({C.dataset[n]}, {'CDR>0' if C.label[n] else 'CDR=0'})\n{title}", fontsize=8.2)
        if k == 0: a.axhspan(77.5, 82.5, color="#2a6f97", alpha=.45); a.axhspan(c - 2.5, c + 2.5, color="#d1495b", alpha=.45); a.axvline(105, color="#2a6f97", ls="--"); a.axvline(cy, color="#d1495b", ls="--")
        if k == 1: a.axhspan(77.5, 82.5, color="#2a6f97", alpha=.45); a.axhspan(c - 2.5, c + 2.5, color="#d1495b", alpha=.45)
        if k == 2: a.axhline(105, color="#2a6f97", ls="--"); a.axhline(cy, color="#d1495b", ls="--")
        a.set_xticks([]); a.set_yticks([])
ax[0, 0].text(2, 168, "blue = fixed (axial 78-82, coronal 105)   red = Canny (axial c±2, coronal)", color="w", fontsize=7.5, bbox=dict(fc="k", alpha=.5))
plt.tight_layout(); fig.savefig(ROOT / "results" / "slice_orientation_check.png", dpi=130)

# ---------------- D) OASIS-1 / OASIS-2 alignment with the 111 duplicated people
rep = pd.read_csv(ROOT / "data" / "combined_full" / "oasis2_overlap_report.csv"); rep = rep[rep.removed]
p1 = dict(zip(C.subject_id, C.mri_path)); z = lambda a: (a.ravel().astype(np.float32) - a.mean()) / (a.std() + 1e-6)
def cc(a, b): return float(np.corrcoef(z(a), z(b))[0, 1])
res = {k: [] for k in ["identity", "flip LR (x)", "flip AP (y)", "flip SI (z)"]}; shiftz = {k: [] for k in range(-8, 9, 2)}; shifty = {k: [] for k in range(-8, 9, 2)}
for r in rep.itertuples():
    p2 = ROOT / "data" / "oasis2" / "registered" / f"{r.oasis2_id}_MR1_t88_masked.nii.gz"
    if not p2.exists() or r.nearest_oasis1_id not in p1: continue
    a, _, _, _ = load(p1[r.nearest_oasis1_id]); b, _, _, _ = load(str(p2))
    ref = (AX(a, 80), COR(a, 105))
    res["identity"].append(np.mean([cc(ref[0], AX(b, 80)), cc(ref[1], COR(b, 105))]))
    bx, by, bz = b[::-1], b[:, ::-1], b[:, :, ::-1]
    res["flip LR (x)"].append(np.mean([cc(ref[0], AX(bx, 80)), cc(ref[1], COR(bx, 105))])); res["flip AP (y)"].append(np.mean([cc(ref[0], AX(by, 80)), cc(ref[1], COR(by, 105))])); res["flip SI (z)"].append(np.mean([cc(ref[0], AX(bz, 80)), cc(ref[1], COR(bz, 105))]))
    for k in shiftz: shiftz[k].append(cc(ref[0], AX(b, 80 + k))); shifty[k].append(cc(ref[1], COR(b, 105 + k)))
print(f"== D) {len(res['identity'])} people present in both datasets (same fixed slices, axial 80 + coronal 105): mean correlation OASIS-1 vs OASIS-2")
for k, v in res.items(): print(f"   {k:12s}: {np.mean(v):.3f}  (min {np.min(v):.3f})")
print("   axial slice shift (OASIS-2 index = 80 + k):", {k: round(float(np.mean(v)), 3) for k, v in shiftz.items()})
print("   coronal slice shift (OASIS-2 index = 105 + k):", {k: round(float(np.mean(v)), 3) for k, v in shifty.items()})
pd.DataFrame({k: v for k, v in res.items()}).to_csv(ROOT / "results" / "slice_alignment_pairs.csv", index=False)
