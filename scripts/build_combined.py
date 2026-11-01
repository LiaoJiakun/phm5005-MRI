"""Combine OASIS-1 (cross-sectional, age>=60) with OASIS-2 (longitudinal, FIRST visit only) and build slices.
Slices (identical preprocessing for both datasets; volumes are in the same 176x208x176 T88-like RAS grid):
  fixed axial 80 | fixed coronal 105 | Canny axial (range 50-110) | Canny coronal (range 66-140)   -> 128x128 uint8
Checks: subject-level uniqueness, no ID overlap between datasets, OASIS-2 Visit==1 / MR Delay==0 only, heuristic cross-dataset near-duplicate scan."""
from pathlib import Path
import numpy as np, pandas as pd, cv2, nibabel as nib
ROOT = Path(__file__).resolve().parent.parent; D = ROOT / "data"; OUT = D / "combined_full"; OUT.mkdir(exist_ok=True)   # OASIS-1 + OASIS-2 PART1 + PART2 (first visit only)
SIZE, FIX_AX, FIX_COR, AX_CAND, COR_CAND = 128, 80, 105, range(50, 111), range(66, 141)

# ---- OASIS-1 ----
o1 = pd.read_csv(D / "cohort" / "cohort_age60plus.csv")
o1 = pd.DataFrame(dict(subject_id=o1.subject_id, dataset="OASIS1", visit=1, age=o1.Age, sex=o1["M/F"], educ=o1.Educ, ses=o1.SES, mmse=o1.MMSE,
                       cdr=o1.CDR, etiv=o1.eTIV, nwbv=o1.nWBV, label=o1.label, mri_path=o1.mri_path))
# ---- OASIS-2: first visit only ----
dem = pd.read_excel(D / "oasis2" / "raw" / "oasis_longitudinal_demographics.xlsx")
n_all, n_subj = len(dem), dem["Subject ID"].nunique()
first = dem[(dem.Visit == 1) & (dem["MR Delay"] == 0)].copy()
assert first["Subject ID"].is_unique and first["MRI ID"].str.endswith("_MR1").all()
reg = {p.name.replace("_t88_masked.nii.gz", ""): str(p) for p in (D / "oasis2" / "registered").glob("*_t88_masked.nii.gz")}
qc = pd.read_csv(D / "oasis2" / "registration_qc.csv").drop_duplicates("subject", keep="last").set_index("subject").ncc
bad = [k for k in reg if qc.get(k, 0) < .4]   # registration failed QC (atlas-space correlation < 0.4; OASIS-1 reference range 0.50-0.78)
print("OASIS-2 excluded for failed atlas registration:", bad); reg = {k: v for k, v in reg.items() if k not in bad}
first["mri_path"] = first["MRI ID"].map(reg); n_nomri = int(first.mri_path.isna().sum())
f = first.dropna(subset=["mri_path", "CDR", "Age"]); f = f[f.Age >= 60]
o2 = pd.DataFrame(dict(subject_id=f["Subject ID"], dataset="OASIS2", visit=1, age=f.Age, sex=f["M/F"], educ=f.EDUC, ses=f.SES, mmse=f.MMSE,
                       cdr=f.CDR, etiv=f.eTIV, nwbv=f.nWBV, label=(f.CDR > 0).astype(int), mri_path=f.mri_path))
comb = pd.concat([o1, o2], ignore_index=True)

# ---- sample-overlap checks ----
assert comb.subject_id.is_unique, "duplicate subject_id in combined cohort"
assert (o2.visit == 1).all() and all(p.split("/")[-1].split("_t88")[0].endswith("_MR1") for p in o2.mri_path), "OASIS-2 must use first visit (MR1) only"
assert set(o1.subject_id).isdisjoint(o2.subject_id) and o1.subject_id.str.startswith("OAS1_").all() and o2.subject_id.str.startswith("OAS2_").all()
assert comb.mri_path.is_unique
print(f"OASIS-2 records: {n_all} sessions / {n_subj} subjects | first visit: {len(first)} | with registered MRI (PART1): {len(first)-n_nomri} | used (age>=60, CDR known): {len(o2)}")
print(f"OASIS-1 used: {len(o1)} | combined: {len(comb)} | ID overlap: 0 (IDs only - image-level overlap is checked below)")

# ---- slices ----
def norm(s):
    s = np.nan_to_num(s.astype(np.float32)); fg = s[s > 0]
    if fg.size < 100: return np.zeros(s.shape, np.uint8)
    lo, hi = np.percentile(fg, [1, 99]); return np.uint8(np.clip((s - lo) / max(hi - lo, 1e-6), 0, 1) * 255)
rs = lambda x: cv2.resize(norm(x), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
ax_ = lambda v, i: rs(v[:, :, i].T); co_ = lambda v, j: rs(v[:, j, :].T[::-1])
def canny(fn, v, cand):
    sc = {i: int(np.count_nonzero(cv2.Canny(cv2.GaussianBlur(fn(v, i), (5, 5), 1), 50, 150))) for i in cand}
    return max(sc, key=lambda i: (sc[i], -i))
def load(p):
    if p.endswith(".nii.gz"): return np.squeeze(np.asarray(nib.load(p).dataobj, dtype=np.float32))
    return np.squeeze(np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj, dtype=np.float32))
S = {k: [] for k in ("fixed_axial", "fixed_coronal", "canny_axial", "canny_coronal", "canny_axial_idx", "canny_coronal_idx")}; ok = []
for k, r in enumerate(comb.itertuples()):
    try:
        v = load(r.mri_path); assert v.shape == (176, 208, 176), v.shape
        ia, ic = canny(ax_, v, AX_CAND), canny(co_, v, COR_CAND)
        S["fixed_axial"].append(ax_(v, FIX_AX)); S["fixed_coronal"].append(co_(v, FIX_COR)); S["canny_axial"].append(ax_(v, ia)); S["canny_coronal"].append(co_(v, ic))
        S["canny_axial_idx"].append(ia); S["canny_coronal_idx"].append(ic); ok.append(True)
    except Exception as e: print("FAILED", r.subject_id, e); ok.append(False)
    if (k + 1) % 50 == 0: print(f"  {k+1}/{len(comb)}", flush=True)
comb = comb[ok].reset_index(drop=True); S = {k: np.array(v) for k, v in S.items()}

# ---- IMAGE-LEVEL OVERLAP CHECK: OASIS-2 first-visit scans vs OASIS-1 --------------------------------------------------------
# ID checks are not enough: a large part of OASIS-2 re-uses participants that are already in OASIS-1 (same brain, different ID).
# Similarity = correlation of the z-scored (fixed axial + fixed coronal) slices. Two DIFFERENT people never exceed ~0.92 in OASIS-1,
# true duplicates are 0.98-0.997. An OASIS-2 subject is removed if  corr > 0.93  OR  (corr > 0.90 and sex, age(+-1 y), eTIV(+-1%) agree).
V = np.concatenate([S["fixed_axial"].reshape(len(comb), -1), S["fixed_coronal"].reshape(len(comb), -1)], 1).astype(np.float32)
V = (V - V.mean(1, keepdims=True)) / (V.std(1, keepdims=True) + 1e-6); V /= np.linalg.norm(V, axis=1, keepdims=True)
Sim = V @ V.T; i1 = np.where(comb.dataset == "OASIS1")[0]; i2 = np.where(comb.dataset == "OASIS2")[0]
within = Sim[np.ix_(i1, i1)].copy(); np.fill_diagonal(within, -1); print("max similarity between two DIFFERENT OASIS-1 people: %.3f" % within.max())
cross = Sim[np.ix_(i2, i1)]; rep = []
for t, a in enumerate(i2):
    b = i1[cross[t].argmax()]; r2, r1 = comb.iloc[a], comb.iloc[b]
    meta = (r2.sex == r1.sex) and abs(r2.age - r1.age) <= 1 and abs(r2.etiv - r1.etiv) / r2.etiv <= .01
    sim = cross[t].max(); rep.append(dict(oasis2_id=r2.subject_id, nearest_oasis1_id=r1.subject_id, image_similarity=sim, sex_age_etiv_match=meta,
                                           cdr_oasis2=r2.cdr, cdr_oasis1=r1.cdr, removed=bool(sim > .93 or (sim > .90 and meta))))
rep = pd.DataFrame(rep); rep.to_csv(OUT / "oasis2_overlap_report.csv", index=False)
drop_ids = set(rep[rep.removed].oasis2_id); keep = ~comb.subject_id.isin(drop_ids).to_numpy()
print(f"OASIS-2 subjects that are the same person as an OASIS-1 subject (removed): {len(drop_ids)} of {len(i2)} | kept (unique to OASIS-2): {len(i2) - len(drop_ids)}")
comb = comb[keep].reset_index(drop=True); S = {k: v[keep] for k, v in S.items()}
# post-check: no remaining cross-dataset pair above the threshold, no near-duplicates inside the final cohort
Sim = Sim[np.ix_(np.where(keep)[0], np.where(keep)[0])]; np.fill_diagonal(Sim, -1)
assert Sim.max() <= .93, f"remaining near-duplicate pair, similarity {Sim.max():.3f}"
print("post-check passed: max similarity between any two subjects of the final cohort = %.3f" % Sim.max())

comb.to_csv(OUT / "combined_cohort.csv", index=False)
np.savez_compressed(OUT / "combined_slices.npz", subject_id=comb.subject_id.to_numpy(), dataset=comb.dataset.to_numpy(), label=comb.label.to_numpy(), **S)
print(comb.groupby(["dataset", "label"]).size().unstack(), "\nfinal N:", len(comb))
