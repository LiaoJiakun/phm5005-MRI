"""Build the Age>=60 OASIS-1 cohort dataset (same filters/selection as oasis1_fixed_vs_canny.py)."""
import re
from pathlib import Path
import numpy as np, pandas as pd, cv2, nibabel as nib
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw"; MRI = ROOT / "data" / "extracted"; OUT = ROOT / "data" / "cohort"
OUT.mkdir(parents=True, exist_ok=True)
MIN_AGE, SEED, SIZE, FIXED, CAND = 60, 42, 128, 80, range(50, 111)

meta = pd.read_excel(DATA / "oasis_cross-sectional.xlsx")
meta.columns = meta.columns.astype(str).str.strip()
n_all = len(meta)
meta["subject_id"] = meta["ID"].astype(str).str.extract(r"(OAS1_\d+)", expand=False)
for c in ("Age", "CDR"): meta[c] = pd.to_numeric(meta[c], errors="coerce")
n_young = int((meta.Age < MIN_AGE).sum()); n_nocdr = int(((meta.Age >= MIN_AGE) & meta.CDR.isna()).sum())
coh = meta.dropna(subset=["subject_id", "Age", "CDR"]); coh = coh[coh.Age >= MIN_AGE].copy()
coh["label"] = (coh.CDR > 0).astype(int)
coh["group"] = np.where(coh.label == 1, "CDR>0", "CDR=0")

rows = [{"subject_id": re.search(r"OAS1_\d+", str(p.relative_to(MRI))).group(), "mri_path": str(p)}
        for p in sorted(MRI.rglob("*t88_masked_gfc.img")) if p.with_suffix(".hdr").exists()]
files = pd.DataFrame(rows).sort_values(["subject_id", "mri_path"]).drop_duplicates("subject_id")
coh = coh.sort_values(["subject_id", "ID"]).drop_duplicates("subject_id")
cohort = coh.merge(files, on="subject_id", how="left", validate="one_to_one")
n_nomri = int(cohort.mri_path.isna().sum()); cohort = cohort.dropna(subset=["mri_path"]).copy()

tr, tmp = train_test_split(cohort, test_size=.30, random_state=SEED, stratify=cohort.label)
va, te = train_test_split(tmp, test_size=.50, random_state=SEED, stratify=tmp.label)
cohort["split"] = cohort.subject_id.map({**{s: "train" for s in tr.subject_id}, **{s: "val" for s in va.subject_id}, **{s: "test" for s in te.subject_id}})

def norm(s):
    s = np.nan_to_num(s.astype(np.float32)); fg = s[s > 0]
    if fg.size < 100: return np.zeros(s.shape, np.uint8)
    lo, hi = np.percentile(fg, [1, 99]); return np.uint8(np.clip((s - lo) / max(hi - lo, 1e-6), 0, 1) * 255)
def sl(v, i): return cv2.resize(norm(v[:, :, i].T), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
def canny(v):
    sc = {i: int(np.count_nonzero(cv2.Canny(cv2.GaussianBlur(sl(v, i), (5, 5), 1), 50, 150))) for i in CAND if i < v.shape[2]}
    return max(sc, key=lambda i: (sc[i], -i))

fx, cn, ci, ok = [], [], [], []
for r in cohort.itertuples():
    try:
        v = np.squeeze(np.asarray(nib.as_closest_canonical(nib.load(r.mri_path)).dataobj, dtype=np.float32))
        k = canny(v); fx.append(sl(v, FIXED)); cn.append(sl(v, k)); ci.append(k); ok.append(True)
    except Exception as e:
        print("FAILED", r.subject_id, e); ok.append(False)
cohort = cohort[ok].copy(); cohort["canny_slice_index"] = ci; cohort["fixed_slice_index"] = FIXED
cohort.to_csv(OUT / "cohort_age60plus.csv", index=False)
np.savez_compressed(OUT / "cohort_slices.npz", subject_id=cohort.subject_id.to_numpy(), label=cohort.label.to_numpy(),
                    split=cohort.split.to_numpy(), fixed=np.stack(fx), canny=np.stack(cn), canny_index=np.array(ci))

print(f"OASIS-1 records: {n_all} | age<{MIN_AGE} removed: {n_young} | age>={MIN_AGE} w/o CDR: {n_nocdr} | no T88 MRI: {n_nomri}")
print(f"Final cohort: {len(cohort)} subjects -> {OUT}")
print(cohort.groupby(["split", "group"]).size().unstack(fill_value=0))
print(cohort.groupby("group")[["Age"]].describe().round(1).to_string())
print(cohort.groupby("group")["M/F"].value_counts().unstack(fill_value=0) if "M/F" in cohort else "")
