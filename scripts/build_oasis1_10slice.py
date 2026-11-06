"""OASIS-1 ONLY (no OASIS-2): 10 consecutive axial slices per subject, fixed vs Canny-selected, rebuilt from the raw T88 volumes.
  fixed : 0-based axial indices 75..84 (inclusive, 10 slices)            [index = 3rd axis of the canonical (R,A,S) volume, v[:, :, i]]
  Canny : centre c = slice with the most Canny edge pixels among 50..110 (ties -> lowest index, as in the previous project); slices c-5 .. c+4
Preprocessing (identical for both): nibabel as_closest_canonical -> v[:, :, i].T ([208 (A-P), 176 (L-R)]) -> 1/99 percentile of the foreground ->
  uint8 -> zero-pad left/right to 208 x 208 -> resize 128 x 128 (INTER_AREA). Empty / constant / non-finite slices are counted and must be zero.
Output: data/oasis1_10slice/oasis1_10slice.npz (local only), results/oasis1_10slice/cohort_audit.json, cohort.csv (subject-level table, no images)."""
import json, re
from pathlib import Path
import numpy as np, pandas as pd, cv2, nibabel as nib
ROOT = Path(__file__).resolve().parent.parent; RAW = ROOT / "data" / "raw"; MRI = ROOT / "data" / "extracted"
OUT = ROOT / "data" / "oasis1_10slice"; RES = ROOT / "results" / "oasis1_10slice"; OUT.mkdir(exist_ok=True, parents=True); RES.mkdir(exist_ok=True, parents=True)
MIN_AGE, SIZE, N_SLICES = 60, 128, 10
FIXED = list(range(75, 85)); CAND = range(50, 111); BEFORE, AFTER = 5, 4          # Canny: c-5 .. c+4
assert len(FIXED) == N_SLICES and BEFORE + AFTER + 1 == N_SLICES

# ---- 1. subjects, scans and labels ----
meta = pd.read_excel(RAW / "oasis_cross-sectional.xlsx"); meta.columns = meta.columns.astype(str).str.strip()
audit = {"clinical_rows": int(len(meta))}
meta["subject_id"] = meta["ID"].astype(str).str.extract(r"(OAS1_\d+)", expand=False)
audit["rows_without_valid_id"] = int(meta.subject_id.isna().sum())
for c in ("Age", "CDR"): meta[c] = pd.to_numeric(meta[c], errors="coerce")
audit["duplicate_ids_in_spreadsheet"] = int(meta.ID.duplicated().sum()); audit["subjects_with_more_than_one_row"] = int(meta.subject_id.duplicated().sum())
audit["excluded_age_under_60"] = int((meta.Age < MIN_AGE).sum())
audit["excluded_age_missing"] = int(meta.Age.isna().sum())
audit["excluded_age60plus_cdr_missing"] = int(((meta.Age >= MIN_AGE) & meta.CDR.isna()).sum())
coh = meta[(meta.Age >= MIN_AGE) & meta.CDR.notna() & meta.subject_id.notna()].copy()
files = pd.DataFrame([{"subject_id": re.search(r"OAS1_\d+", str(p.relative_to(MRI))).group(), "mri_path": str(p)} for p in sorted(MRI.rglob("*t88_masked_gfc.img")) if p.with_suffix(".hdr").exists()])
audit["t88_masked_gfc_scans_found"] = int(len(files)); audit["subjects_with_more_than_one_scan_file"] = int(files.subject_id.duplicated().sum())
files = files.sort_values(["subject_id", "mri_path"]).drop_duplicates("subject_id")           # one scan per person (MR1 sorts first)
coh = coh.sort_values(["subject_id", "ID"]).drop_duplicates("subject_id")
audit["age60plus_cdr_known_before_image_check"] = int(len(coh))
coh = coh.merge(files, on="subject_id", how="left", validate="one_to_one")
audit["excluded_no_image"] = int(coh.mri_path.isna().sum()); coh = coh.dropna(subset=["mri_path"]).reset_index(drop=True)
assert coh.subject_id.is_unique and coh.subject_id.str.startswith("OAS1_").all() and not coh.subject_id.str.startswith("OAS2").any()   # OASIS-1 only
coh["label"] = (coh.CDR > 0).astype(int)
old = pd.read_csv(ROOT / "data" / "cohort" / "cohort_age60plus.csv")
audit["same_subjects_as_previous_cohort_file"] = bool(set(old.subject_id) == set(coh.subject_id))
print(json.dumps(audit, indent=1))

# ---- 2. slices ----
def norm(s):
    s = np.nan_to_num(s.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0); fg = s[s > 0]
    if fg.size < 100: return np.zeros(s.shape, np.uint8), "empty"
    lo, hi = np.percentile(fg, [1, 99])
    if hi <= lo: return np.zeros(s.shape, np.uint8), "constant"
    return np.uint8(np.clip((s - lo) / (hi - lo), 0, 1) * 255), "ok"
flags = {"empty": 0, "constant": 0, "nonfinite_volume": 0}
def ax_sq(v, i):
    s, f = norm(v[:, :, i].T)                              # [208 (y, A-P), 176 (x, L-R)]
    if f != "ok": flags[f] += 1
    h, w = s.shape; pad = (h - w) // 2; assert h >= w and (h - w) % 2 == 0
    return cv2.resize(np.pad(s, ((0, 0), (pad, h - w - pad))), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
def load(p):
    raw = nib.load(p); can = nib.as_closest_canonical(raw)
    return np.squeeze(np.asarray(can.dataobj, dtype=np.float32)), nib.aff2axcodes(raw.affine), nib.aff2axcodes(can.affine)
fixed10, canny10, centers, orient = [], [], [], set()
for n, p in enumerate(coh.mri_path):
    v, ax_in, ax_out = load(p); orient.add((ax_in, ax_out)); assert v.shape == (176, 208, 176), (p, v.shape)
    if not np.isfinite(v).all(): flags["nonfinite_volume"] += 1
    sc = {i: int(np.count_nonzero(cv2.Canny(cv2.GaussianBlur(ax_sq(v, i), (5, 5), 1), 50, 150))) for i in CAND}
    c = max(sc, key=lambda i: (sc[i], -i)); centers.append(c)
    fixed10.append([ax_sq(v, i) for i in FIXED]); canny10.append([ax_sq(v, i) for i in range(c - BEFORE, c + AFTER + 1)])
    if (n + 1) % 50 == 0: print(f"  {n+1}/{len(coh)}", flush=True)
fixed10, canny10, centers = np.array(fixed10), np.array(canny10), np.array(centers)
flags.update(flag_note="counted over all slices inspected incl. the Canny search range; must be 0 for the cohort to be used as is")
audit.update(n_subjects=int(len(coh)), n_cdr0=int((coh.label == 0).sum()), n_cdr_pos=int((coh.label == 1).sum()), cdr_counts={str(k): int(v) for k, v in coh.CDR.value_counts().items()},
             orientation_before_after=[[list(a), list(b)] for a, b in orient], volume_shape=[176, 208, 176], fixed_indices_0based=FIXED,
             canny_search_range=[CAND.start, CAND.stop - 1], canny_window="c-5 .. c+4", canny_center_mean=float(centers.mean()), canny_center_sd=float(centers.std()),
             canny_center_min=int(centers.min()), canny_center_max=int(centers.max()), canny_window_min=int(centers.min() - BEFORE), canny_window_max=int(centers.max() + AFTER),
             preprocessing_flags=flags, fixed_slices_empty_after_resize=int((fixed10.reshape(len(coh), N_SLICES, -1).max(2) == 0).sum()),
             canny_slices_empty_after_resize=int((canny10.reshape(len(coh), N_SLICES, -1).max(2) == 0).sum()))
assert audit["fixed_slices_empty_after_resize"] == 0 and audit["canny_slices_empty_after_resize"] == 0 and sum(v for k, v in flags.items() if k != "flag_note") == 0
np.savez_compressed(OUT / "oasis1_10slice.npz", subject_id=coh.subject_id.to_numpy(), label=coh.label.to_numpy(), cdr=coh.CDR.to_numpy(), fixed10=fixed10, canny10=canny10, canny_center=centers)
coh.drop(columns=["mri_path"]).assign(canny_center=centers).to_csv(RES / "cohort.csv", index=False)
json.dump(audit, open(RES / "cohort_audit.json", "w"), indent=1)
print("saved", fixed10.shape, canny10.shape); print(json.dumps({k: audit[k] for k in ("n_subjects", "n_cdr0", "n_cdr_pos", "canny_center_mean", "canny_center_min", "canny_center_max", "orientation_before_after", "preprocessing_flags")}, indent=1))
