"""Hippocampus-level fixed slices vs the earlier fixed slices (ResNet18, same folds/seeds as every other CV -> paired).
  H1  single axial z=62                          (cf. A: fixed axial 80)
  H2  dual view: axial z=62 + fixed coronal 105  (cf. C: fixed axial 80 + coronal 105)
  H3  late fusion (concat) axial z=60-64         (cf. M2: late fusion fixed 78-82)
Usage: python hippo_cv.py   Output: results/hippo/resnet/ (resumable)"""
import numpy as np, torch
import oasis_lib as L, oasis_5slice as S
torch.set_num_threads(6)
Hz = np.load(L.ROOT / "data" / "combined_full" / "hippo_slices.npz", allow_pickle=True); assert (Hz["subject_id"] == S.SID).all()
hip5 = Hz["hippo5"]; out = L.ROOT / "results" / "hippo" / "resnet"
# H1 + H3 through the 5-slice runner, H2 through the dual-view runner (own result folders, merged at analysis time)
S.run("resnet", out / "slice", arms={"H1: single axial z62": (hip5[:, 2:3], "stack"), "H3: late fusion z60-64": (hip5, "late")})
d = L.load_data(); assert (d["subject_id"] == S.SID).all()
L.run_cv(d, "resnet", out / "dual", experiments={"H2: axial z62 + coronal 105": [hip5[:, 2], d["fixed_coronal"]]}); print("ALL DONE", flush=True)
