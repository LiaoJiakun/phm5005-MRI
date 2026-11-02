"""Second hippocampus-level variant requested after seeing the first: stack z=59-63 (centre 61). Same protocol as hippo_cv.py.
  H4  late fusion (concat) axial z=59-63      H5  single axial z=61
Output: results/hippo/resnet/slice_59_63/ (resumable)"""
import numpy as np, torch
import oasis_lib as L, oasis_5slice as S
torch.set_num_threads(6)
Hz = np.load(L.ROOT / "data" / "combined_full" / "hippo_slices_z59-63.npz", allow_pickle=True); assert (Hz["subject_id"] == S.SID).all()
h = Hz["hippo5"]
S.run("resnet", L.ROOT / "results" / "hippo" / "resnet" / "slice_59_63", arms={"H5: single axial z61": (h[:, 2:3], "stack"), "H4: late fusion z59-63": (h, "late")}); print("ALL DONE", flush=True)
