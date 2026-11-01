"""Repeated stratified 5-fold CV for the 5-slice experiments (single slice, 2.5D, late fusion by concatenation; fixed 78-82 vs Canny c+-2) on the 236-subject cohort. Usage: python five_slice_cv.py cnn|resnet. Output: results/five_slice/<backbone>/ (resumable)."""
import sys, torch
import oasis_5slice as S
backbone = sys.argv[1]; assert backbone in ("cnn", "resnet"); torch.set_num_threads(6)
print("N =", len(S.SID), {k: v[0].shape for k, v in S.ARMS.items()}, flush=True)
S.run(backbone, S.L.ROOT / "results" / "five_slice" / backbone); print("ALL DONE", flush=True)
