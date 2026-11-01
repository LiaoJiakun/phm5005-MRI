"""Mean-pooling late fusion trial on the 236-subject no-overlap cohort (same folds/seeds as five_slice_cv.py).
New arms only: shared encoder per slice, slice features MEAN-POOLED, for the fixed (78-82) and Canny (c±2) slice sets.
The 2.5D arms (and concat late fusion) are re-used from results/five_slice (identical pipeline, folds and seeds)."""
import sys, torch
import oasis_5slice as S
backbone = sys.argv[1]; assert backbone in ("cnn", "resnet"); torch.set_num_threads(6)
arms = {"M2-mean: late fusion (mean pool) fixed 78-82": (S.Z["fixed5"], "late_mean"), "M2-mean: late fusion (mean pool) Canny c±2": (S.Z["canny5"], "late_mean")}
S.run(backbone, S.L.ROOT / "results" / "meanpool" / backbone, arms=arms); print("ALL DONE", flush=True)
