"""Early fusion by SUMMATION on the 236-subject no-overlap cohort (same folds/seeds as five_slice_cv.py -> paired with the earlier arms).
New arms: the 5 slices (fixed 78-82 / Canny c±2) are summed pixel-wise (divided by 5) into ONE image and passed to one encoder.
Re-used from results/five_slice and results/meanpool: early fusion by CONCATENATION (= 2.5D channel stacking), late fusion concat, late fusion mean pool.
Usage: python early_sum_cv.py cnn|resnet   Output: results/early_sum/<backbone>/"""
import sys, torch
import oasis_5slice as S
backbone = sys.argv[1]; assert backbone in ("cnn", "resnet"); torch.set_num_threads(6)
arms = {"M0: early fusion SUM fixed 78-82": (S.Z["fixed5"], "early_sum"), "M0: early fusion SUM Canny c±2": (S.Z["canny5"], "early_sum")}
S.run(backbone, S.L.ROOT / "results" / "early_sum" / backbone, arms=arms); print("ALL DONE", flush=True)
