"""Best model (ResNet18, late fusion by concatenation) on FIXED or CANNY axial slices, baseline augmentation (rotation +-5 deg),
same folds/seeds as five_slice_cv.py. Run with AXIAL5_NPZ=axial5_slices_square.npz to use the square-padded slices.
Usage: AXIAL5_NPZ=axial5_slices_square.npz python square_cv.py fixed|canny [threads]"""
import os, sys, torch
import oasis_5slice as S
which = sys.argv[1]; assert which in ("fixed", "canny"); torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
arms = {"M2: late fusion fixed 78-82": (S.Z["fixed5"], "late")} if which == "fixed" else {"M2: late fusion Canny c±2": (S.Z["canny5"], "late")}
tag = "square" if "square" in os.environ.get("AXIAL5_NPZ", "") else "old"
S.run("resnet", S.L.ROOT / "results" / ("square" if tag == "square" else "square_old") / which, arms=arms); print("ALL DONE", flush=True)
