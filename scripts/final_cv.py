"""Final run: repeated 5-fold subject-level CV on OASIS-1 + OASIS-2 (PART1 + PART2, first visit only).
Usage: python final_cv.py cnn | resnet     (run the two backbones in parallel, 6 threads each)"""
import sys, torch
import oasis_lib as L
backbone = sys.argv[1]; assert backbone in ("cnn", "resnet")
torch.set_num_threads(6)
data = L.load_data()
import numpy as np; print("N =", len(data["subject_id"]), dict(zip(*np.unique(data["strata"], return_counts=True))), flush=True)
L.run_cv(data, backbone, L.ROOT / "results" / "final_cv" / backbone)
print("ALL DONE", flush=True)
