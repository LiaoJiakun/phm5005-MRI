"""Re-run of the two final candidates with NEW fold / validation / model seeds, to check how much of the earlier fixed-slice score was selection luck.
The earlier seeds (7000/8000/9000) were used for every comparison, so the best configuration had been picked on them. These seeds are used only here.
  F  late fusion (concat) fixed axial 78-82      (best earlier model)
  H  late fusion (concat) hippocampus z59-63     (pre-specified, tried once)
3 repeats of stratified 5-fold CV, ResNet18, otherwise identical protocol. Output: results/fresh_seeds/resnet/ (resumable)"""
import numpy as np, torch
import oasis_lib as L, oasis_5slice as S
torch.set_num_threads(6)
L.SEEDS.update(fold=17000, val=18000, model=19000); L.N_REPEATS = 3
h = np.load(L.ROOT / "data" / "combined_full" / "hippo_slices_z59-63.npz", allow_pickle=True); assert (h["subject_id"] == S.SID).all()
S.run("resnet", L.ROOT / "results" / "fresh_seeds" / "resnet", arms={"F: late fusion fixed 78-82": (S.Z["fixed5"], "late"), "H: late fusion hippo z59-63": (h["hippo5"], "late")}); print("ALL DONE", flush=True)
