import sys, torch
import oasis_5slice as S
backbone = sys.argv[1]; assert backbone in ("cnn", "resnet"); torch.set_num_threads(6)
print("N =", len(S.SID), {k: v[0].shape for k, v in S.ARMS.items()}, flush=True)
S.run(backbone, S.L.ROOT / "results" / "five_slice" / backbone); print("ALL DONE", flush=True)
