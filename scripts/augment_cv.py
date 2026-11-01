"""MRI data-augmentation experiment on the best model (ResNet18, late fusion by concatenation, fixed axial slices 78-82).
Same cohort, folds, seeds, architecture and hyper-parameters as five_slice_cv.py; ONLY the training-time augmentation differs. Augmentation is applied to
TRAINING images only, with ONE random parameter set per subject shared by all 5 slices (they are one slab). Validation/test images are never augmented.
  baseline : rotation +-5 deg (what all earlier results used)
  G  geometric : rotation +-10 deg, translation +-5 %, isotropic scale +-5 %    (no elastic deformation, no large zoom: ventricle size / sulcal width are the signal)
  I  intensity : baseline rotation + MRI-specific intensity changes, each with p = 0.5:
        smooth multiplicative bias field (B1 inhomogeneity, +-15 %), gamma/contrast (0.8-1.25), Gaussian noise inside the brain (sigma <= 0.03), mild blur (sigma 0.3-0.8 px)
  GI : G + I
Usage: python augment_cv.py baseline_check | G | I | GI"""
import copy, math, os, random, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms.functional as TF
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset
import oasis_lib as L, oasis_5slice as S

BIL = TF.InterpolationMode.BILINEAR
def rot5(x): return TF.rotate(x, random.uniform(-5, 5), interpolation=BIL)                       # baseline (identical RNG use to oasis_5slice.StackDataset)
def geo(x):
    h, w = x.shape[-2:]
    return TF.affine(x, angle=random.uniform(-10, 10), translate=[round(random.uniform(-.05, .05) * w), round(random.uniform(-.05, .05) * h)],
                     scale=random.uniform(.95, 1.05), shear=[0.0, 0.0], interpolation=BIL)
def intensity(x):
    mask = (x > 0).float()                                                                       # brain mask: the background must stay exactly 0
    if random.random() < .5:                                                                      # smooth bias field, same for all slices
        field = F.interpolate((torch.rand(1, 1, 4, 4) * 2 - 1) * .15, size=x.shape[-2:], mode="bicubic", align_corners=False)[0]
        x = x * torch.exp(field)
    if random.random() < .5:                                                                      # gamma / contrast
        x = x.clamp(0, 1) ** math.exp(random.uniform(math.log(.8), math.log(1.25)))
    if random.random() < .5:                                                                      # noise inside the brain only
        x = x + torch.randn_like(x) * random.uniform(0, .03) * mask
    if random.random() < .5:                                                                      # mild blur (motion / resolution), silhouette restored
        x = TF.gaussian_blur(x, kernel_size=3, sigma=random.uniform(.3, .8)) * mask
    return x.clamp(0, 1)
AUG = {"baseline_check": rot5, "G": geo, "I": lambda x: intensity(rot5(x)), "GI": lambda x: intensity(geo(x))}

class AugDataset(Dataset):
    def __init__(self, arr, idx, backbone, aug_fn): self.arr, self.idx, self.mean, self.std, self.aug_fn = arr, idx, *S.NORM[backbone], aug_fn
    def __len__(self): return len(self.idx)
    def __getitem__(self, k):
        i = self.idx[k]; x = torch.from_numpy(self.arr[i]).float() / 255.
        if self.aug_fn is not None: x = self.aug_fn(x)
        return (x - self.mean) / self.std, torch.tensor(S.Y[i])

def train_aug(arr, method, backbone, tr, va, seed, aug_fn):     # = oasis_5slice.train, only the training dataset differs
    hp = S.HP[backbone]; L.set_seed(seed); g = torch.Generator(); g.manual_seed(seed)
    dtr = DataLoader(AugDataset(arr, tr, backbone, aug_fn), batch_size=hp["batch"], shuffle=True, generator=g); dva = S.loader(arr, va, backbone, False)
    model = S.SliceNet(backbone, arr.shape[1], method).to(S.DEVICE); opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"]); crit = nn.CrossEntropyLoss()
    best, state, wait = float("inf"), None, 0
    for ep in range(hp["epochs"]):
        model.train()
        for x, y in dtr: opt.zero_grad(); crit(model(x.to(S.DEVICE)), y.to(S.DEVICE)).backward(); opt.step()
        model.eval(); vl = 0.0
        with torch.no_grad():
            for x, y in dva: vl += crit(model(x.to(S.DEVICE)), y.to(S.DEVICE)).item() * len(y)
        vl /= len(va)
        if vl < best - 1e-4: best, state, wait = vl, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= hp["patience"]: break
    model.load_state_dict(state); return model, ep + 1

if __name__ == "__main__":
    arm = sys.argv[1]; assert arm in AUG; torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    out = L.ROOT / "results" / os.environ.get("AUG_RESULTS", "augmentation") / arm; out.mkdir(parents=True, exist_ok=True); f_res, f_oof = out / "fold_results.csv", out / "oof_predictions.csv"
    done = pd.read_csv(f_res) if f_res.exists() else pd.DataFrame(columns=["repeat", "fold"]); oof = pd.read_csv(f_oof) if f_oof.exists() else pd.DataFrame(); t0 = time.time()
    arr = S.Z["fixed5"]; data = {"subject_id": S.SID, "dataset": S.DS, "label": S.Y, "strata": np.array([f"{a}_{b}" for a, b in zip(S.DS, S.Y)])}
    for rep in range(1 if arm == "baseline_check" else L.N_REPEATS):
        for fold, (tr, va, te) in enumerate(L.cv_folds(data, rep)):
            if arm == "baseline_check" and fold > 0: break
            if ((done["repeat"] == rep) & (done["fold"] == fold)).any(): continue
            model, ep = train_aug(arr, "late", "resnet", tr, va, L.SEEDS["model"] + rep * 100 + fold, AUG[arm])
            p, y = S.predict(model, S.loader(arr, te, "resnet", False)); pred = (p >= .5).astype(int)
            row = dict(repeat=rep, fold=fold, arm=arm, auc=roc_auc_score(y, p), acc=float((pred == y).mean()), bal_acc=balanced_accuracy_score(y, pred), epochs=ep, n_test=len(te))
            done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
            oof = pd.concat([oof, pd.DataFrame(dict(repeat=rep, fold=fold, arm=arm, subject_id=S.SID[te], dataset=S.DS[te], true=y, prob=p))], ignore_index=True)
            done.to_csv(f_res, index=False); oof.to_csv(f_oof, index=False)
            print(f"[{arm} {(time.time() - t0) / 60:5.1f} min] rep{rep} fold{fold} AUC={row['auc']:.4f} acc={row['acc']:.4f} epochs={ep}", flush=True)
    print("ALL DONE", flush=True)
