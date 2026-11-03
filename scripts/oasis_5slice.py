"""Multi-slice comparison on the 236-subject no-overlap cohort (axial plane).
  Method 1  2.5D CNN        : the 5 slices are stacked as 5 input channels of ONE 2D network (early fusion)
  Method 2  late fusion     : every slice goes through the SAME encoder separately, the 5 feature vectors are concatenated (late fusion)
  Slice sets: fixed 78-82  |  Canny: best slice c (most edges in 50-110) and c-2..c+2
  Reference : single slice (fixed 80, Canny c) through the same pipeline.
Same folds / seeds / inner validation split as the final CV (oasis_lib), so results are paired with the earlier experiments."""
import copy, os, random, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
import torchvision.transforms.functional as TF
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, ResNet18_Weights
import oasis_lib as L

DEVICE, HP = L.DEVICE, L.HPARAMS
Z = np.load(L.ROOT / "data" / "combined_full" / os.environ.get("AXIAL5_NPZ", "axial5_slices.npz"), allow_pickle=True)   # AXIAL5_NPZ=axial5_slices_square.npz -> square-padded axial slices
SID, Y, DS = Z["subject_id"], Z["label"].astype(int), Z["dataset"]
ARMS = {  # name: (array, method)      method "stack" = 2.5D / single slice, "late" = shared encoder + feature concatenation
    "S1: single slice fixed 80":   (Z["fixed5"][:, 2:3], "stack"),
    "S1: single slice Canny c":    (Z["canny5"][:, 2:3], "stack"),
    "M1: 2.5D fixed 78-82":        (Z["fixed5"], "stack"),
    "M2: late fusion fixed 78-82": (Z["fixed5"], "late"),
    "M1: 2.5D Canny c±2":          (Z["canny5"], "stack"),
    "M2: late fusion Canny c±2":   (Z["canny5"], "late"),
}
NORM = {"cnn": (0.5, 0.5), "resnet": (0.449, 0.226)}     # constants (ResNet: grey-level equivalent of the ImageNet statistics)

def cnn_encoder(in_ch):
    return nn.Sequential(nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten()), 64

def resnet_encoder(in_ch):
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1); m.fc = nn.Identity()
    if in_ch != 3:     # first conv: average the pretrained RGB filters, replicate to in_ch channels (rescaled to keep the activation level)
        w = m.conv1.weight.data.mean(1, keepdim=True).repeat(1, in_ch, 1, 1) * (3.0 / in_ch)
        m.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False); m.conv1.weight.data.copy_(w)
    return m, 512

def pool_slice_features(features):
    """Late fusion by MEAN POOLING over slices.
    features: list of k tensors, each [B, d] - the encoder output of one slice (same shared encoder for every slice).
    Returns one [B, d] tensor that is passed to the classifier head (so the head input size does NOT grow with k)."""
    return torch.stack(features, dim=0).mean(dim=0)      # [k, B, d] -> average over the slice axis -> [B, d]

class SliceNet(nn.Module):
    """method: 'stack'     = 2.5D, slices stacked as input channels (early fusion)
               'late'      = shared encoder per slice, features CONCATENATED (head input k*d)
               'late_mean' = shared encoder per slice, features MEAN-POOLED (head input d)
               'early_sum' = the k slices are summed pixel-wise (divided by k) into one 1-channel image, one encoder"""
    def __init__(self, backbone, n_slices, method):
        super().__init__(); self.backbone, self.k, self.method = backbone, n_slices, method
        make = cnn_encoder if backbone == "cnn" else resnet_encoder
        if method == "stack":       self.enc, d = make(n_slices if backbone == "cnn" else n_slices)
        elif method == "early_sum": self.enc, d = make(1)       # early fusion by summation: the k slices are added pixel-wise into ONE image (1 channel)
        else:                 self.enc, d = make(1 if backbone == "cnn" else 3); d = d * n_slices if method == "late" else d
        self.head = (nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 2)) if backbone == "cnn" else nn.Sequential(nn.Dropout(0.3), nn.Linear(d, 2)))
    def forward(self, x):                      # x: [B, k, H, W]
        if self.method == "stack": return self.head(self.enc(x))
        if self.method == "early_sum": return self.head(self.enc(x.mean(1, keepdim=True)))   # sum / k (same as the sum up to a constant, keeps the grey-level range)
        f = [self.enc(x[:, j:j + 1] if self.backbone == "cnn" else x[:, j:j + 1].repeat(1, 3, 1, 1)) for j in range(self.k)]
        return self.head(pool_slice_features(f) if self.method == "late_mean" else torch.cat(f, 1))

class StackDataset(Dataset):
    def __init__(self, arr, idx, backbone, augment): self.arr, self.idx, self.mean, self.std, self.aug = arr, idx, *NORM[backbone], augment
    def __len__(self): return len(self.idx)
    def __getitem__(self, k):
        i = self.idx[k]; x = torch.from_numpy(self.arr[i]).float() / 255.
        if self.aug: x = TF.rotate(x, random.uniform(-5, 5), interpolation=TF.InterpolationMode.BILINEAR)   # same rotation for all slices of a subject
        return (x - self.mean) / self.std, torch.tensor(Y[i])

def loader(arr, idx, backbone, train, seed=0):
    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(StackDataset(arr, idx, backbone, train), batch_size=HP[backbone]["batch"], shuffle=train, generator=g if train else None)

def predict(model, dl):
    model.eval(); p, y = [], []
    with torch.no_grad():
        for x, t in dl: p += model(x.to(DEVICE)).softmax(1)[:, 1].cpu().tolist(); y += t.tolist()
    return np.array(p), np.array(y)

def train(arr, method, backbone, tr, va, seed):
    hp = HP[backbone]; L.set_seed(seed); dtr, dva = loader(arr, tr, backbone, True, seed), loader(arr, va, backbone, False)
    model = SliceNet(backbone, arr.shape[1], method).to(DEVICE); opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"]); crit = nn.CrossEntropyLoss()
    best, state, wait = float("inf"), None, 0
    for ep in range(hp["epochs"]):
        model.train()
        for x, y in dtr: opt.zero_grad(); crit(model(x.to(DEVICE)), y.to(DEVICE)).backward(); opt.step()
        model.eval(); vl = 0.0
        with torch.no_grad():
            for x, y in dva: vl += crit(model(x.to(DEVICE)), y.to(DEVICE)).item() * len(y)
        vl /= len(va)
        if vl < best - 1e-4: best, state, wait = vl, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= hp["patience"]: break
    model.load_state_dict(state); return model, ep + 1

def run(backbone, out_dir, arms=None):
    arms = arms or ARMS
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True); f_res, f_oof = out_dir / "fold_results.csv", out_dir / "oof_predictions.csv"
    done = pd.read_csv(f_res) if f_res.exists() else pd.DataFrame(columns=["repeat", "fold", "exp"]); oof = pd.read_csv(f_oof) if f_oof.exists() else pd.DataFrame(); t0 = time.time()
    data = {"subject_id": SID, "dataset": DS, "label": Y, "strata": np.array([f"{a}_{b}" for a, b in zip(DS, Y)])}
    for rep in range(L.N_REPEATS):
        for fold, (tr, va, te) in enumerate(L.cv_folds(data, rep)):
            for name, (arr, method) in arms.items():
                if ((done["repeat"] == rep) & (done["fold"] == fold) & (done["exp"] == name)).any(): continue
                model, ep = train(arr, method, backbone, tr, va, L.SEEDS["model"] + rep * 100 + fold)
                p, y = predict(model, loader(arr, te, backbone, False)); pred = (p >= .5).astype(int)
                row = dict(repeat=rep, fold=fold, exp=name, auc=roc_auc_score(y, p), acc=float((pred == y).mean()), bal_acc=balanced_accuracy_score(y, pred), epochs=ep, n_test=len(te),
                           params=sum(q.numel() for q in model.parameters()))
                done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
                oof = pd.concat([oof, pd.DataFrame(dict(repeat=rep, fold=fold, exp=name, subject_id=SID[te], dataset=DS[te], true=y, prob=p))], ignore_index=True)
                done.to_csv(f_res, index=False); oof.to_csv(f_oof, index=False)
                print(f"[{backbone} {(time.time() - t0) / 60:6.1f} min] rep{rep} fold{fold} {name:30s} AUC={row['auc']:.3f} epochs={ep} params={row['params']:,}", flush=True)
