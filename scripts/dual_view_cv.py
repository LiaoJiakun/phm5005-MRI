"""Axial vs coronal vs dual-branch (axial+coronal), all with label-independent Canny-selected slices.
Arms: axial | coronal | dual (axial branch + coronal branch) | dual_axax (capacity-matched control: two branches, BOTH fed the axial slice).
Backbones: cnn (small CNN, from scratch) | resnet (ImageNet ResNet18, layer4+head trained, rest frozen).
Same subjects, same repeated 5-fold subject-level folds / val splits / seeds as cv_compare.py (paired design).
Usage: python dual_view_cv.py [cnn] [resnet]      Resumable: results/dual_view/<backbone>_fold_results.csv
"""
import copy, random, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "dual_view"; OUT.mkdir(parents=True, exist_ok=True)
torch.hub.set_dir(str(ROOT / "weights" / "torch_hub"))
REPEATS, FOLDS, BATCH = 2, 5, 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Z = np.load(ROOT / "data" / "cohort" / "cohort_slices_dual.npz", allow_pickle=True)
sid, y_all = Z["subject_id"], Z["label"].astype(int)
IMGS = {"axial": Z["axial"], "coronal": Z["coronal"]}
ARMS = {"axial": ("axial",), "coronal": ("coronal",), "dual": ("axial", "coronal"), "dual_axax": ("axial", "axial")}
CFG = {"cnn": dict(size=128, ch=1, lr=1e-3, wd=1e-4, epochs=35, pat=7), "resnet": dict(size=224, ch=3, lr=1e-4, wd=1e-4, epochs=20, pat=5)}

def cnn_branch():
    return nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten()), 64
def resnet_branch():
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1); m.fc = nn.Identity()
    for n, p in m.named_parameters(): p.requires_grad = n.startswith("layer4.")
    return m, 512

class Net(nn.Module):
    def __init__(s, backbone, n_branch):
        super().__init__()
        bs = [(cnn_branch if backbone == "cnn" else resnet_branch)() for _ in range(n_branch)]
        s.branches = nn.ModuleList([b for b, _ in bs]); d = bs[0][1] * n_branch
        s.head = (nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Dropout(.3), nn.Linear(32, 2)) if backbone == "cnn"
                  else nn.Sequential(nn.Dropout(.3), nn.Linear(d, 2)))
        s.backbone = backbone
    def forward(s, xs): return s.head(torch.cat([b(x) for b, x in zip(s.branches, xs)], 1))
    def set_train(s):
        s.train()
        if s.backbone == "resnet":     # frozen BatchNorm stays in eval mode
            for b in s.branches:
                for n, m in b.named_modules():
                    if isinstance(m, nn.BatchNorm2d) and not n.startswith("layer4"): m.eval()

def tf(backbone, aug):
    c = CFG[backbone]
    norm = transforms.Normalize((.5,), (.5,)) if c["ch"] == 1 else transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    return transforms.Compose([transforms.ToPILImage(), *([transforms.Resize(c["size"])] if c["size"] != 128 else []),
                               *([transforms.RandomRotation(5)] if aug else []), transforms.ToTensor(),
                               *([transforms.Lambda(lambda t: t.repeat(3, 1, 1))] if c["ch"] == 3 else []), norm])
class DS(Dataset):
    def __init__(s, views, idx, t): s.v, s.i, s.t = views, idx, t
    def __len__(s): return len(s.i)
    def __getitem__(s, k):
        i = s.i[k]; return [s.t(v[i]) for v in s.v], torch.tensor(y_all[i])
def collate(b): return [torch.stack(x) for x in zip(*[s[0] for s in b])], torch.stack([s[1] for s in b])

def fit_predict(backbone, arm, tr, va, te, seed):
    c = CFG[backbone]; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    g = torch.Generator(); g.manual_seed(seed); views = [IMGS[v] for v in ARMS[arm]]
    mk = lambda idx, aug, sh: DataLoader(DS(views, idx, tf(backbone, aug)), batch_size=BATCH, shuffle=sh, generator=g if sh else None, collate_fn=collate)
    dtr, dva, dte = mk(tr, True, True), mk(va, False, False), mk(te, False, False)
    model = Net(backbone, len(views)).to(DEVICE)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=c["lr"], weight_decay=c["wd"])
    crit = nn.CrossEntropyLoss(); best, state, wait = 1e9, None, 0
    for ep in range(c["epochs"]):
        model.set_train()
        for xs, y in dtr:
            opt.zero_grad(); crit(model([x.to(DEVICE) for x in xs]), y.to(DEVICE)).backward(); opt.step()
        model.eval(); vl = 0
        with torch.no_grad():
            for xs, y in dva: vl += crit(model([x.to(DEVICE) for x in xs]), y.to(DEVICE)).item() * len(y)
        vl /= len(va)
        if vl < best - 1e-4: best, state, wait = vl, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= c["pat"]: break
    model.load_state_dict(state); model.eval(); p = []
    with torch.no_grad():
        for xs, _ in dte: p += model([x.to(DEVICE) for x in xs]).softmax(1)[:, 1].cpu().tolist()
    return np.array(p), sum(q.numel() for q in model.parameters() if q.requires_grad)

if __name__ == "__main__":
    for backbone in (sys.argv[1:] or ["cnn", "resnet"]):
        f_res, f_oof = OUT / f"{backbone}_fold_results.csv", OUT / f"{backbone}_oof_predictions.csv"
        done = pd.read_csv(f_res) if f_res.exists() else pd.DataFrame(columns=["repeat", "fold", "arm"])
        oof = pd.read_csv(f_oof) if f_oof.exists() else pd.DataFrame()
        t0 = time.time()
        for rep in range(REPEATS):
            for fold, (trv, te) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=100 + rep).split(sid, y_all)):
                tr, va = train_test_split(trv, test_size=.15, stratify=y_all[trv], random_state=200 + rep * 10 + fold)
                for arm in ARMS:
                    if ((done.repeat == rep) & (done.fold == fold) & (done.arm == arm)).any(): continue
                    p, npar = fit_predict(backbone, arm, tr, va, te, rep * 100 + fold)
                    auc = roc_auc_score(y_all[te], p); pred = (p >= .5).astype(int)
                    done = pd.concat([done, pd.DataFrame([dict(repeat=rep, fold=fold, arm=arm, auc=auc, acc=float((pred == y_all[te]).mean()),
                                      bal_acc=balanced_accuracy_score(y_all[te], pred), trainable_params=npar)])], ignore_index=True)
                    oof = pd.concat([oof, pd.DataFrame(dict(repeat=rep, fold=fold, arm=arm, subject_id=sid[te], true=y_all[te], prob=p))], ignore_index=True)
                    done.to_csv(f_res, index=False); oof.to_csv(f_oof, index=False)
                    print(f"[{backbone} {(time.time()-t0)/60:5.1f} min] rep{rep} fold{fold} {arm:10s} AUC={auc:.3f} params={npar:,}", flush=True)
        print(f"{backbone} finished", flush=True)
    print("ALL DONE")
