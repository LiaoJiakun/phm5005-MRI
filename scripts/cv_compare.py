"""Repeated stratified 5-fold subject-level CV on the Age>=60 cohort (n=198).
Compares {small CNN from scratch, ResNet18 full fine-tune, ResNet18 layer4-only} x {fixed slice, Canny slice}.
Every subject is tested exactly once per repeat. Within each training portion, 15% is held out for early stopping.
Folds/val splits are identical for all models and both slice arms (paired design). Resumable: results/cv/fold_results.csv
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
OUT = ROOT / "results" / "cv"; OUT.mkdir(parents=True, exist_ok=True)
torch.hub.set_dir(str(ROOT / "weights" / "torch_hub"))
REPEATS, FOLDS, BATCH = 2, 5, 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
z = np.load(ROOT / "data" / "cohort" / "cohort_slices.npz", allow_pickle=True)
sid, y_all = z["subject_id"], z["label"].astype(int)
ARMS = {"Fixed": z["fixed"], "Canny": z["canny"]}

class SmallCNN(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                              nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                              nn.Linear(64, 32), nn.ReLU(), nn.Dropout(.3), nn.Linear(32, 2))
    def forward(s, x): return s.net(x)

MODELS = {  # name: (builder, image size, channels, lr, wd, epochs, patience, trainable-prefixes)
    "CNN scratch":    dict(size=128, ch=1, lr=1e-3, wd=1e-4, epochs=35, pat=7, train=None),
    "ResNet18 full":  dict(size=224, ch=3, lr=1e-4, wd=1e-4, epochs=20, pat=5, train=None),
    "ResNet18 layer4": dict(size=224, ch=3, lr=1e-4, wd=1e-4, epochs=20, pat=5, train=("layer4.", "fc.")),
}
def build(name):
    if name == "CNN scratch": return SmallCNN().to(DEVICE)
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1); m.fc = nn.Sequential(nn.Dropout(.3), nn.Linear(512, 2))
    tr = MODELS[name]["train"]
    if tr:
        for n, p in m.named_parameters(): p.requires_grad = n.startswith(tr)
    return m.to(DEVICE)
def set_train(model, name):
    model.train()
    if MODELS[name]["train"]:  # keep frozen BatchNorm in eval mode
        for n, mod in model.named_modules():
            if isinstance(mod, nn.BatchNorm2d) and not n.startswith("layer4"): mod.eval()

def tf(name, aug):
    c = MODELS[name]
    norm = transforms.Normalize((.5,), (.5,)) if c["ch"] == 1 else transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    return transforms.Compose([transforms.ToPILImage(), *([transforms.Resize(c["size"])] if c["size"] != 128 else []),
                               *([transforms.RandomRotation(5)] if aug else []), transforms.ToTensor(),
                               *([transforms.Lambda(lambda t: t.repeat(3, 1, 1))] if c["ch"] == 3 else []), norm])
class DS(Dataset):
    def __init__(s, x, y, t): s.x, s.y, s.t = x, y, t
    def __len__(s): return len(s.y)
    def __getitem__(s, i): return s.t(s.x[i]), torch.tensor(s.y[i])
def predict(model, loader):
    model.eval(); p = []
    with torch.no_grad():
        for x, _ in loader: p += model(x.to(DEVICE)).softmax(1)[:, 1].cpu().tolist()
    return np.array(p)

def fit_predict(name, imgs, tr, va, te, seed):
    c = MODELS[name]; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    g = torch.Generator(); g.manual_seed(seed)
    mk = lambda idx, aug, sh: DataLoader(DS(imgs[idx], y_all[idx], tf(name, aug)), batch_size=BATCH, shuffle=sh, generator=g if sh else None)
    dtr, dva, dte = mk(tr, True, True), mk(va, False, False), mk(te, False, False)
    model = build(name); opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=c["lr"], weight_decay=c["wd"])
    crit = nn.CrossEntropyLoss(); best, state, wait = 1e9, None, 0
    for ep in range(c["epochs"]):
        set_train(model, name)
        for x, y in dtr:
            opt.zero_grad(); crit(model(x.to(DEVICE)), y.to(DEVICE)).backward(); opt.step()
        model.eval(); vl = 0
        with torch.no_grad():
            for x, y in dva: vl += crit(model(x.to(DEVICE)), y.to(DEVICE)).item() * len(y)
        vl /= len(va)
        if vl < best - 1e-4: best, state, wait = vl, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= c["pat"]: break
    model.load_state_dict(state); return predict(model, dte)

f_res = OUT / "fold_results.csv"; f_oof = OUT / "oof_predictions.csv"
done = pd.read_csv(f_res) if f_res.exists() else pd.DataFrame(columns=["repeat", "fold", "model", "arm"])
oof = pd.read_csv(f_oof) if f_oof.exists() else pd.DataFrame(columns=["repeat", "model", "arm", "subject_id", "true", "prob"])
t0 = time.time()
for rep in range(REPEATS):
    skf = StratifiedKFold(FOLDS, shuffle=True, random_state=100 + rep)
    for fold, (trv, te) in enumerate(skf.split(sid, y_all)):
        tr, va = train_test_split(trv, test_size=.15, stratify=y_all[trv], random_state=200 + rep * 10 + fold)
        for name in MODELS:
            for arm, imgs in ARMS.items():
                if ((done.repeat == rep) & (done.fold == fold) & (done.model == name) & (done.arm == arm)).any(): continue
                p = fit_predict(name, imgs, tr, va, te, seed=rep * 100 + fold)
                auc = roc_auc_score(y_all[te], p); pred = (p >= .5).astype(int)
                done = pd.concat([done, pd.DataFrame([dict(repeat=rep, fold=fold, model=name, arm=arm, auc=auc, acc=float((pred == y_all[te]).mean()),
                                           bal_acc=balanced_accuracy_score(y_all[te], pred), n_test=len(te))])], ignore_index=True)
                oof = pd.concat([oof, pd.DataFrame(dict(repeat=rep, model=name, arm=arm, subject_id=sid[te], true=y_all[te], prob=p))])
                done.to_csv(f_res, index=False); oof.to_csv(f_oof, index=False)
                print(f"[{(time.time()-t0)/60:5.1f} min] rep{rep} fold{fold} {name:16s} {arm:5s} AUC={auc:.3f}", flush=True)
print("CV finished")
