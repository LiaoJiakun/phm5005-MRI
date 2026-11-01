"""ImageNet-pretrained ResNet18 (full fine-tuning): fixed slice vs Canny slice on the Age>=60 OASIS-1 cohort.

Uses data/cohort/cohort_slices.npz (built by build_cohort.py): same subjects, same train/val/test split,
same hyperparameters for both arms. Repeated over several seeds because the test set is small (n=30).
Output: results/resnet18/ (roc_resnet18.png, metrics csv, predictions, weights)
"""
import copy, random, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

ROOT = Path(__file__).resolve().parent.parent
# usage: python resnet18_fixed_vs_canny.py [full|layer4|fc]
#   full   = fine-tune all layers (results/resnet18)
#   layer4 = freeze everything except the last residual stage + classifier (results/resnet18_layer4)
#   fc     = linear probe: only the new classifier is trained (results/resnet18_fc)
MODE = sys.argv[1] if len(sys.argv) > 1 else "full"
assert MODE in ("full", "layer4", "fc")
OUT = ROOT / "results" / ("resnet18" if MODE == "full" else f"resnet18_{MODE}"); OUT.mkdir(parents=True, exist_ok=True)
torch.hub.set_dir(str(ROOT / "weights" / "torch_hub"))
SEEDS = [42, 43, 44]
IMG, BATCH, EPOCHS, PATIENCE, LR, WD = 224, 16, 20, 5, 1e-4, 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(max(1, torch.get_num_threads()))

z = np.load(ROOT / "data" / "cohort" / "cohort_slices.npz", allow_pickle=True)
split, label = z["split"], z["label"].astype(int)
ARMS = {"Fixed slice 80": z["fixed"], "Canny selected": z["canny"]}
print("Device:", DEVICE, "| N:", {s: int((split == s).sum()) for s in ("train", "val", "test")})

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
def tf(aug):
    return transforms.Compose([transforms.ToPILImage(), transforms.Resize(IMG),
                               *([transforms.RandomRotation(5)] if aug else []),
                               transforms.ToTensor(), transforms.Lambda(lambda t: t.repeat(3, 1, 1)),
                               transforms.Normalize(MEAN, STD)])
class DS(Dataset):
    def __init__(s, imgs, y, aug): s.x, s.y, s.t = imgs, y, tf(aug)
    def __len__(s): return len(s.y)
    def __getitem__(s, i): return s.t(s.x[i]), torch.tensor(s.y[i])
def dl(imgs, part, aug, seed):
    m = split == part; g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(DS(imgs[m], label[m], aug), batch_size=BATCH, shuffle=aug, generator=g if aug else None)

def make_model():
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Sequential(nn.Dropout(.3), nn.Linear(m.fc.in_features, 2))
    if MODE != "full":
        trainable = ("fc.",) if MODE == "fc" else ("layer4.", "fc.")
        for n, p_ in m.named_parameters(): p_.requires_grad = n.startswith(trainable)
    return m.to(DEVICE)

def set_train(model):
    """train mode, but keep frozen BatchNorm layers in eval so their ImageNet statistics stay fixed"""
    model.train()
    if MODE != "full":
        keep = ("layer4",) if MODE == "layer4" else ()
        for n, mod in model.named_modules():
            if isinstance(mod, nn.BatchNorm2d) and not n.startswith(keep or ("\0",)): mod.eval()

def predict(model, loader):
    model.eval(); p, y = [], []
    with torch.no_grad():
        for x, t in loader:
            p += model(x.to(DEVICE)).softmax(1)[:, 1].cpu().tolist(); y += t.tolist()
    return np.array(y), np.array(p)

def run(arm, imgs, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = make_model(); opt = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad], lr=LR, weight_decay=WD)
    crit = nn.CrossEntropyLoss()
    tr, va, te = dl(imgs, "train", True, seed), dl(imgs, "val", False, seed), dl(imgs, "test", False, seed)
    best, state, wait = float("inf"), None, 0
    for ep in range(EPOCHS):
        set_train(model); tl = 0
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE); opt.zero_grad()
            l = crit(model(x), y); l.backward(); opt.step(); tl += l.item() * len(y)
        model.eval(); vl = 0
        with torch.no_grad():
            for x, y in va: vl += crit(model(x.to(DEVICE)), y.to(DEVICE)).item() * len(y)
        vl /= len(va.dataset); print(f"{arm} seed{seed} ep{ep+1:02d} train={tl/len(tr.dataset):.3f} val={vl:.3f}", flush=True)
        if vl < best - 1e-4: best, state, wait = vl, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE: break
    model.load_state_dict(state)
    vy, vp = predict(model, va); ty, tp = predict(model, te)
    return model, vy, vp, ty, tp

rows, curves, preds = [], {a: [] for a in ARMS}, {}
for arm, imgs in ARMS.items():
    for seed in SEEDS:
        model, vy, vp, ty, tp = run(arm, imgs, seed)
        pred = (tp >= .5).astype(int)
        rows.append({"arm": arm, "seed": seed, "val_auc": roc_auc_score(vy, vp), "test_auc": roc_auc_score(ty, tp),
                     "test_acc": float((pred == ty).mean()), "test_bal_acc": balanced_accuracy_score(ty, pred),
                     "test_f1": f1_score(ty, pred, zero_division=0)})
        curves[arm].append(roc_curve(ty, tp)[:2] + (rows[-1]["test_auc"],))
        preds[f"{arm}|{seed}"] = tp
        if seed == SEEDS[0]: torch.save(model.state_dict(), OUT / f"resnet18_{'fixed' if 'Fixed' in arm else 'canny'}_seed{seed}.pt")
        print(rows[-1], flush=True)
res = pd.DataFrame(rows); res.to_csv(OUT / "resnet18_metrics_per_seed.csv", index=False)
summ = res.groupby("arm").agg(["mean", "std"]).drop(columns="seed").round(3); summ.to_csv(OUT / "resnet18_metrics_summary.csv")
print(summ.to_string())
te_ids = z["subject_id"][split == "test"]
pd.DataFrame({"subject_id": te_ids, "true_label": ty, **{k: v for k, v in preds.items()}}).to_csv(OUT / "resnet18_test_predictions.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
for a, (arm, cs) in zip(ax, curves.items()):
    for i, (f, t, auc) in enumerate(cs):
        a.plot(f, t, lw=2.2 if i == 0 else 1, alpha=1 if i == 0 else .5, label=f"seed {SEEDS[i]}: AUC={auc:.3f}")
    aucs = [c[2] for c in cs]
    a.plot([0, 1], [0, 1], "--", c="gray"); a.set_title(f"ResNet18 ({MODE}) - {arm}\nmean AUC={np.mean(aucs):.3f} ± {np.std(aucs, ddof=1):.3f}")
    a.set_xlabel("False positive rate"); a.legend(loc="lower right")
ax[0].set_ylabel("True positive rate"); plt.tight_layout(); fig.savefig(OUT / "roc_resnet18.png", dpi=150)
print("Saved to", OUT)
