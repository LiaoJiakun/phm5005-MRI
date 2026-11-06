"""Method flow diagram and model-structure figure for the OASIS-1 10-slice experiment.
The model figure is drawn from the real modules: layer names and tensor shapes are recorded with forward hooks on a dummy input."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import torch
import torch.nn as nn
import torchvision.models as models

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "oasis1_10slice" / "figures"; FIG.mkdir(parents=True, exist_ok=True)


def box(ax, x, y, w, h, text, fc="#eef3fb", ec="#3b5b92", fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.2))


# ---------------------------------------------------------------- 1. method flow
fig, ax = plt.subplots(figsize=(15, 4.6)); ax.set_xlim(0, 15); ax.set_ylim(0, 4.6); ax.axis("off")
steps = [
    ("OASIS-1 cross-sectional\nT88 masked volumes\n198 subjects, age >= 60\nCDR=0: 98 | CDR>0: 100", "#f4f4f4"),
    ("RAS orientation\naxial slice = v[:, :, z].T\n1/99 percentile scaling\npad to square, 128 x 128", "#f4f4f4"),
    ("Slice selection (no labels)\nFixed: z = 75..84\nCanny: centre c in 50..110\n(max edges, ties -> lowest z)\nslices c-5..c+4", "#fff4e0"),
    ("Subject-level stratified\n5-fold CV (seed 31000)\ninner 15% validation\n(early stopping only)", "#eaf6ea"),
    ("4 models, same folds\nCNN Scratch (BN) | ResNet18\nx Fixed | Canny\nshared encoder per slice\nfeature concatenation", "#eef3fb"),
    ("Subject-level predictions\nthreshold 0.5 (fixed)\nAUC (A) mean fold\n(B) pooled OOF (C) subject avg\npaired Fixed vs Canny", "#f9ecf4"),
]
w, h, y = 2.15, 1.8, 2.3
for i, (t, c) in enumerate(steps):
    x = 0.2 + i * 2.45
    box(ax, x, y, w, h, t, fc=c, fs=9)
    if i: arrow(ax, x - 0.3, y + h / 2, x, y + h / 2)
box(ax, 0.2 + 4 * 2.45, 0.3, 2.15 + 2.45, 0.9,
    "Grad-CAM on unseen test subjects (ResNet18, fold 0)\nhook order checked against slice order; one colour scale per subject", fc="#fdfbe6", fs=9)
arrow(ax, 0.2 + 4 * 2.45 + w / 2, y, 0.2 + 4 * 2.45 + w / 2, 1.25)
ax.set_title("OASIS-1 10-slice experiment: method flow", fontsize=13, weight="bold")
plt.tight_layout(); plt.savefig(FIG / "method_flow.png", dpi=150); plt.close()


# ---------------------------------------------------------------- 2. real model structure
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        def block(i, o): return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU()]
        self.features = nn.Sequential(*block(1, 32), *block(32, 32), nn.MaxPool2d(2, 2), *block(32, 64), *block(64, 64), nn.MaxPool2d(2, 2),
                                      *block(64, 128), *block(128, 128))
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1))
    def forward(self, x): return self.pool(self.features(x))


def shapes(encoder, x, names):
    out = {}
    hooks = [dict(encoder.named_modules())[n].register_forward_hook(lambda m, i, o, n=n: out.__setitem__(n, tuple(o.shape[1:]))) for n in names]
    with torch.no_grad(): encoder.eval()(x)
    for hk in hooks: hk.remove()
    return out


cnn = SimpleCNN()
cnn_names = ["features.2", "features.5", "features.6", "features.9", "features.12", "features.13", "features.16", "features.19", "pool"]
cnn_s = shapes(cnn, torch.zeros(1, 1, 128, 128), cnn_names)
res = models.resnet18(weights=None); res.fc = nn.Identity()
res_names = ["relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool"]
res_s = shapes(res, torch.zeros(1, 3, 128, 128), res_names)
n_cnn = sum(p.numel() for p in cnn.parameters()) + 1280 * 2 + 2
n_res = sum(p.numel() for p in res.parameters()) + 5120 * 2 + 2

cnn_rows = [("input slice", "1x128x128"), ("Conv3x3-BN-ReLU x2", "x".join(map(str, cnn_s["features.5"]))), ("MaxPool 2x2", "x".join(map(str, cnn_s["features.6"]))),
            ("Conv3x3-BN-ReLU x2", "x".join(map(str, cnn_s["features.12"]))), ("MaxPool 2x2", "x".join(map(str, cnn_s["features.13"]))),
            ("Conv3x3-BN-ReLU x2", "x".join(map(str, cnn_s["features.19"]))), ("global avg pool", str(cnn_s["pool"][0]))]
res_rows = [("input slice (grey x3)", "3x128x128"), ("conv1 7x7/2-BN-ReLU", "x".join(map(str, res_s["relu"]))), ("maxpool 3x3/2", "x".join(map(str, res_s["maxpool"]))),
            ("layer1 (2 basic blocks)", "x".join(map(str, res_s["layer1"]))), ("layer2", "x".join(map(str, res_s["layer2"]))),
            ("layer3", "x".join(map(str, res_s["layer3"]))), ("layer4", "x".join(map(str, res_s["layer4"]))), ("global avg pool", "512")]

fig, axes = plt.subplots(1, 2, figsize=(16, 8.5))
for ax, title, rows, d, n_par, colour in [(axes[0], "CNN Scratch (course SimpleCNN + BatchNorm), random init", cnn_rows, 128, n_cnn, "#eaf6ea"),
                                          (axes[1], "ResNet18, ImageNet weights, all layers fine-tuned", res_rows, 512, n_res, "#eef3fb")]:
    ax.set_xlim(0, 10); ax.set_ylim(0, 11); ax.axis("off"); ax.set_title(title, fontsize=11.5, weight="bold")
    # 10 slices -> shared encoder
    for j in range(10):
        box(ax, 0.15 + j * 0.62, 10.0, 0.55, 0.6, f"z{j}", fc="#ffffff", fs=7)
    ax.text(5, 9.65, "10 slices of one subject, each passed separately through ONE shared encoder", ha="center", fontsize=8.5)
    y = 9.2
    for name, shp in rows:
        y -= 0.78
        box(ax, 2.0, y, 6.0, 0.6, f"{name}   ->   {shp}", fc=colour, fs=8.5)
    y -= 0.95
    box(ax, 1.0, y, 8.0, 0.7, f"concatenate 10 x {d} = {10 * d} features (slice order kept)", fc="#fff4e0", fs=9, bold=True)
    y -= 0.95
    box(ax, 1.0, y, 8.0, 0.7, f"Dropout(0.3)  ->  Linear({10 * d}, 2)  ->  softmax  ->  P(CDR>0)", fc="#f9ecf4", fs=9, bold=True)
    ax.text(5, y - 0.45, f"total parameters: {n_par:,}", ha="center", fontsize=9)
plt.suptitle("Model structure (shapes recorded from the actual PyTorch modules, input 128 x 128)", fontsize=12.5)
plt.tight_layout(); plt.savefig(FIG / "model_structure.png", dpi=150); plt.close()
print("params CNN", n_cnn, "ResNet", n_res); print("saved", FIG / "method_flow.png", FIG / "model_structure.png")
