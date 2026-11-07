"""Tables and figures for the nested tuning run (results/oasis1_10slice_tuned), recomputed from the saved predictions.
Compares the tuned models (nested selection, 3-seed ensemble) with the untuned main notebook on the same outer folds."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "oasis1_10slice_tuned"; FIG = T / "figures"; FIG.mkdir(parents=True, exist_ok=True)
MAIN = ROOT / "results" / "oasis1_10slice"
N_BOOT, RNG = 2000, np.random.default_rng(0)
NAMES = {"cnn": "CNN Scratch", "resnet": "ResNet18"}


def boot_ci(y, p):
    v = []
    for _ in range(N_BOOT):
        i = RNG.integers(0, len(y), len(y))
        if len(set(y[i])) == 2: v.append(roc_auc_score(y[i], p[i]))
    return np.percentile(v, [2.5, 97.5])


def paired_boot(y, p1, p2):
    v = []
    for _ in range(N_BOOT):
        i = RNG.integers(0, len(y), len(y))
        if len(set(y[i])) == 2: v.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    return np.percentile(v, [2.5, 97.5])


def fold_aucs(df):
    return df.groupby("fold").apply(lambda g: roc_auc_score(g.label, g.prob))


# ------------------------------------------------------------------ load
inner = pd.read_csv(T / "inner_scores.csv")
final = pd.read_csv(T / "final_predictions.csv")
ens = final.groupby(["backbone", "selection", "fold", "subject_id", "label"], as_index=False).agg(prob=("prob", "mean"), config=("config", "first"), n_seeds=("seed", "nunique"))
single = final[final.seed == 33000 + final.fold]                       # the seed the main notebook used
main = pd.read_csv(MAIN / "predictions.csv")
main["backbone_key"] = main.backbone; main["sel"] = main.selection

# ------------------------------------------------------------------ chosen configurations
im = inner.groupby(["backbone", "selection", "fold", "config"]).auc.mean().reset_index()
chosen = ens.groupby(["backbone", "selection", "fold"]).config.first().reset_index()
chosen = chosen.merge(im, on=["backbone", "selection", "fold", "config"]).rename(columns={"auc": "inner_auc"})
chosen.to_csv(T / "chosen_configs.csv", index=False)
inner_table = im.pivot_table(index=["backbone", "config"], columns="selection", values="auc", aggfunc="mean").round(3)
inner_table.to_csv(T / "inner_scores_mean.csv")

# ------------------------------------------------------------------ main comparison table
rows, fold_rows = [], []
for bb in ["resnet", "cnn"]:
    for sel in ["Fixed", "Canny"]:
        e = ens[(ens.backbone == bb) & (ens.selection == sel)]
        if not len(e): continue
        s = single[(single.backbone == bb) & (single.selection == sel)]
        m = main[(main.backbone == bb) & (main.selection == sel)]
        fe, fs_, fm = fold_aucs(e), fold_aucs(s), fold_aucs(m)
        ci_e, ci_m = boot_ci(e.label.values, e.prob.values), boot_ci(m.label.values, m.prob.values)
        mm = m.merge(e, on="subject_id", suffixes=("_m", "_e"))
        d = paired_boot(mm.label_m.values, mm.prob_e.values, mm.prob_m.values)
        diff = (fe - fm).values
        t_lo, t_hi = stats.t.interval(.95, len(diff) - 1, diff.mean(), stats.sem(diff)) if len(diff) > 1 else (np.nan, np.nan)
        y, p = e.label.values, e.prob.values
        tn, fp, fn, tp = confusion_matrix(y, p >= .5).ravel()
        rows.append({"model": f"{NAMES[bb]} | {sel}",
                     "untuned (A) mean fold AUC": f"{fm.mean():.3f} (sd {fm.std():.3f})",
                     "untuned (B) pooled AUC [95% CI]": f"{roc_auc_score(m.label, m.prob):.3f} [{ci_m[0]:.3f}, {ci_m[1]:.3f}]",
                     "tuned single seed (A)": f"{fs_.mean():.3f}" if len(s) else "",
                     "tuned ensemble (A) mean fold AUC": f"{fe.mean():.3f} (sd {fe.std():.3f})",
                     "tuned ensemble (B) pooled AUC [95% CI]": f"{roc_auc_score(y, p):.3f} [{ci_e[0]:.3f}, {ci_e[1]:.3f}]",
                     "tuned - untuned, mean fold diff [t 95% CI]": f"{diff.mean():+.3f} [{t_lo:+.3f}, {t_hi:+.3f}]",
                     "tuned - untuned, pooled AUC diff [paired bootstrap 95% CI]": f"{roc_auc_score(mm.label_m, mm.prob_e) - roc_auc_score(mm.label_m, mm.prob_m):+.3f} [{d[0]:+.3f}, {d[1]:+.3f}]",
                     "tuned acc / sens / spec at 0.5": f"{(tp + tn) / len(y):.3f} / {tp / (tp + fn):.3f} / {tn / (tn + fp):.3f}",
                     "configs chosen (folds 0-4)": ", ".join(chosen[(chosen.backbone == bb) & (chosen.selection == sel)].sort_values("fold").config)})
        for f in fe.index: fold_rows.append(dict(model=f"{NAMES[bb]} | {sel}", fold=f, untuned_auc=fm[f], tuned_single_seed_auc=fs_.get(f, np.nan), tuned_ensemble_auc=fe[f]))
summary = pd.DataFrame(rows); summary.to_csv(T / "tuned_vs_untuned.csv", index=False)
pd.DataFrame(fold_rows).to_csv(T / "fold_metrics_tuned.csv", index=False)

# ------------------------------------------------------------------ Fixed vs Canny after tuning
pc = []
for bb in ["resnet", "cnn"]:
    a, b = ens[(ens.backbone == bb) & (ens.selection == "Canny")], ens[(ens.backbone == bb) & (ens.selection == "Fixed")]
    if not (len(a) and len(b)): continue
    fa, fb = fold_aucs(a), fold_aucs(b); diff = (fa - fb).values
    lo, hi = stats.t.interval(.95, len(diff) - 1, diff.mean(), stats.sem(diff))
    mm = a.merge(b, on="subject_id", suffixes=("_c", "_f"))
    d = paired_boot(mm.label_c.values, mm.prob_c.values, mm.prob_f.values)
    pc.append(dict(comparison=f"{NAMES[bb]} tuned: Canny - Fixed", mean_fold_diff=round(diff.mean(), 3), fold_ci=f"[{lo:+.3f}, {hi:+.3f}]",
                   wilcoxon_p=round(stats.wilcoxon(diff).pvalue, 3), folds_canny_better=int((diff > 0).sum()),
                   pooled_diff=round(roc_auc_score(mm.label_c, mm.prob_c) - roc_auc_score(mm.label_c, mm.prob_f), 3), bootstrap_ci=f"[{d[0]:+.3f}, {d[1]:+.3f}]"))
pd.DataFrame(pc).to_csv(T / "paired_fixed_vs_canny_tuned.csv", index=False)

# ------------------------------------------------------------------ transparency: every configuration on the outer folds
allc_path = T / "all_config_predictions.csv"
if allc_path.exists():
    allc = pd.read_csv(allc_path)
    t = allc.groupby(["backbone", "selection", "config"]).apply(lambda g: pd.Series(dict(
        n_folds=g.fold.nunique(), mean_fold_auc=g.groupby("fold").apply(lambda h: roc_auc_score(h.label, h.prob)).mean(),
        pooled_auc=roc_auc_score(g.label, g.prob)))).reset_index()
    t.round(3).to_csv(T / "all_configs_outer.csv", index=False)

# ------------------------------------------------------------------ figures
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, bb in zip(axes, ["resnet", "cnn"]):
    for sel, c in [("Fixed", "tab:blue"), ("Canny", "tab:orange")]:
        e = ens[(ens.backbone == bb) & (ens.selection == sel)]; m = main[(main.backbone == bb) & (main.selection == sel)]
        if len(e):
            f, tp_, _ = roc_curve(e.label, e.prob); ax.plot(f, tp_, color=c, lw=2, label=f"{sel} tuned (AUC {roc_auc_score(e.label, e.prob):.3f})")
        if len(m):
            f, tp_, _ = roc_curve(m.label, m.prob); ax.plot(f, tp_, color=c, lw=1.2, ls="--", label=f"{sel} untuned (AUC {roc_auc_score(m.label, m.prob):.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=.8); ax.set_title(f"{NAMES[bb]}: pooled out-of-fold ROC"); ax.set_xlabel("1 - specificity"); ax.set_ylabel("sensitivity"); ax.legend(fontsize=8.5)
plt.tight_layout(); plt.savefig(FIG / "roc_tuned_vs_untuned.png", dpi=140); plt.close()

fr = pd.DataFrame(fold_rows)
if len(fr):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    models = list(dict.fromkeys(fr.model)); x = np.arange(len(models))
    for k, (col, lab, c) in enumerate([("untuned_auc", "untuned (main notebook)", "#9aa9c2"), ("tuned_ensemble_auc", "tuned, 3-seed ensemble", "#2b5ea7")]):
        mean = [fr[fr.model == m][col].mean() for m in models]; sd = [fr[fr.model == m][col].std() for m in models]
        ax.bar(x + (k - .5) * .38, mean, .38, yerr=sd, capsize=4, color=c, label=lab)
        for i, m in enumerate(models): ax.scatter(np.full(5, x[i] + (k - .5) * .38), fr[fr.model == m][col], color="k", s=10, zorder=3)
    ax.axhline(.5, color="gray", lw=.8, ls=":"); ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylabel("test-fold ROC-AUC"); ax.set_ylim(.3, 1)
    ax.set_title("Mean fold AUC (dots = the 5 outer folds; same folds and subjects)"); ax.legend()
    plt.tight_layout(); plt.savefig(FIG / "fold_auc_tuned_vs_untuned.png", dpi=140); plt.close()

fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))
for ax, (bb, sel) in zip(axes, [("cnn", "Fixed"), ("cnn", "Canny"), ("resnet", "Fixed"), ("resnet", "Canny")]):
    e = ens[(ens.backbone == bb) & (ens.selection == sel)]
    if not len(e): ax.axis("off"); continue
    cm = confusion_matrix(e.label, e.prob >= .5); ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm): ax.text(j, i, v, ha="center", va="center", color="white" if v > cm.max() / 2 else "black", fontsize=13)
    tn, fp, fn, tp = cm.ravel()
    ax.set_title(f"{NAMES[bb]} | {sel} (tuned)\nacc {(tp + tn) / cm.sum():.3f} sens {tp / (tp + fn):.2f} spec {tn / (tn + fp):.2f}", fontsize=10)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["CDR=0", "CDR>0"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["CDR=0", "CDR>0"]); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.suptitle("Tuned models: confusion matrices (3-seed ensemble probabilities, threshold 0.5, 198 subjects)")
plt.tight_layout(); plt.savefig(FIG / "confusion_tuned.png", dpi=140); plt.close()

print(summary.T.to_string()); print(pd.DataFrame(pc).to_string()); print(chosen.to_string())
