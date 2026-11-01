from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
R = Path(__file__).resolve().parent.parent / "results" / "single_split_5slice"
BB = {"cnn": "Small CNN", "resnet": "ResNet18"}; EX = ["2.5D fixed 78-82", "Late fusion (mean pool) fixed 78-82", "2.5D Canny c±2", "Late fusion (mean pool) Canny c±2"]
rows, CM = [], {}
for b in BB:
    d = pd.read_csv(R / f"{b}_per_seed.csv"); pr = pd.read_csv(R / f"{b}_test_predictions.csv")
    for e in EX:
        g = d[d.exp == e]; p = pr[pr.exp == e].copy(); r = dict(backbone=b, exp=e)
        for k in ("test_auc", "acc_05", "acc_youden", "sens_05", "spec_05", "sens_youden", "spec_youden", "thr_youden"): r[k] = g[k].mean(); r[k + "_sd"] = g[k].std()
        for tag, thr in (("05", np.full(len(p), .5)), ("youden", p.thr_youden.to_numpy())):          # seed-ensemble: margin averaged over the 3 seeds
            p["m"] = p.prob.to_numpy() - thr; o = p.groupby("subject_id").agg(true=("true", "first"), m=("m", "mean")); pred = (o.m >= 0).astype(int)
            tn = int(((pred == 0) & (o.true == 0)).sum()); fp = int(((pred == 1) & (o.true == 0)).sum()); fn = int(((pred == 0) & (o.true == 1)).sum()); tp = int(((pred == 1) & (o.true == 1)).sum())
            r.update({f"ens_acc_{tag}": (tn + tp) / len(o), f"ens_sens_{tag}": tp / (tp + fn), f"ens_spec_{tag}": tn / (tn + fp)}); CM[(b, e, tag)] = (tn, fp, fn, tp)
        o = p.groupby("subject_id").agg(true=("true", "first"), pr=("prob", "mean")); r["ens_auc"] = roc_auc_score(o.true, o.pr); r["n_test"] = len(o); rows.append(r)
T = pd.DataFrame(rows); T.to_csv(R / "single_split_summary.csv", index=False); pd.set_option("display.width", 250)
print(T[["backbone", "exp", "test_auc", "test_auc_sd", "acc_05", "acc_05_sd", "acc_youden", "acc_youden_sd", "sens_youden", "spec_youden", "thr_youden", "thr_youden_sd", "ens_acc_05", "ens_acc_youden", "n_test"]].round(3).to_string())
fig, ax = plt.subplots(2, 4, figsize=(15.5, 8))
for i, b in enumerate(BB):
    for j, e in enumerate(EX):
        tn, fp, fn, tp = CM[(b, e, "youden")]; m = np.array([[tn, fp], [fn, tp]]); r = T[(T.backbone == b) & (T.exp == e)].iloc[0]; a = ax[i, j]; a.imshow(m / m.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for (p_, q_), v in np.ndenumerate(m): a.text(q_, p_, f"{v}\n({v / m[p_].sum():.0%})", ha="center", va="center", fontsize=12, color="white" if m[p_, q_] / m[p_].sum() > .55 else "#1f2933")
        a.set_xticks([0, 1]); a.set_xticklabels(["CDR=0", "CDR>0"]); a.set_yticks([0, 1]); a.set_yticklabels(["CDR=0", "CDR>0"]); a.set_xlabel("Predicted")
        a.set_title(f"{BB[b]} · {e}\nYouden: acc {r.ens_acc_youden:.3f} (seeds {r.acc_youden:.2f}±{r.acc_youden_sd:.2f})\n0.5: acc {r.ens_acc_05:.3f} (seeds {r.acc_05:.2f}±{r.acc_05_sd:.2f})", fontsize=8.3)
        if j == 0: a.set_ylabel("Actual")
fig.suptitle(f"Single split, no cross-validation (test n = {int(T.n_test[0])}): confusion matrices at the Youden threshold chosen on the validation set (3-seed ensemble)", x=.01, ha="left", fontsize=11.5, weight="bold")
plt.tight_layout(); fig.savefig(R / "single_split_confusion.png", dpi=150)
