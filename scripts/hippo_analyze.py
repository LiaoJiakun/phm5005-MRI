"""Hippocampus-level slices vs the earlier fixed slices. Paired by (repeat, fold): same subjects in every test fold, same seeds.
Reports mean fold AUC, pooled out-of-fold AUC (mean over the 2 repeats) with a subject-level bootstrap 95% CI, and paired fold differences
(t-interval with df=9 and Wilcoxon). Folds overlap in training data, so intervals are optimistic; treat as exploratory.
Output: results/hippo/hippo_summary.csv, hippo_paired.csv"""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
R = __import__("pathlib").Path(__file__).resolve().parent.parent / "results"
SRC = {"final_cv/resnet": None, "five_slice/resnet": None, "hippo/resnet/slice": None, "hippo/resnet/dual": None, "hippo/resnet/slice_59_63": None}
F, O = [], []
for k in SRC:
    if (R / k / "fold_results.csv").exists(): F.append(pd.read_csv(R / k / "fold_results.csv")); O.append(pd.read_csv(R / k / "oof_predictions.csv"))
F, O = pd.concat(F), pd.concat(O)
PAIRS = [("H1: single axial z62", "S1: single slice fixed 80"), ("H5: single axial z61", "S1: single slice fixed 80"),
         ("H3: late fusion z60-64", "M2: late fusion fixed 78-82"), ("H4: late fusion z59-63", "M2: late fusion fixed 78-82"),
         ("H2: axial z62 + coronal 105", "C: fixed axial+coronal")]
rng = np.random.default_rng(0)
def pooled(exp, idx=None):
    a = []
    for rep, g in O[O.exp == exp].groupby("repeat"):
        g = g.sort_values("subject_id").reset_index(drop=True)
        if idx is not None: g = g.iloc[idx]
        a.append(roc_auc_score(g.true, g.prob))
    return np.mean(a)
rows = []
for e in sorted(set(F.exp)):
    n = O[(O.exp == e) & (O.repeat == 0)].shape[0]; boot = [pooled(e, rng.integers(0, n, n)) for _ in range(500)]
    rows.append(dict(exp=e, mean_fold_auc=F[F.exp == e].auc.mean(), pooled_auc=pooled(e), ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5), mean_acc=F[F.exp == e].acc.mean()))
S = pd.DataFrame(rows).round(3); S.to_csv(R / "hippo" / "hippo_summary.csv", index=False); print(S.to_string(index=False))
P = []
for a, b in PAIRS:
    if a not in set(F.exp): continue
    x = F[F.exp == a].set_index(["repeat", "fold"]).auc; y = F[F.exp == b].set_index(["repeat", "fold"]).auc; d = (x - y.loc[x.index]).to_numpy()
    se = d.std(ddof=1) / np.sqrt(len(d)); t = stats.t.ppf(.975, len(d) - 1)
    P.append(dict(new=a, reference=b, mean_diff=d.mean(), ci_lo=d.mean() - t * se, ci_hi=d.mean() + t * se, wilcoxon_p=stats.wilcoxon(d).pvalue, folds_better=int((d > 0).sum()), n=len(d)))
P = pd.DataFrame(P).round(3); P.to_csv(R / "hippo" / "hippo_paired.csv", index=False); print(P.to_string(index=False))
