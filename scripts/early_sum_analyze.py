"""Early fusion SUM vs the earlier fusion methods (paired by repeat/fold; same folds and seeds). Output: results/early_sum/summary.csv, paired.csv"""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
R = __import__("pathlib").Path(__file__).resolve().parent.parent / "results"; rng = np.random.default_rng(0)
NAMES = {"M0: early fusion SUM": "early SUM", "M1: 2.5D": "early CONCAT (2.5D)", "M2: late fusion": "late CONCAT", "M2-mean: late fusion (mean pool)": "late MEAN"}
S, P = [], []
for bb in ("resnet", "cnn"):
    F = pd.concat([pd.read_csv(R / d / bb / "fold_results.csv") for d in ("early_sum", "five_slice", "meanpool")]); O = pd.concat([pd.read_csv(R / d / bb / "oof_predictions.csv") for d in ("early_sum", "five_slice", "meanpool")])
    for sl, tag in (("fixed 78-82", "fixed"), ("Canny c±2", "Canny")):
        ex = {l: next(e for e in F.exp.unique() if e.startswith(k) and e.endswith(sl) and (k != "M2: late fusion" or "mean" not in e)) for k, l in NAMES.items()}
        for l, e in ex.items():
            def pooled(idx=None):
                a = []
                for r, g in O[O.exp == e].groupby("repeat"):
                    g = g.sort_values("subject_id").reset_index(drop=True); g = g if idx is None else g.iloc[idx]; a.append(roc_auc_score(g.true, g.prob))
                return np.mean(a)
            b = [pooled(rng.integers(0, 236, 236)) for _ in range(300)]
            S.append(dict(backbone=bb, slices=tag, fusion=l, mean_fold_auc=F[F.exp == e].auc.mean(), pooled_auc=pooled(), ci_lo=np.percentile(b, 2.5), ci_hi=np.percentile(b, 97.5), acc=F[F.exp == e].acc.mean()))
        for l in ("early CONCAT (2.5D)", "late CONCAT", "late MEAN"):
            x = F[F.exp == ex["early SUM"]].set_index(["repeat", "fold"]).auc; y = F[F.exp == ex[l]].set_index(["repeat", "fold"]).auc; d = (x - y.loc[x.index]).to_numpy()
            se = d.std(ddof=1) / np.sqrt(len(d)); t = stats.t.ppf(.975, len(d) - 1)
            P.append(dict(backbone=bb, slices=tag, comparison=f"early SUM - {l}", mean_diff=d.mean(), ci_lo=d.mean() - t * se, ci_hi=d.mean() + t * se, wilcoxon_p=stats.wilcoxon(d).pvalue, n=len(d)))
S, P = pd.DataFrame(S).round(3), pd.DataFrame(P).round(3); S.to_csv(R / "early_sum" / "summary.csv", index=False); P.to_csv(R / "early_sum" / "paired.csv", index=False)
print(S.to_string(index=False)); print(P.to_string(index=False))
