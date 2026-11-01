"""Sanity audit of the final result: is the high mixed-CV AUC coming from real signal, or from mixing two datasets?
ResNet18 (full fine-tune), experiment A (fixed axial 80) and C (fixed axial+coronal), seed-controlled:
  within-OASIS-1 CV | within-OASIS-2 CV | train OASIS-1 -> test OASIS-2 | train OASIS-2 -> test OASIS-1"""
import sys, numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import oasis_lib as L
torch.set_num_threads(10)
d = L.load_data(); y, ds, ids = d["label"], d["dataset"], d["subject_id"]; out = []
def within(name, which, views):
    idx = np.where(ds == which)[0]; oof = np.zeros(len(y))
    for k, (trv, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=555).split(idx, y[idx])):
        trv, te = idx[trv], idx[te]; tr, va = train_test_split(trv, test_size=.15, stratify=y[trv], random_state=556 + k)
        m, _ = L.train_model(d, views, "resnet", tr, va, seed=1234 + k); p, _ = L.predict_proba(m, L.make_loader(d, views, te, "resnet", False)); oof[te] = p
        print(f"  {name} within-{which} fold{k} AUC={roc_auc_score(y[te], oof[te]):.3f}", flush=True)
    return roc_auc_score(y[idx], oof[idx])
def external(name, src, dst, views):
    s, t = np.where(ds == src)[0], np.where(ds == dst)[0]; tr, va = train_test_split(s, test_size=.15, stratify=y[s], random_state=557)
    m, _ = L.train_model(d, views, "resnet", tr, va, seed=1234); p, _ = L.predict_proba(m, L.make_loader(d, views, t, "resnet", False)); return roc_auc_score(y[t], p)
for name in ("A: fixed axial 80", "C: fixed axial+coronal"):
    v = d["experiments"][name]
    for which in ("OASIS1", "OASIS2"):
        if name.startswith("C") and False: continue
        out.append(dict(exp=name, setting=f"within {which} (5-fold CV)", auc=within(name, which, v)))
    out.append(dict(exp=name, setting="train OASIS1 -> test OASIS2", auc=external(name, "OASIS1", "OASIS2", v)))
    out.append(dict(exp=name, setting="train OASIS2 -> test OASIS1", auc=external(name, "OASIS2", "OASIS1", v)))
    print(pd.DataFrame(out).round(3).to_string(), flush=True)
pd.DataFrame(out).to_csv(L.ROOT / "results" / "final_cv" / "dataset_audit.csv", index=False); print("AUDIT DONE")
