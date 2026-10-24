"""
Two extra experiments behind statements in the report that the main
experiment does not cover.

1. Number of trees on Arrhythmia, the data set where my iForest and
   scikit-learn differ most at t = 100: both implementations with
   t = 25, 100, 400 and 1600 trees (psi = 256), seeds 0-9.
   Writes results/extra_checks_trees.csv.
2. Exact harmonic numbers against the paper's estimate H(i) = ln(i) + 0.5772
   in c(n), on all twelve data sets: each forest (t = 100, psi = 256,
   seeds 0-9) is grown once and scored both ways, so only c(n) differs.
   Writes results/extra_checks_harmonic.csv.

    python extra_checks.py
"""
import single_thread  # sets the thread variables before NumPy loads

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest as SkIsolationForest
from sklearn.metrics import roc_auc_score

import datasets
from iforest import IsolationForest

single_thread.limit()
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
RUNS = 10
SUBSAMPLE = 256


def sd(values):
    return float(np.std(values, ddof=1))


# 1. number of trees on Arrhythmia, my iForest against scikit-learn
X, y = datasets.load("arrhythmia")
rows = []
for t in [25, 100, 400, 1600]:
    ours = [roc_auc_score(y, IsolationForest(n_trees=t, subsample_size=SUBSAMPLE,
                                             random_state=s).fit(X).score(X))
            for s in range(RUNS)]
    sk = []
    for s in range(RUNS):
        model = SkIsolationForest(n_estimators=t, max_samples=SUBSAMPLE, max_features=1.0,
                                  bootstrap=False, random_state=s, n_jobs=1).fit(X)
        sk.append(roc_auc_score(y, -model.score_samples(X)))
    rows.append({"trees": t, "runs": RUNS,
                 "auc_ours_mean": np.mean(ours), "auc_ours_sd": sd(ours),
                 "auc_sklearn_mean": np.mean(sk), "auc_sklearn_sd": sd(sk),
                 "sklearn_minus_ours": np.mean(sk) - np.mean(ours)})
    print(f"Arrhythmia, t = {t:4d}: mine {np.mean(ours):.4f} +/- {sd(ours):.4f}, "
          f"scikit-learn {np.mean(sk):.4f} +/- {sd(sk):.4f}, "
          f"difference {np.mean(sk) - np.mean(ours):+.4f}", flush=True)
pd.DataFrame(rows).to_csv(OUT / "extra_checks_trees.csv", index=False, float_format="%.4f")

# 2. exact harmonic numbers against the paper's estimate, on the same forests
rows = []
for name in datasets.DATASETS:
    X, y = datasets.load(name)
    exact, approx = [], []
    for s in range(RUNS):
        model = IsolationForest(n_trees=100, subsample_size=SUBSAMPLE, random_state=s).fit(X)
        exact.append(roc_auc_score(y, model.score(X)))
        model.harmonic = "approx"          # the same trees; only c(n) changes
        approx.append(roc_auc_score(y, model.score(X)))
    diff = np.array(approx) - np.array(exact)
    rows.append({"dataset": name, "runs": RUNS,
                 "auc_exact_mean": np.mean(exact), "auc_approx_mean": np.mean(approx),
                 "approx_minus_exact_mean": diff.mean(),
                 "approx_minus_exact_max_abs": np.abs(diff).max()})
    print(f"{name:12s} exact {np.mean(exact):.4f}, paper's estimate {np.mean(approx):.4f}, "
          f"mean difference {diff.mean():+.5f}, largest single-run difference "
          f"{np.abs(diff).max():.5f}", flush=True)
pd.DataFrame(rows).to_csv(OUT / "extra_checks_harmonic.csv", index=False, float_format="%.5f")
