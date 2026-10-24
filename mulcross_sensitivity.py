"""
Two checks on Mulcross, the data set where masking matters most.

1. The OpenML file (data/mulcross.arff): our iForest with t = 100 and
   subsample sizes psi from 32 to 2048, ten runs each. Writes
   results/mulcross_psi.csv.
2. Mulcross regenerated from the paper's description alone, with different
   anomaly-cluster widths and two readings of "distance factor = 2", scored by
   our iForest (ten runs) and the exact ORCA score. Writes
   results/mulcross_sensitivity.csv.

    python mulcross_sensitivity.py
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import baselines
import datasets
from datasets import make_mulcross
from iforest import IsolationForest

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def iforest_aucs(X, y, psi=256, runs=10):
    return [roc_auc_score(y, IsolationForest(100, psi, "varying", s).fit(X).score(X))
            for s in range(runs)]


# 1. subsample size on the OpenML file
X, y = datasets.load("mulcross")
rows = []
for psi in [32, 64, 128, 256, 512, 2048]:
    aucs = iforest_aucs(X, y, psi)
    rows.append({"psi": psi, "iforest_auc_mean": np.mean(aucs), "iforest_auc_sd": np.std(aucs, ddof=1)})
    print(f"OpenML Mulcross, psi = {psi:5d}: iForest {np.mean(aucs):.3f} +/- {np.std(aucs, ddof=1):.3f}",
          flush=True)
pd.DataFrame(rows).to_csv(OUT / "mulcross_psi.csv", index=False, float_format="%.4f")

# 2. regenerated Mulcross under other generator settings
# (label, distance_factor as a shift per attribute, cluster_std)
SETTINGS = [
    ("shift 2 per attribute, cluster sd 0.15 (first regeneration)", 2.0, 0.15),
    ("shift 2 per attribute, cluster sd 0.3", 2.0, 0.3),
    ("shift 2 per attribute, cluster sd 0.5", 2.0, 0.5),
    ("shift 2 per attribute, cluster sd 1.0", 2.0, 1.0),
    ("centre distance 2 in total (shift 1 per attribute), cluster sd 0.15", 1.0, 0.15),
]
rows = []
for label, shift, sd in SETTINGS:
    X, y = make_mulcross(distance_factor=shift, cluster_std=sd)
    aucs = iforest_aucs(X, y)
    orca = roc_auc_score(y, baselines.orca_scores(X, 5, None))
    rows.append({"setting": label, "iforest_auc_mean": np.mean(aucs),
                 "iforest_auc_sd": np.std(aucs, ddof=1), "orca_auc": orca})
    print(f"{label:70s} iForest {np.mean(aucs):.3f} +/- {np.std(aucs, ddof=1):.3f}   ORCA {orca:.3f}",
          flush=True)
pd.DataFrame(rows).to_csv(OUT / "mulcross_sensitivity.csv", index=False, float_format="%.4f")
