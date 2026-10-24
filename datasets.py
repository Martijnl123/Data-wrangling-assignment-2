"""
Loaders for the twelve data sets of Table 2 in Liu, Ting and Zhou (2008).

Eleven are the ODDS benchmark versions (Rayana, 2016), which follow the paper's
definitions of the anomaly classes; run get_data.py first to download them into
data/. Mulcross is the synthetic file from OpenML (data set 40897), saved as
data/mulcross.arff. make_mulcross() regenerates a Mulcross-style set from the
paper's description alone; it is available as "mulcross_regenerated".

Every loader returns X (float64, n x d) and y (int, 1 = anomaly, 0 = normal).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io
import scipy.io.arff

DATA_DIR = Path(__file__).resolve().parent / "data"

# Order and descriptions as in Table 2 of the paper
DATASETS = ["http", "forestcover", "mulcross", "smtp", "shuttle", "mammography",
            "annthyroid", "satellite", "pima", "breastw", "arrhythmia", "ionosphere"]

DISPLAY_NAMES = {
    "http": "Http (KDDCUP99)", "forestcover": "ForestCover", "mulcross": "Mulcross",
    "smtp": "Smtp (KDDCUP99)", "shuttle": "Shuttle", "mammography": "Mammography",
    "annthyroid": "Annthyroid", "satellite": "Satellite", "pima": "Pima",
    "breastw": "Breastw", "arrhythmia": "Arrhythmia", "ionosphere": "Ionosphere",
}

# file name in data/ for each downloaded data set
FILES = {
    "http": "http.npz", "forestcover": "forestcover.npz", "smtp": "smtp.npz",
    "shuttle": "shuttle.npz", "mammography": "mammography.npz",
    "annthyroid": "annthyroid.npz", "satellite": "satellite.npz", "pima": "pima.npz",
    "breastw": "breastw.npz", "arrhythmia": "arrhythmia.mat",
    "ionosphere": "ionosphere.mat", "mulcross": "mulcross.arff",
}


def make_mulcross(n=262_144, d=4, contamination=0.10, distance_factor=2.0,
                  n_clusters=2, cluster_std=0.15, seed=2008):
    """Regenerate a Mulcross-style data set (Rocke and Woodruff, 1996).

    The paper only states the generator's basic setting: 10% contamination,
    distance factor 2 and 2 anomaly clusters, with the anomaly clusters denser
    than the normal points (Section 3 and Figure 4). The original generator
    could not be obtained, so the remaining choices follow Figure 4(a):

    * normal points ~ N(0, I_d);
    * each anomaly cluster ~ N(mu_k, cluster_std^2 I_d), with mu_k shifted by
      distance_factor standard deviations on every attribute; the two clusters
      sit on opposite sides of the normal cluster (mu_2 = -mu_1);
    * cluster_std = 0.15 is read off Figure 4(a), where the anomaly clusters
      are roughly 0.15 times as wide as the normal cluster.
    """
    rng = np.random.default_rng(seed)
    n_anom = int(contamination * n)                  # 26 214 for the defaults
    n_norm = n - n_anom
    X_norm = rng.standard_normal((n_norm, d))

    signs = rng.choice([-1.0, 1.0], size=(n_clusters, d))
    if n_clusters >= 2:
        signs[1] = -signs[0]
    centres = distance_factor * signs
    per_cluster = np.full(n_clusters, n_anom // n_clusters)
    per_cluster[: n_anom % n_clusters] += 1
    X_anom = np.vstack([c + cluster_std * rng.standard_normal((m, d))
                        for c, m in zip(centres, per_cluster)])

    X = np.vstack([X_norm, X_anom])
    y = np.r_[np.zeros(n_norm, dtype=int), np.ones(n_anom, dtype=int)]
    order = rng.permutation(n)
    return X[order], y[order]


def load(name, data_dir=DATA_DIR):
    """Return X, y for one of the names in DATASETS."""
    if name == "mulcross_regenerated":
        return make_mulcross()
    path = Path(data_dir) / FILES[name]
    if not path.exists():
        hint = ("download the ARFF from https://www.openml.org/d/40897"
                if name == "mulcross" else "run get_data.py first")
        raise FileNotFoundError(f"{path} not found; {hint}")
    if path.suffix == ".arff":
        data, _ = scipy.io.arff.loadarff(path)
        X = np.column_stack([data[f"V{i}"] for i in range(1, 5)])
        y = np.array([v == b"Anomaly" for v in data["Target"]])
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as z:
            X, y = z["X"], z["y"]
    else:
        m = scipy.io.loadmat(path)
        X, y = m["X"], m["y"]
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel().astype(int)
    if name == "ionosphere":
        # ODDS keeps 33 attributes; the first is binary. The paper removes binary
        # attributes, which leaves the 32 used in Table 2.
        X = X[:, 1:]
    return X, y


def describe(name, X, y):
    return {"dataset": name, "n": X.shape[0], "d": X.shape[1],
            "anomalies": int(y.sum()), "anomaly_pct": round(100 * y.mean(), 2)}
