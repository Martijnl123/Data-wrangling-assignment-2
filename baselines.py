"""
The three competitors of Table 3, rebuilt with scikit-learn.

* LOF  - scikit-learn's LocalOutlierFactor with k = 10, as in the paper.
* ORCA-style k-NN score - ORCA itself (Bay and Schwabacher, 2003) is a C++
         program that is not available in Python, so this is not ORCA. It
         computes a k-NN score of the kind ORCA uses, exactly: the average
         distance to the k = 5 nearest neighbours (the paper uses ORCA's
         default k = 5), after scaling every attribute to [0, 1] (my
         assumption for ORCA's preprocessing). If ORCA's score, distance and
         scaling match these assumptions, its pruning should not change which
         points it reports. The running time says nothing about ORCA's: ORCA
         stops the neighbour search for a point as soon as that point cannot
         be among the top N, while this code finds all k neighbours of every
         point with a k-d tree. With n_reported=None every point keeps its
         score (full ranking); with n_reported=N only the top N keep their
         score, as in ORCA's output, and the rest share one tied lowest score.
* RF   - unsupervised Random Forests following Shi and Horvath (2006): a
         synthetic class drawn uniformly between each attribute's minimum and
         maximum, a 100-tree forest separating real from synthetic points,
         proximities between real points, and Breiman's outlier measure
         n / sum_j prox(i, j)^2.

Every function returns a score where higher means more anomalous.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors


def lof_scores(X, k=10):
    lof = LocalOutlierFactor(n_neighbors=k)
    lof.fit(X)
    return -lof.negative_outlier_factor_


def knn_average_distance(X, k=5):
    span = X.max(axis=0) - X.min(axis=0)
    span[span == 0] = 1.0
    Xs = (X - X.min(axis=0)) / span
    nn = NearestNeighbors(n_neighbors=k).fit(Xs)
    dist, _ = nn.kneighbors()              # query points are not their own neighbour
    return dist.mean(axis=1)


def keep_top(score, n_reported):
    """Keep the n_reported highest scores; tie everything else at the bottom."""
    if n_reported >= len(score):
        return score
    top = np.argpartition(-score, n_reported - 1)[:n_reported]
    out = np.full(len(score), score.min() - 1.0)
    out[top] = score[top]
    return out


def orca_style_scores(X, k=5, n_reported=None):
    """ORCA-style k-NN score, see the module docstring."""
    score = knn_average_distance(X, k)
    return score if n_reported is None else keep_top(score, n_reported)


def rf_scores(X, n_trees=100, random_state=None):
    rng = np.random.default_rng(random_state)
    synthetic = rng.uniform(X.min(axis=0), X.max(axis=0), size=X.shape)
    Z = np.vstack([X, synthetic])
    labels = np.r_[np.ones(X.shape[0]), np.zeros(X.shape[0])]
    seed = None if random_state is None else int(random_state)
    forest = RandomForestClassifier(n_estimators=n_trees, max_features="sqrt",
                                    random_state=seed, n_jobs=1).fit(Z, labels)
    leaves = forest.apply(X)                # leaf index of each real point, per tree
    n = X.shape[0]
    prox = np.zeros((n, n))
    for t in range(leaves.shape[1]):
        prox += leaves[:, t][:, None] == leaves[:, t][None, :]
    prox /= leaves.shape[1]
    np.fill_diagonal(prox, 0.0)
    return n / ((prox ** 2).sum(axis=1) + 1e-12)
