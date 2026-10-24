"""
Isolation Forest written from the pseudocode in

    F. T. Liu, K. M. Ting and Z.-H. Zhou (2008). Isolation Forest.
    Proceedings of the 8th IEEE International Conference on Data Mining, 413-422.

Mapping to the paper:
    Equation 1   -> average_path_length()      c(n)
    Algorithm 1  -> IsolationForest.fit()      iForest(X, t, psi)
    Algorithm 2  -> _build_itree()             iTree(X, e, l)
    Algorithm 3  -> IsolationForest.path_length()   PathLength(x, T, e), vectorised
    Equation 2   -> IsolationForest.score()    s(x, psi) = 2 ** (-E[h(x)] / c(psi))

Only NumPy and SciPy are used.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import digamma

EULER_GAMMA = 0.5772156649015329


def average_path_length(n):
    """c(n), Equation 1: the average path length of an unsuccessful search in a
    binary search tree built on n points, c(n) = 2 H(n-1) - 2 (n-1) / n.

    H(i) is the exact harmonic number, computed as digamma(i + 1) + Euler's
    constant (the paper notes that H(i) can be estimated by ln(i) + 0.5772...).
    c(n) is 0 for n <= 1, and c(2) = 1.
    """
    n = np.asarray(n, dtype=float)
    out = np.zeros_like(n)
    big = n > 1
    harmonic = digamma(n[big]) + EULER_GAMMA          # H(n - 1)
    out[big] = 2.0 * harmonic - 2.0 * (n[big] - 1.0) / n[big]
    return out if out.ndim else float(out)


class _ITree:
    """One isolation tree stored as flat arrays (node 0 is the root).

    External nodes point to themselves and have threshold +inf, so the routing
    loop in path_length() leaves an instance where it is once it reaches an
    external node.
    """

    __slots__ = ("feature", "threshold", "left", "right", "size", "depth",
                 "leaf_value", "max_depth")

    def __init__(self, feature, threshold, left, right, size, depth):
        self.feature = np.asarray(feature, dtype=np.intp)
        self.threshold = np.asarray(threshold, dtype=float)
        self.left = np.asarray(left, dtype=np.intp)
        self.right = np.asarray(right, dtype=np.intp)
        self.size = np.asarray(size, dtype=np.intp)
        self.depth = np.asarray(depth, dtype=np.intp)
        # Algorithm 3, line 2: an external node returns e + c(T.size)
        self.leaf_value = self.depth + average_path_length(self.size)
        self.max_depth = int(self.depth.max())

    @property
    def n_nodes(self):
        return len(self.feature)

    def is_leaf(self):
        return self.left == np.arange(self.n_nodes)


def _build_itree(X, height_limit, rng, split_rule="varying"):
    """Algorithm 2, iTree(X, e, l), on the sub-sample X.

    Stopping rules: the height limit is reached, |X| <= 1 (both in Algorithm 2),
    or all instances in X have the same values (rule (iii) in the text of
    Section 2, which Algorithm 2 leaves out).

    split_rule decides which attribute q can be chosen at a node:
      "varying" - uniformly among the attributes that are not constant in X,
                  so every split really divides X (default; scikit-learn does
                  the same);
      "any"     - uniformly among all attributes, read literally from
                  Algorithm 2, line 5. A constant attribute then gives an
                  empty left child and a wasted level.
    """
    if split_rule not in ("varying", "any"):
        raise ValueError("split_rule must be 'varying' or 'any'")
    n_features = X.shape[1]
    feature, threshold, left, right, size, depth = [], [], [], [], [], []

    def grow(rows, e):
        node = len(feature)
        feature.append(0)            # placeholder values for an external node
        threshold.append(np.inf)
        left.append(node)
        right.append(node)
        size.append(len(rows))
        depth.append(e)

        if e >= height_limit or len(rows) <= 1:              # line 1-2
            return node
        Xn = X[rows]
        lo, hi = Xn.min(axis=0), Xn.max(axis=0)
        varying = np.flatnonzero(hi > lo)
        if varying.size == 0:                                # rule (iii)
            return node
        if split_rule == "varying":
            q = int(varying[rng.integers(varying.size)])
        else:
            q = int(rng.integers(n_features))                # line 5
        p = rng.uniform(lo[q], hi[q])                        # line 6
        goes_left = Xn[:, q] < p                             # line 7-8
        feature[node] = q
        threshold[node] = p
        left[node] = grow(rows[goes_left], e + 1)            # line 9
        right[node] = grow(rows[~goes_left], e + 1)          # line 10
        return node

    grow(np.arange(X.shape[0]), 0)
    return _ITree(feature, threshold, left, right, size, depth)


class IsolationForest:
    """iForest with t trees, each grown on a sub-sample of psi instances
    drawn without replacement (Algorithm 1).

    Parameters
    ----------
    n_trees : int, default 100       t in the paper
    subsample_size : int, default 256   psi in the paper (capped at n)
    split_rule : "varying" or "any"  see _build_itree
    random_state : int or None       seed for numpy.random.default_rng
    """

    def __init__(self, n_trees=100, subsample_size=256, split_rule="varying",
                 random_state=None):
        self.n_trees = n_trees
        self.subsample_size = subsample_size
        self.split_rule = split_rule
        self.random_state = random_state

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        self.psi_ = min(self.subsample_size, n)
        self.height_limit_ = math.ceil(math.log2(self.psi_))   # Algorithm 1, line 2
        self.trees_ = []
        for _ in range(self.n_trees):                          # line 3
            rows = rng.choice(n, size=self.psi_, replace=False)   # line 4
            self.trees_.append(_build_itree(X[rows], self.height_limit_, rng,
                                            self.split_rule))  # line 5
        return self

    def path_length(self, X, batch_size=65536):
        """E[h(x)]: the path length of each instance (Algorithm 3), averaged
        over all trees."""
        X = np.asarray(X, dtype=float)
        total = np.zeros(X.shape[0])
        for start in range(0, X.shape[0], batch_size):
            Xb = X[start:start + batch_size]
            rows = np.arange(Xb.shape[0])
            acc = np.zeros(Xb.shape[0])
            for tree in self.trees_:
                node = np.zeros(Xb.shape[0], dtype=np.intp)
                for _ in range(tree.max_depth):
                    # x_a < T.splitValue -> left, else right (lines 5-8)
                    go_left = Xb[rows, tree.feature[node]] < tree.threshold[node]
                    node = np.where(go_left, tree.left[node], tree.right[node])
                acc += tree.leaf_value[node]
            total[start:start + batch_size] = acc
        return total / len(self.trees_)

    def score(self, X):
        """Anomaly score s(x, psi) from Equation 2. Close to 1 means anomaly,
        well below 0.5 means normal."""
        return 2.0 ** (-self.path_length(X) / average_path_length(self.psi_))
