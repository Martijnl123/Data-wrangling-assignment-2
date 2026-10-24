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
import numbers

import numpy as np
from scipy.special import digamma

EULER_GAMMA = 0.5772156649015329


def average_path_length(n, harmonic="exact"):
    """c(n), Equation 1: the average path length of an unsuccessful search in a
    binary search tree built on n points, c(n) = 2 H(n-1) - 2 (n-1) / n.

    harmonic="exact" uses exact harmonic numbers, H(i) = digamma(i + 1) +
    Euler's constant, so c(2) = 1. harmonic="approx" uses the estimate given in
    the paper, H(i) = ln(i) + 0.5772..., for every n >= 2, which makes
    c(2) = 0.154 (scikit-learn also uses the estimate but sets c(2) = 1).
    In both cases c(n) = 0 for n <= 1.
    """
    if harmonic not in ("exact", "approx"):
        raise ValueError("harmonic must be 'exact' or 'approx'")
    n = np.asarray(n, dtype=float)
    out = np.zeros_like(n)
    big = n > 1
    if harmonic == "exact":
        h = digamma(n[big]) + EULER_GAMMA              # H(n - 1)
    else:
        h = np.log(n[big] - 1.0) + EULER_GAMMA         # the paper's estimate of H(n - 1)
    out[big] = 2.0 * h - 2.0 * (n[big] - 1.0) / n[big]
    return out if out.ndim else float(out)


def _check_data(X, n_features=None):
    """Return X as a 2-D float array, or raise ValueError for input the method
    is not defined for (other shapes, missing or infinite values)."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be a 2-D array (instances x attributes), got {X.ndim}-D")
    if X.shape[1] == 0:
        raise ValueError("X has no attributes")
    if not np.isfinite(X).all():
        raise ValueError("X contains missing (NaN) or infinite values; "
                         "iForest has no rule for them, so remove or impute them first")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(f"X has {X.shape[1]} attributes, "
                         f"but the forest was trained on {n_features}")
    return X


def _check_int(value, name, minimum):
    if (isinstance(value, bool) or not isinstance(value, numbers.Integral)
            or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


class _ITree:
    """One isolation tree stored as flat arrays (node 0 is the root).

    External nodes point to themselves and have threshold +inf, so the routing
    loop in path_length() leaves an instance where it is once it reaches an
    external node.
    """

    __slots__ = ("feature", "threshold", "left", "right", "size", "depth", "max_depth")

    def __init__(self, feature, threshold, left, right, size, depth):
        self.feature = np.asarray(feature, dtype=np.intp)
        self.threshold = np.asarray(threshold, dtype=float)
        self.left = np.asarray(left, dtype=np.intp)
        self.right = np.asarray(right, dtype=np.intp)
        self.size = np.asarray(size, dtype=np.intp)
        self.depth = np.asarray(depth, dtype=np.intp)
        self.max_depth = int(self.depth.max())

    @property
    def n_nodes(self):
        return len(self.feature)

    def is_leaf(self):
        return self.left == np.arange(self.n_nodes)

    def leaf_values(self, harmonic="exact"):
        """Algorithm 3, line 2: an external node returns e + c(T.size).
        (Only the entries of external nodes are ever used.)"""
        return self.depth + average_path_length(self.size, harmonic)


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
    n_trees : int >= 1, default 100        t in the paper
    subsample_size : int >= 2, default 256  psi in the paper (capped at n)
    split_rule : "varying" or "any"         see _build_itree
    random_state : int or None              seed for numpy.random.default_rng
    harmonic : "exact" or "approx"          how c(n) computes H(i), see
                                            average_path_length; it only
                                            affects scoring, not the trees

    The training data need at least two instances, and all values must be
    finite: the paper defines no handling of missing values, so they are
    rejected instead of being passed silently into the comparisons.
    """

    def __init__(self, n_trees=100, subsample_size=256, split_rule="varying",
                 random_state=None, harmonic="exact"):
        self.n_trees = n_trees
        self.subsample_size = subsample_size
        self.split_rule = split_rule
        self.random_state = random_state
        self.harmonic = harmonic

    def _check_params(self):
        _check_int(self.n_trees, "n_trees", 1)
        _check_int(self.subsample_size, "subsample_size", 2)
        if self.split_rule not in ("varying", "any"):
            raise ValueError("split_rule must be 'varying' or 'any'")
        if self.harmonic not in ("exact", "approx"):
            raise ValueError("harmonic must be 'exact' or 'approx'")

    def fit(self, X):
        self._check_params()
        X = _check_data(X)
        n = X.shape[0]
        if n < 2:
            raise ValueError(f"iForest needs at least 2 training instances, got {n} "
                             "(with one instance c(psi) = 0 and the score is undefined)")
        rng = np.random.default_rng(self.random_state)
        self.n_features_in_ = X.shape[1]
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
        if not getattr(self, "trees_", None):
            raise RuntimeError("call fit() before path_length() or score()")
        if self.harmonic not in ("exact", "approx"):
            raise ValueError("harmonic must be 'exact' or 'approx'")
        X = _check_data(X, self.n_features_in_)
        leaf_values = [tree.leaf_values(self.harmonic) for tree in self.trees_]
        total = np.zeros(X.shape[0])
        for start in range(0, X.shape[0], batch_size):
            Xb = X[start:start + batch_size]
            rows = np.arange(Xb.shape[0])
            acc = np.zeros(Xb.shape[0])
            for tree, values in zip(self.trees_, leaf_values):
                node = np.zeros(Xb.shape[0], dtype=np.intp)
                for _ in range(tree.max_depth):
                    # x_a < T.splitValue -> left, else right (lines 5-8)
                    go_left = Xb[rows, tree.feature[node]] < tree.threshold[node]
                    node = np.where(go_left, tree.left[node], tree.right[node])
                acc += values[node]
            total[start:start + batch_size] = acc
        return total / len(self.trees_)

    def score(self, X):
        """Anomaly score s(x, psi) from Equation 2. Close to 1 means anomaly,
        well below 0.5 means normal."""
        return 2.0 ** (-self.path_length(X) / average_path_length(self.psi_, self.harmonic))
