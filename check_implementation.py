"""
Checks that iforest.py does what the paper describes, and that it agrees with
scikit-learn's IsolationForest. Writes results/implementation_checks.txt.

    python check_implementation.py
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest as SkIsolationForest
from sklearn.metrics import roc_auc_score

import datasets
from iforest import IsolationForest, _build_itree, average_path_length

OUT = Path(__file__).resolve().parent / "results"
lines = []


def report(text=""):
    print(text)
    lines.append(text)


def check(condition, text):
    report(("PASS  " if condition else "FAIL  ") + text)
    return condition


all_ok = True

# 1. Equation 1
report("1. c(n), Equation 1")
all_ok &= check(average_path_length(1) == 0.0, "c(1) = 0")
all_ok &= check(abs(average_path_length(2) - 1.0) < 1e-12, "c(2) = 1")
h255 = sum(1.0 / k for k in range(1, 256))
c256 = 2 * h255 - 2 * 255 / 256
all_ok &= check(abs(average_path_length(256) - c256) < 1e-9,
                f"c(256) = {average_path_length(256):.4f} equals 2 H(255) - 2*255/256 with exact H")
approx = 2 * (math.log(255) + 0.5772156649) - 2 * 255 / 256
report(f"      (with the paper's estimate H(i) = ln(i) + 0.5772: c(256) = {approx:.4f})")

# 2. Structure of single trees (Algorithm 2)
report("\n2. Tree structure, Algorithm 2 (psi = 256, l = 8, 200 trees on Satellite)")
X, _ = datasets.load("satellite")
rng = np.random.default_rng(0)
ok_size = ok_nodes = ok_depth = ok_children = ok_route = True
for _ in range(200):
    rows = rng.choice(len(X), 256, replace=False)
    tree = _build_itree(X[rows], 8, rng)
    leaf = tree.is_leaf()
    ok_size &= tree.size[leaf].sum() == 256
    ok_nodes &= tree.n_nodes <= 2 * 256 - 1
    ok_depth &= tree.depth.max() <= 8
    internal = np.flatnonzero(~leaf)
    ok_children &= bool(np.all(tree.depth[tree.left[internal]] == tree.depth[internal] + 1)
                        and np.all(tree.depth[tree.right[internal]] == tree.depth[internal] + 1))
    node = np.zeros(256, dtype=np.intp)
    Xs = X[rows]
    for _ in range(tree.max_depth):
        go_left = Xs[np.arange(256), tree.feature[node]] < tree.threshold[node]
        node = np.where(go_left, tree.left[node], tree.right[node])
    counts = np.bincount(node, minlength=tree.n_nodes)
    ok_route &= bool(np.all(counts[leaf] == tree.size[leaf]))
all_ok &= check(ok_size, "external-node sizes add up to psi in every tree")
all_ok &= check(ok_nodes, "every tree has at most 2 psi - 1 = 511 nodes")
all_ok &= check(ok_depth, "no node is deeper than the height limit l = ceiling(log2 psi) = 8")
all_ok &= check(ok_children, "every internal node has exactly two children one level deeper")
all_ok &= check(ok_route, "routing the sub-sample down the tree reproduces the stored node sizes")

# 3. A fully grown tree isolates every distinct instance (Section 2)
report("\n3. Fully grown trees (no height limit) on 300 distinct points")
Z = np.random.default_rng(1).standard_normal((300, 2))
tree = _build_itree(Z, 10_000, np.random.default_rng(2))
leaf = tree.is_leaf()
all_ok &= check(int(leaf.sum()) == 300 and int((~leaf).sum()) == 299,
                "n = 300 external nodes and n - 1 = 299 internal nodes (2n - 1 in total)")
all_ok &= check(bool(np.all(tree.size[leaf] == 1)), "every external node holds one instance")

# 4. Few and different: an obvious outlier gets a short path (Figure 1)
report("\n4. Figure 1 style example: 135 Gaussian points plus one far point, 1000 trees")
G = np.random.default_rng(3).standard_normal((135, 2))
G[0] = [4.0, 4.0]
forest = IsolationForest(n_trees=1000, subsample_size=135, random_state=4).fit(G)
h = forest.path_length(G)
centre = int(np.argmin(np.linalg.norm(G[1:], axis=1))) + 1
report(f"      E[h] of the far point {h[0]:.2f}, of the most central point {h[centre]:.2f}")
all_ok &= check(h[0] < h[centre], "the anomaly has the shorter average path")
s = forest.score(G)
all_ok &= check(s[0] > 0.6 and s[centre] < 0.5, f"score of far point {s[0]:.2f} > 0.6, central point {s[centre]:.2f} < 0.5")

# 5. Agreement with scikit-learn
report("\n5. Agreement with scikit-learn's IsolationForest (psi = 256)")
report("   Expected path lengths with 2000 trees each, so that tree-to-tree randomness averages out:")
for name in ["mammography", "satellite", "arrhythmia"]:
    X, y = datasets.load(name)
    ours = IsolationForest(n_trees=2000, subsample_size=256, random_state=0).fit(X)
    sk = SkIsolationForest(n_estimators=2000, max_samples=256, random_state=0, n_jobs=1).fit(X)
    s_ours, s_sk = ours.score(X), -sk.score_samples(X)
    rho = spearmanr(s_ours, s_sk).statistic
    gap = np.abs(s_ours - s_sk).mean()
    report(f"   {name:12s} Spearman rho = {rho:.4f}, mean |s_ours - s_sklearn| = {gap:.4f}, "
           f"AUC ours {roc_auc_score(y, s_ours):.4f} vs sklearn {roc_auc_score(y, s_sk):.4f}")
    all_ok &= check(rho > 0.99, f"{name}: rankings agree (rho > 0.99)")

report("\nALL CHECKS PASSED" if all_ok else "\nSOME CHECKS FAILED")
OUT.mkdir(exist_ok=True)
(OUT / "implementation_checks.txt").write_text("\n".join(lines) + "\n")
