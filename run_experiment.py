"""
Reproduces the main experiment of Liu, Ting and Zhou (2008), Section 5.1 and
Table 3: AUC and processing time of iForest on twelve data sets, with ORCA,
LOF and Random Forests as competitors.

    python run_experiment.py                         # everything
    python run_experiment.py --methods iforest sklearn
    python run_experiment.py --datasets pima breastw --runs 3

Each call starts results/runs.csv again; --append adds to it instead.

Methods
    iforest        my implementation (iforest.py), split_rule="varying"
    iforest_any    my implementation, split_rule="any" (Algorithm 2 read literally)
    sklearn        scikit-learn's IsolationForest with the same parameters
    orca_style     ORCA-style k-NN score (average distance to the k = 5 nearest
                   neighbours, computed exactly), every point ranked; not ORCA
                   itself, see baselines.py
    orca_style_n8  the same score, but only the top N = n / 8 points ranked, as
                   the paper describes ORCA's output
    lof, rf        the other competitors in baselines.py (rf only for n <= 1000,
                   as in the paper)

Repetitions
    The randomised methods (iforest, iforest_any, sklearn, rf) run --runs times
    (default 10) with seeds 0, 1, ...; their AUC and time are averaged over
    these runs. The deterministic methods (orca_style, orca_style_n8, lof) give
    the same AUC every time, so they run --timing-repeats times (default 3)
    only to average their CPU time; the script checks that the AUC is the same
    in every repeat. In runs.csv the column "run" is the seed for randomised
    methods and the repeat number for deterministic ones.

Outputs in results/: runs.csv (one row per run), summary.csv, comparison.csv,
datasets.csv, environment.txt. Times are CPU seconds (time.process_time) on
one thread; environment.txt lists the thread pools as measured at run time.
"""
import single_thread  # sets the thread variables before NumPy loads

import argparse
import platform
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import IsolationForest as SkIsolationForest
from sklearn.metrics import roc_auc_score

import baselines
import datasets
from iforest import IsolationForest

single_thread.limit()

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"

# Parameters of the experiment (Section 4.1 and 5 of the paper)
N_TREES = 100          # t
SUBSAMPLE = 256        # psi
LOF_K = 10
ORCA_K = 5             # ORCA's default k; N = n / 8 reported anomalies
RF_TREES = 100
RF_MAX_N = 1000        # the paper only ran RF on the four data sets below 1000 points

ALL_METHODS = ["iforest", "iforest_any", "sklearn", "orca_style", "orca_style_n8", "lof", "rf"]
RANDOMISED = {"iforest", "iforest_any", "sklearn", "rf"}


def cpu_time(fn, *args, **kwargs):
    start = time.process_time()
    out = fn(*args, **kwargs)
    return out, time.process_time() - start


def run_one(method, X, y, seed):
    """Returns (auc, train_seconds, eval_seconds); train is None when a method
    has no separate training stage."""
    if method in ("iforest", "iforest_any"):
        rule = "varying" if method == "iforest" else "any"
        forest = IsolationForest(n_trees=N_TREES, subsample_size=SUBSAMPLE,
                                 split_rule=rule, random_state=seed)
        model, t_train = cpu_time(forest.fit, X)
        score, t_eval = cpu_time(model.score, X)
    elif method == "sklearn":
        model = SkIsolationForest(n_estimators=N_TREES, max_samples=SUBSAMPLE,
                                  max_features=1.0, bootstrap=False,
                                  random_state=seed, n_jobs=1)
        _, t_train = cpu_time(model.fit, X)
        neg, t_eval = cpu_time(model.score_samples, X)
        score = -neg
    elif method == "lof":
        score, t_eval = cpu_time(baselines.lof_scores, X, LOF_K)
        t_train = None
    elif method == "orca_style":
        score, t_eval = cpu_time(baselines.orca_style_scores, X, ORCA_K, None)
        t_train = None
    elif method == "orca_style_n8":
        score, t_eval = cpu_time(baselines.orca_style_scores, X, ORCA_K, X.shape[0] // 8)
        t_train = None
    elif method == "rf":
        score, t_eval = cpu_time(baselines.rf_scores, X, RF_TREES, seed)
        t_train = None
    else:
        raise ValueError(method)
    return roc_auc_score(y, score), t_train, t_eval


def write_environment(runs, timing_repeats):
    cpu = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        cpu = platform.processor() or "unknown"
    pools = "\n".join("  " + line for line in single_thread.pools())
    text = (f"date: {date.today()}\nplatform: {platform.platform()}\ncpu: {cpu}\n"
            f"python: {platform.python_version()}\nnumpy: {np.__version__}\n"
            f"scipy: {scipy.__version__}\nscikit-learn: {sklearn.__version__}\n"
            f"pandas: {pd.__version__}\n"
            f"runs per randomised method: {runs} (seeds 0-{runs - 1})\n"
            f"timing repeats per deterministic method: {timing_repeats}\n"
            f"scikit-learn n_jobs: 1\n"
            f"thread pools at run time (threadpoolctl):\n{pools}\n"
            f"largest thread count of any pool: {single_thread.max_threads()}\n")
    (OUT / "environment.txt").write_text(text)


def summarise():
    runs = pd.read_csv(OUT / "runs.csv")
    runs["total_s"] = runs["train_s"].fillna(0) + runs["eval_s"]
    summary = (runs.groupby(["dataset", "method"])
               .agg(runs=("auc", "size"), auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
                    train_s=("train_s", "mean"), eval_s=("eval_s", "mean"),
                    total_s=("total_s", "mean"), total_s_sd=("total_s", "std"))
               .reset_index())
    order = {name: i for i, name in enumerate(datasets.DATASETS)}
    summary = summary.sort_values(["dataset", "method"],
                                  key=lambda c: c.map(order) if c.name == "dataset" else c)
    summary.to_csv(OUT / "summary.csv", index=False, float_format="%.4f")

    paper = pd.read_csv(ROOT / "paper_table3.csv")
    props = pd.read_csv(OUT / "datasets.csv")
    wide = summary.pivot(index="dataset", columns="method", values="auc_mean")
    sd = summary.pivot(index="dataset", columns="method", values="auc_sd")
    tot = summary.pivot(index="dataset", columns="method", values="total_s")
    tr = summary.pivot(index="dataset", columns="method", values="train_s")
    ev = summary.pivot(index="dataset", columns="method", values="eval_s")
    comp = paper.merge(props, on="dataset", how="left")
    get = lambda frame, col: comp["dataset"].map(frame[col]) if col in frame else np.nan
    comp["auc_iforest_ours"] = get(wide, "iforest")
    comp["auc_iforest_ours_sd"] = get(sd, "iforest")
    comp["auc_iforest_diff"] = comp["auc_iforest_ours"] - comp["auc_iforest"]
    comp["auc_sklearn"] = get(wide, "sklearn")
    comp["auc_sklearn_sd"] = get(sd, "sklearn")
    comp["auc_iforest_any"] = get(wide, "iforest_any")
    comp["auc_orca_style_ours"] = get(wide, "orca_style")
    comp["auc_orca_style_n8_ours"] = get(wide, "orca_style_n8")
    comp["auc_lof_ours"] = get(wide, "lof")
    comp["auc_rf_ours"] = get(wide, "rf")
    comp["time_iforest_train_ours"] = get(tr, "iforest")
    comp["time_iforest_eval_ours"] = get(ev, "iforest")
    comp["time_iforest_total_ours"] = get(tot, "iforest")
    comp["time_sklearn_total"] = get(tot, "sklearn")
    comp["time_orca_style_ours"] = get(tot, "orca_style")
    comp["time_lof_ours"] = get(tot, "lof")
    comp["time_rf_ours"] = get(tot, "rf")
    comp.to_csv(OUT / "comparison.csv", index=False, float_format="%.4f")
    return comp


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=datasets.DATASETS)
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    parser.add_argument("--runs", type=int, default=10,
                        help="runs (seeds) per randomised method")
    parser.add_argument("--timing-repeats", type=int, default=3,
                        help="repeats per deterministic method, to average its CPU time")
    parser.add_argument("--append", action="store_true",
                        help="add to results/runs.csv instead of starting it again")
    parser.add_argument("--fresh", action="store_true",
                        help="start results/runs.csv again (the default; kept for compatibility)")
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    runs_file = OUT / "runs.csv"
    if runs_file.exists() and not args.append:
        runs_file.unlink()
    write_environment(args.runs, args.timing_repeats)

    props = []
    for name in datasets.DATASETS:
        X, y = datasets.load(name)
        props.append(datasets.describe(name, X, y))
    pd.DataFrame(props).rename(columns={"n": "n_ours", "d": "d_ours"}).to_csv(
        OUT / "datasets.csv", index=False)

    for name in args.datasets:
        X, y = datasets.load(name)
        for method in args.methods:
            if method == "rf" and X.shape[0] > RF_MAX_N:
                continue
            randomised = method in RANDOMISED
            repeats = range(args.runs) if randomised else range(args.timing_repeats)
            rows = []
            for run in repeats:
                auc, t_train, t_eval = run_one(method, X, y, run)
                rows.append({"dataset": name, "method": method, "run": run, "auc": auc,
                             "train_s": t_train, "eval_s": t_eval})
            frame = pd.DataFrame(rows)
            if not randomised and frame.auc.nunique() != 1:
                raise RuntimeError(f"{method} on {name} is not deterministic: {frame.auc.tolist()}")
            frame.to_csv(runs_file, mode="a", header=not runs_file.exists(), index=False)
            print(f"{name:12s} {method:14s} AUC {frame.auc.mean():.4f} +/- "
                  f"{frame.auc.std() if len(frame) > 1 else 0:.4f}"
                  f"  time {frame.train_s.fillna(0).mean() + frame.eval_s.mean():8.2f}s"
                  f"  ({len(frame)} {'runs' if randomised else 'timing repeats'})", flush=True)
    summarise()


if __name__ == "__main__":
    main()
