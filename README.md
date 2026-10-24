# Reproducing "Isolation Forest" (Liu, Ting and Zhou, ICDM 2008)

Data Wrangling, assignment 2, task 2: a from-scratch NumPy implementation of
iForest, checked against scikit-learn, and a rerun of the paper's main
experiment (Section 5.1, Table 3) on all twelve data sets, with ORCA, LOF and
Random Forests as competitors.

## Result

My iForest matches the paper's AUC to two decimals on 10 of the 12 data sets
and comes within 0.013 on Mulcross; Smtp is the one clear gap. The best method
per data set is the same as in Table 3.

| Data set | n | d | Paper AUC | Mine (mean ± sd, 10 runs) | scikit-learn |
| --- | --- | --- | --- | --- | --- |
| Http | 567,498 | 3 | 1.00 | 1.000 ± 0.000 | 1.000 |
| ForestCover | 286,048 | 10 | 0.88 | 0.882 ± 0.032 | 0.877 |
| Mulcross | 262,144 | 4 | 0.97 | 0.957 ± 0.009 | 0.961 |
| Smtp | 95,156 | 3 | 0.88 | 0.908 ± 0.007 | 0.904 |
| Shuttle | 49,097 | 9 | 1.00 | 0.997 ± 0.001 | 0.997 |
| Mammography | 11,183 | 6 | 0.86 | 0.861 ± 0.006 | 0.862 |
| Annthyroid | 7,200 | 6 | 0.82 | 0.821 ± 0.018 | 0.818 |
| Satellite | 6,435 | 36 | 0.71 | 0.708 ± 0.018 | 0.701 |
| Pima | 768 | 8 | 0.67 | 0.674 ± 0.012 | 0.671 |
| Breastw | 683 | 9 | 0.99 | 0.986 ± 0.002 | 0.987 |
| Arrhythmia | 452 | 274 | 0.80 | 0.796 ± 0.010 | 0.809 |
| Ionosphere | 351 | 32 | 0.85 | 0.849 ± 0.005 | 0.846 |

The competitors, processing times and every single run are in `results/`.

## Run it

```
pip install -r requirements.txt
python get_data.py               # checks the data files in data/ (downloads the ODDS files if missing)
python check_implementation.py   # unit checks and agreement with scikit-learn, about 30 s
python run_experiment.py         # the main experiment, 10 runs per data set, about 7 min on one core
python mulcross_sensitivity.py   # Mulcross under other subsample sizes and generator settings
```

All scripts run single-threaded. `run_experiment.py` starts `results/runs.csv`
again on every call; `--append` adds to it instead. Tested with Python 3.13,
NumPy 2.5, SciPy 1.18, scikit-learn 1.9 and pandas 3.0.

## Layout

```
iforest.py                 iForest from the paper: Algorithms 1-3 and Equations 1-2
datasets.py                loads the twelve data sets; can also regenerate Mulcross
baselines.py               ORCA score (exact k-NN), LOF and unsupervised Random Forests
run_experiment.py          the experiment of Section 5.1 / Table 3
check_implementation.py    checks on the implementation
mulcross_sensitivity.py    Mulcross with other subsample sizes and generator settings
get_data.py                downloads the ODDS data and checks all file hashes
paper_table3.csv           the numbers reported in Tables 2 and 3 of the paper
data/                      the twelve data sets
results/comparison.csv     my results next to the paper's, one row per data set
results/runs.csv           AUC and CPU time of every single run
results/summary.csv        mean and standard deviation per data set and method
```

## Parameters

| Parameter | Value |
| --- | --- |
| number of trees t | 100 |
| subsample size psi | 256, drawn without replacement |
| height limit l | ceiling(log2 psi) = 8 |
| split attribute | uniform among the attributes that are not constant in the node (`split_rule="varying"`); `"any"` reads Algorithm 2 literally |
| split value | uniform between the attribute's minimum and maximum in the node |
| stopping rules | height limit, one instance left, or all instances identical |
| c(n) | 2 H(n-1) - 2(n-1)/n with exact harmonic numbers; c(1) = 0, c(2) = 1 |
| score | s(x, psi) = 2^(-E[h(x)] / c(psi)) |
| runs | 10 per data set, seeds 0-9 |
| ORCA | k = 5, average distance to the k nearest neighbours, attributes scaled to [0, 1]; all points ranked, and top N = n/8 only |
| LOF | k = 10 |
| RF | 100 trees, max_features = sqrt(d), uniform synthetic class of size n, Breiman's outlier measure, only for n <= 1000 |

## Data

Eleven data sets are the ODDS benchmark versions (Rayana, 2016), taken from the
[ADBench](https://github.com/Minqi824/ADBench) and
[PyOD](https://github.com/yzhao062/pyod) repositories. Mulcross is
[OpenML data set 40897](https://www.openml.org/d/40897). `get_data.py` checks
every file against its SHA-256 hash.

## Reference

Liu, F. T., Ting, K. M. and Zhou, Z.-H. (2008). Isolation Forest. *Proceedings
of the 8th IEEE International Conference on Data Mining*, 413-422.
[doi:10.1109/ICDM.2008.17](https://doi.org/10.1109/ICDM.2008.17)
