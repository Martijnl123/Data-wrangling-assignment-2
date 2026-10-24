# Reproducing Table 3 of "Isolation Forest" (Liu, Ting and Zhou, ICDM 2008)

Own NumPy implementation of iForest, checked against scikit-learn, run on the
twelve data sets of the paper with ORCA, LOF and Random Forests as competitors.

## Run it

```
pip install numpy scipy scikit-learn pandas
python get_data.py               # downloads the ODDS files and checks all hashes (everything is already in data/)
python check_implementation.py   # unit checks and agreement with scikit-learn, about 30 s
python run_experiment.py --fresh # the main experiment, 10 runs per data set
python mulcross_sensitivity.py   # Mulcross under other subsample sizes and generator settings
```

All scripts run single-threaded. Results land in `results/`.

## Files

| File | What it does |
| --- | --- |
| `iforest.py` | iForest from the paper: Algorithms 1-3 and Equations 1-2 |
| `datasets.py` | loads the twelve data sets; can also regenerate Mulcross |
| `baselines.py` | ORCA score (exact k-NN), LOF and unsupervised Random Forests |
| `run_experiment.py` | the experiment of Section 5.1 / Table 3 |
| `check_implementation.py` | checks on the implementation |
| `mulcross_sensitivity.py` | Mulcross with other subsample sizes and generator settings |
| `get_data.py` | downloads the ODDS data (via the ADBench and PyOD repositories) and checks the Mulcross file from OpenML |
| `paper_table3.csv` | the numbers reported in Tables 2 and 3 of the paper |
| `results/comparison.csv` | our results next to the paper's, one row per data set |
| `results/runs.csv` | AUC and CPU time of every single run |
| `results/summary.csv` | mean and standard deviation per data set and method |

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
| Mulcross | the OpenML file (data set 40897) in `data/mulcross.arff`; the regenerated version (shift 2 on every attribute, cluster sd 0.15, seed 2008) is only used in `mulcross_sensitivity.py` |
