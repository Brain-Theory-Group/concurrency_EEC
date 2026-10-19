"""SI Section S1 (relationship between the EEC and burstiness): Table S1,
the data of Fig. S1, and the numbers quoted in Section S1 and in the
first paragraph of Section 4 of the main text.

Writes data/results.json. Runtime about 1 min.

Pairs: the intersecting pairs of the datasets of Tables 2, 3, and 4 (edge
pairs sharing a node; e-h3 pairs sharing one node; h3-h3 pairs sharing one or
two nodes), with the main-text theta and Delta.

Burstiness of a (hyper)edge: B = (sigma - m) / (sigma + m) of its
inter-event times (lib/events.burstiness).

Per dataset and pair type:
  B_mean            mean B over the (hyper)edges in at least one intersecting
                    pair (the x axis of Fig. S1, overline{B} in Table S1);
  eec_mean          mean EEC over the intersecting pairs (y axis of Fig. S1);
  r_mean, r_max     Pearson correlation, across the intersecting pairs,
                    between the EEC and the mean (max) of the two B values;
  r_mean_n, r_max_n the same, partial on ln of the smaller and ln of the
                    larger event count of the two (hyper)edges.
The correlations are computed from the covariance matrix of the five
per-pair variables (EEC, mean B, max B, ln n_min, ln n_max), accumulated
block by block.

data/results.json has two parts:
  "cases"   the values above for each pair type and dataset (Table S1 and
            the points of Fig. S1);
  "quoted"  the numbers quoted in the text: how many of the 25 cases have
            |r_mean| <= 0.30 (after rounding to two decimals), the cases above
            0.30, the ranges of r_mean_n and r_max_n, and the across-dataset
            correlations of Fig. S1 with each dataset left out in turn (the
            text quotes the h3-h3 value without NDC-substances).

    python3 s1_burstiness/burstiness_analysis.py
"""

import os
import sys
import json

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))

import numpy as np  # noqa: E402

import config  # noqa: E402
from events import observation_period, window_counts, burstiness  # noqa: E402
from pairs import load_pairs  # noqa: E402
from eec import Windowed, iter_labelled  # noqa: E402
from eec import pearson_from_cov, partial_from_cov  # noqa: E402


def analyze(name, pair_type):
    folder = config.dataset_folder(name)
    _, (ta, tb), L = load_pairs(folder, pair_type, config.THETA)
    L = np.where(L >= 1, L, -1).astype(np.int8)        # intersecting pairs only
    T = observation_period(folder)
    delta = config.DELTA[name]
    A = Windowed(window_counts(ta, T, delta))
    B = A if tb is ta else Windowed(window_counts(tb, T, delta))
    Ba = np.array([burstiness(t) for t in ta])
    Bb = Ba if tb is ta else np.array([burstiness(t) for t in tb])
    lna = np.log([len(t) for t in ta])
    lnb = lna if tb is ta else np.log([len(t) for t in tb])
    n, s, P = 0, np.zeros(5), np.zeros((5, 5))
    used_a, used_b = np.zeros(len(ta), bool), np.zeros(len(tb), bool)
    for _, eec, i, j in iter_labelled(A, B, L, "eec"):
        V = np.column_stack([eec, 0.5 * (Ba[i] + Bb[j]), np.maximum(Ba[i], Bb[j]),
                             np.minimum(lna[i], lnb[j]), np.maximum(lna[i], lnb[j])])
        n += len(V)
        s += V.sum(axis=0)
        P += V.T @ V
        used_a[i] = True
        used_b[j] = True
    mu = s / n
    C = P / n - np.outer(mu, mu)                       # population covariance
    if tb is ta:
        B_mean = float(Ba[used_a | used_b].mean())
    else:
        B_mean = float(np.concatenate([Ba[used_a], Bb[used_b]]).mean())
    return {"pair_type": pair_type, "dataset": name, "n_pairs": int(n),
            "B_mean": B_mean, "eec_mean": float(mu[0]),
            "r_mean": pearson_from_cov(C, 0, 1), "r_max": pearson_from_cov(C, 0, 2),
            "r_mean_n": partial_from_cov(C, 0, 1, (3, 4)),
            "r_max_n": partial_from_cov(C, 0, 2, (3, 4))}


def main():
    res = {}
    for pt in config.PAIR_TYPES:
        for name in config.TABLE_OF[pt]:
            res[f"{pt}/{name}"] = analyze(name, pt)
            print(f"{pt:6s} {name:24s} pairs={res[f'{pt}/{name}']['n_pairs']:>10,d}",
                  flush=True)
    # Numbers quoted in the text.
    v = list(res.values())
    rm = np.array([x["r_mean"] for x in v])
    num = {
        "n_cases": len(v),
        "n_abs_r_mean_le_0.30": int((np.abs(np.round(rm, 2)) <= 0.30).sum()),
        "r_mean_above_0.30": {f"{x['pair_type']}/{x['dataset']}": x["r_mean"]
                              for x in v if abs(round(x["r_mean"], 2)) > 0.30},
        "r_mean_n_range": [min(x["r_mean_n"] for x in v), max(x["r_mean_n"] for x in v)],
        "r_max_n_range": [min(x["r_max_n"] for x in v), max(x["r_max_n"] for x in v)],
        "across_datasets": {},
    }
    for pt in config.PAIR_TYPES:
        names = config.TABLE_OF[pt]
        Bm = np.array([res[f"{pt}/{n}"]["B_mean"] for n in names])
        E = np.array([res[f"{pt}/{n}"]["eec_mean"] for n in names])
        num["across_datasets"][pt] = {
            "r": float(np.corrcoef(Bm, E)[0, 1]),
            "r_without": {n: float(np.corrcoef(np.delete(Bm, k), np.delete(E, k))[0, 1])
                          for k, n in enumerate(names)}}
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    with open(os.path.join(HERE, "data", "results.json"), "w") as f:
        json.dump({"cases": res, "quoted": num}, f, indent=1)
    print("wrote data/results.json")


if __name__ == "__main__":
    main()
