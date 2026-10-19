"""Main text, Section 3: the EEC of every pair of frequent (hyper)edges,
summarized per pair class. Gives the values of Tables 2, 3, and 4 and the
data drawn in Figs. 2, 3, and 4 (plot_figs2to4.py).

Writes data/eec_results.json. Runtime about 12 min (8 cores, 16 GB),
almost all of it tags-stack-overflow.

For each dataset with a Delta (lib/config.py) and each pair type
  edge   frequent edge x frequent edge                (Table 2, Figs. 2, 3)
  e-h3   frequent edge x frequent size-3 hyperedge    (Table 3)
  h3-h3  frequent size-3 x frequent size-3 hyperedge  (Table 4, Fig. 4)
the script
  1. loads the frequent (hyper)edges (theta = 50 for edges, 25 for size-3
     hyperedges) and classifies every pair (lib/pairs.py);
  2. counts the events in the l_max = floor(T / Delta) windows of width
     Delta that lie inside the observation period [0, T) (Eq. (1)) and
     computes the EEC S_ij of every classified pair (lib/eec.py, Eqs. (4)-(5));
  3. summarizes the EEC per class: number of pairs n, mean mu, and Cohen's d
     of the comparisons of the tables; applies the sample-size criteria;
  4. for edge and h3-h3 pairs, stores what the figures draw: the percentiles
     1, 5, 25, 50, 75, and 95 of each class (box plots and x ranges) and,
     for edge pairs, the histogram counts on a fixed grid of bin width 0.025
     (Fig. 2). The per-pair values themselves (up to 1.3 x 10^8 per dataset)
     are not stored.
The datasets that pass the criteria must be config.TABLE2/3/4, which the SI
analyses use; the script stops otherwise.

data/eec_results.json:
  {dataset: {"delta": .., "T": .., "<pair type>": {
      "E_a", "E_b"       numbers of frequent (hyper)edges of the two kinds
                         (E_a = E_freq of Table 2; E_b = E_a except for e-h3);
      "n"                number of pairs per group (dj, int, open, closed; for
                         h3-h3 int1, int2, int1_open, ..., see pairs.GROUPS),
                         which the sample-size criteria use;
      "in_table"         whether the dataset is a row of the table;
      "subclasses"       whether the open/closed columns (Tables 2, 3) or the
                         five classes (Fig. 4) are shown;
    and, for the datasets in the table only,
      "mu"               mean EEC of the groups shown in the table;
      "d"                Cohen's d of the comparisons shown: int-dj, open-dj,
                         closed-dj (edge, e-h3), int1-dj, int2-int1 (h3-h3);
      "percentiles", "hist"   the classes drawn in the figures (see above)}}}

    python3 tables2to4_figs2to4_eec/compute_eec.py
"""

import os
import sys
import json
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))

import numpy as np  # noqa: E402

import config  # noqa: E402
from events import observation_period, window_counts  # noqa: E402
from pairs import load_pairs, GROUPS, N_LABELS  # noqa: E402
from eec import (Windowed, iter_labelled, check_eec_identity, moments,  # noqa: E402
                 group_moments, qualifies, mean, cohen_d)

OUT = os.path.join(HERE, "data", "eec_results.json")
PERCENTILES = [1, 5, 25, 50, 75, 95]
HIST_EDGES = np.linspace(-1.0, 1.0, 81)       # Fig. 2: bins of width 0.025
COMPARISONS = {"edge": [("int", "dj"), ("open", "dj"), ("closed", "dj")],
               "e-h3": [("int", "dj"), ("open", "dj"), ("closed", "dj")],
               "h3-h3": [("int1", "dj"), ("int2", "int1")]}


def analyze(name, pair_type, T):
    folder = config.dataset_folder(name)
    _, (ta, tb), L = load_pairs(folder, pair_type, config.THETA)
    res = {"E_a": len(ta), "E_b": len(tb), "in_table": False, "subclasses": False}
    if len(ta) == 0 or len(tb) == 0 or not (L >= 0).any():
        return res
    delta = config.DELTA[name]
    A = Windowed(window_counts(ta, T, delta))
    B = A if tb is ta else Windowed(window_counts(tb, T, delta))
    n_lab = N_LABELS[pair_type]
    M = np.zeros((n_lab, 3))
    with_dist = pair_type in ("edge", "h3-h3")
    values = {k: [] for k in range(n_lab)}
    for lab, val, _, _ in iter_labelled(A, B, L, "eec"):
        M += moments(lab, val, n_lab)
        if with_dist:
            for k in range(n_lab):
                values[k].append(val[lab == k].astype(np.float32))
    g = group_moments(M, pair_type)
    res["n"] = {k: int(v[0]) for k, v in g.items()}
    res["in_table"], res["subclasses"] = (bool(x) for x in
                                          qualifies(pair_type, len(ta), len(tb), g))
    if not res["in_table"]:
        return res                       # only the sample sizes are needed
    # Groups and comparisons shown in the tables: for edge and e-h3 pairs,
    # open and closed only where the open/closed columns are present.
    groups = list(GROUPS[pair_type])
    comps = COMPARISONS[pair_type]
    if pair_type != "h3-h3" and not res["subclasses"]:
        groups, comps = ["dj", "int"], comps[:1]
    if pair_type == "h3-h3":
        groups = ["dj", "int1", "int2"] + (groups[4:] if res["subclasses"] else [])
    res["mu"] = {k: mean(g[k]) for k in groups}
    res["d"] = {f"{a}-{b}": cohen_d(g[a], g[b]) for a, b in comps}
    # Figure data: Fig. 2 (edge: dj, int), Fig. 3 (edge with open/closed:
    # dj, open, closed), Fig. 4 (h3-h3 with all five classes).
    shown = {"edge": ["dj", "int"] + (["open", "closed"] if res["subclasses"] else []),
             "h3-h3": (["dj", "int1_open", "int1_closed", "int2_open", "int2_closed"]
                       if res["subclasses"] else []),
             "e-h3": []}[pair_type]
    if shown:
        res["percentiles"], res["hist"] = {}, {}
        for grp in shown:
            v = np.concatenate([x for k in GROUPS[pair_type][grp] for x in values[k]])
            res["percentiles"][grp] = np.percentile(v, PERCENTILES).tolist()
            if pair_type == "edge" and grp in ("dj", "int"):
                res["hist"][grp] = np.histogram(v, bins=HIST_EDGES)[0].tolist()
    return res


def main():
    check_eec_identity()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out = {"_percentiles": PERCENTILES, "_hist_edges": HIST_EDGES.tolist()}
    for name in config.DATASETS:
        if name not in config.DELTA:
            continue                     # threads-ask-ubuntu: no frequent edge
        T = observation_period(config.dataset_folder(name))
        out[name] = {"delta": config.DELTA[name], "T": T}
        for pt in config.PAIR_TYPES:
            t0 = time.time()
            out[name][pt] = analyze(name, pt, T)
            print(f"{name:24s} {pt:6s} n={out[name][pt].get('n', {})}  "
                  f"({time.time() - t0:.0f} s)", flush=True)
    for pt in config.PAIR_TYPES:
        rows = [n for n in config.DATASETS if n in out and out[n][pt]["in_table"]]
        assert rows == config.TABLE_OF[pt], (pt, rows)
    with open(OUT, "w") as f:
        json.dump(out, f)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
