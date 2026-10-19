"""SI Section S2 (robustness against theta and Delta): Tables 2, 3, and 4 of
the main text recomputed for six combinations of theta and Delta (Tables
S2-S4), and the numbers quoted in Section S2 and in the second paragraph of
Section 4 of the main text.

Writes data/results.json. Runtime about 5 min.

For each dataset of Tables 2 (edge pairs), 3 (e-h3 pairs), and 4 (h3-h3
pairs):
    theta in {theta_0, 2 theta_0},  theta_0 = 50 (edges), 25 (size-3);
    Delta in {Delta_0 / 3, Delta_0, 3 Delta_0},  Delta_0 = config.DELTA.
Delta_0 / 3 is rounded to the nearest integer, and skipped if Delta_0 < 3
(DAWN, whose Delta_0 = 1 is the native resolution). The pair classes depend
on theta only, so they are computed once per theta; the EEC is then
summarized per class for each Delta. A combination is listed in Tables S2-S4
if it passes the sample-size criteria of the main text (lib/eec.qualifies);
the combination (theta_0, Delta_0) reproduces Tables 2-4.

data/results.json:
  "rows"    {"<pair type>/<dataset>": [one entry per (theta, Delta): theta,
            delta, E_a, E_b, in_table, subclasses, and, for the combinations
            in the table, mu and d of the columns of the table, as in
            tables2to4_figs2to4_eec/data/eec_results.json]};
  "quoted"  for edge pairs, the range of d per network, the smallest
            d_closed-dj, and whether intersecting-closed pairs have the
            largest mean EEC in every combination; for e-h3 pairs, whether d
            keeps its sign, whether intersecting-closed pairs are the most
            concurrent class, and the range of d_open-dj; for h3-h3 pairs,
            whether d_int2-int1 is always positive and the largest d_int1-dj.

    python3 s2_theta_delta/theta_delta_sweep.py
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
from pairs import load_pairs, N_LABELS  # noqa: E402
from eec import (Windowed, iter_labelled, moments, group_moments,  # noqa: E402
                 qualifies, mean, cohen_d)

THETA_FACTORS = (1, 2)
COMPARISONS = {"edge": [("int", "dj"), ("open", "dj"), ("closed", "dj")],
               "e-h3": [("int", "dj"), ("open", "dj"), ("closed", "dj")],
               "h3-h3": [("int1", "dj"), ("int2", "int1")]}


def deltas(name):
    """Delta values examined for a dataset: Delta_0/3 (if >= 1 is possible),
    Delta_0, and 3 Delta_0."""
    d0 = config.DELTA[name]
    return ([int(round(d0 / 3))] if d0 >= 3 else []) + [d0, 3 * d0]


def run(name, pair_type):
    folder = config.dataset_folder(name)
    T = observation_period(folder)
    rows = []
    for tf in THETA_FACTORS:
        theta = {k: v * tf for k, v in config.THETA.items()}
        _, (ta, tb), L = load_pairs(folder, pair_type, theta)
        for delta in deltas(name):
            row = {"theta": theta, "delta": delta, "E_a": len(ta), "E_b": len(tb),
                   "in_table": False, "subclasses": False}
            if len(ta) and len(tb) and (L >= 0).any():
                A = Windowed(window_counts(ta, T, delta))
                B = A if tb is ta else Windowed(window_counts(tb, T, delta))
                M = np.zeros((N_LABELS[pair_type], 3))
                for lab, val, _, _ in iter_labelled(A, B, L, "eec"):
                    M += moments(lab, val, N_LABELS[pair_type])
                g = group_moments(M, pair_type)
                ok, sub = qualifies(pair_type, len(ta), len(tb), g)
                row.update(in_table=bool(ok), subclasses=bool(sub))
                if ok:
                    # Only the columns of Tables S2-S4: for edge and e-h3
                    # pairs, open and closed only where they are shown.
                    if pair_type == "h3-h3":
                        groups, comps = ["dj", "int1", "int2"], COMPARISONS[pair_type]
                    elif sub:
                        groups, comps = ["dj", "int", "open", "closed"], COMPARISONS[pair_type]
                    else:
                        groups, comps = ["dj", "int"], COMPARISONS[pair_type][:1]
                    row["mu"] = {k: mean(g[k]) for k in groups}
                    row["d"] = {f"{a}-{b}": cohen_d(g[a], g[b]) for a, b in comps}
            rows.append(row)
        del L
    return rows


def quoted(res):
    def listed(pt, name):
        return [r for r in res[f"{pt}/{name}"] if r["in_table"]]

    def closed_largest(r):
        return r["mu"]["closed"] > max(r["mu"]["open"], r["mu"]["dj"])

    q = {"edge": {}, "e-h3": {}, "h3-h3": {}}
    for name in config.TABLE2:
        v = listed("edge", name)
        q["edge"][name] = {
            "d_range": [min(r["d"]["int-dj"] for r in v), max(r["d"]["int-dj"] for r in v)],
            "d_closed_dj_min": min((r["d"]["closed-dj"] for r in v if r["subclasses"]),
                                   default=None),
            "closed_largest_everywhere": all(closed_largest(r) for r in v if r["subclasses"])}
    q["edge"]["_d_closed_dj_min_over_networks"] = min(
        x["d_closed_dj_min"] for x in q["edge"].values() if x["d_closed_dj_min"] is not None)
    for name in config.TABLE3:
        v = listed("e-h3", name)
        main = next(r for r in v if r["theta"] == config.THETA
                    and r["delta"] == config.DELTA[name])
        sign0 = np.sign(main["d"]["int-dj"])
        q["e-h3"][name] = {
            "d_same_sign_everywhere": all(np.sign(r["d"]["int-dj"]) == sign0 for r in v),
            "closed_largest_everywhere": all(closed_largest(r) for r in v if r["subclasses"]),
            "d_open_dj_range": [min((r["d"]["open-dj"] for r in v if r["subclasses"]), default=None),
                                max((r["d"]["open-dj"] for r in v if r["subclasses"]), default=None)]}
    for name in config.TABLE4:
        v = listed("h3-h3", name)
        q["h3-h3"][name] = {
            "d_int2_int1_all_positive": all(r["d"]["int2-int1"] > 0 for r in v),
            "d_int1_dj_max": max(r["d"]["int1-dj"] for r in v)}
    return q


def main():
    res = {}
    for pt in config.PAIR_TYPES:
        for name in config.TABLE_OF[pt]:
            t0 = time.time()
            res[f"{pt}/{name}"] = run(name, pt)
            print(f"{pt:6s} {name:24s} ({time.time() - t0:.0f} s)", flush=True)
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    with open(os.path.join(HERE, "data", "results.json"), "w") as f:
        json.dump({"rows": res, "quoted": quoted(res)}, f, indent=1)
    print("wrote data/results.json")


if __name__ == "__main__":
    main()
