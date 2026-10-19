"""Main text, Table 1: the 22 temporal networks (dyadic slices).

Writes data/table1.json, one entry per dataset in the order of Table 1.
Runtime about 1 min.

Entries, computed from the dyadic events only:
  N           nodes taking part in at least one dyadic event;
  E           distinct dyadic edges;
  E_freq      frequent edges, i.e., edges with >= theta = 50 events;
  time_res    native time resolution (from the data documentation; META);
  T           observation span: the time of the last dyadic event (the
              first event of a dataset is at t = 0), converted to the unit
              T_unit (d = days, y = years) with the divisor in META;
  n_dj, n_int disjoint and intersecting pairs of frequent edges.

    python3 table1_datasets/table1.py
"""

import os
import sys
import json

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))

import config  # noqa: E402
from events import load_frequent  # noqa: E402
from pairs import shared_node_counts  # noqa: E402

# Native time resolution, and the unit and divisor used for the span T.
META = {
    "ia-reality-call": ("1 s", "d", 86400), "copenhagen-calls": ("1 s", "d", 86400),
    "college-msg": ("1 s", "d", 86400), "friends-family-call": ("1 s", "d", 86400),
    "social-evolution-call": ("1 s", "d", 86400),
    "contact-high-school": ("20 s", "d", 4320), "contact-primary-school": ("20 s", "d", 4320),
    "DAWN": ("1 q", "y", 4),
    "NDC-classes": ("1 d", "y", 365), "NDC-substances": ("1 d", "y", 365),
    "coauth-DBLP": ("1 y", "y", 1), "coauth-MAG-Geology": ("1 y", "y", 1),
    "coauth-MAG-History": ("1 y", "y", 1),
    "congress-bills": ("1 d", "y", 365), "email-Enron": ("1 d", "y", 365),
    "email-Eu": ("1 s", "y", 86400 * 365),
    "tags-ask-ubuntu": ("1 h", "y", 24 * 365), "tags-math-sx": ("1 h", "y", 24 * 365),
    "tags-stack-overflow": ("1 h", "y", 24 * 365),
    "threads-ask-ubuntu": ("1 h", "y", 24 * 365), "threads-math-sx": ("1 h", "y", 24 * 365),
    "threads-stack-overflow": ("1 h", "y", 24 * 365),
}


def row(name):
    nodes, times = load_frequent(config.dataset_folder(name), 2, 1)   # all dyadic edges
    freq = [n for n, t in zip(nodes, times) if len(t) >= config.THETA[2]]
    res, unit, div = META[name]
    n_int = 0
    if len(freq) >= 2:
        S = shared_node_counts(freq, freq)
        n_int = int(((S.row < S.col) & (S.data >= 1)).sum())
    return {"dataset": name, "N": len({v for t in nodes for v in t}), "E": len(nodes),
            "E_freq": len(freq), "time_res": res,
            "T": max(int(t.max()) for t in times) / div, "T_unit": unit,
            "n_dj": len(freq) * (len(freq) - 1) // 2 - n_int, "n_int": n_int}


def main():
    rows = [row(name) for name in config.DATASETS]
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    with open(os.path.join(HERE, "data", "table1.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print("wrote data/table1.json")


if __name__ == "__main__":
    main()
