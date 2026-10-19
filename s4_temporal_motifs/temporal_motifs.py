"""SI Section S4 (comparison with temporal motifs): Table S6 and the numbers
quoted in Section S4 and in the Discussion of the main text.

Writes data/results.json. Runtime about 2 min.

Networks: those of Table 2 except DAWN, whose quarterly time stamps give
many simultaneous events at a node, which these motifs do not allow.

Temporal motifs of Kovanen et al. (J. Stat. Mech. P11005, 2011), on all
dyadic events of a network (all edges, not only frequent ones), events
treated as instantaneous and undirected:
  * two events are Delta_t-adjacent if they share a node and
    0 < t_2 - t_1 <= Delta_t (equal time stamps are never adjacent);
  * a two-event subgraph is valid if the two events are consecutive among
    all events of their shared node (no event of that node in between);
  * chain motif:  events on (v1, v) and then (v, v2), v1 != v2;
    repeat motif: two events on the same edge.
Delta_t = Delta of the main text (config.DELTA).

R = (count in the data) / (mean count over 10 time-shuffled networks), in
which the time stamps of all events are randomly permuted across all events
(each event keeps its edge). R' is the same ratio when the two events need
not be consecutive, i.e., all pairs of Delta_t-adjacent events are counted.
The z-score is (count - mean) / standard deviation over the 10 shuffles.
The random generator of each network is seeded with SEED and the position of
the network in the list, so each network's numbers can be rerun alone.

data/results.json:
  "networks"  per network: the counts N, the ratios R, and the z-scores of
              the chain and repeat motifs, with (keys *_all) and without the
              consecutive-events condition (Table S6 lists R);
  "quoted"    the ranges of R in the text, the smallest |z|, the number of
              networks with d > 0 in Table 2, and the Pearson correlation,
              across the networks, between R_chain (R_repeat) and the
              disjoint-versus-intersecting d of Table 2, read from
              tables2to4_figs2to4_eec/data/eec_results.json.

    python3 s4_temporal_motifs/temporal_motifs.py
"""

import os
import sys
import json
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))

import numpy as np  # noqa: E402
from scipy import stats as sps  # noqa: E402

import config  # noqa: E402
from events import load_frequent  # noqa: E402

NETWORKS = [n for n in config.TABLE2 if n != "DAWN"]
N_SHUFFLE = 10
SEED = 20261003
EEC_RESULTS = os.path.join(HERE, "..", "tables2to4_figs2to4_eec", "data",
                           "eec_results.json")


def consecutive_counts(node_a, node_b, edge, t, delta):
    """Chain and repeat motifs (consecutive events of the shared node).

    Each event is listed once under each of its two nodes; sorting by
    (node, time, edge) puts the events of a node in time order, and two
    neighbors in this order are consecutive events of that node.
    """
    node = np.concatenate([node_a, node_b])
    ev_edge = np.concatenate([edge, edge])
    ev_t = np.concatenate([t, t])
    order = np.lexsort((ev_edge, ev_t, node))
    node, ev_edge, ev_t = node[order], ev_edge[order], ev_t[order]
    dt = ev_t[1:] - ev_t[:-1]
    adj = (node[1:] == node[:-1]) & (dt > 0) & (dt <= delta)
    same = ev_edge[:-1][adj] == ev_edge[1:][adj]
    return int((~same).sum()), int(same.sum())


def _pairs_within(group, t, delta):
    """Number of event pairs in the same group with 0 < t2 - t1 <= delta."""
    order = np.lexsort((t, group))
    g, s = group[order].astype(np.int64), t[order].astype(np.int64)
    span = int(s.max() - s.min()) + 2 * int(delta) + 2
    key = g * span + (s - s.min())
    lo = np.searchsorted(key, key, side="right")
    hi = np.searchsorted(key, key + delta, side="right")
    return int((hi - lo).sum())


def all_pair_counts(node_a, node_b, edge, t, delta):
    """Chain and repeat counts without the consecutive-events condition.
    A same-edge pair is seen at both of its nodes, a chain pair at its one
    shared node, hence chain = (pairs at nodes) - 2 (pairs on edges)."""
    s_node = _pairs_within(np.concatenate([node_a, node_b]), np.concatenate([t, t]), delta)
    s_edge = _pairs_within(edge, t, delta)
    return s_node - 2 * s_edge, s_edge


def analyze(name, rng):
    nodes, times = load_frequent(config.dataset_folder(name), 2, 1)
    counts = np.array([len(x) for x in times])
    edge = np.repeat(np.arange(len(nodes)), counts)
    na = np.repeat(np.array([u for u, _ in nodes]), counts)
    nb = np.repeat(np.array([v for _, v in nodes]), counts)
    t = np.concatenate(times)
    delta = config.DELTA[name]

    def four(tt):
        return np.array(consecutive_counts(na, nb, edge, tt, delta)
                        + all_pair_counts(na, nb, edge, tt, delta), dtype=float)

    obs = four(t)
    null = np.array([four(rng.permutation(t)) for _ in range(N_SHUFFLE)])
    mu, sd = null.mean(axis=0), null.std(axis=0, ddof=1)
    res = {"delta": delta, "n_events": int(len(t))}
    for k, key in enumerate(["chain", "repeat", "chain_all", "repeat_all"]):
        res["N_" + key] = int(obs[k])
        res["R_" + key] = float(obs[k] / mu[k])
        res["z_" + key] = float((obs[k] - mu[k]) / sd[k])
    return res


def main():
    res = {}
    for k, name in enumerate(NETWORKS):
        t0 = time.time()
        res[name] = analyze(name, np.random.default_rng([SEED, k]))
        r = res[name]
        print(f"{name:24s} R_chain={r['R_chain']:.2f} R_repeat={r['R_repeat']:.2f} "
              f"R'_chain={r['R_chain_all']:.2f} R'_repeat={r['R_repeat_all']:.2f} "
              f"({time.time() - t0:.0f} s)", flush=True)
    with open(EEC_RESULTS) as f:
        eec = json.load(f)
    d = [eec[name]["edge"]["d"]["int-dj"] for name in NETWORKS]
    col = lambda c: np.array([res[n][c] for n in NETWORKS])  # noqa: E731
    num = {
        "R_repeat_range": [col("R_repeat").min(), col("R_repeat").max()],
        "n_chain_under": int((col("R_chain") < 1).sum()),
        "R_chain_under_range": [col("R_chain")[col("R_chain") < 1].min(),
                                col("R_chain")[col("R_chain") < 1].max()],
        "R_chain_over": {n: res[n]["R_chain"] for n in NETWORKS if res[n]["R_chain"] > 1},
        "R_chain_all_range": [col("R_chain_all").min(), col("R_chain_all").max()],
        "min_abs_z_chain": float(np.abs(col("z_chain")).min()),
        "min_abs_z_repeat": float(np.abs(col("z_repeat")).min()),
        "n_networks_d_positive": int((np.array(d) > 0).sum()),
    }
    for c in ("R_chain", "R_repeat"):
        r, p = sps.pearsonr(col(c), d)
        num[f"pearson_{c}_vs_d"] = {"r": float(r), "p": float(p)}
    num = {k: (list(map(float, v)) if isinstance(v, list) else v) for k, v in num.items()}
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    with open(os.path.join(HERE, "data", "results.json"), "w") as f:
        json.dump({"networks": res, "quoted": num}, f, indent=1)
    print("wrote data/results.json")


if __name__ == "__main__":
    main()
