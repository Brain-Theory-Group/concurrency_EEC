"""Loading time-stamped events and binning them into time windows.

Every dataset folder (see README.md, Section 2) holds two parallel files:

    edges_list.txt     "eid size n1 n2 [n3 ...]"   one row per (hyper)edge
    edges_timings.txt  "eid size t1 t2 ..."         one row per (hyper)edge

Time stamps are integers in the native time unit of the dataset, shifted so
that the first event of the dataset is at t = 0.

The loaders follow the code of the first submission of the paper; the
co-occurrence relation is returned as integer keys, which the vectorized
closure tests in pairs.py need, and it is computed with int32 counts (the
first-submission code used int8, which dropped node pairs co-occurring in a
multiple of 256 hyperedges).
"""

import os

import numpy as np
import scipy.sparse as sp


def load_frequent(folder, k, theta):
    """Frequent (hyper)edges of size k, i.e., those with >= theta events.

    Returns two parallel lists, in the order of edges_timings.txt:
      nodes : tuple of the k node ids of each (hyper)edge, sorted;
      times : np.ndarray (int64) of the event times of each (hyper)edge,
              in the order stored on disk (nondecreasing).
    The edge ids of the files are not needed downstream and are dropped.
    theta = 1 returns every (hyper)edge of size k with at least one event.
    """
    nodes_of = {}
    with open(os.path.join(folder, "edges_list.txt")) as f:
        for line in f:
            p = line.split()
            if p and int(p[1]) == k:
                nodes_of[int(p[0])] = tuple(sorted(int(x) for x in p[2:2 + k]))
    nodes, times = [], []
    with open(os.path.join(folder, "edges_timings.txt")) as f:
        for line in f:
            p = line.split()
            if not p or int(p[1]) != k:
                continue
            eid = int(p[0])
            if eid not in nodes_of or len(p) - 2 < theta:
                continue
            nodes.append(nodes_of[eid])
            times.append(np.array(p[2:], dtype=np.int64))
    return nodes, times


def static_dyadic_edges(folder):
    """All dyadic edges of the aggregate network (any number of events), as a
    set of (u, v) with u < v. Used to decide whether an intersecting edge
    pair {(v, v1), (v, v2)} is closed, i.e., whether (v1, v2) is an edge.
    Larger hyperedges are ignored here (Section 3.1.2)."""
    out = set()
    with open(os.path.join(folder, "edges_list.txt")) as f:
        for line in f:
            p = line.split()
            if p and int(p[1]) == 2:
                u, v = int(p[2]), int(p[3])
                out.add((min(u, v), max(u, v)))
    return out


def cooccurring_pairs(folder, nodes_kept):
    """Pairs of nodes that co-occur in at least one static hyperedge of any
    size, restricted to nodes_kept. Returned as a sorted int64 array of keys
    u * K + v (u < v), with K = max(nodes_kept) + 1, together with K.

    Used to decide whether an e-h3 or h3-h3 pair is closed (Section 3.2.1):
    the pair is closed if an unshared node of one hyperedge co-occurs with an
    unshared node of the other in some static hyperedge of any size.
    """
    kept = np.array(sorted(nodes_kept), dtype=np.int64)
    pos = {int(v): i for i, v in enumerate(kept)}
    rows, cols, h = [], [], 0
    with open(os.path.join(folder, "edges_list.txt")) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            members = [pos[int(x)] for x in p[2:2 + int(p[1])] if int(x) in pos]
            if len(members) >= 2:
                rows += members
                cols += [h] * len(members)
                h += 1
    K = int(kept.max()) + 1 if len(kept) else 1
    if not rows:
        return np.zeros(0, dtype=np.int64), K
    M = sp.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)),
                      shape=(len(kept), h))
    C = sp.triu(M @ M.T, k=1).tocoo()
    keys = kept[C.row] * K + kept[C.col]
    return np.unique(keys), K


def observation_period(folder):
    """T: the observation period is [0, T), with T one plus the time of the
    latest event of the dataset (events of any size). Time stamps are
    integers and the first event of a dataset is at t = 0."""
    latest = 0
    with open(os.path.join(folder, "edges_timings.txt")) as f:
        for line in f:
            p = line.split()
            if len(p) > 2:
                latest = max(latest, max(int(x) for x in p[2:]))
    return latest + 1


def window_counts(times, T, delta):
    """Event counts in the time windows of Eq. (1).

    The observation period [0, T) is partitioned into the windows
    [(l - 1) Delta, l Delta), l = 1, ..., l_max, where l_max = floor(T / Delta)
    is the largest l whose window lies inside the observation period. Row i,
    column l - 1 is the number of events of (hyper)edge i in window l; events
    after the last full window, t >= l_max Delta, are not counted.
    """
    n_win = T // delta
    X = np.zeros((len(times), n_win), dtype=np.float64)
    for i, ts in enumerate(times):
        ts = ts[ts < n_win * delta]
        np.add.at(X[i], ts // delta, 1.0)
    return X


def burstiness(ts):
    """Burstiness B = (sigma - m) / (sigma + m) of the inter-event times
    (Goh and Barabasi, EPL 2008), with the population standard deviation."""
    tau = np.diff(np.sort(np.asarray(ts, dtype=np.float64)))
    m, s = tau.mean(), tau.std()
    return float((s - m) / (s + m))
