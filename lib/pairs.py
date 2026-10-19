"""Classification of pairs of frequent (hyper)edges (Fig. 1B; Sections 3.1.2
and 3.2.1).

Each function returns a label matrix L: L[i, j] is the class of the pair
made of (hyper)edge i of the first list and (hyper)edge j of the second.
Pairs that do not enter the analysis have label -1: for pairs within one
list, the diagonal and the lower triangle (each unordered pair is counted
once, as i < j); for e-h3 pairs, an edge lying inside the size-3 hyperedge.

  edge pairs (labels_edge):
      0 disjoint, 1 intersecting-open, 2 intersecting-closed.
      {(v, v1), (v, v2)} is closed iff (v1, v2) is a dyadic edge of the
      aggregate network (any number of events; larger hyperedges ignored).
  e-h3 pairs (labels_eh3):
      0 disjoint, 1 intersecting-open, 2 intersecting-closed; intersecting
      means exactly one shared node. Closed iff the unshared node of the edge
      co-occurs with one of the two unshared nodes of the size-3 hyperedge in
      some static hyperedge of any size.
  h3-h3 pairs (labels_h3h3):
      0 disjoint, 1 1-intersecting-open, 2 1-intersecting-closed,
      3 2-intersecting-open, 4 2-intersecting-closed. Closed iff an unshared
      node of one hyperedge co-occurs with an unshared node of the other in
      some static hyperedge of any size. Table 4 merges 1 + 2 (int_1) and
      3 + 4 (int_2); Fig. 4 shows all five classes.

These are the rules of the first submission of the paper, implemented in a
vectorized way: the shared-node counts come from a sparse incidence-matrix
product, and the closure tests are set-membership tests on integer keys.
"""

import numpy as np
import scipy.sparse as sp

from events import load_frequent, static_dyadic_edges, cooccurring_pairs

NO_PAIR = -1


def _incidence(nodes, index):
    rows = np.repeat(np.arange(len(nodes)), [len(t) for t in nodes])
    cols = np.array([index[v] for t in nodes for v in t], dtype=np.int64)
    return sp.csr_matrix((np.ones(len(cols), dtype=np.int16), (rows, cols)),
                         shape=(len(nodes), len(index)))


def shared_node_counts(nodes_a, nodes_b):
    """Sparse (COO) matrix of the number of nodes shared by (hyper)edge i of
    nodes_a and (hyper)edge j of nodes_b; pairs sharing no node are absent."""
    allv = sorted({v for t in nodes_a for v in t} | {v for t in nodes_b for v in t})
    index = {v: k for k, v in enumerate(allv)}
    return (_incidence(nodes_a, index) @ _incidence(nodes_b, index).T).tocoo()


def _within_list_matrix(n):
    """Labels for pairs within one list: 0 (disjoint) for i < j, else -1.
    Intersecting pairs are overwritten by the callers."""
    L = np.full((n, n), NO_PAIR, dtype=np.int8)
    L[np.triu_indices(n, k=1)] = 0
    return L


def labels_edge(folder, nodes):
    """Edge pairs: 0 disjoint, 1 intersecting-open, 2 intersecting-closed."""
    n = len(nodes)
    L = _within_list_matrix(n)
    if n < 2:
        return L
    S = shared_node_counts(nodes, nodes)
    sel = (S.row < S.col) & (S.data == 1)
    i, j = S.row[sel], S.col[sel]
    nd = np.array(nodes, dtype=np.int64)
    ai, bi, aj, bj = nd[i, 0], nd[i, 1], nd[j, 0], nd[j, 1]
    shared = np.where((ai == aj) | (ai == bj), ai, bi)
    u1 = np.where(shared == ai, bi, ai)          # unshared node of edge i
    u2 = np.where(shared == aj, bj, aj)          # unshared node of edge j
    lo, hi = np.minimum(u1, u2), np.maximum(u1, u2)
    static = static_dyadic_edges(folder)
    closed = np.fromiter(((int(x), int(y)) in static for x, y in zip(lo, hi)),
                         dtype=bool, count=len(lo))
    L[i, j] = np.where(closed, 2, 1)
    return L


def _closed(keys, K, A, B):
    """For each row r, whether some node in A[r] co-occurs with some node in
    B[r]. A and B are int64 arrays of node ids padded with -1."""
    out = np.zeros(len(A), dtype=bool)
    for p in range(A.shape[1]):
        for q in range(B.shape[1]):
            a, b = A[:, p], B[:, q]
            ok = (a >= 0) & (b >= 0)
            k = np.minimum(a, b) * K + np.maximum(a, b)
            out |= ok & np.isin(k, keys)
    return out


def _unshared(nd_x, nd_y):
    """Nodes of each row of nd_x that are absent from the same row of nd_y,
    padded with -1 (shape kept)."""
    absent = np.ones(nd_x.shape, dtype=bool)
    for c in range(nd_y.shape[1]):
        absent &= nd_x != nd_y[:, [c]]
    return np.where(absent, nd_x, -1)


def labels_eh3(folder, nodes2, nodes3):
    """e-h3 pairs: 0 disjoint, 1 intersecting-open, 2 intersecting-closed;
    -1 if the edge lies inside the hyperedge (two shared nodes)."""
    L = np.zeros((len(nodes2), len(nodes3)), dtype=np.int8)
    if len(nodes2) == 0 or len(nodes3) == 0:
        return L
    S = shared_node_counts(nodes2, nodes3)
    L[S.row[S.data >= 2], S.col[S.data >= 2]] = NO_PAIR
    sel = S.data == 1
    i, j = S.row[sel], S.col[sel]
    n2 = np.array(nodes2, dtype=np.int64)[i]
    n3 = np.array(nodes3, dtype=np.int64)[j]
    keys, K = cooccurring_pairs(folder, {v for t in nodes2 for v in t}
                                | {v for t in nodes3 for v in t})
    closed = _closed(keys, K, _unshared(n2, n3), _unshared(n3, n2))
    L[i, j] = np.where(closed, 2, 1)
    return L


def labels_h3h3(folder, nodes3):
    """h3-h3 pairs: 0 disjoint, 1/2 1-intersecting open/closed,
    3/4 2-intersecting open/closed."""
    n = len(nodes3)
    L = _within_list_matrix(n)
    if n < 2:
        return L
    S = shared_node_counts(nodes3, nodes3)
    sel = (S.row < S.col) & (S.data <= 2)
    i, j, sh = S.row[sel], S.col[sel], S.data[sel].astype(np.int8)
    nd = np.array(nodes3, dtype=np.int64)
    keys, K = cooccurring_pairs(folder, {v for t in nodes3 for v in t})
    closed = _closed(keys, K, _unshared(nd[i], nd[j]), _unshared(nd[j], nd[i]))
    # sh = 1 -> 1 (open) or 2 (closed); sh = 2 -> 3 or 4.
    L[i, j] = 2 * sh - 1 + closed
    return L


# Class groups used by the tables (label values that are merged).
GROUPS = {
    "edge": {"dj": [0], "int": [1, 2], "open": [1], "closed": [2]},
    "e-h3": {"dj": [0], "int": [1, 2], "open": [1], "closed": [2]},
    "h3-h3": {"dj": [0], "int": [1, 2, 3, 4], "int1": [1, 2], "int2": [3, 4],
              "int1_open": [1], "int1_closed": [2],
              "int2_open": [3], "int2_closed": [4]},
}
N_LABELS = {"edge": 3, "e-h3": 3, "h3-h3": 5}


def load_pairs(folder, pair_type, theta):
    """Frequent (hyper)edges of one dataset and the labels of their pairs.

    pair_type is "edge", "e-h3", or "h3-h3"; theta = {2: theta_2, 3: theta_3}
    gives the event thresholds. Returns three items:
      (nodes_a, nodes_b)  node tuples of the first and second list of
                          (hyper)edges (the same list for edge and h3-h3 pairs;
                          edges and size-3 hyperedges for e-h3 pairs);
      (times_a, times_b)  their event times;
      L                   the label matrix (rows: first list, columns: second).
    """
    if pair_type == "edge":
        nodes, times = load_frequent(folder, 2, theta[2])
        return (nodes, nodes), (times, times), labels_edge(folder, nodes)
    if pair_type == "e-h3":
        n2, t2 = load_frequent(folder, 2, theta[2])
        n3, t3 = load_frequent(folder, 3, theta[3])
        return (n2, n3), (t2, t3), labels_eh3(folder, n2, n3)
    nodes, times = load_frequent(folder, 3, theta[3])
    return (nodes, nodes), (times, times), labels_h3h3(folder, nodes)
