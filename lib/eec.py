"""The edge-event correlation (EEC), the two other window-count
similarities of the paper, per-class summaries with Cohen's d, and the
sample-size criteria.

Notation (Section 2.1). For (hyper)edge i, x_i = (e_i(1), ..., e_i(L)) is the
vector of its event counts in the L windows of width Delta
(events.window_counts), and <.> is the average over the L windows.

  cosine similarity, Eq. (2):
      C_ij = <x_i x_j> / sqrt(<x_i^2> <x_j^2>)
  EEC (baseline-corrected cosine similarity), Eqs. (4) and (5):
      S_ij = (C_ij - C0_ij) / (1 - C0_ij)
           = (<x_i x_j> - <x_i><x_j>) / (sqrt(<x_i^2><x_j^2>) - <x_i><x_j>),
      where C0_ij = sqrt(a_i a_j / ((a_i + F_i)(a_j + F_j))) (Eq. (3)) is the
      limit of C_ij for independent sequences with window-count means a and
      Fano factors F, estimated as a_i = <x_i>, F_i = Var(x_i) / a_i
      (population variance). check_eec_identity() verifies that the two
      expressions agree.
  Pearson correlation coefficient (PCC), the first alternative definition of
  the EEC in Appendix B.1:
      r_ij = (<x_i x_j> - <x_i><x_j>) / sqrt(Var(x_i) Var(x_j)).

The EEC lies in [-1, 1]. If its denominator is zero (only when the two count
vectors are constant and proportional), it is set to 0. Values are clipped to
[-1, 1] to remove rounding errors.

All three are computed for blocks of rows against all columns, so that only
the int8 label matrix of size N_a x N_b is ever held in memory in full.
"""

import numpy as np

import config
from pairs import GROUPS

MEASURES = ("eec", "cosine", "pearson")   # cosine: only inside the EEC check


class Windowed:
    """Window counts of a list of (hyper)edges with the per-row moments."""

    def __init__(self, X):
        self.X = X
        self.L = X.shape[1]
        self.m = X.mean(axis=1)            # <x_i>
        self.q = (X * X).mean(axis=1)      # <x_i^2>


def block(A, B, rows, measure="eec"):
    """Similarity of rows `rows` (a slice) of A against all rows of B."""
    dot = (A.X[rows] @ B.X.T) / A.L                       # <x_i x_j>
    if measure == "cosine":
        num = dot
        den = np.sqrt(A.q[rows, None] * B.q[None, :])
    else:
        ab = A.m[rows, None] * B.m[None, :]
        num = dot - ab                                    # covariance
        if measure == "eec":
            den = np.sqrt(A.q[rows, None] * B.q[None, :]) - ab
        elif measure == "pearson":
            den = np.sqrt((A.q[rows] - A.m[rows] ** 2)[:, None]
                          * (B.q - B.m ** 2)[None, :])
        else:
            raise ValueError(measure)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, 0.0)
    return np.clip(out, -1.0, 1.0, out=out)


def iter_labelled(A, B, L, measure="eec", cells=8_000_000):
    """Yield (labels, values, i, j) for the labelled pairs (L >= 0), block by
    block of rows; i and j are the row and column indices of the pairs."""
    step = max(1, cells // max(1, B.X.shape[0]))
    for a in range(0, A.X.shape[0], step):
        b = min(a + step, A.X.shape[0])
        lab = L[a:b]
        keep = lab >= 0
        if not keep.any():
            continue
        vals = block(A, B, slice(a, b), measure)
        ii, jj = np.nonzero(keep)
        yield lab[keep], vals[keep], ii + a, jj


def check_eec_identity(seed=0):
    """The closed form of S_ij equals (C - C0) / (1 - C0) with the empirical
    means and Fano factors (Eqs. (3)-(5))."""
    rng = np.random.default_rng(seed)
    X = rng.negative_binomial(0.3, 0.2, size=(40, 200)).astype(np.float64)
    X[X.sum(axis=1) == 0, 0] = 1.0
    W = Windowed(X)
    C = block(W, W, slice(None), "cosine")
    a = X.mean(axis=1)
    F = X.var(axis=1) / a
    C0 = np.sqrt(np.outer(a, a) / np.outer(a + F, a + F))
    iu = np.triu_indices(40, k=1)
    err = np.max(np.abs(((C - C0) / (1 - C0))[iu] - block(W, W, slice(None))[iu]))
    assert err < 1e-10, err
    return err


# ---------------------------------------------------------------------------
# Per-class summaries and Cohen's d (Section 2.2)
# ---------------------------------------------------------------------------
# Cohen's d = (mu_1 - mu_2) / sqrt((sigma_1^2 + sigma_2^2) / 2) with sample
# (ddof = 1) variances. With up to 10^8 pairs per dataset, values are not
# stored; a class is summarized by its moments (n, sum, sum of squares),
# which add up across blocks of pairs and merge across classes.


def moments(labels, values, n_labels):
    """Per-label (n, sum, sum of squares), shape (n_labels, 3)."""
    labels = labels.astype(np.int64)
    return np.column_stack([
        np.bincount(labels, minlength=n_labels),
        np.bincount(labels, weights=values, minlength=n_labels),
        np.bincount(labels, weights=values * values, minlength=n_labels),
    ]).astype(np.float64)


def merged(M, labels):
    """Moments of the union of the given labels."""
    return M[list(labels)].sum(axis=0)


def mean(m):
    return float(m[1] / m[0]) if m[0] > 0 else float("nan")


def var(m):
    n, s, ss = m
    return float((ss - s * s / n) / (n - 1)) if n > 1 else float("nan")


def cohen_d(m1, m2):
    """Cohen's d of class 1 versus class 2 from their moments."""
    if m1[0] < 2 or m2[0] < 2:
        return float("nan")
    pooled = np.sqrt((var(m1) + var(m2)) / 2.0)
    return float((mean(m1) - mean(m2)) / pooled) if pooled > 0 else float("nan")


def pearson_from_cov(C, a, b):
    """Pearson correlation of variables a and b from a covariance matrix."""
    return float(C[a, b] / np.sqrt(C[a, a] * C[b, b]))


def partial_from_cov(C, a, b, controls):
    """Partial correlation of a and b controlling for `controls`, from the
    inverse of the covariance matrix of (a, b, controls)."""
    idx = [a, b] + list(controls)
    P = np.linalg.inv(C[np.ix_(idx, idx)])
    return float(-P[0, 1] / np.sqrt(P[0, 0] * P[1, 1]))


# ---------------------------------------------------------------------------
# Sample-size criteria (Sections 3.1.1 and 3.2.1)
# ---------------------------------------------------------------------------
# A dataset (or, in SI Section S2, a combination of theta and Delta) enters
#   Table 2 (edge)   if E_freq >= 50, n_dj >= 20, n_int >= 20;
#   Table 3 (e-h3)   if E_2,freq + E_3,freq >= 40, n_dj >= 20, n_int >= 20;
#   Table 4 (h3-h3)  if E_3,freq >= 40, n_dj >= 20, n_int1 >= 20, n_int2 >= 20.
# The open/closed columns of Tables 2 and 3 need, in addition, n_open >= 10
# and n_closed >= 10; the five-class comparison of Fig. 4 needs at least 10
# pairs in each of the four intersecting h3-h3 subclasses.


def group_moments(M, pair_type):
    """Moments (n, sum, sum of squares) of every group of pairs.GROUPS, from
    the per-label moments M (rows = labels of lib/pairs.py)."""
    M = np.asarray(M, dtype=np.float64)
    return {g: merged(M, labs) for g, labs in GROUPS[pair_type].items()}


def qualifies(pair_type, E_a, E_b, g):
    """(enters the table, has the open/closed columns or the five classes).
    g is the output of group_moments()."""
    n = {k: v[0] for k, v in g.items()}
    if pair_type == "edge":
        row = (E_a >= config.MIN_E_FREQ_EDGES and n["dj"] >= config.MIN_PAIRS
               and n["int"] >= config.MIN_PAIRS)
    elif pair_type == "e-h3":
        row = (E_a + E_b >= config.MIN_E_FREQ_HYPER and n["dj"] >= config.MIN_PAIRS
               and n["int"] >= config.MIN_PAIRS)
    else:
        row = (E_a >= config.MIN_E_FREQ_HYPER and n["dj"] >= config.MIN_PAIRS
               and n["int1"] >= config.MIN_PAIRS and n["int2"] >= config.MIN_PAIRS)
        sub = ("int1_open", "int1_closed", "int2_open", "int2_closed")
        return row, row and all(n[c] >= config.MIN_SUBCLASS for c in sub)
    return row, row and n["open"] >= config.MIN_SUBCLASS \
        and n["closed"] >= config.MIN_SUBCLASS
