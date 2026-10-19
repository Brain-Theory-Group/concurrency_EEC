"""Event-time similarity measures of Appendix B (module; imported by
compute_alternatives.py, not run on its own).

Two measures work on the raw event times rather than on window counts, and
are computed pair by pair with numba:

  EMD (Appendix B.2; Sihn and Kim 2019): the earth mover's distance
      between the event times of two (hyper)edges viewed as distributions
      with mass 1/N_i on each time stamp. On the real line it equals the
      Wasserstein-1 distance, the integral of |F_i(t) - F_j(t)|, where F_i is
      the empirical distribution function; emd_sorted() evaluates this
      integral exactly by merging the two sorted arrays (checked against
      scipy.stats.wasserstein_distance by check_emd()).

  STTC (Appendix B.3; Cutts and Eglen 2014): with coincidence window
      Delta_t,
          STTC = (1/2) [ (P_i - T_j) / (1 - P_i T_j) + (P_j - T_i) / (1 - P_j T_i) ],
      P_i = fraction of the events of i within Delta_t of an event of j,
      T_i = fraction of [0, T) within Delta_t of an event of i
            (tiled_fraction()).
      A term whose denominator is zero is set to 0 and counted as undefined.

class_moments() walks over all labelled pairs of a label matrix
(lib/pairs.py) and returns, per class, Welford's (count, mean, M2) of the
measure; for the STTC it also returns the sums needed for the Pearson
correlation between the STTC and the EEC across all labelled pairs, and the
number of undefined terms. Rows are processed in parallel.
"""

import numpy as np
import numba

EMD, STTC = 0, 1


@numba.njit(cache=False)
def emd_sorted(a, b):
    """Integral of |F_a - F_b| for sorted arrays a and b."""
    n, m = a.shape[0], b.shape[0]
    i = 0
    j = 0
    total = 0.0
    prev = min(a[0], b[0])
    while i < n or j < m:
        if j >= m or (i < n and a[i] <= b[j]):
            x = a[i]
        else:
            x = b[j]
        total += abs(i / n - j / m) * (x - prev)
        prev = x
        while i < n and a[i] == x:
            i += 1
        while j < m and b[j] == x:
            j += 1
    return total


@numba.njit(cache=False)
def coincident_fraction(a, b, dt):
    """Fraction of the sorted events a within dt of some event of sorted b."""
    j = 0
    m = b.shape[0]
    hit = 0
    for i in range(a.shape[0]):
        while j < m and b[j] < a[i] - dt:
            j += 1
        if j < m and b[j] <= a[i] + dt:
            hit += 1
    return hit / a.shape[0]


def tiled_fraction(ts, dt, T):
    """Fraction of [0, T) covered by the union of [t - dt, t + dt]."""
    lo = np.clip(ts - dt, 0, T)
    hi = np.clip(ts + dt, 0, T)
    total, cur_lo, cur_hi = 0.0, lo[0], hi[0]
    for a, b in zip(lo[1:], hi[1:]):
        if a > cur_hi:
            total += cur_hi - cur_lo
            cur_lo, cur_hi = a, b
        else:
            cur_hi = max(cur_hi, b)
    total += cur_hi - cur_lo
    return total / T


@numba.njit(cache=False)
def _sttc(a, b, Ta, Tb, dt):
    """STTC of one pair and its number of undefined terms (0, 1, or 2)."""
    s = 0.0
    und = 0
    P = coincident_fraction(a, b, dt)
    den = 1.0 - P * Tb
    if den == 0.0:
        und += 1
    else:
        s += (P - Tb) / den
    P = coincident_fraction(b, a, dt)
    den = 1.0 - P * Ta
    if den == 0.0:
        und += 1
    else:
        s += (P - Ta) / den
    return 0.5 * s, und


@numba.njit(parallel=True, cache=False)
def class_moments(ta, oa, tb, ob, Ta, Tb, Xa, ma, qa, Xb, mb, qb, L, n_class,
                  measure, dt):
    """Per-class Welford moments of the measure over pairs with L >= 0.

    ta/oa, tb/ob: sorted event times of all (hyper)edges flattened, with
    offsets; Ta, Tb: tiled fractions (STTC only); Xa, ma, qa (Xb, mb, qb):
    window counts with their means and mean squares (STTC only, for the EEC
    in the correlation). Returns cnt, mean, M2 of shape (n_class,) per row
    summed later, the correlation sums (n, sx, sy, sxx, syy, sxy) of
    (STTC, EEC), and the number of undefined STTC terms.
    """
    na = L.shape[0]
    nwin = Xa.shape[1]
    cnt = np.zeros((na, n_class))
    mean = np.zeros((na, n_class))
    m2 = np.zeros((na, n_class))
    cor = np.zeros((na, 6))
    und = np.zeros(na)
    # Rows are visited in an interleaved order (0, n-1, 1, n-2, ...): for pairs
    # within one list only j > i is labelled, so early rows carry most of the
    # work, and the contiguous chunks of prange would leave one thread with
    # most of it. The per-row accumulators make the result order-independent.
    for r in numba.prange(na):
        i = r // 2 if r % 2 == 0 else na - 1 - r // 2
        a = ta[oa[i]:oa[i + 1]]
        for j in range(L.shape[1]):
            c = L[i, j]
            if c < 0:
                continue
            b = tb[ob[j]:ob[j + 1]]
            if measure == EMD:
                x = emd_sorted(a, b)
            else:
                x, u = _sttc(a, b, Ta[i], Tb[j], dt)
                und[i] += u
                dot = 0.0
                for k in range(nwin):
                    dot += Xa[i, k] * Xb[j, k]
                ab = ma[i] * mb[j]
                den = np.sqrt(qa[i] * qb[j]) - ab
                e = (dot / nwin - ab) / den if den > 0 else 0.0
                e = min(1.0, max(-1.0, e))
                cor[i, 0] += 1.0
                cor[i, 1] += x
                cor[i, 2] += e
                cor[i, 3] += x * x
                cor[i, 4] += e * e
                cor[i, 5] += x * e
            cnt[i, c] += 1.0
            delta = x - mean[i, c]
            mean[i, c] += delta / cnt[i, c]
            m2[i, c] += delta * (x - mean[i, c])
    return cnt, mean, m2, cor.sum(axis=0), und.sum()


def combine(parts):
    """Merge Welford accumulators (count, mean, M2) (Chan et al.)."""
    n, mu, m2 = 0.0, 0.0, 0.0
    for nb, mb, m2b in parts:
        if nb == 0:
            continue
        tot = n + nb
        d = mb - mu
        mu += d * nb / tot
        m2 += m2b + d * d * n * nb / tot
        n = tot
    return n, mu, m2


def to_sums(w):
    """Welford (count, mean, M2) -> (count, sum, sum of squares), the moment
    format of lib/moments.py."""
    n, mu, m2 = w
    return np.array([n, n * mu, m2 + n * mu * mu])


def flatten(times):
    """Sorted float64 event times of all (hyper)edges and their offsets."""
    ts = [np.sort(np.asarray(t, dtype=np.float64)) for t in times]
    off = np.zeros(len(ts) + 1, dtype=np.int64)
    off[1:] = np.cumsum([len(t) for t in ts])
    return np.concatenate(ts), off


def check_emd(times, n_pairs=2000, seed=0):
    """emd_sorted() equals scipy.stats.wasserstein_distance."""
    from scipy.stats import wasserstein_distance
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_pairs):
        i, j = rng.choice(len(times), size=2, replace=False)
        a = np.sort(times[i].astype(np.float64))
        b = np.sort(times[j].astype(np.float64))
        ref = wasserstein_distance(a, b)
        worst = max(worst, abs(emd_sorted(a, b) - ref) / max(ref, 1e-12))
    assert worst < 1e-9, worst
    return worst
