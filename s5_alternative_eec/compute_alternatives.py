"""Appendix B and SI Section S5: the comparisons of Tables 2-4 repeated with
the alternative definitions of the EEC (Tables S7-S9), and the numbers
quoted in Appendix B and Section S5.

Writes data/results.json. Runtime about 15 min (8 cores), most of it the
EMD and the STTC of tags-stack-overflow.

For every dataset and pair type of Tables 2, 3, and 4 (main-text theta,
Delta, and pair classes), and for every classified pair, the script computes
  pearson  the Pearson correlation coefficient (PCC) of the window counts
           (Appendix B.1);
  emd      the earth mover's distance of the event times (Appendix B.2);
  sttc     the spike time tiling coefficient with Delta_t = Delta over the
           observation period [0, T) (Appendix B.3).
The EMD and the STTC are computed pair by pair (event_measures.py). For
each measure, Cohen's d is computed for the comparisons of the main tables:
int-dj, open-dj, closed-dj (edge and e-h3 pairs; open-dj and closed-dj only
where the main table has them) and int1-dj, int2-int1 (h3-h3 pairs). The EMD
is a distance, so a negative d means that the first class is more concurrent.

data/results.json:
  "cases"   per pair type and dataset: Cohen's d of each measure ("d", the
            values of Tables S7-S9), and, for the STTC, its Pearson
            correlation with the EEC across all classified pairs, the
            fraction of undefined terms, and the fraction of (hyper)edges
            whose coincidence windows cover the whole period;
  "quoted"  the comparisons in which an alternative disagrees in sign with
            the EEC (for the EMD, agrees) and, per pair type, the range of
            the STTC-EEC correlation over the datasets other than DAWN and
            the values for DAWN.

    python3 s5_alternative_eec/compute_alternatives.py
"""

import os
import sys
import json
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import config  # noqa: E402
from events import load_frequent, observation_period, window_counts  # noqa: E402
from pairs import load_pairs, GROUPS, N_LABELS  # noqa: E402
from eec import Windowed, iter_labelled, moments, cohen_d  # noqa: E402
import event_measures as em  # noqa: E402

EEC_RESULTS = os.path.join(HERE, "..", "tables2to4_figs2to4_eec", "data",
                           "eec_results.json")
MEASURES = ("pearson", "emd", "sttc")
COMPARISONS = {"edge": [("int", "dj"), ("open", "dj"), ("closed", "dj")],
               "e-h3": [("int", "dj"), ("open", "dj"), ("closed", "dj")],
               "h3-h3": [("int1", "dj"), ("int2", "int1")]}


def welford_from_sums(M):
    """(n, sum, sum of squares) -> (n, mean, M2) per label."""
    return [[n, s / n if n else 0.0, ss - s * s / n if n else 0.0] for n, s, ss in M]


def group_sums(W, pair_type):
    """Welford (n, mean, M2) per label -> (n, sum, sum of squares) per group."""
    out = {}
    for g, labs in GROUPS[pair_type].items():
        n, mu, m2 = em.combine([W[k] for k in labs])
        out[g] = np.array([n, n * mu, m2 + n * mu * mu])
    return out


def d_values(W, pair_type, subclasses):
    g = group_sums(W, pair_type)
    out = {}
    for k, (a, b) in enumerate(COMPARISONS[pair_type]):
        if pair_type != "h3-h3" and k > 0 and not subclasses:
            continue
        out[f"{a}-{b}"] = cohen_d(g[a], g[b])
    return out


def analyze(name, pair_type):
    folder = config.dataset_folder(name)
    _, (ta, tb), L = load_pairs(folder, pair_type, config.THETA)
    T = observation_period(folder)
    delta = config.DELTA[name]
    A = Windowed(window_counts(ta, T, delta))
    B = A if tb is ta else Windowed(window_counts(tb, T, delta))
    n_lab = N_LABELS[pair_type]
    res = {"moments": {}}
    for measure in ("pearson",):
        M = np.zeros((n_lab, 3))
        for lab, val, _, _ in iter_labelled(A, B, L, measure):
            M += moments(lab, val, n_lab)
        res["moments"][measure] = welford_from_sums(M)
    fa, oa = em.flatten(ta)
    fb, ob = (fa, oa) if tb is ta else em.flatten(tb)
    dt = float(delta)
    tile = lambda ts: np.array([em.tiled_fraction(np.sort(t.astype(float)), dt, float(T))  # noqa: E731
                                for t in ts])
    Tfa = tile(ta)
    Tfb = Tfa if tb is ta else tile(tb)
    for measure, mid in (("emd", em.EMD), ("sttc", em.STTC)):
        cnt, mean, m2, cor, und = em.class_moments(
            fa, oa, fb, ob, Tfa, Tfb, A.X, A.m, A.q, B.X, B.m, B.q, L, n_lab, mid, dt)
        res["moments"][measure] = [list(em.combine(zip(cnt[:, c], mean[:, c], m2[:, c])))
                                   for c in range(n_lab)]
        if mid == em.STTC:
            n, sx, sy, sxx, syy, sxy = cor
            res["r_sttc_eec"] = float((sxy / n - sx * sy / n ** 2) / np.sqrt(
                (sxx / n - (sx / n) ** 2) * (syy / n - (sy / n) ** 2)))
            res["frac_undefined_terms"] = float(und / (2 * n))
            tiled = Tfa if tb is ta else np.concatenate([Tfa, Tfb])
            res["frac_fully_tiled"] = float(np.mean(tiled >= 1.0))
    return res


def main():
    em.check_emd(load_frequent(config.dataset_folder("DAWN"), 2, config.THETA[2])[1])
    with open(EEC_RESULTS) as f:
        eec = json.load(f)
    cases, quoted = {}, {"sign_disagreements": {}, "sttc": {}}
    for pt in config.PAIR_TYPES:
        r_other = []
        for name in config.TABLE_OF[pt]:
            t0 = time.time()
            x = analyze(name, pt)
            sub = eec[name][pt]["subclasses"]
            mom = x.pop("moments")       # per-class moments; only d is kept
            x["d"] = {m: d_values(mom[m], pt, sub) for m in MEASURES}
            d_eec = eec[name][pt]["d"]
            for m in MEASURES:
                flip = -1 if m == "emd" else 1
                for k, v in x["d"][m].items():
                    if np.sign(flip * v) != np.sign(d_eec[k]):
                        quoted["sign_disagreements"].setdefault(m, []).append(
                            f"{pt}/{name}: d_{k} = {v:.2f} vs EEC {d_eec[k]:.2f}")
            if name == "DAWN":
                quoted["sttc"].setdefault(pt, {})["DAWN"] = {
                    k: x[k] for k in ("r_sttc_eec", "frac_undefined_terms", "frac_fully_tiled")}
            else:
                r_other.append(x["r_sttc_eec"])
            cases[f"{pt}/{name}"] = x
            print(f"{pt:6s} {name:24s} r(STTC, EEC)={x['r_sttc_eec']:+.2f} "
                  f"undefined={x['frac_undefined_terms']:.2f} ({time.time() - t0:.0f} s)",
                  flush=True)
        quoted["sttc"].setdefault(pt, {})["r_range_without_DAWN"] = [min(r_other), max(r_other)]
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    with open(os.path.join(HERE, "data", "results.json"), "w") as f:
        json.dump({"cases": cases, "quoted": quoted}, f, indent=1)
    print("wrote data/results.json")


if __name__ == "__main__":
    main()
