"""SI Section S3 (controlling for the activity of the two (hyper)edges):
Table S5 and the numbers quoted in Section S3 and in the third paragraph of
Section 4 of the main text.

Writes data/results.json. Runtime about 3 min.

For every dataset and pair type of Tables 2, 3, and 4 (main-text theta and
Delta), all classified pairs enter an ordinary least-squares regression
(Eq. (S2))

  S_ij = b0 + sum_c beta_c D^c_ij
            + g1 ln n_(1) + g2 ln n_(2) + h1 B_(1) + h2 B_(2) + error,

where D^c_ij are dummy variables of the pair classes (disjoint pairs are the
reference class), n_(1) >= n_(2) are the numbers of events of the two
(hyper)edges, and B_(1) and B_(2) are the burstiness of the (hyper)edge with
n_(1) and n_(2) events (if n_(1) = n_(2), B_(1) >= B_(2)). Two models:
  two-class model:   D^int (intersecting; for h3-h3, 1- or 2-intersecting);
  three-class model: D^open and D^closed (edge and e-h3 pairs), or D^int1 and
                     D^int2 (h3-h3 pairs; open and closed are merged).
A dummy coefficient is converted to the units of Cohen's d by dividing it
by the pooled standard deviation of the EEC of the two compared classes:
  d^adj_{int-dj} = beta_int / s(int, dj),  d^adj_{open-dj} = beta_open / s(open, dj),
  d^adj_{closed-dj} = beta_closed / s(closed, dj),
  d^adj_{int1-dj} = beta_int1 / s(int1, dj),
  d^adj_{int2-int1} = (beta_int2 - beta_int1) / s(int2, int1).
Without the four covariates these equal Cohen's d of Tables 2-4 exactly;
the script computes both from the same normal equations and reports the
covariate-free value as d.

data/results.json:
  "cases"   per pair type and dataset: n per class; d and d^adj of the
            comparisons of Table S5 (d_int_dj, dadj_int_dj; d_c1, dadj_c1,
            d_c2, dadj_c2, where c1 and c2 are open-dj and closed-dj for edge
            and e-h3 pairs and int1-dj and int2-int1 for h3-h3 pairs);
            three_class; and the coefficients of the three-class model;
  "quoted"  per pair type, the number of comparisons, how many keep the sign
            of d, and how many shrink in magnitude; for edge pairs, whether
            d^adj of int vs dj and of closed vs dj is positive everywhere.

The normal equations X'X b = X'y are accumulated block by block, so the
10^8 pairs of tags-stack-overflow are never held in memory at once.

    python3 s3_activity_controls/controlled_regression.py
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
from events import observation_period, window_counts, burstiness  # noqa: E402
from pairs import load_pairs  # noqa: E402
from eec import Windowed, iter_labelled  # noqa: E402
from eec import var  # noqa: E402

# Columns of the design matrix: intercept, class dummies c1 and c2 of the
# three-class model, the dummy of the two-class model, and the covariates.
COLS = ["one", "c1", "c2", "int", "ln_n1", "ln_n2", "B1", "B2"]
COV = [4, 5, 6, 7]


def three_class(pair_type, labels):
    """Labels of lib/pairs.py -> 0 dj, 1 open (int1), 2 closed (int2)."""
    if pair_type == "h3-h3":
        return np.array([0, 1, 1, 2, 2])[labels]
    return labels


def analyze(name, pair_type):
    folder = config.dataset_folder(name)
    _, (ta, tb), L = load_pairs(folder, pair_type, config.THETA)
    T = observation_period(folder)
    delta = config.DELTA[name]
    A = Windowed(window_counts(ta, T, delta))
    Bw = A if tb is ta else Windowed(window_counts(tb, T, delta))
    lna = np.log([len(t) for t in ta])
    lnb = lna if tb is ta else np.log([len(t) for t in tb])
    Ba = np.array([burstiness(t) for t in ta])
    Bb = Ba if tb is ta else np.array([burstiness(t) for t in tb])
    XtX, Xty, mom = np.zeros((8, 8)), np.zeros(8), np.zeros((3, 3))
    for lab, y, i, j in iter_labelled(A, Bw, L, "eec", cells=4_000_000):
        c = three_class(pair_type, lab.astype(np.int64))
        la, lb, ba, bb = lna[i], lnb[j], Ba[i], Bb[j]
        a_first = (la > lb) | ((la == lb) & (ba >= bb))
        X = np.column_stack([
            np.ones_like(y), c == 1, c == 2, c >= 1,
            np.where(a_first, la, lb), np.where(a_first, lb, la),
            np.where(a_first, ba, bb), np.where(a_first, bb, ba)]).astype(np.float64)
        XtX += X.T @ X
        Xty += X.T @ y
        for k in range(3):
            yk = y[c == k]
            mom[k] += (len(yk), yk.sum(), (yk * yk).sum())

    def ols(cols):
        idx = np.array(cols)
        return dict(zip([COLS[k] for k in cols],
                        np.linalg.solve(XtX[np.ix_(idx, idx)], Xty[idx])))

    def s(m1, m2):
        return float(np.sqrt((var(m1) + var(m2)) / 2.0))

    res = {"n": [int(m[0]) for m in mom]}
    s_int = s(mom[1] + mom[2], mom[0])
    res["d_int_dj"] = ols([0, 3])["int"] / s_int
    res["dadj_int_dj"] = ols([0, 3] + COV)["int"] / s_int
    if pair_type == "h3-h3":
        three = all(m[0] >= config.MIN_PAIRS for m in mom)
    else:
        three = mom[0][0] >= config.MIN_PAIRS and min(mom[1][0], mom[2][0]) >= config.MIN_SUBCLASS
    res["three_class"] = bool(three)
    if three:
        raw, adj = ols([0, 1, 2]), ols([0, 1, 2] + COV)
        res["coef_three_class"] = adj
        if pair_type == "h3-h3":
            s10, s21 = s(mom[1], mom[0]), s(mom[2], mom[1])
            res["d_c1"], res["dadj_c1"] = raw["c1"] / s10, adj["c1"] / s10
            res["d_c2"] = (raw["c2"] - raw["c1"]) / s21
            res["dadj_c2"] = (adj["c2"] - adj["c1"]) / s21
        else:
            s10, s20 = s(mom[1], mom[0]), s(mom[2], mom[0])
            res["d_c1"], res["dadj_c1"] = raw["c1"] / s10, adj["c1"] / s10
            res["d_c2"], res["dadj_c2"] = raw["c2"] / s20, adj["c2"] / s20
    return res


def main():
    res = {}
    for pt in config.PAIR_TYPES:
        for name in config.TABLE_OF[pt]:
            t0 = time.time()
            res[f"{pt}/{name}"] = analyze(name, pt)
            print(f"{pt:6s} {name:24s} ({time.time() - t0:.0f} s)", flush=True)
    # Numbers quoted in the text: for each pair type, the comparisons in the
    # table, how many keep the sign of d, and how many shrink in magnitude.
    num = {}
    for pt in config.PAIR_TYPES:
        pairs = []
        for name in config.TABLE_OF[pt]:
            x = res[f"{pt}/{name}"]
            pairs.append((x["d_int_dj"], x["dadj_int_dj"]))
            if x["three_class"]:
                pairs += [(x["d_c1"], x["dadj_c1"]), (x["d_c2"], x["dadj_c2"])]
        d, da = np.array(pairs).T
        num[pt] = {"n_comparisons": len(pairs),
                   "n_same_sign": int((np.sign(d) == np.sign(da)).sum()),
                   "n_smaller_magnitude": int((np.abs(da) < np.abs(d)).sum())}
    num["edge"]["dadj_int_dj_all_positive"] = all(
        res[f"edge/{n}"]["dadj_int_dj"] > 0 for n in config.TABLE2)
    num["edge"]["dadj_closed_dj_all_positive"] = all(
        res[f"edge/{n}"]["dadj_c2"] > 0 for n in config.TABLE2 if res[f"edge/{n}"]["three_class"])
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    with open(os.path.join(HERE, "data", "results.json"), "w") as f:
        json.dump({"cases": res, "quoted": num}, f, indent=1)
    print("wrote data/results.json")


if __name__ == "__main__":
    main()
