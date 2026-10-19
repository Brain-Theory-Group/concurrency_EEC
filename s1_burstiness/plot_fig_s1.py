"""SI Section S1, Fig. S1: mean EEC versus mean burstiness across datasets.

One point per dataset: the mean EEC over intersecting pairs (eec_mean)
against the mean burstiness of the (hyper)edges in those pairs (B_mean), for
(a) edge pairs, (b) e-h3 pairs, and (c) h3-h3 pairs; the panel title gives
the Pearson correlation r across datasets. Reads data/results.json
(burstiness_analysis.py) and writes fig_s1.pdf into the folder of the SI
(config.SI_FIG_DIR). Runtime: seconds.

The label offsets below only place the dataset names so that they do not
overlap; they carry no information.

    python3 s1_burstiness/plot_fig_s1.py
"""

import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
import config  # noqa: E402

FIG = os.path.join(config.SI_FIG_DIR, "fig_s1.pdf")

KINDS = ["edge", "e-h3", "h3-h3"]
COLOR = {"edge": "#1f5fa8", "e-h3": "#c2571a", "h3-h3": "#2a8a5a"}
ABBR = {
    "ia-reality-call": "reality", "college-msg": "college",
    "friends-family-call": "ff-call", "contact-high-school": "HS",
    "contact-primary-school": "PS", "DAWN": "DAWN",
    "congress-bills": "congress", "email-Eu": "email-Eu",
    "tags-ask-ubuntu": "ubuntu", "tags-math-sx": "math-sx",
    "tags-stack-overflow": "SO", "NDC-substances": "NDC-sub",
}
# Label position of each point: (dx, dy) in points from the point, and the
# horizontal alignment of the text. Default: (3, 2, "left").
LABEL = {
    ("edge", "ia-reality-call"): (-4, 0, "right"),
    ("edge", "college-msg"): (4, 0, "left"),
    ("edge", "friends-family-call"): (4, -5, "left"),
    ("edge", "contact-high-school"): (4, -4, "left"),
    ("edge", "contact-primary-school"): (4, 5, "left"),
    ("edge", "congress-bills"): (4, 0, "left"),
    ("edge", "tags-ask-ubuntu"): (4, 0, "left"),
    ("edge", "tags-math-sx"): (4, 3, "left"),
    ("edge", "tags-stack-overflow"): (-4, 0, "right"),
    ("e-h3", "contact-high-school"): (4, 5, "left"),
    ("e-h3", "contact-primary-school"): (-4, -5, "right"),
    ("e-h3", "email-Eu"): (-9.5, -6, "left"),
    ("e-h3", "tags-ask-ubuntu"): (0, 6, "center"),
    ("e-h3", "tags-math-sx"): (12, -24, "left"),
    ("e-h3", "tags-stack-overflow"): (-4, 0, "right"),
    ("h3-h3", "tags-ask-ubuntu"): (-6.5, 6, "left"),
    ("h3-h3", "tags-math-sx"): (-4.0, -0.1, "right"),
    ("h3-h3", "tags-stack-overflow"): (-4, -4, "right"),
}
ARROW = {("e-h3", "tags-math-sx")}          # label connected by an arrow
YTICKS = {"edge": [0, 0.05, 0.1, 0.15, 0.2], "e-h3": [0, 0.1, 0.2, 0.3, 0.4]}
XLEFT0 = {"h3-h3"}                           # x axis starts at 0

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 7.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def tick_label(v, _pos):
    return "0" if abs(v) < 1e-12 else f"{v:g}"


def main(path=FIG):
    with open(os.path.join(HERE, "data", "results.json")) as f:
        res = json.load(f)["cases"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35))
    for ax, kind, letter in zip(axes, KINDS, "abc"):
        rows = [r for r in res.values() if r["pair_type"] == kind]
        x = np.array([r["B_mean"] for r in rows])
        y = np.array([r["eec_mean"] for r in rows])
        ax.scatter(x, y, s=18, c=COLOR[kind], zorder=3)
        for xx, yy, r in zip(x, y, rows):
            dx, dy, ha = LABEL.get((kind, r["dataset"]), (3, 2, "left"))
            arrow = (dict(arrowstyle="->", color="black", lw=0.6,
                          shrinkA=1, shrinkB=3)
                     if (kind, r["dataset"]) in ARROW else None)
            ax.annotate(ABBR[r["dataset"]], (xx, yy), fontsize=5.5,
                        xytext=(dx, dy), textcoords="offset points",
                        ha=ha, va="center", arrowprops=arrow)
        ax.set_xlabel(r"$\overline{B}$")
        ax.set_ylabel("EEC mean")
        ax.set_title(f"$r = {np.corrcoef(x, y)[0, 1]:.2f}$", pad=3)
        ax.text(-0.26, 1.04, f"({letter})", transform=ax.transAxes,
                fontsize=13.5, va="bottom")
        ax.margins(x=0.25, y=0.15)
        if kind in YTICKS:
            ax.set_yticks(YTICKS[kind])
        if kind in XLEFT0:
            ax.set_xlim(left=0)
        ax.xaxis.set_major_formatter(FuncFormatter(tick_label))
        ax.yaxis.set_major_formatter(FuncFormatter(tick_label))
    fig.tight_layout(pad=0.4, h_pad=1.0)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
