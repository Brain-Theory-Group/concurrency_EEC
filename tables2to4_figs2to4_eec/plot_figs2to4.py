"""Main text, Figs. 2, 3, and 4: distributions of the EEC per pair class.

Reads data/eec_results.json (compute_eec.py) and writes, into the folder of
the main text (config.MAIN_FIG_DIR),
  fig2.pdf   Fig. 2: histograms of the EEC of disjoint and intersecting
                  edge pairs, one panel per network of Table 2, annotated
                  with the two means and Cohen's d;
  fig3.pdf   Fig. 3: box plots of the EEC of disjoint, intersecting-open,
                  and intersecting-closed edge pairs for the networks with the
                  three-class columns in Table 2;
  fig4.pdf   Fig. 4: box plots of the EEC of the five h3-h3 pair classes
                  for the hypergraphs with at least 10 pairs in each
                  intersecting subclass.
Runtime: seconds.

The figures are drawn from stored summaries, not from the per-pair values:
  * histograms: counts on the fixed grid of bin width 0.025; a panel uses
    the bins from x_lo to 1 and normalizes them to a probability density, as
    plt.hist(..., density=True) would with those bins;
  * box plots: box = 25th to 75th percentile, line = median, whiskers = 5th
    and 95th percentiles (as stated in the captions), no outliers; drawn with
    Axes.bxp from the stored percentiles.
x_lo is 0 if no plotted class has values below 0 at the relevant percentile
(the 1st for Fig. 2, the 5th for Fig. 3), and otherwise that percentile
rounded down to a multiple of 0.1.

The layout (panel grid, colors, font sizes) is that of the figures of the
first submission, with the x axis extended below 0 because the EEC can be
negative.

    python3 tables2to4_figs2to4_eec/plot_figs2to4.py
"""

import os
import sys
import json
import math

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import config  # noqa: E402

SRC = os.path.join(HERE, "data", "eec_results.json")
FIGS = config.MAIN_FIG_DIR
BIN = 0.025


def pct(entry, group, p, summary):
    """Stored percentile p (one of summary['_percentiles']) of a group."""
    return entry["percentiles"][group][summary["_percentiles"].index(p)]


def x_lower(lows):
    lo = min(lows)
    return 0.0 if lo >= 0 else math.floor(lo * 10) / 10


def ticks(lo):
    """Multiples of 0.2 from the first one >= lo up to 1."""
    k0 = int(np.ceil(round(lo / 0.2, 9)))
    return [round(0.2 * k, 10) for k in range(k0, 6)]


def tick_labels(ts):
    return ["0" if abs(t) < 1e-9 else "1" if abs(t - 1) < 1e-9 else f"{t:.1f}"
            for t in ts]


def box_stats(entry, group, summary, label):
    """Statistics for Axes.bxp: whiskers at the 5th and 95th percentiles."""
    p = lambda q: pct(entry, group, q, summary)  # noqa: E731
    return {"label": label, "whislo": p(5), "q1": p(25), "med": p(50),
            "q3": p(75), "whishi": p(95), "fliers": []}


def fig2(summary):
    names = config.TABLE2
    edges = np.array(summary["_hist_edges"])
    lo = x_lower([pct(summary[n]["edge"], g, 1, summary)
                  for n in names for g in ("dj", "int")])
    first = int(np.argmin(np.abs(edges - lo)))       # grid edge at x_lo
    bins = edges[first:]
    ncols, nrows = 3, math.ceil(len(names) / 3)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.4 * nrows))
    label_fs, title_fs, legend_fs, annot_fs = 20, 20, 18, 15
    for k, name in enumerate(names):
        ax = axes[k // ncols, k % ncols]
        e = summary[name]["edge"]
        for g, color, lab in (("dj", "#4169E1", "dj"), ("int", "#FF6347", "int")):
            c = np.array(e["hist"][g][first:], dtype=float)
            dens = c / (c.sum() * BIN)
            ax.hist(bins[:-1], bins=bins, weights=dens, color=color, alpha=0.55,
                    label=lab)
        ax.set_xlim(lo, 1.0)
        tk = ticks(lo)
        ax.set_xticks(tk)
        ax.set_xticklabels(tick_labels(tk))
        ax.set_xlabel("EEC", fontsize=label_fs)
        if k % ncols == 0:
            ax.set_ylabel("probability density", fontsize=label_fs)
        ax.set_title(f"({'ABCDEFGHIJKL'[k]}) {name}", fontsize=title_fs)
        ax.tick_params(axis="both", labelsize=label_fs * 0.6)
        if k == 0:
            ax.legend(loc="lower right", fontsize=legend_fs, frameon=False,
                      bbox_to_anchor=(0.97, 0.05))
        x_anchor, ha = (0.05, "left") if name == "DAWN" else (0.95, "right")
        ax.text(x_anchor, 0.95,
                (f"$\\mu_{{\\rm dj}}={e['mu']['dj']:.3f}$\n"
                 f"$\\mu_{{\\rm int}}={e['mu']['int']:.3f}$\n"
                 f"$d={e['d']['int-dj']:.2f}$"),
                transform=ax.transAxes, fontsize=annot_fs, family="monospace",
                verticalalignment="top", horizontalalignment=ha)
    for k in range(len(names), nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig2.pdf"), format="pdf")
    plt.close(fig)


def boxplot(summary, pair_type, names, classes, colors, labels, centre, gap,
            figsize, out, xlim_from_data, legend_loc, legend_frame):
    stats, positions, box_colors, centers, pos = [], [], [], [], 0.0
    for name in names:
        e = summary[name][pair_type]
        for c in classes:
            stats.append(box_stats(e, c, summary, ""))
            positions.append(pos)
            box_colors.append(colors[c])
            pos += 1.0
        centers.append(pos - len(classes) + centre)
        pos += gap
    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.bxp(stats, positions=positions, widths=0.85, patch_artist=True,
                showfliers=False, vert=False)
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.5)
    ax.set_yticks(centers)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.axvline(0, color="grey", linewidth=0.6, linestyle=":")
    ax.set_ylim(positions[-1] + 1, -1)
    if xlim_from_data:
        ax.set_xlabel("EEC", fontsize=15)
        lo = x_lower([s["whislo"] for s in stats])
        xmax = 1.0 if max(s["whishi"] for s in stats) <= 1.0 else 1.015
        ax.set_xlim(lo - 0.015, xmax)
        tk = ticks(lo)
        ax.set_xticks(tk)
        ax.set_xticklabels(tick_labels(tk))
    else:
        ax.set_xlabel("EEC")
    handles = [Patch(facecolor=colors[c], label=labels[c], alpha=0.75,
                     edgecolor="black") for c in classes]
    ax.legend(handles=handles, loc=legend_loc, fontsize=9, frameon=legend_frame)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, out), format="pdf")
    plt.close(fig)


def main():
    with open(SRC) as f:
        summary = json.load(f)
    fig2(summary)
    three = [n for n in config.TABLE2 if summary[n]["edge"]["subclasses"]]
    boxplot(summary, "edge", three, ["dj", "open", "closed"],
            {"dj": "#4169E1", "open": "#FF6347", "closed": "#2E8B57"},
            {"dj": "dj", "open": "int$_\\mathrm{open}$",
             "closed": "int$_\\mathrm{closed}$"},
            centre=1.0, gap=1.3, figsize=(9, max(4.5, 0.55 * len(three) + 1.5)),
            out="fig3.pdf", xlim_from_data=True, legend_loc="upper right",
            legend_frame=False)
    five = [n for n in config.TABLE4 if summary[n]["h3-h3"]["subclasses"]]
    boxplot(summary, "h3-h3", five,
            ["dj", "int1_open", "int1_closed", "int2_open", "int2_closed"],
            {"dj": "#4169E1", "int1_open": "#FFB347", "int1_closed": "#FF6347",
             "int2_open": "#7AC074", "int2_closed": "#2E8B57"},
            {"dj": "dj", "int1_open": "int$_{1,\\mathrm{open}}$",
             "int1_closed": "int$_{1,\\mathrm{closed}}$",
             "int2_open": "int$_{2,\\mathrm{open}}$",
             "int2_closed": "int$_{2,\\mathrm{closed}}$"},
            centre=2.0, gap=1.6, figsize=(10, max(4.5, 0.7 * len(five) + 1.5)),
            out="fig4.pdf", xlim_from_data=False, legend_loc="lower right",
            legend_frame=True)
    print(f"wrote fig2.pdf, fig3.pdf, fig4.pdf in {FIGS};",
          "Fig. 3 networks:", len(three), "; Fig. 4 hypergraphs:", five)


if __name__ == "__main__":
    main()
