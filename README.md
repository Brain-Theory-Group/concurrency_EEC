# Code for "Quantifying concurrency in event-based temporal network and hypergraph data"

J. Kang, H.-H. Jo, and N. Masuda.

This folder contains the code behind every table, figure, and number of the
paper: the main text, including its appendices, and the Supplementary
Information (SI). The scripts compute exactly the results the paper reports
and nothing else.

Every script begins with a docstring that states which table, figure, or
numbers it produces, what it computes and how, what it writes, and how long
it takes. Fig. 1 is a schematic drawn by hand; no code produces it.

---

## 1. Folders

```
revision_code/
  README.md                  this file
  requirements.txt           Python packages
  run_all.sh                 every analysis script, in order

  lib/                       shared modules (imported, not run)
  data_conversion/           raw downloads -> standard data format (already done)

  table1_datasets/           Table 1                              (main text, Section 3.1.1)
  tables2to4_figs2to4_eec/   Tables 2, 3, 4 and Figs. 2, 3, 4      (main text, Section 3)
  s1_burstiness/             SI Section S1: Table S1, Fig. S1     (main text, Section 4)
  s2_theta_delta/            SI Section S2: Tables S2-S4          (main text, Section 4)
  s3_activity_controls/      SI Section S3: Table S5              (main text, Section 4)
  s4_temporal_motifs/        SI Section S4: Table S6              (main text, Section 5)
  s5_alternative_eec/        SI Section S5: Tables S7-S9          (main text, Appendix B)
```

### `lib/`: the modules shared by all analyses

Every analysis folder loads the same data, classifies pairs of (hyper)edges
by the same rules, and computes the same edge-event correlation (EEC). These
steps are written once, in `lib/`, so that all results use identical
definitions. The scripts import the modules; nothing in `lib/` is run on its
own.

| module | what it contains |
|---|---|
| `config.py` | Settings only, no computation: where the data are and where the figures go; the 22 datasets of Table 1; the window width Delta of each dataset; the event thresholds theta (50 for edges, 25 for size-3 hyperedges); the sample-size criteria; and the datasets of Tables 2-4, which the SI analyses use. |
| `events.py` | Reading a dataset: the frequent (hyper)edges and their event times; the observation period [0, T); the event counts in the time windows (Eq. (1)); the dyadic edges and the node co-occurrences of the aggregate network, used to decide whether a pair is open or closed; and the burstiness B of a (hyper)edge. |
| `pairs.py` | The classification of every pair of frequent (hyper)edges (Fig. 1B): disjoint, intersecting-open, or intersecting-closed for edge pairs and e-h3 pairs; disjoint, 1-intersecting, or 2-intersecting, each open or closed, for h3-h3 pairs. |
| `eec.py` | The EEC (Eqs. (2)-(5)) and the Pearson correlation coefficient (Appendix B.1), computed block by block for all classified pairs; per-class summaries (number of pairs, mean, variance); Cohen's d; partial correlations; and the sample-size criteria that decide which datasets enter Tables 2-4. |

### `data_conversion/`: preparing the five pairwise networks

The analyses read every dataset in one standard format (Section 2 below).
The five scripts in `data_conversion/` produced that format for the five
pairwise temporal networks (the first five rows of Table 1) from the files
as downloaded from their sources, which are kept in `data/raw/`: they keep
calls and messages between study participants, drop self-loops, make each
record undirected, and shift the time stamps to start at 0. The converted
data are already in `data/networks/`, so these scripts need not be rerun;
they are kept to document the conversion. They are unchanged since the first
submission of the paper. See `data_conversion/README.md`.

### The analysis folders

Each folder computes one group of results and is named after them. It holds
its scripts and a `data/` folder with what the scripts compute (JSON). These
files contain every value of the corresponding tables (unrounded) and, under
the key `"quoted"`, the numbers that the text quotes (ranges, counts of
datasets, correlations). The tables of the paper were typed from these
files.

| folder | what it computes |
|---|---|
| `table1_datasets/` | Table 1: for each of the 22 datasets, the numbers of nodes, edges, and frequent edges, the time resolution, the observation span, and the numbers of disjoint and intersecting pairs of frequent edges. |
| `tables2to4_figs2to4_eec/` | The main results (Section 3): the EEC of every pair of frequent (hyper)edges in every dataset, summarized per pair class into the means and Cohen's d of Tables 2 (edge pairs), 3 (e-h3 pairs), and 4 (h3-h3 pairs), and the distributions shown in Figs. 2-4. |
| `s1_burstiness/` | SI Section S1: the correlation between the EEC of intersecting pairs and the burstiness of the two (hyper)edges, with and without controlling for their numbers of events (Table S1), and its comparison across datasets (Fig. S1). |
| `s2_theta_delta/` | SI Section S2: Tables 2-4 recomputed for two thresholds theta and three window widths Delta (Tables S2-S4). |
| `s3_activity_controls/` | SI Section S3: Cohen's d adjusted, by linear regression, for the numbers of events and the burstiness of the two (hyper)edges (Table S5). |
| `s4_temporal_motifs/` | SI Section S4: the standard analysis of two-event temporal motifs (Kovanen et al., 2011) on the temporal networks, compared with time-shuffled networks (Table S6). |
| `s5_alternative_eec/` | Appendix B and SI Section S5: Cohen's d of Tables 2-4 under three alternative definitions of the EEC, the Pearson correlation coefficient, the earth mover's distance, and the spike time tiling coefficient (Tables S7-S9). |

## 2. Data

The scripts read the folder `data/` next to this folder (set the environment
variable `NETWORKEEC_DATA` to use another location). It is described in
`data/README.md`:

```
data/
  networks/      the five pairwise temporal networks (Table 1, first five rows)
  hypergraphs/   the 17 temporal hypergraphs of Benson et al., PNAS 115, E11221 (2018)
  raw/           the original downloads of the five networks (used only by data_conversion/)
```

Every dataset folder holds two parallel files with one row per (hyper)edge:

| file | format |
|---|---|
| `edges_list.txt` | `eid size n1 n2 [n3 ...]`: the id, the size, and the nodes of the (hyper)edge |
| `edges_timings.txt` | `eid size t1 t2 ...`: the times of its events, as integers in the native time unit of the dataset; the first event of the dataset is at 0 |

## 3. Running the code

Run the scripts from this folder, in the order of `run_all.sh`, or run

```bash
sh run_all.sh
```

which takes about 40 min on 8 cores with 16 GB of memory (most of it for
tags-stack-overflow, whose 8,126 frequent edges give about 10^8 pairs).

| result | folder | scripts, in order | output | time |
|---|---|---|---|---|
| Table 1 | `table1_datasets/` | `table1.py` | `data/table1.json` | 1 min |
| Tables 2-4, Figs. 2-4 | `tables2to4_figs2to4_eec/` | `compute_eec.py`, `plot_figs2to4.py` | `data/eec_results.json`; `fig2.pdf`, `fig3.pdf`, `fig4.pdf` | 12 min |
| Table S1, Fig. S1 | `s1_burstiness/` | `burstiness_analysis.py`, `plot_fig_s1.py` | `data/results.json`; `fig_s1.pdf` | 1 min |
| Tables S2-S4 | `s2_theta_delta/` | `theta_delta_sweep.py` | `data/results.json` | 5 min |
| Table S5 | `s3_activity_controls/` | `controlled_regression.py` | `data/results.json` | 3 min |
| Table S6 | `s4_temporal_motifs/` | `temporal_motifs.py` | `data/results.json` | 2 min |
| Tables S7-S9 | `s5_alternative_eec/` | `compute_alternatives.py` (with the module `event_measures.py`) | `data/results.json` | 15 min |

`compute_eec.py` must run before `s4_temporal_motifs/` and
`s5_alternative_eec/`, which compare their results with the EEC. It also
checks that the datasets passing the sample-size criteria are those listed in
`lib/config.py`, which the SI analyses use.

The figure scripts write the figure files into the folders of the
manuscript, `../revision_main/` (Figs. 2-4) and `../revision_SI/` (Fig. S1),
so that each figure exists only once. If those folders do not exist, for
example in a copy of this folder downloaded on its own, the figures go to
`revision_code/figures/`. The environment variables `NETWORKEEC_MAIN_FIGS`
and `NETWORKEEC_SI_FIGS` choose other folders.

## 4. The computation in brief

For each dataset, with its window width Delta (`lib/config.py`):

1. **Frequent (hyper)edges** (`events.load_frequent`): edges with at least
   50 events and size-3 hyperedges with at least 25.
2. **Pair classes** (`pairs.py`): every pair of frequent (hyper)edges is
   disjoint or intersecting; intersecting pairs are open or closed, and
   h3-h3 pairs are 1- or 2-intersecting (Fig. 1B; Sections 3.1.2, 3.2.1).
3. **Window counts** (`events.window_counts`, Eq. (1)): the observation
   period [0, T), with T one plus the time of the latest event of the
   dataset, is divided into l_max = floor(T / Delta) windows of width Delta,
   and the events of each (hyper)edge are counted in each window.
4. **EEC** (`eec.py`, Eqs. (2)-(5)): the cosine similarity C_ij of two count
   vectors, corrected by its value C0_ij for independent event sequences,
   S_ij = (C_ij - C0_ij) / (1 - C0_ij)
   = Cov(x_i, x_j) / (sqrt(<x_i^2><x_j^2>) - <x_i><x_j>).
5. **Comparison** (`eec.py`): Cohen's d between pair classes, for the
   datasets that pass the sample-size criteria.

Only the label matrix of the pairs (one byte per pair) is held in memory in
full; the EEC is computed for blocks of pairs and summarized per class as it
goes.

## 5. Requirements

Python 3.9 or later with numpy, scipy, matplotlib, and numba
(`requirements.txt`; tested with Python 3.9.6, numpy 2.0.2, scipy 1.13.1,
matplotlib 3.9.4, and numba 0.60.0). numba is used for the pair-by-pair
earth mover's distance and spike time tiling coefficient in
`s5_alternative_eec/`.

## 6. Reproducibility

All computations are deterministic except for the time-shuffled networks of
`s4_temporal_motifs/`, whose random number generator is seeded separately
for each network. Rerunning the scripts therefore reproduces every file in
the `data/` folders exactly. The figure files differ between runs only in
their creation-date metadata.
