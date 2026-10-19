"""Configuration shared by every script: where the data are, which datasets
are analyzed, and the parameters of the paper.

Nothing in this module computes anything. Every number in it is a choice
stated in the paper:

  * DELTA     the time-window width Delta of each dataset (column "Delta" of
              Table 2; the same value is used for the hypergraph analyses);
  * THETA     the threshold theta on the number of events for a frequent
              edge (50) or a frequent size-3 hyperedge (25);
  * MIN_*     the sample-size criteria of Sections 3.1.1 and 3.2.1;
  * TABLE2/3/4  the datasets that pass those criteria, i.e., the rows of
              Tables 2, 3, and 4. They are written out here because the SI
              sections and Appendix B analyze exactly these datasets;
              tables2to4_figs2to4_eec/compute_eec.py re-derives them from the
              criteria and stops if they differ.
"""

import os

# ---------------------------------------------------------------------------
# Location of the data
# ---------------------------------------------------------------------------
# The data folder (see README.md) has the subfolders networks/ and
# hypergraphs/ and sits next to this code folder: <project>/data and
# <project>/revision_code. The environment variable NETWORKEEC_DATA points to
# another location.

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _find_data_dir():
    env = os.environ.get("NETWORKEEC_DATA")
    candidates = ([env] if env else []) + [os.path.join(CODE_DIR, "..", "data")]
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "networks")) \
                and os.path.isdir(os.path.join(c, "hypergraphs")):
            return os.path.abspath(c)
    raise FileNotFoundError(
        "data folder not found; set NETWORKEEC_DATA to the folder that "
        "contains networks/ and hypergraphs/")


DATA_DIR = _find_data_dir()

# Where the figure scripts write the figure files. In the authors' project the
# figures go straight into the folders of the manuscript, <project>/revision_main
# (main text) and <project>/revision_SI (SI), so that each figure file exists
# only once. Where those folders do not exist (e.g., in a copy of this code
# folder downloaded from GitHub), the figures go to revision_code/figures/.
# The environment variables NETWORKEEC_MAIN_FIGS and NETWORKEEC_SI_FIGS
# choose other folders.


def _fig_dir(env, manuscript_folder):
    d = os.environ.get(env) or os.path.join(CODE_DIR, "..", manuscript_folder)
    if not os.path.isdir(d):
        d = os.path.join(CODE_DIR, "figures")
        os.makedirs(d, exist_ok=True)
    return os.path.abspath(d)


MAIN_FIG_DIR = _fig_dir("NETWORKEEC_MAIN_FIGS", "revision_main")
SI_FIG_DIR = _fig_dir("NETWORKEEC_SI_FIGS", "revision_SI")

# The five genuinely pairwise temporal networks live in data/networks/; the
# 17 temporal hypergraphs (Benson et al., PNAS 2018) in data/hypergraphs/.
PAIRWISE_NETWORKS = (
    "ia-reality-call", "copenhagen-calls", "college-msg",
    "friends-family-call", "social-evolution-call",
)


def dataset_folder(name):
    """Absolute path of the folder holding edges_list.txt and
    edges_timings.txt of a dataset."""
    sub = "networks" if name in PAIRWISE_NETWORKS else "hypergraphs"
    return os.path.join(DATA_DIR, sub, name)


# ---------------------------------------------------------------------------
# Datasets and parameters
# ---------------------------------------------------------------------------

# The 22 datasets of Table 1, in the order of Table 1.
DATASETS = [
    "ia-reality-call", "copenhagen-calls", "college-msg",
    "friends-family-call", "social-evolution-call",
    "contact-high-school", "contact-primary-school", "DAWN",
    "NDC-classes", "NDC-substances",
    "coauth-DBLP", "coauth-MAG-Geology", "coauth-MAG-History",
    "congress-bills", "email-Enron", "email-Eu",
    "tags-ask-ubuntu", "tags-math-sx", "tags-stack-overflow",
    "threads-ask-ubuntu", "threads-math-sx", "threads-stack-overflow",
]

# Time-window width Delta in the native time unit of each dataset
# (Section 3.1.1): such that the observation period contains about 200
# windows (Delta ~ T/200), except Delta = 1 for datasets whose native
# resolution is already coarse (e.g., DAWN, quarters; coauth-*, years), and a
# natural unit for the datasets with time stamps in seconds: one hour for
# ia-reality-call and copenhagen-calls, one day for college-msg,
# friends-family-call, social-evolution-call, and email-Eu.
DELTA = {
    "ia-reality-call": 3600, "copenhagen-calls": 3600,
    "college-msg": 86400, "friends-family-call": 86400,
    "social-evolution-call": 86400,
    "contact-high-school": 90, "contact-primary-school": 30, "DAWN": 1,
    "NDC-classes": 200, "NDC-substances": 200,
    "coauth-DBLP": 1, "coauth-MAG-Geology": 1, "coauth-MAG-History": 1,
    "congress-bills": 60, "email-Enron": 7, "email-Eu": 86400,
    "tags-ask-ubuntu": 400, "tags-math-sx": 325, "tags-stack-overflow": 400,
    "threads-math-sx": 311, "threads-stack-overflow": 400,
}
# threads-ask-ubuntu has no frequent edge (Table 1), so no Delta is needed.

# Frequent (hyper)edges: at least THETA[k] events on a hyperedge of size k.
THETA = {2: 50, 3: 25}

# Sample-size criteria (Sections 3.1.1 and 3.2.1).
MIN_E_FREQ_EDGES = 50   # Table 2: E_freq >= 50
MIN_E_FREQ_HYPER = 40   # Table 3: E_2,freq + E_3,freq >= 40; Table 4: E_3,freq >= 40
MIN_PAIRS = 20          # n_dj, n_int (and n_int1, n_int2 for Table 4) >= 20
MIN_SUBCLASS = 10       # n_open, n_closed (and the four h3-h3 subclasses) >= 10

# Rows of Tables 2, 3, and 4 (re-derived and checked by compute_eec.py).
TABLE2 = ["ia-reality-call", "college-msg", "friends-family-call",
          "contact-high-school", "contact-primary-school", "DAWN",
          "congress-bills", "email-Eu", "tags-ask-ubuntu", "tags-math-sx",
          "tags-stack-overflow"]
TABLE3 = ["contact-high-school", "contact-primary-school", "DAWN",
          "NDC-substances", "email-Eu", "tags-ask-ubuntu", "tags-math-sx",
          "tags-stack-overflow"]
TABLE4 = ["DAWN", "NDC-substances", "email-Eu", "tags-ask-ubuntu",
          "tags-math-sx", "tags-stack-overflow"]

# Pair types, as named in the paper.
PAIR_TYPES = ("edge", "e-h3", "h3-h3")
TABLE_OF = {"edge": TABLE2, "e-h3": TABLE3, "h3-h3": TABLE4}
