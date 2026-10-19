#!/bin/sh
# Every analysis script of the paper, in order. Run from this folder
# (revision_code/): sh run_all.sh
# About 40 min on 8 cores with 16 GB of memory.
set -e
python3 table1_datasets/table1.py
python3 tables2to4_figs2to4_eec/compute_eec.py
python3 tables2to4_figs2to4_eec/plot_figs2to4.py
python3 s1_burstiness/burstiness_analysis.py
python3 s1_burstiness/plot_fig_s1.py
python3 s2_theta_delta/theta_delta_sweep.py
python3 s3_activity_controls/controlled_regression.py
python3 s4_temporal_motifs/temporal_motifs.py
python3 s5_alternative_eec/compute_alternatives.py
