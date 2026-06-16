# TN-Reg-Benchmark
# Code author: Xupeng Zhang (xzhan419@jh.edu)
# Johns Hopkins University
# See README for the full author list and citation.

"""Figure generation for the paper (reads the metric CSVs, writes PNGs).

Modules:
  overview_figures    Fig.2 (intensity) + Fig.3 (geometry) + per-metric supp figs.
  coupling_figures    Fig.5 (predicted-volume vs distance coupling).
  stratified_figures  Fig.6 (contrast-stratified) + Fig.7 (FOV-stratified paired).
  case_figures        Per-case operating-regime / failure-mode grids (Fig.4/Fig.8),
                      parameterized by a de-identified case id.
  figure1_pipeline    Fig.1 pipeline illustration assets, parameterized similarly.
"""
