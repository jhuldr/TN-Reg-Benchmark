# TN-Reg-Benchmark
# Code author: Xupeng Zhang (xzhan419@jh.edu)
# Johns Hopkins University
# See README for the full author list and citation.

"""ROI-centered evaluation package for the TN MRI-MRA registration benchmark.

Submodules
----------
config              Paths (env-var driven), method registry, ROI/metric constants.
io_utils            NIfTI/DICOM loading, ROI sampling grid, centroid lookup.
metrics.track_a_intensity   Local image-based metrics (Vessel AUC, ROI NMI, contrast).
metrics.track_b_geometry    Vessel-proximity / geometry metrics + predicted volume.
compute_metrics     Main loop: cohort x 6 methods -> trackA / trackB / skip-log CSVs.
stratify            Contrast tiers, FOV r_min, paired Affine-vs-SyN, bootstrap CIs.
make_tables         Overall metric tables (Table III) -> CSV + LaTeX.
figures.*           Paper figures (Fig.2/3/5/6/7, supplementary, per-case, Fig.1).
"""
