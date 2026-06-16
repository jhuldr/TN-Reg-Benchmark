# TN-Reg-Benchmark
# Code author: Xupeng Zhang (xzhan419@jh.edu)
# Johns Hopkins University
# See README for the full author list and citation.

"""Central configuration for the ROI-centered evaluation.

All paths are driven by the ``REG_ROOT`` environment variable (the project root
that holds preprocessed data, registration outputs, and VesselFM predictions).
Set it before running, e.g.::

    export REG_ROOT=/path/to/TN_Reg
    export VFM_ROOT=/path/to/vesselfm_results   # optional, defaults under REG_ROOT

The expected on-disk layout is documented in eval/README.md.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths (override via environment variables)
# --------------------------------------------------------------------------
REG_ROOT = Path(os.environ.get("REG_ROOT", os.path.expanduser("~/TN_Reg")))

# Clinician-annotated ROI crops + centroid CSVs (TIFF stacks)
TN_ROOT = Path(os.environ.get("TN_ROOT", REG_ROOT / "2020-22_MRIs_Cropped_and_Segmentations"))
# Preprocessed cropped MRI used as the spatial reference for ROI sampling
CROPPED_MRI = REG_ROOT / "data/preprocessed_cropped/mri"
# Raw DICOM (only headers/affine are read, to map clinician centroids into RAS)
DICOM_ROOT = REG_ROOT / "data/raw/mri"
# Whole-brain warped-MRA outputs, one subdir per method
OUT_BASE = REG_ROOT / "outputs/MRIfixed_MRAmoving"
# VesselFM whole-brain vessel predictions
VFM_BASE = Path(os.environ.get("VFM_ROOT", os.path.expanduser("~/vesselfm_results")))
# Where eval CSVs and figures are written
EVAL_DIR = REG_ROOT / "eval_result"
PAPER_DIR = EVAL_DIR / "paperimage"

# --------------------------------------------------------------------------
# Method registry (the six evaluated registration pipelines)
# --------------------------------------------------------------------------
METHOD_ORDER = [
    "ANTs_Affine", "ANTs_SyN", "ConvexAdam", "FireANTs", "EasyReg", "SynthMorph",
]

# Short labels used in figures/tables (matches the manuscript)
METHOD_ABBR = {
    "ANTs_Affine": "Affine", "ANTs_SyN": "SyN", "ConvexAdam": "CvxAd",
    "FireANTs": "Fire", "EasyReg": "Easy", "SynthMorph": "Synth",
}

# Warped-MRA directory per method, relative to OUT_BASE
WARPED_SUBDIR = {
    "ANTs_Affine": "ANTs_result/warped",
    "ANTs_SyN": "ANTs_result_syn_fixedmask/warped",
    "ConvexAdam": "ConvexAdam_result/warped",
    "FireANTs": "FireANTs_result/warped",
    "EasyReg": "EasyReg/warped",
    "SynthMorph": "SynthMorph/warped",
}
# VesselFM prediction directory per method, relative to VFM_BASE
VFM_SUBDIR = {
    "ANTs_Affine": "vesselfm_ANTs_Affine_fixed",
    "ANTs_SyN": "vesselfm_ants_fixed",
    "ConvexAdam": "vesselfm_ConvexAdam_fixed",
    "FireANTs": "vesselfm_FireANTs_fixed",
    "EasyReg": "vesselfm_EasyReg_fixed",
    "SynthMorph": "vesselfm_SynthMorph_fixed",
}
WARPED_PAT = "{cid}_reg_Warped.nii.gz"
VFM_PAT = "{cid}_vessel_mask.nii.gz"


def method_dirs() -> dict[str, Path]:
    """Absolute warped-MRA dir per method."""
    return {m: OUT_BASE / sub for m, sub in WARPED_SUBDIR.items()}


def vfm_dirs(require_nonempty: bool = True) -> dict[str, Path]:
    """Absolute VesselFM dir per method (optionally only those that exist)."""
    out = {}
    for m in METHOD_ORDER:
        d = VFM_BASE / VFM_SUBDIR[m]
        if not require_nonempty or (d.exists() and any(d.glob("*.nii.gz"))):
            out[m] = d
    return out


# --------------------------------------------------------------------------
# ROI / metric constants (manuscript Section II-D, III-A)
# --------------------------------------------------------------------------
CUBE_SIZE = 48          # matched 48^3 trigeminal ROI
SPACING = 0.47          # mm, isotropic ROI sampling grid
PERCENTILE_THR = 75.0   # threshold percentile for the auxiliary vessel-Dice
MIN_PATCH_NZ = 0.05     # skip ROIs with <5% non-zero warped-MRA coverage

# Stratification thresholds (manuscript Section II-E)
CONTRAST_LOW = 1.1      # contrast <= 1.1 -> Low tier
CONTRAST_HIGH = 1.3     # contrast > 1.3 -> High tier (Mid in between)
FOV_RMIN_BAD = 0.76     # r_min <= 0.76 (20th percentile) -> Bad FOV

# Bootstrap (manuscript Section II-E)
N_BOOT = 1000
BOOT_SEED = 404

# Consistent per-method colors (the published-figure palette)
COLORS = {
    "ANTs_Affine": "#4C78A8", "ANTs_SyN": "#72B7B2",
    "ConvexAdam": "#54A24B", "FireANTs": "#9D755D",
    "EasyReg": "#B279A2", "SynthMorph": "#E45756",
}
# Tier colors for the contrast-stratified figure
TIER_COLORS = ["#4C78A8", "#888888", "#E45756"]
TIER_LABELS = ["Low", "Mid", "High"]

SIDES = ("ipsi", "contra")
SIDE_NAME = {"ipsi": "Ipsilateral", "contra": "Contralateral"}
