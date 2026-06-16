# TN-Reg-Benchmark
# Code author: Xupeng Zhang (xzhan419@jh.edu)
# Johns Hopkins University
# See README for the full author list and citation.

"""Shared helpers for figure generation: data loading, FOV ratios, style."""
from __future__ import annotations

import numpy as np
import pandas as pd
import nibabel as nib

from .. import config

METHOD_ORDER = config.METHOD_ORDER
ABBR = config.METHOD_ABBR
COLORS = config.COLORS


def apply_paper_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 10, "figure.dpi": 120,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def load_tracks(eval_dir=None):
    """Load Track A / Track B CSVs; derive symmetric_mm. Returns (df_a, df_b)."""
    eval_dir = eval_dir or config.EVAL_DIR
    df_a = pd.read_csv(eval_dir / "eval_trackA_all_methods.csv", dtype={"mrn": str})
    df_b = pd.read_csv(eval_dir / "eval_trackB_all_methods.csv", dtype={"mrn": str})
    df_a["mrn"] = df_a["mrn"].str.zfill(8)
    df_b["mrn"] = df_b["mrn"].str.zfill(8)
    if "symmetric_mm" not in df_b.columns:
        df_b["symmetric_mm"] = (df_b["gt2pred_mean_mm"] + df_b["pred2gt_mean_mm"]) / 2
    return df_a, df_b


def compute_fov(mrns, cropped_mri=None, cropped_mra=None) -> pd.DataFrame:
    """Per-case MRI/MRA FOV ratios -> df with min_ratio (manuscript Eq. 2)."""
    cropped_mri = cropped_mri or config.CROPPED_MRI
    cropped_mra = cropped_mra or (config.REG_ROOT / "data/preprocessed_cropped/mra")
    rows = []
    for mrn in sorted(set(mrns)):
        mri_path = cropped_mri / f"{mrn}.nii.gz"
        mra_path = cropped_mra / f"{mrn}.nii.gz"
        if not mri_path.exists() or not mra_path.exists():
            continue
        mri_nii, mra_nii = nib.load(str(mri_path)), nib.load(str(mra_path))
        mri_fov = np.array(mri_nii.shape) * np.abs(np.diag(mri_nii.affine)[:3])
        mra_fov = np.array(mra_nii.shape) * np.abs(np.diag(mra_nii.affine)[:3])
        ratio = mri_fov / mra_fov
        rows.append({"mrn": str(mrn).zfill(8), "min_ratio": float(ratio.min())})
    return pd.DataFrame(rows)


def assign_contrast_tier(df_a):
    """Tier each ROI by ANTs_SyN local contrast (canonical reference)."""
    bins = [-np.inf, config.CONTRAST_LOW, config.CONTRAST_HIGH, np.inf]
    df_c = df_a[df_a["method"] == "ANTs_SyN"][["mrn", "side", "contrast_A"]].copy()
    df_c["tier"] = pd.cut(df_c["contrast_A"], bins=bins, labels=config.TIER_LABELS)
    return df_c
