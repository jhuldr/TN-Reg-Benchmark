# TN-Reg-Benchmark
# Code author: Xupeng Zhang (xzhan419@jh.edu)
# Johns Hopkins University
# See README for the full author list and citation.

"""Track A: local image-based metrics inside the trigeminal ROI.

Computes, on the warped MRA sampled within the matched 48^3 ROI:
  * vessel_AUC   - separability of clinician vessel voxels vs background (Eq. 3)
  * roi_NMI      - normalized mutual information between MRI and warped MRA (Eq. 4)
  * contrast_A   - local vessel-to-background intensity ratio (Eq. 1)
  * auxiliary diagnostics (CNR, Dice@p75, intensity ratio, MI, coverage)

The ground-truth mask encodes: 1 = trigeminal nerve, 2 = vessel.
"""
from __future__ import annotations

import numpy as np

from .common import dice_score
from .. import config

TRACK_A_COLS = [
    "vessel_AUC", "vessel_CNR", "vessel_Dice",
    "intensity_ratio", "roi_MI", "roi_NMI", "contrast_A", "patch_nz_frac",
]


def mutual_information(a, b, bins=32) -> float:
    af, bf = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    mask = (af != 0) | (bf != 0)
    if mask.sum() < 50:
        return 0.0
    af, bf = af[mask], bf[mask]
    h, _, _ = np.histogram2d(af, bf, bins=bins)
    s = h.sum()
    if s <= 0:
        return 0.0
    pxy = h / s
    px, py = pxy.sum(1), pxy.sum(0)
    nz = pxy > 0
    denom = (px[:, None] * py[None, :])[nz]
    return float(np.sum(pxy[nz] * np.log(pxy[nz] / denom)))


def normalized_mutual_information(a, b, bins=32) -> float:
    af, bf = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    mask = (af != 0) | (bf != 0)
    if mask.sum() < 50:
        return 0.0
    af, bf = af[mask], bf[mask]
    h, _, _ = np.histogram2d(af, bf, bins=bins)
    s = h.sum()
    if s <= 0:
        return 0.0
    pxy = h / s
    px, py = pxy.sum(1), pxy.sum(0)

    def H(p):
        nz = p[p > 0]
        return float(-np.sum(nz * np.log(nz)))

    h_ab = H(pxy.ravel())
    return float((H(px) + H(py)) / h_ab) if h_ab > 0 else 0.0


def robust_sigma(nerve_vals, labeled_vals, min_n=20, floor=1e-3) -> float:
    """Robust noise estimate for CNR (MAD of nerve voxels, with fallbacks)."""
    nv_nz = nerve_vals[nerve_vals > 0]
    sigma = 0.0
    if nv_nz.size >= min_n:
        sigma = float(1.4826 * np.median(np.abs(nv_nz - np.median(nv_nz))))
    if sigma < floor:
        sigma = float(nv_nz.std()) if nv_nz.size > 0 else 0.0
    if sigma < floor:
        lv_nz = labeled_vals[labeled_vals > 0]
        sigma = float(lv_nz.std()) if lv_nz.size > 0 else 0.0
    return sigma if sigma >= floor else float("nan")


def compute_track_a(mra_patch, mri_patch, gt_mask) -> dict:
    """Image-based metrics for one (case, side, method) ROI."""
    nerve_mask = (gt_mask >= 0.5) & (gt_mask < 1.5)
    vessel_mask = (gt_mask >= 1.5)
    labeled_mask = (gt_mask >= 0.5)
    nerve_vals = mra_patch[nerve_mask]
    vessel_vals = mra_patch[vessel_mask]
    labeled_vals = mra_patch[labeled_mask]
    nz_frac = float((mra_patch > 0).mean())
    nerve_mean = float(nerve_vals.mean()) if nerve_vals.size > 0 else 0.0
    vessel_mean = float(vessel_vals.mean()) if vessel_vals.size > 0 else 0.0

    r = {"patch_nz_frac": nz_frac, "nerve_mean": nerve_mean, "vessel_mean": vessel_mean}
    if nz_frac < config.MIN_PATCH_NZ:
        for k in ["vessel_AUC", "vessel_CNR", "vessel_Dice",
                  "intensity_ratio", "roi_MI", "roi_NMI", "contrast_A"]:
            r[k] = np.nan
        return r

    # Auxiliary vessel Dice at the p75 intensity threshold
    mra_nz = mra_patch[mra_patch > 0]
    thr = np.percentile(mra_nz, config.PERCENTILE_THR) if mra_nz.size > 10 else 0.0
    r["vessel_Dice"] = dice_score(mra_patch > thr, vessel_mask)

    # CNR
    sigma = robust_sigma(nerve_vals, labeled_vals)
    r["vessel_CNR"] = abs(vessel_mean - nerve_mean) / sigma if np.isfinite(sigma) and sigma > 0 else np.nan

    # Intensity ratio + local contrast (Eq. 1)
    r["intensity_ratio"] = vessel_mean / max(nerve_mean, 1e-6)
    bg = mra_patch[~nerve_mask & ~vessel_mask]
    r["contrast_A"] = float(vessel_mean / bg.mean()) if bg.mean() > 0 and vessel_vals.size > 0 else 0.0

    # Vessel AUC (Eq. 3): rank-based separability of vessel vs non-vessel labeled voxels
    if labeled_mask.sum() > 10:
        vlabel = (gt_mask[labeled_mask] >= 1.5).astype(int)
        mra_l = mra_patch[labeled_mask]
        pos, neg = mra_l[vlabel == 1], mra_l[vlabel == 0]
        if pos.size > 0 and neg.size > 0:
            total = sum((neg < p).sum() + 0.5 * (neg == p).sum() for p in pos)
            r["vessel_AUC"] = float(total / (pos.size * neg.size))
        else:
            r["vessel_AUC"] = 0.5
    else:
        r["vessel_AUC"] = 0.5

    # ROI NMI (Eq. 4)
    if mri_patch is not None:
        r["roi_MI"] = mutual_information(mri_patch, mra_patch)
        r["roi_NMI"] = normalized_mutual_information(mri_patch, mra_patch)
    else:
        r["roi_MI"] = r["roi_NMI"] = np.nan
    return r
