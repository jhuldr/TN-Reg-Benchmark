# TN-Reg-Benchmark
# Code author: Xupeng Zhang (xzhan419@jh.edu)
# Johns Hopkins University
# See README for the full author list and citation.

"""Track B: segmentation-derived vessel-proximity / geometry metrics.

Given the clinician vessel annotation and the VesselFM prediction sampled in the
same ROI, computes directional surface distances and predicted vessel volume:
  * gt2pred_mean_mm  -  d(ann->Pred), annotation covered by prediction (Eq. 5)
  * pred2gt_mean_mm  -  d(Pred->ann), prediction extent vs annotation  (Eq. 6)
  * d_sym            -  symmetric average (Eq. 7), derived downstream
  * pred_vox         -  predicted vessel volume |V_Pred| (auxiliary, Section III-D)

Predicted volume is tracked because, under partial clinical annotations, the
one-sided distances are confounded by how voluminous the prediction is.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation
from scipy.ndimage import label as ndlabel

from .common import dice_score
from .. import config

TRACK_B_COLS = [
    "dice", "dilated_dice", "gt2pred_mean_mm", "gt2pred_median_mm", "gt2pred_p95_mm",
    "pred2gt_mean_mm", "hit_1mm", "hit_2mm", "hit_3mm",
    "cc_dice", "n_pred_cc", "contrast_B", "gt_vox", "pred_vox",
]


def compute_track_b(gt_vessel, pred_vessel, mra_crop, gt_nerve, spacing=config.SPACING) -> dict:
    """Vessel-proximity metrics for one (case, side, method) ROI."""
    out: dict = {}
    out["dice"] = dice_score(pred_vessel, gt_vessel)

    gt_dil = binary_dilation(gt_vessel, iterations=2)
    pred_dil = binary_dilation(pred_vessel, iterations=2)
    out["dilated_dice"] = dice_score(pred_dil, gt_dil)

    if gt_vessel.sum() > 0 and pred_vessel.sum() > 0:
        pred_dt = distance_transform_edt(~pred_vessel) * spacing
        gt_dt = distance_transform_edt(~gt_vessel) * spacing
        d_gt2pred = pred_dt[gt_vessel]
        d_pred2gt = gt_dt[pred_vessel]
        out["gt2pred_mean_mm"] = float(d_gt2pred.mean())
        out["gt2pred_median_mm"] = float(np.median(d_gt2pred))
        out["gt2pred_p95_mm"] = float(np.percentile(d_gt2pred, 95))
        out["pred2gt_mean_mm"] = float(d_pred2gt.mean())
        out["hit_1mm"] = float((d_gt2pred < 1.0).mean())
        out["hit_2mm"] = float((d_gt2pred < 2.0).mean())
        out["hit_3mm"] = float((d_gt2pred < 3.0).mean())
    else:
        for k in ["gt2pred_mean_mm", "gt2pred_median_mm", "gt2pred_p95_mm",
                  "pred2gt_mean_mm", "hit_1mm", "hit_2mm", "hit_3mm"]:
            out[k] = np.nan

    # Connected-component Dice against the component nearest the annotation
    if gt_vessel.sum() > 0 and pred_vessel.sum() > 0:
        gt_cen = np.array(np.where(gt_vessel)).mean(axis=1)
        labeled, n_cc = ndlabel(pred_vessel)
        best = 0.0
        for cc in range(1, n_cc + 1):
            cc_mask = (labeled == cc)
            if np.linalg.norm(np.array(np.where(cc_mask)).mean(axis=1) - gt_cen) * spacing < 5.0:
                best = max(best, dice_score(cc_mask, gt_vessel))
        out["cc_dice"] = float(best)
        out["n_pred_cc"] = int(n_cc)
    else:
        out["cc_dice"] = np.nan
        out["n_pred_cc"] = 0

    bg = mra_crop[~gt_vessel & ~gt_nerve]
    vl = mra_crop[gt_vessel]
    out["contrast_B"] = float(vl.mean() / bg.mean()) if bg.mean() > 0 and vl.size > 0 else 0.0
    out["gt_vox"] = int(gt_vessel.sum())
    out["pred_vox"] = int(pred_vessel.sum())
    return out


def add_symmetric_distance(df):
    """Add d_sym_mm = 0.5*(gt2pred + pred2gt) to a Track B dataframe (Eq. 7)."""
    df = df.copy()
    df["d_sym_mm"] = 0.5 * (df["gt2pred_mean_mm"] + df["pred2gt_mean_mm"])
    return df
