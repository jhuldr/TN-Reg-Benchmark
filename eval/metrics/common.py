# TN-Reg-Benchmark
# Code author: Xupeng Zhang (xzhan419@jh.edu)
# Johns Hopkins University
# See README for the full author list and citation.

"""Shared metric primitives."""
from __future__ import annotations

import numpy as np


def dice_score(pred, gt) -> float:
    """Sorensen-Dice overlap between two boolean masks."""
    pred_b, gt_b = np.asarray(pred, bool), np.asarray(gt, bool)
    inter = (pred_b & gt_b).sum()
    denom = pred_b.sum() + gt_b.sum()
    return float(2 * inter / denom) if denom > 0 else 0.0
