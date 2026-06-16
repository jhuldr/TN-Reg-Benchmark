# TN-Reg-Benchmark
# Code author: Xupeng Zhang (xzhan419@jh.edu)
# Johns Hopkins University
# See README for the full author list and citation.

"""Stratified statistics with patient-level bootstrap confidence intervals.

Reproduces the inferential numbers reported in the manuscript:
  * volume-distance Spearman correlations (Section III-D)
  * Mid-tier proportion with AUC_SyN > 0.60 (Section III-F)
  * FOV-stratified paired Affine-vs-SyN median delta (Section III-G)

Patient-level resampling (1000 resamples) accounts for the two ROIs per patient.

Run:  REG_ROOT=/path/to/TN_Reg python -m eval.stratify
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.stats import spearmanr

from . import config
from .figures._common import load_tracks, compute_fov


def _resample_patients(df, rng, patient_col="mrn"):
    patients = df[patient_col].unique()
    sampled = rng.choice(patients, size=len(patients), replace=True)
    counts = pd.Series(sampled).value_counts()
    out = []
    for p, c in counts.items():
        rows = df[df[patient_col] == p]
        for _ in range(c):
            out.append(rows)
    return pd.concat(out, ignore_index=True)


def volume_distance_correlations(df_b, rng, n_boot=config.N_BOOT):
    """Spearman rho (log volume vs distance) with bootstrap CI, per distance."""
    df_b = df_b.copy()
    df_b["symmetric_mm"] = 0.5 * (df_b["gt2pred_mean_mm"] + df_b["pred2gt_mean_mm"])
    cols = {"d_GT->Pred": "gt2pred_mean_mm", "d_Pred->GT": "pred2gt_mean_mm", "d_sym": "symmetric_mm"}
    results = {}
    for label, dist_col in cols.items():
        sub = df_b[["pred_vox", dist_col, "mrn"]].dropna()
        sub = sub[sub["pred_vox"] > 0]
        rho, _ = spearmanr(np.log(sub["pred_vox"]), sub[dist_col])
        rhos = []
        for _ in range(n_boot):
            sb = _resample_patients(sub, rng)
            rb, _ = spearmanr(np.log(sb["pred_vox"]), sb[dist_col])
            rhos.append(rb)
        lo, hi = np.percentile(rhos, [2.5, 97.5])
        results[label] = (rho, lo, hi, len(sub))
    return results


def midtier_auc_proportion(df_a, rng, n_boot=config.N_BOOT):
    """Proportion of Mid-tier ANTs_SyN ROIs with Vessel AUC > 0.60, + bootstrap CI."""
    syn = df_a[df_a["method"] == "ANTs_SyN"].dropna(subset=["contrast_A", "vessel_AUC"])
    mid = syn[(syn["contrast_A"] > config.CONTRAST_LOW) & (syn["contrast_A"] <= config.CONTRAST_HIGH)]
    point = (mid["vessel_AUC"] > 0.60).mean()
    props = []
    for _ in range(n_boot):
        mb = _resample_patients(mid, rng)
        if len(mb):
            props.append((mb["vessel_AUC"] > 0.60).mean())
    lo, hi = np.percentile(props, [2.5, 97.5])
    return point, lo, hi, len(mid)


def fov_paired_delta(df_b, df_fov, rng, n_boot=config.N_BOOT):
    """Median paired (SyN-Affine) GT->Pred delta per FOV group, + bootstrap CI."""
    thresh = df_fov["min_ratio"].quantile(0.20)
    bad_mrns = set(df_fov[df_fov["min_ratio"] <= thresh]["mrn"])
    aff = df_b[df_b.method == "ANTs_Affine"][["mrn", "side", "gt2pred_mean_mm"]].rename(columns={"gt2pred_mean_mm": "aff"})
    syn = df_b[df_b.method == "ANTs_SyN"][["mrn", "side", "gt2pred_mean_mm"]].rename(columns={"gt2pred_mean_mm": "syn"})
    paired = aff.merge(syn, on=["mrn", "side"], how="inner")
    paired = paired[paired["mrn"].isin(df_fov["mrn"])].copy()
    paired["fov_group"] = np.where(paired["mrn"].isin(bad_mrns), "Bad", "Good")
    paired["delta"] = paired["syn"] - paired["aff"]
    results = {}
    for grp, sub in paired.groupby("fov_group"):
        point = sub["delta"].median()
        deltas = [_resample_patients(sub, rng)["delta"].median() for _ in range(n_boot)]
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        results[grp] = (point, lo, hi, len(sub))
    return results, thresh


def main():
    rng = default_rng(config.BOOT_SEED)
    df_a, df_b = load_tracks()
    df_fov = compute_fov(df_a["mrn"].unique())

    print("=" * 70)
    print(f"Bootstrap CIs (N={config.N_BOOT}, seed={config.BOOT_SEED}, patient-level resampling)")
    print("=" * 70)

    print("\n1. Volume-distance Spearman correlations")
    for label, (rho, lo, hi, n) in volume_distance_correlations(df_b, rng).items():
        print(f"   {label:12s}  rho = {rho:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  (N={n})")

    print("\n2. Mid-tier proportion AUC_SyN > 0.60")
    p, lo, hi, n = midtier_auc_proportion(df_a, rng)
    print(f"   {p:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (N={n} Mid-tier ROIs)")

    print("\n3. FOV-stratified paired Affine-vs-SyN delta (GT->Pred)")
    res, thresh = fov_paired_delta(df_b, df_fov, rng)
    print(f"   FOV threshold (20th pct min_ratio): {thresh:.3f}")
    for grp, (point, lo, hi, n) in res.items():
        print(f"   {grp}-FOV: median delta = {point:+.3f} mm  95% CI [{lo:+.3f}, {hi:+.3f}]  (N={n})")


if __name__ == "__main__":
    main()
