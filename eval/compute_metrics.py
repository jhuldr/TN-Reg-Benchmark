"""Compute Track A + Track B ROI metrics for all six registration methods.

For every evaluable (case, side):
  * sample each method's warped MRA in the matched 48^3 ROI and compute Track A;
  * for methods with a VesselFM prediction, sample it in the same ROI and compute
    Track B; every skipped (case, side, method) is recorded with a reason.

Outputs (under ``$REG_ROOT/eval_result``):
  eval_trackA_all_methods.csv, eval_trackB_all_methods.csv, eval_skip_log.csv

Usage:
    REG_ROOT=/path/to/TN_Reg python -m eval.compute_metrics
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import nibabel as nib
from tifffile import imread

from . import config, io_utils
from .metrics import compute_track_a, compute_track_b, TRACK_A_COLS

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - progress bar is optional
    def tqdm(it, **_kw):
        return it


def _log_skip(skip_log, mrn, side, method, track, reason, **extra):
    entry = {"mrn": str(mrn).zfill(8), "side": side, "method": method,
             "track": track, "reason": reason, "detail": ""}
    if extra:
        entry["detail"] = "; ".join(f"{k}={v}" for k, v in extra.items())
    skip_log.append(entry)


def run(verbose: bool = True):
    """Evaluate the cohort and return (df_a, df_b, df_skip)."""
    sides = io_utils.load_side_centroids()
    methods = config.method_dirs()
    vfm = config.vfm_dirs(require_nonempty=True)
    cases = io_utils.discover_cases()
    if verbose:
        print(f"Evaluable (mrn, side) pairs: {len(cases)}")
        print(f"Track B methods with VesselFM: {list(vfm)}")

    results_a, results_b, skip_log = [], [], []

    for mrn, side in tqdm(cases, desc="Evaluating"):
        mrn_pad = str(mrn).zfill(8)
        csv_id = mrn.lstrip("0") or "0"
        side_name, centroids = sides[side]

        csv_coord = centroids.get(csv_id)
        if csv_coord is None:
            _log_skip(skip_log, mrn, side, "*", "*", "missing_centroid_csv")
            continue
        dicom_dir = io_utils.find_dicom_dir(mrn)
        if dicom_dir is None:
            _log_skip(skip_log, mrn, side, "*", "*", "missing_dicom_dir")
            continue

        try:
            dicom_vol = io_utils.load_dicom_volume(io_utils.Path(dicom_dir))
            ref_nii = io_utils.load_reference(mrn)
            grid = io_utils.build_sampling_grid(io_utils.csv_to_zyx(csv_coord), dicom_vol, ref_nii)

            tiff_path = config.TN_ROOT / f"202022_{side_name}_Target/{mrn}_mask.tiff"
            if not tiff_path.exists():
                _log_skip(skip_log, mrn, side, "*", "*", "missing_gt_tiff", path=str(tiff_path))
                continue
            label_raw = imread(str(tiff_path))
            gt_mask = label_raw.astype(np.float32)
            gt_vessel = (label_raw == 2)
            gt_nerve = (label_raw == 1)

            input_path = config.TN_ROOT / f"202022_{side_name}_Input/{mrn}_cropped.tiff"
            mri_patch = imread(str(input_path)).astype(np.float32) if input_path.exists() else None

            # ---- Track A: all six methods ----
            for mname, mdir in methods.items():
                warped_path = mdir / config.WARPED_PAT.format(cid=mrn)
                row = {"mrn": mrn_pad, "side": side, "method": mname}
                if not warped_path.exists():
                    _log_skip(skip_log, mrn, side, mname, "A", "missing_warped_mra", path=str(warped_path))
                    row.update({k: np.nan for k in TRACK_A_COLS})
                else:
                    try:
                        mra_data = nib.load(str(warped_path)).get_fdata()
                        mra_patch = io_utils.patch_to_tn(io_utils.sample_volume(mra_data, grid, order=1))
                        row.update(compute_track_a(mra_patch, mri_patch, gt_mask))
                    except Exception as e:
                        _log_skip(skip_log, mrn, side, mname, "A", "track_a_exception", error=str(e)[:100])
                        row.update({k: np.nan for k in TRACK_A_COLS})
                results_a.append(row)

            # ---- Track B: methods with VesselFM predictions ----
            if gt_vessel.sum() == 0:
                for mname in vfm:
                    _log_skip(skip_log, mrn, side, mname, "B", "gt_vessel_empty")
                continue

            for mname, vfm_dir in vfm.items():
                vfm_path = vfm_dir / config.VFM_PAT.format(cid=mrn)
                warped_path = methods[mname] / config.WARPED_PAT.format(cid=mrn)
                if not vfm_path.exists():
                    _log_skip(skip_log, mrn, side, mname, "B", "missing_vfm_mask", path=str(vfm_path))
                    continue
                if not warped_path.exists():
                    _log_skip(skip_log, mrn, side, mname, "B", "missing_warped_mra", path=str(warped_path))
                    continue
                try:
                    vfm_data = nib.load(str(vfm_path)).get_fdata()
                    vfm_crop = io_utils.patch_to_tn(io_utils.sample_volume(vfm_data, grid, order=0))
                    pred_vessel = (vfm_crop > 0)
                    if pred_vessel.sum() == 0:
                        _log_skip(skip_log, mrn, side, mname, "B", "pred_vessel_empty")
                        continue
                    mra_data = nib.load(str(warped_path)).get_fdata()
                    mra_crop = io_utils.patch_to_tn(io_utils.sample_volume(mra_data, grid, order=1))
                    row = {"mrn": mrn_pad, "side": side, "method": mname}
                    row.update(compute_track_b(gt_vessel, pred_vessel, mra_crop, gt_nerve))
                    results_b.append(row)
                except Exception as e:
                    _log_skip(skip_log, mrn, side, mname, "B", "track_b_exception", error=str(e)[:100])
        except Exception as e:
            _log_skip(skip_log, mrn, side, "*", "*", "outer_exception", error=str(e)[:100])

    df_a = pd.DataFrame(results_a)
    df_b = pd.DataFrame(results_b)
    df_skip = pd.DataFrame(skip_log)
    for df in (df_a, df_b, df_skip):
        if "mrn" in df.columns:
            df["mrn"] = df["mrn"].astype(str).str.zfill(8)
    return df_a, df_b, df_skip


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=None,
                    help="Output directory for CSVs (default: $REG_ROOT/eval_result)")
    args = ap.parse_args()

    out_dir = io_utils.Path(args.out_dir) if args.out_dir else config.EVAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df_a, df_b, df_skip = run(verbose=True)
    df_a.to_csv(out_dir / "eval_trackA_all_methods.csv", index=False)
    df_b.to_csv(out_dir / "eval_trackB_all_methods.csv", index=False)
    df_skip.to_csv(out_dir / "eval_skip_log.csv", index=False)

    print("=" * 60)
    print(f"Track A: {len(df_a)} rows | {df_a['mrn'].nunique() if len(df_a) else 0} MRNs")
    print(f"Track B: {len(df_b)} rows | {df_b['mrn'].nunique() if len(df_b) else 0} MRNs")
    print(f"Skips:   {len(df_skip)} entries")
    print(f"Saved CSVs to {out_dir}")


if __name__ == "__main__":
    main()
