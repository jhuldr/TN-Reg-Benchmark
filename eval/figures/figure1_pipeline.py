# TN-Reg-Benchmark
# Code author: Xupeng Zhang (xzhan419@jh.edu)
# Johns Hopkins University
# See README for the full author list and citation.

"""Figure 1 pipeline-illustration assets (manuscript Fig. 1).

Renders the eight numbered panels (raw MRI/MRA -> labels -> warped MRA -> ROI ->
VesselFM overlay) for a single example case. Output filenames are generic
(``1_MRI_axial.png`` ... ``8_MRI_wholebrain_with_ROI_box.png``); no patient
identifier is written into the images or filenames.

Run:
    REG_ROOT=/path/to/TN_Reg python -m eval.figures.figure1_pipeline \
        --case-id <CASE_ID> --side contra
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence

from .. import config
from .._common_vis import (load_volume, load_dicom_volume, case_to_csv_id, csv_to_zyx,
                           read_centroid, map_center_between_volumes,
                           find_best_dicom_series_dir, patch_to_tn)
from ..io_utils import build_sampling_grid, sample_volume, find_dicom_dir


def _load_tiff_stack(path):
    im = Image.open(path)
    return np.stack([np.array(fr, dtype=np.float32) for fr in ImageSequence.Iterator(im)], axis=0)


def _save_slice(arr, sl_idx, axis, sp_h, sp_w, out_path, cmap="gray",
                overlay=None, overlay_color=(1, 0.3, 0.3, 0.6)):
    sl = arr[sl_idx] if axis == 0 else (arr[:, sl_idx] if axis == 1 else arr[:, :, sl_idx])
    v1, v2 = (np.percentile(sl[sl > 0], [1, 99]) if (sl > 0).sum() > 10 else (0, 1))
    aspect = sp_w / sp_h
    fig, ax = plt.subplots(figsize=(6, 6 * aspect if aspect < 1 else 6 / aspect))
    ax.imshow(sl.T, cmap=cmap, origin="lower", vmin=v1, vmax=v2, aspect=aspect)
    if overlay is not None:
        ov_sl = overlay[sl_idx] if axis == 0 else (overlay[:, sl_idx] if axis == 1 else overlay[:, :, sl_idx])
        rgba = np.zeros((*ov_sl.shape, 4)); rgba[ov_sl > 0] = overlay_color
        ax.imshow(np.transpose(rgba, (1, 0, 2)), origin="lower", aspect=aspect)
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight", pad_inches=0, facecolor="black")
    plt.close(fig)


def render_figure1_assets(case_id, side, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    cropped_mra = config.REG_ROOT / "data/preprocessed_cropped/mra"
    methods = config.method_dirs()
    vfm_dir = config.VFM_BASE / config.VFM_SUBDIR["ANTs_SyN"]

    mri_nii = nib.load(str(config.CROPPED_MRI / f"{case_id}.nii.gz"))
    mra_nii = nib.load(str(cropped_mra / f"{case_id}.nii.gz"))
    warped_nii = nib.load(str(methods["ANTs_SyN"] / config.WARPED_PAT.format(cid=case_id)))
    vfm_nii = nib.load(str(vfm_dir / f"{case_id}_vessel_mask.nii.gz"))
    mri, mra = mri_nii.get_fdata(), mra_nii.get_fdata()
    warped, vfm = warped_nii.get_fdata(), vfm_nii.get_fdata()
    sp_mri = np.abs(np.diag(mri_nii.affine)[:3])
    sp_mra = np.abs(np.diag(mra_nii.affine)[:3])

    side_name = config.SIDE_NAME[side]
    csv_id = case_to_csv_id(case_id)
    csv_coord = read_centroid(config.TN_ROOT / f"202022_centroids_{side_name.lower()}.csv", csv_id)
    dicom_dir = find_best_dicom_series_dir(config.DICOM_ROOT / case_id)
    dicom_vol = load_volume(dicom_dir, input_kind="dicom")
    nifti_vol_mri = load_volume(config.CROPPED_MRI / f"{case_id}.nii.gz",
                                input_kind="nifti", nifti_axis_order="xyz_to_zyx")
    nifti_vol_mra = load_volume(cropped_mra / f"{case_id}.nii.gz",
                                input_kind="nifti", nifti_axis_order="xyz_to_zyx")
    center_dicom = csv_to_zyx(csv_coord)
    cmri = map_center_between_volumes(center_dicom, dicom_vol, nifti_vol_mri)
    cmra = map_center_between_volumes(center_dicom, dicom_vol, nifti_vol_mra)
    cx_mri, cy_mri, cz_mri = int(cmri[2]), int(cmri[1]), int(cmri[0])
    cz_mra = int(cmra[0])

    # Assets 1, 2: raw MRI / MRA axial at the centroid
    _save_slice(mri, cz_mri, 2, sp_mri[0], sp_mri[1], out_dir / "1_MRI_axial.png")
    _save_slice(mra, cz_mra, 2, sp_mra[0], sp_mra[1], out_dir / "2_MRA_original_axial.png")

    # Assets 3, 3b: MRI with nerve / vessel annotation
    label = nib.load(str(config.REG_ROOT / f"nii_Label/{case_id}/{case_id}_{side}_label.nii.gz")).get_fdata()
    _save_slice(mri, cz_mri, 2, sp_mri[0], sp_mri[1], out_dir / "3_MRI_with_label.png",
                overlay=(label == 1).astype(float), overlay_color=(1, 0.8, 0.2, 0.7))
    _save_slice(mri, cz_mri, 2, sp_mri[0], sp_mri[1], out_dir / "3b_MRI_with_vessel_label.png",
                overlay=(label == 2).astype(float), overlay_color=(0, 0.9, 1.0, 0.7))

    # Asset 4: warped MRA blended onto MRI
    asp = sp_mri[2] / sp_mri[1]
    fig, ax = plt.subplots(figsize=(6, 6))
    mri_sl, warp_sl = mri[:, :, cz_mri], warped[:, :, cz_mri]
    v1, v2 = np.percentile(mri_sl[mri_sl > 0], [1, 99])
    v3, v4 = np.percentile(warp_sl[warp_sl > 0], [1, 99])
    ax.imshow(mri_sl.T, cmap="gray", origin="lower", vmin=v1, vmax=v2, aspect=asp)
    warp_norm = np.clip((warp_sl - v3) / (v4 - v3 + 1e-6), 0, 1)
    rgba = np.zeros((*warp_sl.shape, 4)); rgba[..., 0] = warp_norm; rgba[..., 3] = warp_norm * 0.5
    ax.imshow(np.transpose(rgba, (1, 0, 2)), origin="lower", aspect=asp)
    ax.axis("off"); plt.tight_layout(pad=0)
    plt.savefig(str(out_dir / "4_MRI_plus_warped_MRA_blend.png"), dpi=200,
                bbox_inches="tight", pad_inches=0, facecolor="black"); plt.close(fig)

    # Asset 5: warped MRA with VesselFM overlay
    _save_slice(warped, cz_mri, 2, sp_mri[0], sp_mri[1], out_dir / "5_warped_MRA_with_vesselFM.png",
                overlay=(vfm > 0).astype(float), overlay_color=(1, 0.2, 0.2, 0.6))

    # Asset 6: 48^3 ROI crop with nerve + vessel labels
    tn_input = _load_tiff_stack(config.TN_ROOT / f"202022_{side_name}_Input/{case_id}_cropped.tiff")
    tn_label = _load_tiff_stack(config.TN_ROOT / f"202022_{side_name}_Target/{case_id}_mask.tiff")
    fig, ax = plt.subplots(figsize=(4, 4))
    sl_tn = tn_input[24]
    v1, v2 = np.percentile(sl_tn[sl_tn > 0], [1, 99])
    ax.imshow(sl_tn, cmap="gray", vmin=v1, vmax=v2, aspect="equal")
    lbl = tn_label[24]
    for val, color in [(1, [1, 0.8, 0.2, 0.7]), (2, [0, 0.9, 1.0, 0.7])]:
        ov = np.zeros((*lbl.shape, 4)); ov[lbl == val] = color; ax.imshow(ov, aspect="equal")
    ax.add_patch(plt.Rectangle((0, 0), 47, 47, fill=False, edgecolor="yellow", linewidth=3))
    ax.axis("off"); plt.tight_layout(pad=0)
    plt.savefig(str(out_dir / "6_ROI_crop_48x48_with_labels.png"), dpi=200,
                bbox_inches="tight", pad_inches=0, facecolor="black"); plt.close(fig)

    # Asset 7: 48^3 ROI warped MRA with GT (cyan) vs VesselFM (red)
    grid = build_sampling_grid(csv_to_zyx(csv_coord),
                               load_dicom_volume(Path(find_dicom_dir(case_id))), mri_nii)
    warp_crop = patch_to_tn(sample_volume(warped, grid, order=1))
    vfm_crop = patch_to_tn(sample_volume(vfm, grid, order=0))
    gt_vessel = (tn_label == 2).astype(float)
    fig, ax = plt.subplots(figsize=(4, 4))
    sl = warp_crop[24]
    v1, v2 = np.percentile(sl[sl > 0], [1, 99])
    ax.imshow(sl, cmap="gray", vmin=v1, vmax=v2, aspect="equal")
    gt_ov = np.zeros((*gt_vessel[24].shape, 4)); gt_ov[gt_vessel[24] > 0] = [0, 0.9, 1.0, 0.7]
    ax.imshow(gt_ov, aspect="equal")
    vfm_ov = np.zeros((*vfm_crop[24].shape, 4)); vfm_ov[vfm_crop[24] > 0] = [1, 0.3, 0.3, 0.5]
    ax.imshow(vfm_ov, aspect="equal")
    ax.add_patch(plt.Rectangle((0, 0), 47, 47, fill=False, edgecolor="yellow", linewidth=3))
    ax.axis("off"); plt.tight_layout(pad=0)
    plt.savefig(str(out_dir / "7_ROI_warped_MRA_with_VFM_vs_GT.png"), dpi=200,
                bbox_inches="tight", pad_inches=0, facecolor="black"); plt.close(fig)

    # Asset 8: whole-brain MRI with the ROI box
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mri[:, :, cz_mri].T, cmap="gray", origin="lower", aspect=asp,
              vmin=np.percentile(mri[mri > 0], 1), vmax=np.percentile(mri[mri > 0], 99))
    hx = (config.CUBE_SIZE * config.SPACING) / 2 / sp_mri[0]
    hy = (config.CUBE_SIZE * config.SPACING) / 2 / sp_mri[1]
    ax.add_patch(plt.Rectangle((cx_mri - hx, cy_mri - hy), hx * 2, hy * 2,
                               fill=False, edgecolor="yellow", linewidth=3))
    ax.axis("off"); plt.tight_layout(pad=0)
    plt.savefig(str(out_dir / "8_MRI_wholebrain_with_ROI_box.png"), dpi=200,
                bbox_inches="tight", pad_inches=0, facecolor="black"); plt.close(fig)
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case-id", required=True, help="Zero-padded case id (data lookup only)")
    ap.add_argument("--side", required=True, choices=["ipsi", "contra"])
    ap.add_argument("--out-dir", default=None, help="default: $REG_ROOT/eval_result/figure1_assets")
    args = ap.parse_args()
    out_dir = (config.EVAL_DIR / "figure1_assets") if args.out_dir is None \
        else __import__("pathlib").Path(args.out_dir)
    from ._common import apply_paper_style
    apply_paper_style()
    render_figure1_assets(args.case_id, args.side, out_dir)
    print(f"Figure 1 assets saved to: {out_dir}")


if __name__ == "__main__":
    main()
