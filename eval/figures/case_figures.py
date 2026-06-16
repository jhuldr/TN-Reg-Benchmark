"""Per-case operating-regime / failure-mode grids (manuscript Fig. 4 / Fig. 8).

Renders, for one (case, side), a column grid of [Fixed MRI, Original MRA, + the
six warped-MRA methods] over three rows: whole-brain slice with the ROI box, the
matched 48^3 trigeminal ROI, and the ROI with GT-vessel (cyan) / VesselFM (red)
overlays.

DE-IDENTIFICATION: the figure title uses ``--label`` (e.g. "Representative Case A"),
never the MRN, and the output filename is whatever you pass via ``--out-name``.
No patient identifier is written into the image or filename.

Run:
    REG_ROOT=/path/to/TN_Reg python -m eval.figures.case_figures \
        --case-id <CASE_ID> --side contra --label "Representative Case B" \
        --out-name representative_case_B.png
"""
from __future__ import annotations

import argparse

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from skimage import measure
from scipy.ndimage import center_of_mass
from tifffile import imread

from .. import config
from .._common_vis import (load_volume, case_to_csv_id, read_centroid, csv_to_zyx,
                           map_center_between_volumes, find_best_dicom_series_dir)
from ..io_utils import build_sampling_grid, sample_volume, patch_to_tn
from ._common import apply_paper_style, ABBR

CUBE_SIZE = config.CUBE_SIZE
SPACING = config.SPACING
FIG_METHODS = config.METHOD_ORDER


def _pad_slice_to_fov(sl, sp_h, sp_w, target_h, target_w):
    h, w = sl.shape
    th, tw = int(round(target_h / sp_h)), int(round(target_w / sp_w))
    pad_h, pad_w = max(0, th - h), max(0, tw - w)
    before_h, before_w = pad_h // 2, pad_w // 2
    padded = np.pad(sl, ((before_h, pad_h - before_h), (before_w, pad_w - before_w)), mode="constant")
    return padded, before_h, before_w


def render_case(case_id, side, display_label, out_name, out_dir):
    methods = config.method_dirs()
    vfm = {m: config.VFM_BASE / config.VFM_SUBDIR[m] for m in FIG_METHODS}

    mri_nii = nib.load(str(config.CROPPED_MRI / f"{case_id}.nii.gz"))
    mri = mri_nii.get_fdata(); sp_mri = np.abs(np.diag(mri_nii.affine)[:3])
    cropped_mra = config.REG_ROOT / "data/preprocessed_cropped/mra"
    mra_nii = nib.load(str(cropped_mra / f"{case_id}.nii.gz"))
    mra = mra_nii.get_fdata(); sp_mra = np.abs(np.diag(mra_nii.affine)[:3])

    side_name = config.SIDE_NAME[side]
    csv_id = case_to_csv_id(case_id)
    csv_coord = read_centroid(config.TN_ROOT / f"202022_centroids_{side_name.lower()}.csv", csv_id)

    dicom_dir = find_best_dicom_series_dir(config.DICOM_ROOT / case_id)
    dicom_vol = load_volume(dicom_dir, input_kind="dicom")
    nifti_vol_mri = load_volume(config.CROPPED_MRI / f"{case_id}.nii.gz",
                                input_kind="nifti", nifti_axis_order="xyz_to_zyx")
    center_dicom = csv_to_zyx(csv_coord)
    center_mri_int = map_center_between_volumes(center_dicom, dicom_vol, nifti_vol_mri)
    cmri_nib = (int(center_mri_int[2]), int(center_mri_int[1]), int(center_mri_int[0]))

    grid = build_sampling_grid(csv_to_zyx(csv_coord), dicom_vol, mri_nii)
    grid_mra = build_sampling_grid(csv_to_zyx(csv_coord), dicom_vol, mra_nii)

    label_tiff = imread(str(config.TN_ROOT / f"202022_{side_name}_Target/{case_id}_mask.tiff"))
    gt_vessel_tn = (label_tiff == 2)

    roi_mm = CUBE_SIZE * SPACING
    target_fov = [mri.shape[0] * sp_mri[0], mri.shape[1] * sp_mri[1]]
    warped_whole, method_roi = {}, {}
    for m in FIG_METHODS:
        wp = methods[m] / f"{case_id}_reg_Warped.nii.gz"
        if not wp.exists():
            continue
        wnii = nib.load(str(wp))
        warped_whole[m] = (wnii.get_fdata(), np.abs(np.diag(wnii.affine)[:3]))
        target_fov[0] = max(target_fov[0], wnii.shape[0] * abs(wnii.affine[0, 0]))
        target_fov[1] = max(target_fov[1], wnii.shape[1] * abs(wnii.affine[1, 1]))
        mra_crop = patch_to_tn(sample_volume(wnii.get_fdata(), grid, order=1))
        vp = vfm[m] / f"{case_id}_vessel_mask.nii.gz"
        vfm_crop = patch_to_tn(sample_volume(nib.load(str(vp)).get_fdata(), grid, order=0)) if vp.exists() else None
        method_roi[m] = {"mra": mra_crop, "vfm": vfm_crop}

    mri_tn = patch_to_tn(sample_volume(mri, grid, order=1))
    orig_mra_tn = patch_to_tn(sample_volume(mra, grid_mra, order=1))

    # Pick the z-slice with GT voxels and the most agreeing method predictions
    if gt_vessel_tn.sum() > 0:
        gt_z = gt_vessel_tn.sum(axis=(1, 2))
        vfm_z = np.zeros((len(FIG_METHODS), gt_vessel_tn.shape[0]), dtype=int)
        for i, m in enumerate(FIG_METHODS):
            if method_roi.get(m, {}).get("vfm") is not None:
                vfm_z[i] = (method_roi[m]["vfm"] > 0).sum(axis=(1, 2))
        score = ((gt_z > 0) * 1_000_000 + (vfm_z > 0).sum(0) * 10_000 + gt_z * 100 + vfm_z.sum(0))
        cz = int(np.argmax(score))
        slice_gt = gt_vessel_tn[cz]
        cy, cx = ([int(round(v)) for v in center_of_mass(slice_gt.astype(float))]
                  if slice_gt.sum() > 0 else (24, 24))
    else:
        cz, cy, cx = 24, 24, 24

    zc = np.broadcast_to(np.arange(mri.shape[2]), mri.shape).astype(np.float32).copy()
    mri_z = int(round(patch_to_tn(sample_volume(zc, grid, order=1))[cz, cy, cx]))

    # Assemble columns: (label, whole_vol, whole_sp, whole_z, tn_patch, vfm_patch, center)
    columns = [("Fixed MRI", mri, sp_mri, mri_z, mri_tn, None, cmri_nib),
               ("Original MRA", mra, sp_mra, mri_z, orig_mra_tn, None, cmri_nib)]
    for m in FIG_METHODS:
        if m in warped_whole and m in method_roi:
            wdata, wsp = warped_whole[m]
            columns.append((ABBR[m], wdata, wsp, mri_z, method_roi[m]["mra"], method_roi[m]["vfm"], cmri_nib))

    n_cols = len(columns)
    fig = plt.figure(figsize=(1.15 * n_cols, 5.8))
    gs = fig.add_gridspec(3, n_cols, height_ratios=[1, 1, 1], hspace=0.06, wspace=0.05)

    for ci, (name, whole_vol, whole_sp, whole_z, tn_patch, vfm_patch, center) in enumerate(columns):
        # Row 0: whole-brain with ROI box
        ax0 = fig.add_subplot(gs[0, ci])
        whole_z = int(np.clip(whole_z, 0, whole_vol.shape[2] - 1))
        sl = whole_vol[:, :, whole_z]
        sl_padded, pad_h0, pad_w0 = _pad_slice_to_fov(sl, whole_sp[0], whole_sp[1], target_fov[0], target_fov[1])
        v1, v2 = (np.percentile(sl[sl > 0], [1, 99]) if (sl > 0).sum() > 10 else (0, 1))
        ax0.imshow(sl_padded.T, cmap="gray", origin="lower", vmin=v1, vmax=v2, aspect=whole_sp[1] / whole_sp[0])
        hx, hy = (roi_mm / 2) / whole_sp[0], (roi_mm / 2) / whole_sp[1]
        ax0.add_patch(mpatches.Rectangle((center[0] + pad_h0 - hx, center[1] + pad_w0 - hy),
                                         hx * 2, hy * 2, linewidth=1.3, edgecolor="#FFD700", facecolor="none"))
        ax0.set_title(name, fontsize=10, fontweight="bold"); ax0.axis("off")

        # Rows 1 (ROI intensity) and 2 (ROI + overlays)
        roi_sl = np.rot90(tn_patch[cz, :, :], 2)
        v1r, v2r = np.percentile(roi_sl, [1, 99])
        for row in (1, 2):
            ax = fig.add_subplot(gs[row, ci])
            ax.imshow(roi_sl, cmap="gray", origin="lower", vmin=v1r, vmax=v2r, interpolation="bicubic")
            if row == 2:
                gt_sl = np.rot90(gt_vessel_tn[cz, :, :], 2)
                if gt_sl.sum() > 0:
                    fill = np.zeros((*gt_sl.shape, 4)); fill[gt_sl] = [0.0, 0.9, 1.0, 0.22]
                    ax.imshow(fill, origin="lower")
                    for c in measure.find_contours(gt_sl.astype(float), 0.5):
                        ax.plot(c[:, 1], c[:, 0], color="#00E5FF", linewidth=1.3, alpha=0.95)
                if vfm_patch is not None:
                    vfm_sl = np.rot90(vfm_patch[cz, :, :] > 0, 2)
                    if vfm_sl.sum() > 0:
                        fill = np.zeros((*vfm_sl.shape, 4)); fill[vfm_sl] = [1.0, 0.15, 0.35, 0.22]
                        ax.imshow(fill, origin="lower")
                        for c in measure.find_contours(vfm_sl.astype(float), 0.5):
                            ax.plot(c[:, 1], c[:, 0], color="#FF1744", linewidth=1.3, alpha=0.95)
            ax.axis("off")

    fig.text(0.012, 0.79, "Whole-brain", rotation=90, va="center", ha="center", fontweight="bold", fontsize=9)
    fig.text(0.012, 0.50, "TN 48³ ROI", rotation=90, va="center", ha="center", fontweight="bold", fontsize=9)
    fig.text(0.012, 0.21, "TN 48³ ROI + overlay", rotation=90, va="center", ha="center", fontweight="bold", fontsize=9)
    legend = [
        Line2D([0], [0], color="#FFD700", linewidth=1.8, label="TN ROI"),
        Line2D([0], [0], color="#00E5FF", linewidth=1.8, label="GT vessel (annotated segment)"),
        Line2D([0], [0], color="#FF1744", linewidth=1.8, label="VesselFM prediction (full trace)"),
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=9,
               frameon=True, framealpha=0.95)
    # De-identified title: never the MRN.
    fig.suptitle(f"{display_label} — Axial", fontsize=10, fontweight="bold", y=0.97)
    plt.subplots_adjust(left=0.04, right=0.99, top=0.93, bottom=0.06)
    out = out_dir / out_name
    plt.savefig(str(out), dpi=600, bbox_inches="tight"); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case-id", required=True, help="Zero-padded case id (data lookup only)")
    ap.add_argument("--side", required=True, choices=["ipsi", "contra"])
    ap.add_argument("--label", required=True, help="De-identified title, e.g. 'Representative Case A'")
    ap.add_argument("--out-name", required=True, help="Output PNG filename (no MRN, please)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out_dir = config.PAPER_DIR if args.out_dir is None else __import__("pathlib").Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_style()
    out = render_case(args.case_id, args.side, args.label, args.out_name, out_dir)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
