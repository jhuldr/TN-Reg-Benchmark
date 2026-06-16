# TN-Reg-Benchmark
# Code author: Xupeng Zhang (xzhan419@jh.edu)
# Johns Hopkins University
# See README for the full author list and citation.

"""I/O and ROI-sampling helpers shared by the metric computation.

These map a clinician-provided trigeminal centroid (in DICOM index space) onto a
fixed 48^3 physical sampling grid, then sample any warped volume on that grid so
that all methods are compared inside the identical ROI.
"""
from __future__ import annotations

import csv
import os
import re
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates

from . import config

# The low-level DICOM/coordinate helpers live in the visualization package.
_VIS_DIR = str(Path(__file__).resolve().parents[1] / "visualization")
if _VIS_DIR not in sys.path:
    sys.path.insert(0, _VIS_DIR)
from standardize_case_view import load_dicom_volume, csv_to_zyx, patch_to_tn  # noqa: E402


def read_all_centroids(csv_path) -> dict[str, tuple[int, int, int]]:
    """Parse a centroid CSV keyed by (unpadded) MRN -> (i, j, k)."""
    centroids: dict[str, tuple[int, int, int]] = {}
    with open(csv_path) as f:
        for row in csv.reader(f):
            if not row or not row[0].strip():
                continue
            head = row[0].strip()
            if not re.fullmatch(r"\d+", head):
                continue
            nums = []
            for token in row[1:]:
                for part in re.split(r"[,\s]+", token.strip()):
                    if part and re.fullmatch(r"-?\d+", part):
                        nums.append(int(part))
            if len(nums) >= 3:
                centroids[head] = (nums[0], nums[1], nums[2])
    return centroids


def load_side_centroids() -> dict[str, tuple[str, dict]]:
    """Return {side: (SideName, {mrn: centroid})} for ipsi/contra."""
    ipsi = read_all_centroids(config.TN_ROOT / "202022_centroids_ipsilateral.csv")
    contra = read_all_centroids(config.TN_ROOT / "202022_centroids_contralateral.csv")
    return {"ipsi": ("Ipsilateral", ipsi), "contra": ("Contralateral", contra)}


def build_sampling_grid(center_zyx, dicom_vol, ref_nii,
                        cube_size=config.CUBE_SIZE, spacing=config.SPACING):
    """Build a cube_size^3 grid of fractional voxel indices into ``ref_nii``.

    The grid is centered on ``center_zyx`` (DICOM index space) and oriented by
    the DICOM RAS axes at ``spacing`` mm steps, then mapped into the NIfTI's
    index frame so any aligned volume can be sampled consistently.
    """
    ras_coord = dicom_vol.affine_internal_to_ras @ np.array([*center_zyx, 1.0])
    nifti_aff_inv = np.linalg.inv(ref_nii.affine)
    daff = dicom_vol.affine_internal_to_ras
    dz = daff[:3, 0] / np.linalg.norm(daff[:3, 0])
    dy = daff[:3, 1] / np.linalg.norm(daff[:3, 1])
    dx = daff[:3, 2] / np.linalg.norm(daff[:3, 2])
    idx = np.arange(cube_size, dtype=np.float64) - cube_size / 2
    zz, yy, xx = np.meshgrid(idx, idx, idx, indexing="ij")
    ras_pts = (ras_coord[:3][None, None, None, :]
               + zz[..., None] * dz[None, None, None, :] * spacing
               + yy[..., None] * dy[None, None, None, :] * spacing
               + xx[..., None] * dx[None, None, None, :] * spacing)
    pts_flat = ras_pts.reshape(-1, 3)
    ijk_flat = (nifti_aff_inv[:3, :3] @ pts_flat.T + nifti_aff_inv[:3, 3:4]).T
    return ijk_flat.reshape(cube_size, cube_size, cube_size, 3)


def sample_volume(data, ijk_grid, order=1):
    """Sample ``data`` at the fractional indices in ``ijk_grid`` (order=1 linear,
    order=0 nearest for label/mask volumes)."""
    return map_coordinates(
        data, [ijk_grid[..., 0], ijk_grid[..., 1], ijk_grid[..., 2]],
        order=order, mode="nearest")


def find_dicom_dir(mrn) -> str | None:
    """Locate the largest DICOM series directory for a case (>50 files)."""
    case_dir = config.DICOM_ROOT / str(mrn)
    if not case_dir.exists():
        return None
    for root, _dirs, files in os.walk(case_dir):
        if len([f for f in files if not f.startswith(".")]) > 50:
            return root
    return None


def discover_cases() -> list[tuple[str, str]]:
    """Return evaluable (mrn, side) pairs: have preprocessed MRI, a TIFF target,
    and a matching centroid."""
    sides = load_side_centroids()
    mri_cases = {f.name.replace(".nii.gz", "") for f in config.CROPPED_MRI.glob("*.nii.gz")}
    cases: list[tuple[str, str]] = []
    for side, (side_name, centroids) in sides.items():
        label_dir = config.TN_ROOT / f"202022_{side_name}_Target"
        label_mrns = {f.stem.replace("_mask", "") for f in label_dir.glob("*.tiff")}
        for mrn in sorted(mri_cases & label_mrns):
            csv_id = mrn.lstrip("0") or "0"
            if csv_id in centroids:
                cases.append((mrn, side))
    return cases


def load_reference(mrn):
    """Load the cropped-MRI reference NIfTI for a case."""
    return nib.load(str(config.CROPPED_MRI / f"{mrn}.nii.gz"))
