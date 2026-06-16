#!/usr/bin/env python3
"""
Render standardized views for one TN case.

Supported display modes:
- IPL: columns I / P / L
- RAS: columns Axial / Coronal / Sagittal

Key features:
- Can render from either:
  (1) DICOM series (when --reg-nifti not provided), OR
  (2) NIfTI volume (when --reg-nifti provided)
- When using NIfTI and centroid-space=raw/auto:
  it will map centroid from RAW DICOM space -> NIfTI space via physical transform.
- RAW DICOM is searched under:
    --raw-dicom-root/<case_id>/... (auto find best series dir)
  Default raw root is <reg-root>/data/raw/mri (your symlink "path2")
- --out-dir: write outputs to a custom directory instead of preview_<case>/
- --compare-nifti: optional second NIfTI for side-by-side MRI vs MRA comparison

Author: Xupeng Zhang
Johns Hopkins University
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
from PIL import Image, ImageDraw

_IGNORE_FILES = {"DICOMDIR", "LOCKFILE", "VERSION"}

from standardize_case_view import (  # same directory import
    VolumeData,
    case_to_csv_id,
    csv_to_zyx,
    extract_iso_cube,
    load_tiff_stack,
    load_volume,
    map_center_between_volumes,
    normalize_2d,
    overlay_label,
    read_centroid,
    reorient_patch_dst_to_src,
    reorient_patch_src_to_dst,
    tn_to_patch,
    volume_transform,
    zscore,
)


# -------------------------
# DICOM series auto finder
# -------------------------
def _count_dicom_like_files(d: Path, min_bytes: int = 512) -> int:
    if not d.exists() or not d.is_dir():
        return 0
    cnt = 0
    try:
        for f in d.iterdir():
            if not f.is_file():
                continue
            if f.name in _IGNORE_FILES:
                continue
            suf = f.suffix.lower()
            if suf in {".dcm", ".ima", ""}:
                try:
                    if f.stat().st_size > min_bytes:
                        cnt += 1
                except Exception:
                    pass
    except Exception:
        return 0
    return cnt


def find_best_dicom_series_dir(
    case_root: Path,
    max_depth: int = 8,
    min_files: int = 10,
    min_bytes: int = 512,
) -> Optional[Path]:
    if not case_root.exists():
        return None

    best_dir = case_root
    best_score = _count_dicom_like_files(case_root, min_bytes=min_bytes)

    stack: List[tuple[Path, int]] = [(case_root, 0)]
    seen: set[Path] = set()

    while stack:
        d, depth = stack.pop()
        if d in seen:
            continue
        seen.add(d)
        if depth > max_depth:
            continue

        score = _count_dicom_like_files(d, min_bytes=min_bytes)
        if score > best_score:
            best_score = score
            best_dir = d

        try:
            for sub in d.iterdir():
                if sub.is_dir():
                    stack.append((sub, depth + 1))
        except Exception:
            pass

    if best_score < min_files:
        return None
    return best_dir


# -------------------------
# Views / rendering helpers
# -------------------------
def to_ipl(arr_zyx: np.ndarray) -> np.ndarray:
    out = np.transpose(arr_zyx, (1, 2, 0))
    out = np.flip(out, axis=2)
    return out


def project_patch_label_to_full_zyx(
    patch_lbl_zyx: np.ndarray,
    center_zyx: Tuple[float, float, float],
    vol_shape_zyx: Tuple[int, int, int],
    spacing_zyx: Tuple[float, float, float],
    iso_mm: float,
) -> np.ndarray:
    sz, sy, sx = spacing_zyx
    z_size, y_size, x_size = vol_shape_zyx
    idx = np.arange(patch_lbl_zyx.shape[0], dtype=np.float32) - 24.0
    oz = idx * (iso_mm / sz)
    oy = idx * (iso_mm / sy)
    ox = idx * (iso_mm / sx)
    zz, yy, xx = np.meshgrid(oz, oy, ox, indexing="ij")
    cz, cy, cx = center_zyx

    mask = patch_lbl_zyx > 0
    if not np.any(mask):
        return np.zeros(vol_shape_zyx, dtype=np.uint8)

    zf = cz + zz[mask]
    yf = cy + yy[mask]
    xf = cx + xx[mask]
    zi = np.clip(np.rint(zf).astype(np.int32), 0, z_size - 1)
    yi = np.clip(np.rint(yf).astype(np.int32), 0, y_size - 1)
    xi = np.clip(np.rint(xf).astype(np.int32), 0, x_size - 1)
    lbl = patch_lbl_zyx[mask].astype(np.uint8)

    out = np.zeros(vol_shape_zyx, dtype=np.uint8)
    flat = out.reshape(-1)
    flat_idx = zi * (y_size * x_size) + yi * x_size + xi
    np.maximum.at(flat, flat_idx, lbl)
    return out


def stamp_letter(im: Image.Image, ch: str) -> Image.Image:
    d = ImageDraw.Draw(im)
    d.rectangle((2, 2, 26, 24), fill=(0, 0, 0))
    d.text((8, 5), ch, fill=(255, 255, 255))
    return im


def _stamp_tag(im: Image.Image, tag: str) -> Image.Image:
    """Stamp a tag label (e.g. 'MRA_warped', 'MRI_fixed') in the top-left."""
    d = ImageDraw.Draw(im)
    d.rectangle((2, 2, 130, 24), fill=(0, 0, 0))
    d.text((8, 5), tag, fill=(255, 255, 255))
    return im


def draw_physical_box_ipl(
    ims: list[Image.Image],
    center_ipl: Tuple[int, int, int],
    half_ipl: Tuple[float, float, float],
    color: Tuple[int, int, int] = (255, 210, 50),
) -> list[Image.Image]:
    ci, cp, cl = center_ipl
    hi, hp, hl = half_ipl
    i0, i1 = ci - hi, ci + hi
    p0, p1 = cp - hp, cp + hp
    l0, l1 = cl - hl, cl + hl

    draw_i = ImageDraw.Draw(ims[0])
    w0, h0 = ims[0].size
    draw_i.rectangle(
        (float(np.clip(l0, 0, w0 - 1)), float(np.clip(p0, 0, h0 - 1)),
         float(np.clip(l1, 0, w0 - 1)), float(np.clip(p1, 0, h0 - 1))),
        outline=color, width=2,
    )

    draw_p = ImageDraw.Draw(ims[1])
    w1, h1 = ims[1].size
    draw_p.rectangle(
        (float(np.clip(l0, 0, w1 - 1)), float(np.clip(i0, 0, h1 - 1)),
         float(np.clip(l1, 0, w1 - 1)), float(np.clip(i1, 0, h1 - 1))),
        outline=color, width=2,
    )

    draw_l = ImageDraw.Draw(ims[2])
    w2, h2 = ims[2].size
    draw_l.rectangle(
        (float(np.clip(p0, 0, w2 - 1)), float(np.clip(i0, 0, h2 - 1)),
         float(np.clip(p1, 0, w2 - 1)), float(np.clip(i1, 0, h2 - 1))),
        outline=color, width=2,
    )
    return ims


def three_planes_raw(arr_ipl: np.ndarray, center_ipl: Tuple[int, int, int]):
    i, p, l = center_ipl
    i = int(np.clip(i, 0, arr_ipl.shape[0] - 1))
    p = int(np.clip(p, 0, arr_ipl.shape[1] - 1))
    l = int(np.clip(l, 0, arr_ipl.shape[2] - 1))
    im_i = Image.fromarray(normalize_2d(arr_ipl[i, :, :]), mode="L").convert("RGB")
    im_p = Image.fromarray(normalize_2d(arr_ipl[:, p, :]), mode="L").convert("RGB")
    im_l = Image.fromarray(normalize_2d(arr_ipl[:, :, l]), mode="L").convert("RGB")
    return [im_i, im_p, im_l]


def three_planes_overlay(arr_ipl: np.ndarray, lbl_ipl: np.ndarray, center_ipl: Tuple[int, int, int]):
    i, p, l = center_ipl
    i = int(np.clip(i, 0, arr_ipl.shape[0] - 1))
    p = int(np.clip(p, 0, arr_ipl.shape[1] - 1))
    l = int(np.clip(l, 0, arr_ipl.shape[2] - 1))
    im_i = Image.fromarray(overlay_label(normalize_2d(arr_ipl[i, :, :]), lbl_ipl[i, :, :]), mode="RGB")
    im_p = Image.fromarray(overlay_label(normalize_2d(arr_ipl[:, p, :]), lbl_ipl[:, p, :]), mode="RGB")
    im_l = Image.fromarray(overlay_label(normalize_2d(arr_ipl[:, :, l]), lbl_ipl[:, :, l]), mode="RGB")
    return [im_i, im_p, im_l]


def draw_physical_box_ras(
    ims: list[Image.Image],
    center_zyx: Tuple[int, int, int],
    half_zyx: Tuple[float, float, float],
    color: Tuple[int, int, int] = (255, 210, 50),
) -> list[Image.Image]:
    cz, cy, cx = center_zyx
    hz, hy, hx = half_zyx
    z0, z1 = cz - hz, cz + hz
    y0, y1 = cy - hy, cy + hy
    x0, x1 = cx - hx, cx + hx

    draw_a = ImageDraw.Draw(ims[0])
    w0, h0 = ims[0].size
    draw_a.rectangle(
        (float(np.clip(x0, 0, w0 - 1)), float(np.clip(y0, 0, h0 - 1)),
         float(np.clip(x1, 0, w0 - 1)), float(np.clip(y1, 0, h0 - 1))),
        outline=color, width=2,
    )

    draw_c = ImageDraw.Draw(ims[1])
    w1, h1 = ims[1].size
    draw_c.rectangle(
        (float(np.clip(x0, 0, w1 - 1)), float(np.clip(z0, 0, h1 - 1)),
         float(np.clip(x1, 0, w1 - 1)), float(np.clip(z1, 0, h1 - 1))),
        outline=color, width=2,
    )

    draw_s = ImageDraw.Draw(ims[2])
    w2, h2 = ims[2].size
    draw_s.rectangle(
        (float(np.clip(y0, 0, w2 - 1)), float(np.clip(z0, 0, h2 - 1)),
         float(np.clip(y1, 0, w2 - 1)), float(np.clip(z1, 0, h2 - 1))),
        outline=color, width=2,
    )
    return ims


def three_planes_raw_ras(arr_zyx: np.ndarray, center_zyx: Tuple[int, int, int]):
    z, y, x = center_zyx
    z = int(np.clip(z, 0, arr_zyx.shape[0] - 1))
    y = int(np.clip(y, 0, arr_zyx.shape[1] - 1))
    x = int(np.clip(x, 0, arr_zyx.shape[2] - 1))
    im_a = Image.fromarray(normalize_2d(arr_zyx[z, :, :]), mode="L").convert("RGB")
    im_c = Image.fromarray(normalize_2d(arr_zyx[:, y, :]), mode="L").convert("RGB")
    im_s = Image.fromarray(normalize_2d(arr_zyx[:, :, x]), mode="L").convert("RGB")
    return [im_a, im_c, im_s]


def three_planes_overlay_ras(arr_zyx: np.ndarray, lbl_zyx: np.ndarray, center_zyx: Tuple[int, int, int]):
    z, y, x = center_zyx
    z = int(np.clip(z, 0, arr_zyx.shape[0] - 1))
    y = int(np.clip(y, 0, arr_zyx.shape[1] - 1))
    x = int(np.clip(x, 0, arr_zyx.shape[2] - 1))
    im_a = Image.fromarray(overlay_label(normalize_2d(arr_zyx[z, :, :]), lbl_zyx[z, :, :]), mode="RGB")
    im_c = Image.fromarray(overlay_label(normalize_2d(arr_zyx[:, y, :]), lbl_zyx[:, y, :]), mode="RGB")
    im_s = Image.fromarray(overlay_label(normalize_2d(arr_zyx[:, :, x]), lbl_zyx[:, :, x]), mode="RGB")
    return [im_a, im_c, im_s]


def rotate_tiles_180(ims: list[Image.Image]) -> list[Image.Image]:
    return [im.rotate(180, expand=False) for im in ims]


# -------------------------
# Compare: MRI vs MRA side-by-side
# -------------------------
def build_two_volume_compare(
    volA: VolumeData,
    centerA_zyx: Tuple[float, float, float],
    volB: VolumeData,
    centerB_zyx: Tuple[float, float, float],
    iso_mm: float,
    view: str,
    ras_rotate_180: bool,
    cell: int = 300,
    tagA: str = "A",
    tagB: str = "B",
) -> Image.Image:
    """
    Compare image: top row = volA three-plane views, bottom row = volB.
    Each row shows A/C/S (RAS) or I/P/L (IPL) with a physical box overlay.
    """
    half_mm = (48 / 2.0) * iso_mm

    def _clamp(c, shape):
        return (
            int(np.clip(round(c[0]), 0, shape[0] - 1)),
            int(np.clip(round(c[1]), 0, shape[1] - 1)),
            int(np.clip(round(c[2]), 0, shape[2] - 1)),
        )

    if view == "ras":
        cA = _clamp(centerA_zyx, volA.vol.shape)
        cB = _clamp(centerB_zyx, volB.vol.shape)

        imsA = three_planes_raw_ras(volA.vol, cA)
        imsB = three_planes_raw_ras(volB.vol, cB)

        hA = (half_mm / volA.spacing_zyx[0], half_mm / volA.spacing_zyx[1], half_mm / volA.spacing_zyx[2])
        hB = (half_mm / volB.spacing_zyx[0], half_mm / volB.spacing_zyx[1], half_mm / volB.spacing_zyx[2])

        imsA = draw_physical_box_ras(imsA, cA, hA)
        imsB = draw_physical_box_ras(imsB, cB, hB)

        if ras_rotate_180:
            imsA = rotate_tiles_180(imsA)
            imsB = rotate_tiles_180(imsB)

        letters = ["A", "C", "S"]

    elif view == "ipl":
        def _zyx_to_ipl(c_zyx, vol_shape):
            return (
                int(round(c_zyx[1])),
                int(round(c_zyx[2])),
                int(round((vol_shape[0] - 1) - c_zyx[0])),
            )

        cA_ipl = _zyx_to_ipl(centerA_zyx, volA.vol.shape)
        cB_ipl = _zyx_to_ipl(centerB_zyx, volB.vol.shape)

        fullA = to_ipl(volA.vol)
        fullB = to_ipl(volB.vol)

        imsA = three_planes_raw(fullA, cA_ipl)
        imsB = three_planes_raw(fullB, cB_ipl)

        hA_ipl = (half_mm / volA.spacing_zyx[1], half_mm / volA.spacing_zyx[2], half_mm / volA.spacing_zyx[0])
        hB_ipl = (half_mm / volB.spacing_zyx[1], half_mm / volB.spacing_zyx[2], half_mm / volB.spacing_zyx[0])

        imsA = draw_physical_box_ipl(imsA, cA_ipl, hA_ipl)
        imsB = draw_physical_box_ipl(imsB, cB_ipl, hB_ipl)

        letters = ["I", "P", "L"]
    else:
        raise ValueError(f"Unsupported view: {view}")

    gap = 12
    w = cell * 3
    h = cell * 2 + gap
    canvas = Image.new("RGB", (w, h), (12, 12, 12))

    for i, im in enumerate(imsA):
        resized = im.resize((cell, cell), Image.BILINEAR)
        resized = stamp_letter(resized, letters[i])
        resized = _stamp_tag(resized, tagA)
        canvas.paste(resized, (i * cell, 0))

    for i, im in enumerate(imsB):
        resized = im.resize((cell, cell), Image.BILINEAR)
        resized = stamp_letter(resized, letters[i])
        resized = _stamp_tag(resized, tagB)
        canvas.paste(resized, (i * cell, cell + gap))

    return canvas


# -------------------------
# Main rendering
# -------------------------
def build_side(
    side: str,
    case_id: str,
    csv_coord: Tuple[int, int, int],
    vol_data: VolumeData,
    tn_input: np.ndarray,
    tn_label: np.ndarray,
    iso_mm: float,
    shift_zyx: Tuple[float, float, float],
    center_zyx_override: Tuple[float, float, float] | None = None,
    raw_to_source_transform: np.ndarray | None = None,
    view: str = "ipl",
    ras_rotate_180: bool = False,
    align_lower_to_upper: bool = False,
    cell: int = 300,
) -> Tuple[Image.Image, float]:
    vol = vol_data.vol
    sz, sy, sx = vol_data.spacing_zyx

    if center_zyx_override is None:
        center_zyx_base = csv_to_zyx(csv_coord)
        cz = center_zyx_base[0] + shift_zyx[0]
        cy = center_zyx_base[1] + shift_zyx[1]
        cx = center_zyx_base[2] + shift_zyx[2]
        center_zyx = (cz, cy, cx)
    else:
        center_zyx = center_zyx_override
        cz, cy, cx = center_zyx

    center_ipl_full = (
        int(round(cy)),
        int(round(cx)),
        int(round((vol.shape[0] - 1) - cz)),
    )

    patch_iso_source = extract_iso_cube(vol, center_zyx, vol_data.spacing_zyx, iso_mm, size=48)
    if raw_to_source_transform is None:
        patch_iso_raw = patch_iso_source
    else:
        patch_iso_raw = reorient_patch_dst_to_src(patch_iso_source, raw_to_source_transform)

    patch_lbl_raw = tn_to_patch(tn_label)
    if raw_to_source_transform is None:
        patch_lbl_source = patch_lbl_raw
    else:
        patch_lbl_source = reorient_patch_src_to_dst(patch_lbl_raw, raw_to_source_transform)

    tn_patch_raw = tn_to_patch(tn_input)
    tn_lbl_raw = tn_to_patch(tn_label)
    if raw_to_source_transform is None:
        tn_patch_source = tn_patch_raw
        tn_lbl_source = tn_lbl_raw
    else:
        tn_patch_source = reorient_patch_src_to_dst(tn_patch_raw, raw_to_source_transform)
        tn_lbl_source = reorient_patch_src_to_dst(tn_lbl_raw, raw_to_source_transform)

    if align_lower_to_upper:
        patch_for_rows = patch_iso_source
        patch_lbl_for_rows = patch_lbl_source
        tn_patch_for_rows = tn_patch_source
        tn_lbl_for_rows = tn_lbl_source
    else:
        patch_for_rows = patch_iso_raw
        patch_lbl_for_rows = patch_lbl_raw
        tn_patch_for_rows = tn_patch_raw
        tn_lbl_for_rows = tn_lbl_raw

    full_lbl_zyx = project_patch_label_to_full_zyx(
        patch_lbl_source, center_zyx, vol.shape, vol_data.spacing_zyx, iso_mm,
    )

    half_mm = (48 / 2.0) * iso_mm

    if view == "ipl":
        patch_ipl = to_ipl(patch_for_rows)
        patch_lbl_ipl = to_ipl(patch_lbl_for_rows)
        tn_patch_ipl = to_ipl(tn_patch_for_rows)
        tn_lbl_ipl = to_ipl(tn_lbl_for_rows)
        full_lbl_ipl = to_ipl(full_lbl_zyx)

        center_ipl_patch = (23, 23, 23)
        score = float((zscore(patch_ipl) * zscore(tn_patch_ipl)).mean())

        full_ipl = to_ipl(vol)
        ims_full_raw = three_planes_raw(full_ipl, center_ipl_full)
        ims_full_lbl = three_planes_overlay(full_ipl, full_lbl_ipl, center_ipl_full)

        half_ipl = (half_mm / sy, half_mm / sx, half_mm / sz)
        ims_full_raw = draw_physical_box_ipl(ims_full_raw, center_ipl_full, half_ipl)
        ims_full_lbl = draw_physical_box_ipl(ims_full_lbl, center_ipl_full, half_ipl)

        rows = [
            (False, ims_full_raw),
            (True, ims_full_lbl),
            (False, three_planes_raw(patch_ipl, center_ipl_patch)),
            (True, three_planes_overlay(patch_ipl, patch_lbl_ipl, center_ipl_patch)),
            (False, three_planes_raw(tn_patch_ipl, center_ipl_patch)),
            (True, three_planes_overlay(tn_patch_ipl, tn_lbl_ipl, center_ipl_patch)),
        ]
        letters = ["I", "P", "L"]

    elif view == "ras":
        center_ras_full = (
            int(np.clip(round(cz), 0, vol.shape[0] - 1)),
            int(np.clip(round(cy), 0, vol.shape[1] - 1)),
            int(np.clip(round(cx), 0, vol.shape[2] - 1)),
        )
        center_ras_patch = (24, 24, 24)
        score = float((zscore(patch_for_rows) * zscore(tn_patch_for_rows)).mean())

        ims_full_raw = three_planes_raw_ras(vol, center_ras_full)
        ims_full_lbl = three_planes_overlay_ras(vol, full_lbl_zyx, center_ras_full)

        half_ras_zyx = (half_mm / sz, half_mm / sy, half_mm / sx)
        ims_full_raw = draw_physical_box_ras(ims_full_raw, center_ras_full, half_ras_zyx)
        ims_full_lbl = draw_physical_box_ras(ims_full_lbl, center_ras_full, half_ras_zyx)

        rows = [
            (False, ims_full_raw),
            (True, ims_full_lbl),
            (False, three_planes_raw_ras(patch_for_rows, center_ras_patch)),
            (True, three_planes_overlay_ras(patch_for_rows, patch_lbl_for_rows, center_ras_patch)),
            (False, three_planes_raw_ras(tn_patch_for_rows, center_ras_patch)),
            (True, three_planes_overlay_ras(tn_patch_for_rows, tn_lbl_for_rows, center_ras_patch)),
        ]
        if ras_rotate_180:
            rows = [(is_label_row, rotate_tiles_180(ims)) for is_label_row, ims in rows]
        letters = ["A", "C", "S"]

    else:
        raise ValueError(f"Unsupported view: {view}")

    title_h = 0
    row_gap = 12
    fig_w = cell * 3
    fig_h = title_h + len(rows) * cell + (len(rows) - 1) * row_gap
    canvas = Image.new("RGB", (fig_w, fig_h), (18, 18, 18))

    y = title_h
    for is_label_row, ims in rows:
        for i, im in enumerate(ims):
            resized = im.resize((cell, cell), Image.NEAREST if is_label_row else Image.BILINEAR)
            resized = stamp_letter(resized, letters[i])
            canvas.paste(resized, (i * cell, y))
        y += cell + row_gap

    return canvas, score


def make_overview(left: Image.Image, right: Image.Image) -> Image.Image:
    gap = 16
    out = Image.new("RGB", (left.width + gap + right.width, max(left.height, right.height)), (12, 12, 12))
    out.paste(left, (0, 0))
    out.paste(right, (left.width + gap, 0))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case-id", required=True)
    # p.add_argument("--reg-root", default="/path/to/TN_Reg")
    # p.add_argument(
    #     "--tn-seg-root",
    #     default="/path/to/TN_Seg",
    # )
    p.add_argument("--reg-root", required=True, help="Project root directory")
    p.add_argument("--tn-seg-root", required=True, help="TN segmentation root directory")
    p.add_argument("--iso-mm", type=float, default=0.47)
    p.add_argument("--shift-dz", type=float, default=0.0)
    p.add_argument("--shift-dy", type=float, default=0.0)
    p.add_argument("--shift-dx", type=float, default=0.0)
    p.add_argument("--suffix", default="")

    p.add_argument("--view", default="ipl", choices=["ipl", "ras"])
    p.add_argument("--ras-rotate-180", action="store_true")
    p.add_argument("--align-lower-to-upper", action="store_true")

    p.add_argument(
        "--centroid-space", default="auto", choices=["auto", "source", "raw"],
    )

    p.add_argument("--raw-dicom-root", default="")
    p.add_argument("--raw-series-max-depth", type=int, default=8)
    p.add_argument("--raw-series-min-files", type=int, default=10)
    p.add_argument("--raw-series-min-bytes", type=int, default=512)

    p.add_argument("--reg-nifti", default="")
    p.add_argument("--nifti-axis-order", default="xyz_to_zyx", choices=["xyz_to_zyx", "zyx"])

    # ── NEW parameters ──
    p.add_argument("--out-dir", default="",
                   help="Output directory. If empty, uses preview_<case>/standardized/.")
    p.add_argument("--compare-nifti", default="",
                   help="Second NIfTI for MRI-vs-MRA side-by-side comparison.")
    p.add_argument("--compare-nifti-axis-order", default="xyz_to_zyx",
                   choices=["xyz_to_zyx", "zyx"])

    args = p.parse_args()

    case_id = args.case_id.strip()
    reg_root = Path(args.reg_root).resolve()
    tn_root = Path(args.tn_seg_root).resolve()

    # ── Output directory ──
    if args.out_dir.strip():
        out_dir = Path(args.out_dir.strip())
        if not out_dir.is_absolute():
            out_dir = (reg_root / out_dir).resolve()
    else:
        out_dir = reg_root / f"preview_{case_id}" / "standardized"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_id = case_to_csv_id(case_id)
    ipsi_csv = read_centroid(tn_root / "202022_centroids_ipsilateral.csv", csv_id)
    contra_csv = read_centroid(tn_root / "202022_centroids_contralateral.csv", csv_id)

    # ── Load target volume ──
    if args.reg_nifti.strip():
        nifti_path = Path(args.reg_nifti.strip())
        if not nifti_path.is_absolute():
            candidate = reg_root / nifti_path
            nifti_path = candidate if candidate.exists() else nifti_path
        nifti_path = nifti_path.resolve()
        if not nifti_path.exists():
            raise FileNotFoundError(f"NIfTI volume not found: {nifti_path}")

        vol_data = load_volume(nifti_path, input_kind="nifti", nifti_axis_order=args.nifti_axis_order)
        volume_kind = "nifti"
        volume_source = str(nifti_path)

    else:
        default_series = reg_root / "data_mri" / case_id / "A5IKQO4I" / "5N3PYHRX"
        if default_series.exists():
            dicom_dir = default_series
        else:
            case_root = reg_root / "data_mri" / case_id
            found = find_best_dicom_series_dir(case_root)
            if found is None:
                raise FileNotFoundError(
                    f"DICOM series directory not found.\n"
                    f"Tried default: {default_series}\n"
                    f"And auto-search under: {case_root}"
                )
            dicom_dir = found

        vol_data = load_volume(dicom_dir, input_kind="dicom")
        volume_kind = "dicom"
        volume_source = str(dicom_dir)

    shift_zyx = (args.shift_dz, args.shift_dy, args.shift_dx)

    def shifted_csv_center_zyx(csv_coord: Tuple[int, int, int]) -> Tuple[float, float, float]:
        base = csv_to_zyx(csv_coord)
        return (base[0] + shift_zyx[0], base[1] + shift_zyx[1], base[2] + shift_zyx[2])

    # ── RAW DICOM -> NIfTI centroid mapping ──
    centroid_space_used = "source"
    raw_volume_source = ""
    raw_to_source_transform: np.ndarray | None = None
    ipsi_center_override: Tuple[float, float, float] | None = None
    contra_center_override: Tuple[float, float, float] | None = None

    want_raw_centroid = args.centroid_space == "raw" or (args.centroid_space == "auto" and volume_kind == "nifti")
    if want_raw_centroid and volume_kind == "nifti":
        if args.raw_dicom_root.strip():
            raw_root = Path(args.raw_dicom_root.strip())
            if not raw_root.is_absolute():
                raw_root = (reg_root / raw_root).resolve()
            raw_root = raw_root.resolve()
        else:
            raw_root = (reg_root / "data" / "raw" / "mri").resolve()

        case_root = raw_root / case_id
        series_dir = find_best_dicom_series_dir(
            case_root, max_depth=args.raw_series_max_depth,
            min_files=args.raw_series_min_files, min_bytes=args.raw_series_min_bytes,
        )

        if series_dir is None:
            if args.centroid_space == "raw":
                raise FileNotFoundError(
                    f"Raw DICOM series not found under: {case_root}\n"
                    f"Tip: pass --raw-dicom-root explicitly or increase --raw-series-max-depth"
                )
        else:
            raw_vol = load_volume(series_dir, input_kind="dicom")
            raw_to_source_transform = volume_transform(raw_vol, vol_data)
            centroid_space_used = "raw"
            raw_volume_source = str(series_dir)

            ipsi_center_override = map_center_between_volumes(
                shifted_csv_center_zyx(ipsi_csv), raw_vol, vol_data,
            )
            contra_center_override = map_center_between_volumes(
                shifted_csv_center_zyx(contra_csv), raw_vol, vol_data,
            )

    # ── Load TN patches + labels ──
    ipsi_in = load_tiff_stack(tn_root / "202022_Ipsilateral_Input" / f"{case_id}_cropped.tiff")
    ipsi_lb = load_tiff_stack(tn_root / "202022_Ipsilateral_Target" / f"{case_id}_mask.tiff")
    contra_in = load_tiff_stack(tn_root / "202022_Contralateral_Input" / f"{case_id}_cropped.tiff")
    contra_lb = load_tiff_stack(tn_root / "202022_Contralateral_Target" / f"{case_id}_mask.tiff")

    contra_fig, contra_score = build_side(
        "contra", case_id, contra_csv, vol_data, contra_in, contra_lb, args.iso_mm,
        shift_zyx=shift_zyx, center_zyx_override=contra_center_override,
        raw_to_source_transform=raw_to_source_transform, view=args.view,
        ras_rotate_180=args.ras_rotate_180, align_lower_to_upper=args.align_lower_to_upper,
    )

    ipsi_fig, ipsi_score = build_side(
        "ipsi", case_id, ipsi_csv, vol_data, ipsi_in, ipsi_lb, args.iso_mm,
        shift_zyx=shift_zyx, center_zyx_override=ipsi_center_override,
        raw_to_source_transform=raw_to_source_transform, view=args.view,
        ras_rotate_180=args.ras_rotate_180, align_lower_to_upper=args.align_lower_to_upper,
    )

    overview = make_overview(ipsi_fig, contra_fig)

    # ── Save outputs + manifest ──
    suffix = f"_{args.suffix}" if args.suffix else ""
    view_tag = args.view.upper()

    ipsi_out = out_dir / f"{case_id}_ipsi_{view_tag}{suffix}.png"
    contra_out = out_dir / f"{case_id}_contra_{view_tag}{suffix}.png"
    overview_out = out_dir / f"{case_id}_{view_tag}_overview{suffix}.png"
    manifest = out_dir / f"{case_id}_{view_tag}_manifest{suffix}.txt"

    ipsi_fig.save(ipsi_out)
    contra_fig.save(contra_out)
    overview.save(overview_out)

    with manifest.open("w") as f:
        f.write(f"{view_tag} view manifest\n")
        f.write(f"case_id={case_id}\n")
        f.write(f"csv_case_id={csv_id}\n")
        f.write(f"view={args.view}\n")
        if args.view == "ipl":
            f.write("columns=I,P,L\n")
            f.write("reorient=[z,y,x]->transpose(1,2,0)->flip(axis2)\n")
        else:
            f.write("columns=A,C,S\n")
            f.write("planes=axial[z,:,:],coronal[:,y,:],sagittal[:,:,x]\n")
            f.write(f"ras_rotate_180={args.ras_rotate_180}\n")

        lower_rows_space = "source" if args.align_lower_to_upper else "raw"
        f.write(f"align_lower_to_upper={args.align_lower_to_upper}\n")
        f.write(f"lower_rows_space={lower_rows_space}\n")
        f.write(f"volume_kind={volume_kind}\n")
        f.write(f"volume_source={volume_source}\n")
        if volume_kind == "nifti":
            f.write(f"nifti_axis_order={args.nifti_axis_order}\n")
        f.write(f"centroid_space={centroid_space_used}\n")
        if args.raw_dicom_root.strip():
            f.write(f"raw_dicom_root_arg={args.raw_dicom_root.strip()}\n")
        else:
            f.write(f"raw_dicom_root_default={(reg_root / 'data' / 'raw' / 'mri').resolve()}\n")
        if raw_volume_source:
            f.write(f"raw_volume_source={raw_volume_source}\n")
        if raw_to_source_transform is not None:
            f.write("raw_to_source_transform=\n")
            for row in raw_to_source_transform:
                f.write(",".join(f"{float(v):.8f}" for v in row) + "\n")
            if ipsi_center_override is not None:
                f.write(
                    f"ipsi_center_mapped_zyx=({ipsi_center_override[0]:.6f},{ipsi_center_override[1]:.6f},{ipsi_center_override[2]:.6f})\n"
                )
            if contra_center_override is not None:
                f.write(
                    f"contra_center_mapped_zyx=({contra_center_override[0]:.6f},{contra_center_override[1]:.6f},{contra_center_override[2]:.6f})\n"
                )
        f.write(f"shift_dz={args.shift_dz}\n")
        f.write(f"shift_dy={args.shift_dy}\n")
        f.write(f"shift_dx={args.shift_dx}\n")
        f.write(f"ipsi_score={ipsi_score:.6f}\n")
        f.write(f"contra_score={contra_score:.6f}\n")

    print(ipsi_out)
    print(contra_out)
    print(overview_out)
    print(manifest)

    # ── Optional: MRI vs MRA compare images ──
    if args.compare_nifti.strip():
        cmp_path = Path(args.compare_nifti.strip())
        if not cmp_path.is_absolute():
            candidate = reg_root / cmp_path
            cmp_path = candidate if candidate.exists() else cmp_path
        if not cmp_path.exists():
            raise FileNotFoundError(f"Compare NIfTI not found: {cmp_path}")

        cmp_vol = load_volume(
            cmp_path, input_kind="nifti", nifti_axis_order=args.compare_nifti_axis_order,
        )

        # Centers in vol_data (the primary volume, e.g. MRA warped)
        def _center_for_vol(csv_coord, override):
            if override is not None:
                return override
            base = csv_to_zyx(csv_coord)
            return (base[0] + shift_zyx[0], base[1] + shift_zyx[1], base[2] + shift_zyx[2])

        ipsi_center_src = _center_for_vol(ipsi_csv, ipsi_center_override)
        contra_center_src = _center_for_vol(contra_csv, contra_center_override)

        # Map centers from vol_data space -> compare volume space
        ipsi_center_cmp = map_center_between_volumes(ipsi_center_src, vol_data, cmp_vol)
        contra_center_cmp = map_center_between_volumes(contra_center_src, vol_data, cmp_vol)

        tagA = "MRA_warped" if "MRA" in (args.suffix or "").upper() else "VOL_A"
        tagB = "MRI_fixed"

        ipsi_cmp_fig = build_two_volume_compare(
            volA=vol_data, centerA_zyx=ipsi_center_src,
            volB=cmp_vol, centerB_zyx=ipsi_center_cmp,
            iso_mm=args.iso_mm, view=args.view, ras_rotate_180=args.ras_rotate_180,
            tagA=tagA, tagB=tagB,
        )
        contra_cmp_fig = build_two_volume_compare(
            volA=vol_data, centerA_zyx=contra_center_src,
            volB=cmp_vol, centerB_zyx=contra_center_cmp,
            iso_mm=args.iso_mm, view=args.view, ras_rotate_180=args.ras_rotate_180,
            tagA=tagA, tagB=tagB,
        )

        ipsi_cmp_out = out_dir / f"{case_id}_ipsi_{view_tag}_mri_vs_mra{suffix}.png"
        contra_cmp_out = out_dir / f"{case_id}_contra_{view_tag}_mri_vs_mra{suffix}.png"

        ipsi_cmp_fig.save(ipsi_cmp_out)
        contra_cmp_fig.save(contra_cmp_out)

        print(ipsi_cmp_out)
        print(contra_cmp_out)


if __name__ == "__main__":
    main()