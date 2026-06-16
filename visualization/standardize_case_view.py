#!/usr/bin/env python3
"""
Standardized visualization for TN centroid/crop alignment.

Outputs per case:
1) {case}_ipsi_STANDARD.png
2) {case}_contra_STANDARD.png
3) {case}_STANDARD_overview.png
4) {case}_STANDARD_manifest.txt

Layout (each side figure): 5 rows x 3 columns
- Row 1: Full brain with physical prediction box
- Row 2: REG isotropic crop (raw)
- Row 3: REG isotropic crop + mapped TN label
- Row 4: TN original crop (raw)
- Row 5: TN original crop + TN label

Author: Xupeng Zhang
Johns Hopkins University
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageSequence


LONG_VR = {b"OB", b"OD", b"OF", b"OL", b"OW", b"SQ", b"UC", b"UR", b"UT", b"UN"}


@dataclass
class DicomSlice:
    instance: Optional[int]
    ipp: Optional[List[float]]
    iop: Optional[List[float]]
    pixel_spacing: Optional[List[float]]
    pixel: np.ndarray  # 2D float32


@dataclass
class VolumeData:
    vol: np.ndarray  # [z, y, x]
    spacing_zyx: Tuple[float, float, float]  # [sz, sy, sx] in mm
    direction_text: str  # human-readable axis directions in RAS
    tile_corner_text: Dict[str, str]  # per-view single-letter corner label
    affine_internal_to_ras: Optional[np.ndarray] = None  # 4x4, internal [z,y,x,1] -> RAS
    source_kind: str = "unknown"


def _dominant_axis_code_ras(v: np.ndarray) -> str:
    i = int(np.argmax(np.abs(v)))
    if i == 0:
        return "R" if v[0] >= 0 else "L"
    if i == 1:
        return "A" if v[1] >= 0 else "P"
    return "S" if v[2] >= 0 else "I"


def _full_axis_code_ras(v: np.ndarray) -> str:
    comps = [
        ("R" if v[0] >= 0 else "L", abs(float(v[0]))),
        ("A" if v[1] >= 0 else "P", abs(float(v[1]))),
        ("S" if v[2] >= 0 else "I", abs(float(v[2]))),
    ]
    comps.sort(key=lambda t: t[1], reverse=True)
    return "".join([c for c, _ in comps])


def _neg_code(code: str) -> str:
    table = {"R": "L", "L": "R", "A": "P", "P": "A", "S": "I", "I": "S"}
    return "".join(table.get(ch, ch) for ch in code)


def _parse_dicom_explicit_le(path: Path) -> DicomSlice:
    data = path.read_bytes()
    off = 132 if data[128:132] == b"DICM" else 0

    while off + 8 <= len(data):
        g, e = struct.unpack_from("<HH", data, off)
        if g != 0x0002:
            break
        off += 4
        vr = data[off : off + 2]
        off += 2
        if vr in LONG_VR:
            off += 2
            ln = struct.unpack_from("<I", data, off)[0]
            off += 4
        else:
            ln = struct.unpack_from("<H", data, off)[0]
            off += 2
        off += ln

    tags: Dict[Tuple[int, int], Tuple[bytes, bytes]] = {}
    pixel_blob: Optional[bytes] = None
    while off + 8 <= len(data):
        g, e = struct.unpack_from("<HH", data, off)
        off += 4
        vr = data[off : off + 2]
        off += 2
        if vr in LONG_VR:
            off += 2
            ln = struct.unpack_from("<I", data, off)[0]
            off += 4
        else:
            ln = struct.unpack_from("<H", data, off)[0]
            off += 2
        if off + ln > len(data):
            break
        val = data[off : off + ln]
        off += ln
        tags[(g, e)] = (vr, val)
        if (g, e) == (0x7FE0, 0x0010):
            pixel_blob = val
            break

    def get_int(tag: Tuple[int, int], default: Optional[int] = None) -> Optional[int]:
        item = tags.get(tag)
        if not item:
            return default
        vr, val = item
        if vr == b"US":
            return int(struct.unpack("<H", val[:2])[0])
        if vr == b"SS":
            return int(struct.unpack("<h", val[:2])[0])
        try:
            return int(val.decode("ascii", "ignore").strip("\x00 ").split("\\")[0])
        except Exception:
            return default

    def get_floats(tag: Tuple[int, int]) -> Optional[List[float]]:
        item = tags.get(tag)
        if not item:
            return None
        _, val = item
        txt = val.decode("ascii", "ignore").strip("\x00 ")
        try:
            return [float(x) for x in txt.split("\\") if x]
        except Exception:
            return None

    rows = get_int((0x0028, 0x0010))
    cols = get_int((0x0028, 0x0011))
    pixel_rep = get_int((0x0028, 0x0103), 0)
    instance = get_int((0x0020, 0x0013), None)
    ipp = get_floats((0x0020, 0x0032))
    iop = get_floats((0x0020, 0x0037))
    pixel_spacing = get_floats((0x0028, 0x0030))

    slope = 1.0
    inter = 0.0
    if (0x0028, 0x1053) in tags:
        try:
            slope = float(tags[(0x0028, 0x1053)][1].decode("ascii", "ignore").strip("\x00 "))
        except Exception:
            pass
    if (0x0028, 0x1052) in tags:
        try:
            inter = float(tags[(0x0028, 0x1052)][1].decode("ascii", "ignore").strip("\x00 "))
        except Exception:
            pass

    if rows is None or cols is None or pixel_blob is None:
        raise ValueError(f"Missing pixel metadata in {path}")

    dt = np.int16 if pixel_rep == 1 else np.uint16
    arr = np.frombuffer(pixel_blob, dtype=dt, count=rows * cols).reshape(rows, cols).astype(np.float32)
    arr = arr * slope + inter

    return DicomSlice(
        instance=instance,
        ipp=ipp,
        iop=iop,
        pixel_spacing=pixel_spacing,
        pixel=arr,
    )


def load_dicom_volume(dicom_series_dir: Path) -> VolumeData:
    dicom_files = sorted([p for p in dicom_series_dir.iterdir() if p.is_file() and p.name.startswith("I")])
    slices = [_parse_dicom_explicit_le(p) for p in dicom_files]
    slices = [s for s in slices if s.instance is not None or s.ipp is not None]
    if not slices:
        raise ValueError(f"No DICOM slices found in {dicom_series_dir}")

    direction_text = "DIR(+x,+y,+z)=unknown"
    tile_corner_text: Dict[str, str] = {
        "axial": "AX",
        "coronal": "CO",
        "sagittal": "SA",
    }
    row_lps: Optional[np.ndarray] = None
    col_lps: Optional[np.ndarray] = None
    if slices[0].iop is None:
        slices.sort(key=lambda s: s.instance if s.instance is not None else 0)
        proj = np.array([float(i) for i in range(len(slices))], dtype=np.float64)
    else:
        row_lps = np.array(slices[0].iop[:3], dtype=np.float64)
        col_lps = np.array(slices[0].iop[3:6], dtype=np.float64)
        # DICOM orientation is in LPS; convert vectors to RAS for human-readable labels.
        lps_to_ras = np.diag([-1.0, -1.0, 1.0])
        row_ras = lps_to_ras @ row_lps
        col_ras = lps_to_ras @ col_lps
        normal_ras = lps_to_ras @ np.cross(row_lps, col_lps)
        direction_text = (
            "DIR(+x,+y,+z)="
            f"{_dominant_axis_code_ras(row_ras)},"
            f"{_dominant_axis_code_ras(col_ras)},"
            f"{_dominant_axis_code_ras(normal_ras)}"
            " | full="
            f"{_full_axis_code_ras(row_ras)},"
            f"{_full_axis_code_ras(col_ras)},"
            f"{_full_axis_code_ras(normal_ras)}"
        )
        # For display arrays:
        # axial [y,x]    : left=-row, up=-col
        # coronal [z,x]  : left=-row, up=-normal
        # sagittal [z,y] : left=-col, up=-normal
        row_code = _full_axis_code_ras(row_ras)
        col_code = _full_axis_code_ras(col_ras)
        nor_code = _full_axis_code_ras(normal_ras)
        # Single-letter corner label rule: use the dominant "up direction" letter per view.
        up_ax = _dominant_axis_code_ras(-col_ras)     # axial up
        up_co = _dominant_axis_code_ras(-normal_ras)  # coronal up
        up_sa = _dominant_axis_code_ras(-normal_ras)  # sagittal up
        tile_corner_text = {
            "axial": up_ax,
            "coronal": up_co,
            "sagittal": up_sa,
        }

        normal = np.cross(row_lps, col_lps)
        proj = []
        for s in slices:
            if s.ipp is not None:
                proj.append(float(np.dot(np.array(s.ipp, dtype=np.float64), normal)))
            else:
                proj.append(float(s.instance if s.instance is not None else 0))
        proj = np.array(proj, dtype=np.float64)
        order = np.argsort(proj)
        proj = proj[order]
        slices = [slices[i] for i in order]

    vol = np.stack([s.pixel for s in slices], axis=0)
    if slices[0].pixel_spacing is None:
        raise ValueError("Missing PixelSpacing in DICOM")
    sy = float(slices[0].pixel_spacing[0])
    sx = float(slices[0].pixel_spacing[1])
    if len(proj) >= 2:
        sz = float(np.median(np.abs(np.diff(proj))))
    else:
        sz = sy

    affine_internal_to_ras: Optional[np.ndarray] = None
    if row_lps is not None and col_lps is not None and slices[0].ipp is not None:
        lps_to_ras = np.diag([-1.0, -1.0, 1.0])
        normal_lps = np.cross(row_lps, col_lps)
        m_xyz = np.eye(4, dtype=np.float64)
        # DICOM voxel xyz order here means [x(col), y(row), z(slice)].
        m_xyz[:3, 0] = lps_to_ras @ (row_lps * sx)
        m_xyz[:3, 1] = lps_to_ras @ (col_lps * sy)
        m_xyz[:3, 2] = lps_to_ras @ (normal_lps * sz)
        m_xyz[:3, 3] = lps_to_ras @ np.array(slices[0].ipp, dtype=np.float64)
        # Internal array is [z,y,x], so convert [z,y,x] -> [x,y,z] first.
        perm_zyx_to_xyz = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        affine_internal_to_ras = m_xyz @ perm_zyx_to_xyz
    return VolumeData(
        vol=vol,
        spacing_zyx=(sz, sy, sx),
        direction_text=direction_text,
        tile_corner_text=tile_corner_text,
        affine_internal_to_ras=affine_internal_to_ras,
        source_kind="dicom",
    )


def _nifti_dtype(datatype: int, endian: str) -> np.dtype:
    table = {
        2: np.uint8,
        4: np.int16,
        8: np.int32,
        16: np.float32,
        64: np.float64,
        256: np.int8,
        512: np.uint16,
        768: np.uint32,
    }
    if datatype not in table:
        raise ValueError(f"Unsupported NIfTI datatype code: {datatype}")
    return np.dtype(table[datatype]).newbyteorder(endian)


def load_nifti_volume(nifti_path: Path, nifti_axis_order: str = "xyz_to_zyx") -> VolumeData:
    if not nifti_path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {nifti_path}")

    opener = gzip.open if str(nifti_path).endswith(".gz") else open
    with opener(nifti_path, "rb") as f:
        blob = f.read()
    if len(blob) < 352:
        raise ValueError(f"NIfTI file too small: {nifti_path}")

    hdr = blob[:348]
    sizeof_hdr_le = struct.unpack("<I", hdr[0:4])[0]
    if sizeof_hdr_le == 348:
        endian = "<"
    else:
        sizeof_hdr_be = struct.unpack(">I", hdr[0:4])[0]
        if sizeof_hdr_be == 348:
            endian = ">"
        else:
            raise ValueError(f"Invalid NIfTI header size in {nifti_path}: {sizeof_hdr_le}")

    dim = struct.unpack(endian + "8h", hdr[40:56])
    ndim = int(dim[0])
    if ndim < 3:
        raise ValueError(f"NIfTI is not 3D: {nifti_path}")
    if ndim >= 4 and int(dim[4]) > 1:
        raise ValueError(f"NIfTI 4D volume is not supported: {nifti_path}")

    nx = int(dim[1])
    ny = int(dim[2])
    nz = int(dim[3])
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError(f"Invalid NIfTI dims in {nifti_path}: {(nx, ny, nz)}")

    pixdim = struct.unpack(endian + "8f", hdr[76:108])
    datatype = struct.unpack(endian + "h", hdr[70:72])[0]
    vox_offset = struct.unpack(endian + "f", hdr[108:112])[0]
    sform_code = struct.unpack(endian + "h", hdr[254:256])[0]
    srow_x = struct.unpack(endian + "4f", hdr[280:296])
    srow_y = struct.unpack(endian + "4f", hdr[296:312])
    srow_z = struct.unpack(endian + "4f", hdr[312:328])

    offset = int(round(float(vox_offset)))
    dtype = _nifti_dtype(int(datatype), endian)
    nvox = int(nx * ny * nz)
    need = offset + nvox * dtype.itemsize
    if need > len(blob):
        raise ValueError(f"NIfTI data section is incomplete: {nifti_path}")

    # NIfTI stores axis-0 as the fastest-changing index.
    data_xyz = np.frombuffer(blob, dtype=dtype, count=nvox, offset=offset).reshape((nx, ny, nz), order="F")
    data_xyz = data_xyz.astype(np.float32, copy=False)

    axis_mode = nifti_axis_order.strip().lower()
    if axis_mode not in {"xyz_to_zyx", "zyx"}:
        raise ValueError(f"Unsupported nifti_axis_order: {nifti_axis_order}")

    if axis_mode == "xyz_to_zyx":
        vol = np.transpose(data_xyz, (2, 1, 0))
        spacing_zyx = (
            float(abs(pixdim[3])) if float(abs(pixdim[3])) > 0 else 1.0,
            float(abs(pixdim[2])) if float(abs(pixdim[2])) > 0 else 1.0,
            float(abs(pixdim[1])) if float(abs(pixdim[1])) > 0 else 1.0,
        )
    else:
        vol = data_xyz
        spacing_zyx = (
            float(abs(pixdim[1])) if float(abs(pixdim[1])) > 0 else 1.0,
            float(abs(pixdim[2])) if float(abs(pixdim[2])) > 0 else 1.0,
            float(abs(pixdim[3])) if float(abs(pixdim[3])) > 0 else 1.0,
        )

    affine_xyz_to_ras = np.eye(4, dtype=np.float64)
    if int(sform_code) > 0:
        affine_xyz_to_ras[0, :] = np.array(srow_x, dtype=np.float64)
        affine_xyz_to_ras[1, :] = np.array(srow_y, dtype=np.float64)
        affine_xyz_to_ras[2, :] = np.array(srow_z, dtype=np.float64)
    else:
        # Fallback when no sform is present.
        affine_xyz_to_ras[0, 0] = float(abs(pixdim[1])) if float(abs(pixdim[1])) > 0 else 1.0
        affine_xyz_to_ras[1, 1] = float(abs(pixdim[2])) if float(abs(pixdim[2])) > 0 else 1.0
        affine_xyz_to_ras[2, 2] = float(abs(pixdim[3])) if float(abs(pixdim[3])) > 0 else 1.0

    direction_text = f"DIR(+x,+y,+z)=unknown | NIFTI axis_order={axis_mode}"
    tile_corner_text: Dict[str, str] = {
        "axial": "AX",
        "coronal": "CO",
        "sagittal": "SA",
    }
    vi = affine_xyz_to_ras[:3, 0]
    vj = affine_xyz_to_ras[:3, 1]
    vk = affine_xyz_to_ras[:3, 2]
    if axis_mode == "xyz_to_zyx":
        x_ras, y_ras, z_ras = vi, vj, vk
    else:
        # Treat input axis order as [z,y,x] when explicitly requested.
        z_ras, y_ras, x_ras = vi, vj, vk

    direction_text = (
        "DIR(+x,+y,+z)="
        f"{_dominant_axis_code_ras(x_ras)},"
        f"{_dominant_axis_code_ras(y_ras)},"
        f"{_dominant_axis_code_ras(z_ras)}"
        " | full="
        f"{_full_axis_code_ras(x_ras)},"
        f"{_full_axis_code_ras(y_ras)},"
        f"{_full_axis_code_ras(z_ras)}"
        f" | NIFTI axis_order={axis_mode}"
    )
    tile_corner_text = {
        "axial": _dominant_axis_code_ras(-y_ras),
        "coronal": _dominant_axis_code_ras(-z_ras),
        "sagittal": _dominant_axis_code_ras(-z_ras),
    }

    affine_internal_to_ras: Optional[np.ndarray] = None
    if axis_mode == "xyz_to_zyx":
        perm_zyx_to_xyz = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        affine_internal_to_ras = affine_xyz_to_ras @ perm_zyx_to_xyz

    return VolumeData(
        vol=vol,
        spacing_zyx=spacing_zyx,
        direction_text=direction_text,
        tile_corner_text=tile_corner_text,
        affine_internal_to_ras=affine_internal_to_ras,
        source_kind="nifti",
    )


def load_volume(source: Path, input_kind: str = "auto", nifti_axis_order: str = "xyz_to_zyx") -> VolumeData:
    kind = input_kind.strip().lower()
    if kind not in {"auto", "dicom", "nifti"}:
        raise ValueError(f"Unsupported input_kind: {input_kind}")

    if kind == "auto":
        if source.is_dir():
            kind = "dicom"
        elif source.is_file() and (str(source).endswith(".nii") or str(source).endswith(".nii.gz")):
            kind = "nifti"
        else:
            raise ValueError(f"Could not infer volume type from source: {source}")

    if kind == "dicom":
        return load_dicom_volume(source)
    return load_nifti_volume(source, nifti_axis_order=nifti_axis_order)


def volume_transform(src: VolumeData, dst: VolumeData) -> np.ndarray:
    """Return 4x4 transform mapping src internal [z,y,x,1] -> dst internal [z,y,x,1]."""
    if src.affine_internal_to_ras is None:
        raise ValueError("Source volume has no affine_internal_to_ras")
    if dst.affine_internal_to_ras is None:
        raise ValueError("Destination volume has no affine_internal_to_ras")
    return np.linalg.inv(dst.affine_internal_to_ras) @ src.affine_internal_to_ras


def map_center_between_volumes(
    center_src_zyx: Tuple[float, float, float],
    src: VolumeData,
    dst: VolumeData,
) -> Tuple[float, float, float]:
    t = volume_transform(src, dst)
    q = t @ np.array([center_src_zyx[0], center_src_zyx[1], center_src_zyx[2], 1.0], dtype=np.float64)
    return float(q[0]), float(q[1]), float(q[2])


def _axis_map_from_linear(src_to_dst_linear: np.ndarray) -> Tuple[Tuple[int, int, int], Tuple[bool, bool, bool]]:
    """
    For src->dst transform, return:
    - perm[src_axis] = dst_axis
    - flips[src_axis] = True if dst axis is opposite direction.
    """
    l = np.asarray(src_to_dst_linear, dtype=np.float64)
    if l.shape != (3, 3):
        raise ValueError("Expected 3x3 linear transform")

    perm = [-1, -1, -1]
    flips = [False, False, False]
    used_dst: set[int] = set()
    for src_ax in range(3):
        col = np.abs(l[:, src_ax])
        dst_ax = int(np.argmax(col))
        if dst_ax in used_dst:
            raise ValueError("Transform is not a one-to-one axis permutation")
        if float(col[dst_ax]) < 0.8:
            raise ValueError("Transform is not near axis-aligned permutation/flip")
        used_dst.add(dst_ax)
        perm[src_ax] = dst_ax
        flips[src_ax] = bool(l[dst_ax, src_ax] < 0)
    return (int(perm[0]), int(perm[1]), int(perm[2])), (bool(flips[0]), bool(flips[1]), bool(flips[2]))


def reorient_patch_dst_to_src(patch_dst: np.ndarray, src_to_dst: np.ndarray) -> np.ndarray:
    """
    Reorient patch from dst axes into src axes, using src->dst transform.
    Suitable when values should be compared in src orientation.
    """
    perm, flips = _axis_map_from_linear(src_to_dst[:3, :3])
    out = np.transpose(patch_dst, perm)
    for ax, do_flip in enumerate(flips):
        if do_flip:
            out = np.flip(out, axis=ax)
    return out


def reorient_patch_src_to_dst(patch_src: np.ndarray, src_to_dst: np.ndarray) -> np.ndarray:
    """Reorient patch from src axes into dst axes, inverse of reorient_patch_dst_to_src."""
    return reorient_patch_dst_to_src(patch_src, np.linalg.inv(src_to_dst))


def load_tiff_stack(path: Path) -> np.ndarray:
    im = Image.open(path)
    return np.stack([np.array(fr, dtype=np.float32) for fr in ImageSequence.Iterator(im)], axis=0)


def case_to_csv_id(case_id: str) -> str:
    s = case_id.lstrip("0")
    return s if s else "0"


def parse_centroid_row(row: List[str]) -> Optional[Tuple[str, Tuple[int, int, int]]]:
    if not row:
        return None
    head = row[0].strip()
    if not head or not re.fullmatch(r"\d+", head):
        return None
    nums: List[int] = []
    for token in row[1:]:
        for part in re.split(r"[,\s]+", token.strip()):
            if part:
                if re.fullmatch(r"-?\d+", part):
                    nums.append(int(part))
    if len(nums) < 3:
        return None
    return head, (nums[0], nums[1], nums[2])  # csv order: z, x, y


def read_centroid(csv_path: Path, case_id_csv: str) -> Tuple[int, int, int]:
    with csv_path.open() as f:
        reader = csv.reader(f)
        for row in reader:
            parsed = parse_centroid_row(row)
            if parsed is None:
                continue
            cid, coords = parsed
            if cid == case_id_csv:
                return coords
    raise ValueError(f"Case {case_id_csv} not found in {csv_path}")


def normalize_2d(img: np.ndarray) -> np.ndarray:
    arr = img.astype(np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    out = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (out * 255).astype(np.uint8)


def overlay_label(gray_u8: np.ndarray, lbl_2d: np.ndarray) -> np.ndarray:
    rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1).astype(np.float32)
    alpha = 0.62
    color1 = np.array([255, 70, 70], dtype=np.float32)   # label 1
    color2 = np.array([70, 220, 255], dtype=np.float32)  # label 2
    m1 = lbl_2d == 1
    m2 = lbl_2d == 2
    if np.any(m1):
        rgb[m1] = (1 - alpha) * rgb[m1] + alpha * color1
    if np.any(m2):
        rgb[m2] = (1 - alpha) * rgb[m2] + alpha * color2
    return np.clip(rgb, 0, 255).astype(np.uint8)


def zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-6)


def trilinear(vol: np.ndarray, z: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    z = np.clip(z, 0, vol.shape[0] - 1)
    y = np.clip(y, 0, vol.shape[1] - 1)
    x = np.clip(x, 0, vol.shape[2] - 1)
    z0 = np.floor(z).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)
    z1 = np.clip(z0 + 1, 0, vol.shape[0] - 1)
    y1 = np.clip(y0 + 1, 0, vol.shape[1] - 1)
    x1 = np.clip(x0 + 1, 0, vol.shape[2] - 1)
    dz = (z - z0).astype(np.float32)
    dy = (y - y0).astype(np.float32)
    dx = (x - x0).astype(np.float32)

    c000 = vol[z0, y0, x0]
    c001 = vol[z0, y0, x1]
    c010 = vol[z0, y1, x0]
    c011 = vol[z0, y1, x1]
    c100 = vol[z1, y0, x0]
    c101 = vol[z1, y0, x1]
    c110 = vol[z1, y1, x0]
    c111 = vol[z1, y1, x1]

    c00 = c000 * (1 - dx) + c001 * dx
    c01 = c010 * (1 - dx) + c011 * dx
    c10 = c100 * (1 - dx) + c101 * dx
    c11 = c110 * (1 - dx) + c111 * dx
    c0 = c00 * (1 - dy) + c01 * dy
    c1 = c10 * (1 - dy) + c11 * dy
    c = c0 * (1 - dz) + c1 * dz
    return c.astype(np.float32)


def extract_iso_cube(
    vol: np.ndarray,
    center_zyx: Tuple[float, float, float],
    spacing_zyx: Tuple[float, float, float],
    out_spacing_mm: float,
    size: int = 48,
) -> np.ndarray:
    sz, sy, sx = spacing_zyx
    # center24 convention: offsets -24..23 for size 48
    idx = np.arange(size, dtype=np.float32) - (size // 2)
    oz = idx * (out_spacing_mm / sz)
    oy = idx * (out_spacing_mm / sy)
    ox = idx * (out_spacing_mm / sx)
    zz, yy, xx = np.meshgrid(oz, oy, ox, indexing="ij")
    cz, cy, cx = center_zyx
    return trilinear(vol, cz + zz, cy + yy, cx + xx)


def csv_to_zyx(csv_coord: Tuple[int, int, int]) -> Tuple[float, float, float]:
    # csv order = (z, x, y)
    z, x, y = csv_coord
    return float(z), float(y), float(x)


def tn_to_patch(lbl_tn: np.ndarray) -> np.ndarray:
    # From prior verification:
    # patch->TN = transpose(1,2,0) + flip(axis=2)
    # inverse:
    return np.transpose(np.flip(lbl_tn, axis=2), (2, 0, 1))


def patch_to_tn(patch: np.ndarray) -> np.ndarray:
    return np.flip(np.transpose(patch, (1, 2, 0)), axis=2)


def build_side_figure(
    side_name: str,
    case_id: str,
    csv_coord: Tuple[int, int, int],
    volume: VolumeData,
    tn_input: np.ndarray,
    tn_label: np.ndarray,
    out_spacing_mm: float,
    cell_size: int = 300,
) -> Tuple[Image.Image, float, Tuple[float, float, float]]:
    vol = volume.vol
    sz, sy, sx = volume.spacing_zyx
    center_zyx = csv_to_zyx(csv_coord)
    cz, cy, cx = center_zyx

    patch_iso = extract_iso_cube(vol, center_zyx, volume.spacing_zyx, out_spacing_mm, size=48)
    lbl_patch = tn_to_patch(tn_label)

    score = float((zscore(patch_to_tn(patch_iso)) * zscore(tn_input)).mean())

    # full-brain display planes at centroid
    zi = int(round(cz))
    yi = int(round(cy))
    xi = int(round(cx))
    zi = int(np.clip(zi, 0, vol.shape[0] - 1))
    yi = int(np.clip(yi, 0, vol.shape[1] - 1))
    xi = int(np.clip(xi, 0, vol.shape[2] - 1))

    full_ax = Image.fromarray(normalize_2d(vol[zi, :, :]), mode="L").convert("RGB")
    full_co = Image.fromarray(normalize_2d(vol[:, yi, :]), mode="L").convert("RGB")
    full_sa = Image.fromarray(normalize_2d(vol[:, :, xi]), mode="L").convert("RGB")

    # physical half-size: (48/2) * out_spacing
    half_mm = (48 / 2.0) * out_spacing_mm
    hz = half_mm / sz
    hy = half_mm / sy
    hx = half_mm / sx

    z0, z1 = cz - hz, cz + hz
    y0, y1 = cy - hy, cy + hy
    x0, x1 = cx - hx, cx + hx
    box_color = (255, 210, 50)
    dot_color = (255, 70, 70)

    draw_ax = ImageDraw.Draw(full_ax)
    draw_co = ImageDraw.Draw(full_co)
    draw_sa = ImageDraw.Draw(full_sa)
    draw_ax.rectangle((x0, y0, x1, y1), outline=box_color, width=2)
    draw_co.rectangle((x0, z0, x1, z1), outline=box_color, width=2)
    draw_sa.rectangle((y0, z0, y1, z1), outline=box_color, width=2)
    draw_ax.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), outline=dot_color, width=2)
    draw_co.ellipse((cx - 3, cz - 3, cx + 3, cz + 3), outline=dot_color, width=2)
    draw_sa.ellipse((cy - 3, cz - 3, cy + 3, cz + 3), outline=dot_color, width=2)

    c = 24
    reg_raw_ax = Image.fromarray(normalize_2d(patch_iso[c, :, :]), mode="L").convert("RGB")
    reg_raw_co = Image.fromarray(normalize_2d(patch_iso[:, c, :]), mode="L").convert("RGB")
    reg_raw_sa = Image.fromarray(normalize_2d(patch_iso[:, :, c]), mode="L").convert("RGB")

    reg_lbl_ax = Image.fromarray(overlay_label(normalize_2d(patch_iso[c, :, :]), lbl_patch[c, :, :]), mode="RGB")
    reg_lbl_co = Image.fromarray(overlay_label(normalize_2d(patch_iso[:, c, :]), lbl_patch[:, c, :]), mode="RGB")
    reg_lbl_sa = Image.fromarray(overlay_label(normalize_2d(patch_iso[:, :, c]), lbl_patch[:, :, c]), mode="RGB")

    tn_raw_ax = Image.fromarray(normalize_2d(tn_input[c, :, :]), mode="L").convert("RGB")
    tn_raw_co = Image.fromarray(normalize_2d(tn_input[:, c, :]), mode="L").convert("RGB")
    tn_raw_sa = Image.fromarray(normalize_2d(tn_input[:, :, c]), mode="L").convert("RGB")

    tn_lbl_ax = Image.fromarray(overlay_label(normalize_2d(tn_input[c, :, :]), tn_label[c, :, :]), mode="RGB")
    tn_lbl_co = Image.fromarray(overlay_label(normalize_2d(tn_input[:, c, :]), tn_label[:, c, :]), mode="RGB")
    tn_lbl_sa = Image.fromarray(overlay_label(normalize_2d(tn_input[:, :, c]), tn_label[:, :, c]), mode="RGB")

    rows = [
        ("Full Brain + Physical Box", [full_ax, full_co, full_sa]),
        ("REG Iso Crop (Raw)", [reg_raw_ax, reg_raw_co, reg_raw_sa]),
        ("REG Iso Crop + Mapped TN Label", [reg_lbl_ax, reg_lbl_co, reg_lbl_sa]),
        ("TN Original Crop (Raw)", [tn_raw_ax, tn_raw_co, tn_raw_sa]),
        ("TN Original Crop + TN Label", [tn_lbl_ax, tn_lbl_co, tn_lbl_sa]),
    ]

    def stamp_corner_text(im: Image.Image, txt: str) -> Image.Image:
        draw_local = ImageDraw.Draw(im)
        # Small opaque box at top-left for readability across all intensities.
        draw_local.rectangle((2, 2, 26, 24), fill=(0, 0, 0))
        draw_local.text((6, 5), txt, fill=(255, 255, 255))
        return im

    title_h = 102
    row_gap = 24
    fig_w = cell_size * 3
    fig_h = title_h + len(rows) * cell_size + (len(rows) - 1) * row_gap
    canvas = Image.new("RGB", (fig_w, fig_h), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)

    draw.text(
        (10, 8),
        f"{case_id} | {side_name.upper()} | CSV(z,x,y)={csv_coord} | center zyx=({cz:.2f},{cy:.2f},{cx:.2f})",
        fill=(255, 255, 255),
    )
    draw.text(
        (10, 28),
        f"spacing [sz,sy,sx]=[{sz:.3f},{sy:.3f},{sx:.3f}] mm | iso={out_spacing_mm:.2f} mm | score={score:.4f}",
        fill=(225, 225, 225),
    )
    draw.text(
        (10, 48),
        "Axial / Coronal / Sagittal   |   label1=red, label2=cyan",
        fill=(210, 210, 210),
    )
    draw.text(
        (10, 66),
        volume.direction_text,
        fill=(245, 245, 245),
    )

    y = title_h
    for row_name, ims in rows:
        draw.text((10, y - 18), row_name, fill=(255, 255, 255))
        for col, im in enumerate(ims):
            resized = im.resize((cell_size, cell_size), Image.NEAREST if "Label" in row_name else Image.BILINEAR)
            if col == 0:
                resized = stamp_corner_text(resized, volume.tile_corner_text["axial"])
            elif col == 1:
                resized = stamp_corner_text(resized, volume.tile_corner_text["coronal"])
            else:
                resized = stamp_corner_text(resized, volume.tile_corner_text["sagittal"])
            canvas.paste(resized, (col * cell_size, y))
        y += cell_size + row_gap

    return canvas, score, center_zyx


def make_overview(left: Image.Image, right: Image.Image) -> Image.Image:
    gap = 16
    w = left.width + gap + right.width
    h = max(left.height, right.height) + 34
    out = Image.new("RGB", (w, h), (12, 12, 12))
    d = ImageDraw.Draw(out)
    d.text((10, 8), "Standardized View: left=ipsi, right=contra", fill=(255, 255, 255))
    out.paste(left, (0, 34))
    out.paste(right, (left.width + gap, 34))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate standardized TN/REG visualization for one case.")
    parser.add_argument("--case-id", required=True, help="Case id with leading zero, e.g. <CASE_ID>")
    # parser.add_argument(
    #     "--reg-root",
    #     default="/path/to/TN_Reg",
    #     help="Path to REG root directory",
    # )
    # parser.add_argument(
    #     "--tn-seg-root",
    #     default="/path/to/TN_Seg",
    #     help="Path to TN segmentation directory",
    # )
    p.add_argument("--reg-root", required=True, help="Project root directory")
    p.add_argument("--tn-seg-root", required=True, help="TN segmentation root directory")

    parser.add_argument("--iso-mm", type=float, default=0.47, help="Target isotropic spacing in mm")
    parser.add_argument(
        "--reg-nifti",
        default="",
        help="Optional NIfTI volume path. When set, this is used instead of the default DICOM series.",
    )
    parser.add_argument(
        "--nifti-axis-order",
        default="xyz_to_zyx",
        choices=["xyz_to_zyx", "zyx"],
        help="Interpretation of NIfTI array axes before internal [z,y,x] processing.",
    )
    args = parser.parse_args()

    case_id = args.case_id.strip()
    reg_root = Path(args.reg_root)
    tn_root = Path(args.tn_seg_root)
    out_dir = reg_root / f"preview_{case_id}" / "standardized"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_id = case_to_csv_id(case_id)
    ipsi_centroid = read_centroid(tn_root / "202022_centroids_ipsilateral.csv", csv_id)
    contra_centroid = read_centroid(tn_root / "202022_centroids_contralateral.csv", csv_id)
    if args.reg_nifti.strip():
        nifti_path = Path(args.reg_nifti.strip())
        if not nifti_path.is_absolute():
            candidate = reg_root / nifti_path
            nifti_path = candidate if candidate.exists() else nifti_path
        if not nifti_path.exists():
            raise FileNotFoundError(f"NIfTI volume not found: {nifti_path}")
        vol = load_volume(
            nifti_path,
            input_kind="nifti",
            nifti_axis_order=args.nifti_axis_order,
        )
        volume_source = str(nifti_path)
        volume_kind = "nifti"
    else:
        dicom_series_dir = reg_root / "data_mri" / case_id / "A5IKQO4I" / "5N3PYHRX"
        if not dicom_series_dir.exists():
            raise FileNotFoundError(f"DICOM series directory not found: {dicom_series_dir}")
        vol = load_volume(dicom_series_dir, input_kind="dicom")
        volume_source = str(dicom_series_dir)
        volume_kind = "dicom"

    ipsi_in = load_tiff_stack(tn_root / "202022_Ipsilateral_Input" / f"{case_id}_cropped.tiff")
    ipsi_lb = load_tiff_stack(tn_root / "202022_Ipsilateral_Target" / f"{case_id}_mask.tiff")
    contra_in = load_tiff_stack(tn_root / "202022_Contralateral_Input" / f"{case_id}_cropped.tiff")
    contra_lb = load_tiff_stack(tn_root / "202022_Contralateral_Target" / f"{case_id}_mask.tiff")

    ipsi_fig, ipsi_score, ipsi_center = build_side_figure(
        "ipsi", case_id, ipsi_centroid, vol, ipsi_in, ipsi_lb, args.iso_mm
    )
    contra_fig, contra_score, contra_center = build_side_figure(
        "contra", case_id, contra_centroid, vol, contra_in, contra_lb, args.iso_mm
    )
    overview = make_overview(ipsi_fig, contra_fig)

    ipsi_out = out_dir / f"{case_id}_ipsi_STANDARD.png"
    contra_out = out_dir / f"{case_id}_contra_STANDARD.png"
    overview_out = out_dir / f"{case_id}_STANDARD_overview.png"
    manifest_out = out_dir / f"{case_id}_STANDARD_manifest.txt"

    ipsi_fig.save(ipsi_out)
    contra_fig.save(contra_out)
    overview.save(overview_out)

    sz, sy, sx = vol.spacing_zyx
    with manifest_out.open("w") as f:
        f.write("Standardized visualization manifest\n")
        f.write(f"case_id={case_id}\n")
        f.write(f"csv_case_id={csv_id}\n")
        f.write(f"spacing_zyx_mm={sz:.6f},{sy:.6f},{sx:.6f}\n")
        f.write(f"direction_text={vol.direction_text}\n")
        f.write(f"volume_kind={volume_kind}\n")
        f.write(f"volume_source={volume_source}\n")
        if volume_kind == "nifti":
            f.write(f"nifti_axis_order={args.nifti_axis_order}\n")
        f.write(f"iso_target_mm={args.iso_mm:.4f}\n")
        f.write("csv_order=z,x,y\n")
        f.write("mapping_to_zyx=(z, y, x)\n")
        f.write("patch_to_tn=transpose(1,2,0)+flip(axis=2)\n")
        f.write("tn_to_patch=transpose(flip(axis=2),(2,0,1))\n")
        f.write(f"ipsi_centroid_csv={ipsi_centroid}\n")
        f.write(f"contra_centroid_csv={contra_centroid}\n")
        f.write(f"ipsi_center_zyx={ipsi_center}\n")
        f.write(f"contra_center_zyx={contra_center}\n")
        f.write(f"ipsi_score={ipsi_score:.6f}\n")
        f.write(f"contra_score={contra_score:.6f}\n")

    print(str(ipsi_out))
    print(str(contra_out))
    print(str(overview_out))
    print(str(manifest_out))


if __name__ == "__main__":
    main()
