"""Thin re-export of the low-level visualization helpers used by case/Fig.1 code.

These live in the repository's ``visualization`` package (standardize_case_view.py,
render_case_ipl_view.py); this module makes them importable as ``eval._common_vis``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_VIS_DIR = str(Path(__file__).resolve().parents[1] / "visualization")
if _VIS_DIR not in sys.path:
    sys.path.insert(0, _VIS_DIR)

from standardize_case_view import (  # noqa: E402
    load_volume,
    load_dicom_volume,
    case_to_csv_id,
    csv_to_zyx,
    read_centroid,
    map_center_between_volumes,
    patch_to_tn,
)
from render_case_ipl_view import find_best_dicom_series_dir  # noqa: E402

__all__ = [
    "load_volume", "load_dicom_volume", "case_to_csv_id", "csv_to_zyx",
    "read_centroid", "map_center_between_volumes", "patch_to_tn",
    "find_best_dicom_series_dir",
]
