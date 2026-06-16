#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# TN-Reg-Benchmark
# Code author: Xupeng Zhang (xzhan419@jh.edu)
# Johns Hopkins University
# See README for the full author list and citation.


"""
Build a unified TN centroid index from the provided TN CSVs.

This script is designed for the TN CSV format you described:
1) case id column is MRN (may have BOM, may be "MRN"/"mrn"/etc.)
2) coordinate columns are sagittal, coronal, axial (may have different cases)
3) MRN may have leading zeros, while file names may or may not; we normalize.

Output columns:
- case_id   : normalized id (leading zeros removed if numeric)
- mrn_raw   : raw MRN string from CSV (kept for traceability)
- side      : "ipsi" or "contra"
- ci,cj,ck  : internal voxel-like coordinates (mapped from CSV columns)
- source_csv: which CSV file the row came from

NOTE on coordinate mapping:
- For ("sagittal","coronal","axial"), by default we map:
    sagittal -> ci (i)
    coronal  -> cj (j)
    axial    -> ck (k)
- If you later find axes swapped, change --scc-to-ijk (default "0,1,2").

Example:
python code/labels/build_tn_centroid_index.py \
  --root /path/to/TN_Reg \
  --out results/labels/tn_centroids_index.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd


def normalize_case_id(x: str) -> str:
    """
    Normalize case id for matching across files:
    - strip whitespace
    - if purely digits: convert to int then back to str (removes leading zeros)
    - else: lstrip leading zeros (but keep non-empty)
    """
    s = str(x).strip()
    if s == "":
        return s
    if s.isdigit():
        try:
            return str(int(s))
        except Exception:
            # extremely large numeric strings: fallback
            return s.lstrip("0") or "0"
    return s.lstrip("0") or s


def parse_perm(perm_str: str) -> Tuple[int, int, int]:
    parts = [p.strip() for p in perm_str.split(",")]
    if len(parts) != 3:
        raise ValueError(f"--scc-to-ijk must have 3 ints like '0,1,2', got: {perm_str}")
    perm = tuple(int(p) for p in parts)
    if sorted(perm) != [0, 1, 2]:
        raise ValueError(f"--scc-to-ijk must be a permutation of 0,1,2, got: {perm}")
    return perm  # type: ignore


def find_id_col(cols: List[str]) -> Optional[str]:
    # Prefer MRN for your dataset, but keep compatibility
    candidates = ["MRN", "mrn", "case_id", "patient_id", "id", "ID"]
    for c in candidates:
        if c in cols:
            return c
    # last resort: fuzzy match
    for c in cols:
        if c.lower().strip() in ("mrn", "case_id", "patient_id", "id"):
            return c
    return None


def find_coord_cols(cols: List[str]) -> Optional[Tuple[str, str, str]]:
    candidates = [
        ("sagittal", "coronal", "axial"),
        ("Sagittal", "Coronal", "Axial"),
        ("SAGITTAL", "CORONAL", "AXIAL"),
        ("x", "y", "z"),
        ("X", "Y", "Z"),
        ("i", "j", "k"),
        ("I", "J", "K"),
        ("cx", "cy", "cz"),
    ]
    colset = set(cols)
    for tpl in candidates:
        if all(c in colset for c in tpl):
            return tpl
    # case-insensitive fallback
    lower_map = {c.lower().strip(): c for c in cols}
    for tpl in [("sagittal", "coronal", "axial"), ("x", "y", "z"), ("i", "j", "k")]:
        if all(k in lower_map for k in tpl):
            return (lower_map[tpl[0]], lower_map[tpl[1]], lower_map[tpl[2]])
    return None


def read_centroid_csv(p: Path, scc_to_ijk: Tuple[int, int, int]) -> pd.DataFrame:
    """
    Read one centroid CSV and return standardized columns.
    """
    df = pd.read_csv(p, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    id_col = find_id_col(df.columns.tolist())
    if id_col is None:
        raise ValueError(f"[{p}] cannot find case id column. Columns={list(df.columns)}")

    coord_cols = find_coord_cols(df.columns.tolist())
    if coord_cols is None:
        raise ValueError(f"[{p}] cannot find coord columns. Columns={list(df.columns)}")

    c0, c1, c2 = coord_cols
    # If it's sagittal/coronal/axial, allow perm mapping
    coord_names = [c0.lower(), c1.lower(), c2.lower()]
    cols = [c0, c1, c2]

    if coord_names == ["sagittal", "coronal", "axial"]:
        perm = scc_to_ijk
    else:
        perm = (0, 1, 2)

    ci_col = cols[perm[0]]
    cj_col = cols[perm[1]]
    ck_col = cols[perm[2]]

    mrn_raw = df[id_col].astype(str).str.strip()
    case_id = mrn_raw.map(normalize_case_id)

    out = pd.DataFrame(
        {
            "case_id": case_id,
            "mrn_raw": mrn_raw,
            "ci": pd.to_numeric(df[ci_col], errors="coerce"),
            "cj": pd.to_numeric(df[cj_col], errors="coerce"),
            "ck": pd.to_numeric(df[ck_col], errors="coerce"),
            "source_csv": str(p),
        }
    )

    # Drop rows without id or coords
    out = out.dropna(subset=["case_id", "ci", "cj", "ck"])
    out = out[out["case_id"].astype(str).str.len() > 0].copy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Project root, e.g. /path/to/TN_Reg")
    ap.add_argument(
        "--ipsi-csv",
        type=str,
        default="2020-22_MRIs_Cropped_and_Segmentations/202022_centroids_ipsilateral.csv",
        help="Relative to --root",
    )
    ap.add_argument(
        "--contra-csv",
        type=str,
        default="2020-22_MRIs_Cropped_and_Segmentations/202022_centroids_contralateral.csv",
        help="Relative to --root",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="results/labels/tn_centroids_index.csv",
        help="Output CSV path relative to --root",
    )
    ap.add_argument(
        "--scc-to-ijk",
        type=str,
        default="0,1,2",
        help="Permutation mapping for (sagittal,coronal,axial)->(ci,cj,ck). Default 0,1,2",
    )
    ap.add_argument(
        "--dedup",
        action="store_true",
        help="If set, keep only one row per (case_id, side) by averaging coords.",
    )

    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    ipsi_csv = (root / args.ipsi_csv).resolve()
    contra_csv = (root / args.contra_csv).resolve()
    out_csv = (root / args.out).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not ipsi_csv.exists():
        raise FileNotFoundError(f"ipsi_csv not found: {ipsi_csv}")
    if not contra_csv.exists():
        raise FileNotFoundError(f"contra_csv not found: {contra_csv}")

    scc_to_ijk = parse_perm(args.scc_to_ijk)

    df_i = read_centroid_csv(ipsi_csv, scc_to_ijk)
    df_i["side"] = "ipsi"

    df_c = read_centroid_csv(contra_csv, scc_to_ijk)
    df_c["side"] = "contra"

    df = pd.concat([df_i, df_c], ignore_index=True)

    # Optional: deduplicate (some CSVs may contain multiple centroids per case)
    if args.dedup:
        df = (
            df.groupby(["case_id", "side"], as_index=False)
            .agg(
                mrn_raw=("mrn_raw", "first"),
                ci=("ci", "mean"),
                cj=("cj", "mean"),
                ck=("ck", "mean"),
                source_csv=("source_csv", "first"),
            )
            .copy()
        )

    # Sort for readability
    df = df.sort_values(["case_id", "side"]).reset_index(drop=True)

    df.to_csv(out_csv, index=False)
    print(f"[OK] wrote {len(df)} rows -> {out_csv}")
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    main()
