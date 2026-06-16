#!/usr/bin/env python3
# TN-Reg-Benchmark
# Code author: Xupeng Zhang (xzhan419@jh.edu)
# Johns Hopkins University
# See README for the full author list and citation.

"""
Batch render TN compare figures by calling an existing per-case renderer script.

It will generate:
- TN vs whole-brain MRI (baseline)
- TN vs reg-warped/prealigned volume for each registration method

Outputs (default):
  results/figures/MRI/<case_id>.png
  results/figures/<METHOD>/<case_id>.png

This script does NOT change your orientation logic.
It simply calls your renderer with:
  --view ipl
  --align-lower-to-upper
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def read_case_ids_from_index(index_csv: Path) -> list[str]:
    df = pd.read_csv(index_csv)
    if "case_id" not in df.columns:
        raise ValueError(f"index csv missing 'case_id' column: {index_csv}")
    case_ids = sorted({str(x).strip() for x in df["case_id"].tolist() if str(x).strip()})
    return case_ids


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def resolve_mri_wholebrain(root: Path, case_id: str) -> Path:
    """
    Prefer preprocessed whole-brain MRI NIfTI so we don't depend on DICOM series paths.
    """
    candidates = [
        root / "data" / "preprocessed" / "mri" / f"{case_id}.nii.gz",
        root / "preprocessed_ras" / "mri" / f"{case_id}.nii.gz",
        root / "backup" / "data" / "preprocessed" / "mri" / f"{case_id}.nii.gz",
    ]
    p = first_existing(candidates)
    if p is None:
        raise FileNotFoundError(
            f"Cannot find whole-brain MRI nifti for case={case_id}. Tried:\n  "
            + "\n  ".join(str(x) for x in candidates)
        )
    return p


def resolve_method_volume(root: Path, method: str, case_id: str) -> Path:
    """
    Resolve the volume to visualize for a method.

    Your current outputs tree shows:
      outputs/MRIfixed_MRAmoving/ANTs_result/warped/...
      outputs/MRIfixed_MRAmoving/FireANTs_result/warped/...
      outputs/MRIfixed_MRAmoving/ConvexAdam_result/prealigned/...

    We'll glob robustly because filenames can vary slightly.
    """
    method = method.strip()
    base = root / "outputs" / "MRIfixed_MRAmoving" / f"{method}_result"

    subdir = None
    if method.lower() == "ants":
        subdir = "warped"
    elif method.lower() == "fireants":
        subdir = "warped"
    elif method.lower() == "convexadam":
        subdir = "prealigned"
    else:
        # fallback: try warped then prealigned
        for cand in ["warped", "prealigned"]:
            if (base / cand).exists():
                subdir = cand
                break

    if subdir is None:
        raise FileNotFoundError(f"Cannot find method output folder for {method}: {base}")

    d = base / subdir
    if not d.exists():
        raise FileNotFoundError(f"Method folder missing: {d}")

    patterns = [
        f"{case_id}*Warped*.nii*",
        f"{case_id}*warped*.nii*",
        f"{case_id}*prealign*.nii*",
        f"{case_id}*prealigned*.nii*",
        f"{case_id}*.nii*",
    ]
    matches: list[Path] = []
    for pat in patterns:
        matches = sorted(d.glob(pat))
        if matches:
            break

    if not matches:
        raise FileNotFoundError(f"No volume found for method={method} case={case_id} under {d}")

    return matches[0]


def run_renderer(
    *,
    root: Path,
    renderer: Path,
    case_id: str,
    tn_seg_root: Path,
    reg_nifti: Optional[Path],
    view: str,
    iso_mm: float,
    align_lower_to_upper: bool,
    ras_rotate_180: bool,
    suffix: str,
) -> tuple[Path, Path]:
    """
    Call your existing per-case renderer and return (overview_png, manifest_txt).
    We parse its stdout: it prints 4 paths (ipsi, contra, overview, manifest).
    """
    cmd = [
        sys.executable,
        str(renderer),
        "--case-id",
        case_id,
        "--reg-root",
        str(root),
        "--tn-seg-root",
        str(tn_seg_root),
        "--iso-mm",
        str(iso_mm),
        "--view",
        view,
        "--suffix",
        suffix,
    ]

    if reg_nifti is not None:
        cmd += ["--reg-nifti", str(reg_nifti)]
        # keep your internal behavior: nifti -> centroid-space auto mapping from raw dicom if available
        cmd += ["--centroid-space", "auto", "--raw-reg-root", str(root)]

    if align_lower_to_upper:
        cmd.append("--align-lower-to-upper")

    if view == "ras" and ras_rotate_180:
        cmd.append("--ras-rotate-180")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"[Renderer failed] case={case_id} suffix={suffix}\n"
            f"CMD: {' '.join(cmd)}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )

    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    # Expect last 4 lines are outputs (per your script)
    if len(lines) < 4:
        raise RuntimeError(
            f"[Renderer output parse failed] case={case_id}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    ipsi_out = Path(lines[-4])
    contra_out = Path(lines[-3])
    overview_out = Path(lines[-2])
    manifest_out = Path(lines[-1])

    # basic sanity
    if not overview_out.exists():
        raise RuntimeError(f"overview png not found after render: {overview_out}")
    if not manifest_out.exists():
        raise RuntimeError(f"manifest not found after render: {manifest_out}")

    return overview_out, manifest_out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Project root, e.g. /path/to/TN_Reg")
    p.add_argument(
        "--index",
        default="results/labels/tn_centroids_index.csv",
        help="TN centroid index csv (built earlier).",
    )
    p.add_argument(
        "--renderer",
        default="code/visualization/render_case_ipl_view.py",
        help="Your per-case renderer script path.",
    )
    p.add_argument(
        "--tn-seg-root",
        default="2020-22_MRIs_Cropped_and_Segmentations",
        help="Folder containing TN TIFF inputs/targets and centroid CSVs.",
    )
    p.add_argument(
        "--out-root",
        default="results/figures",
        help="Where to store final figures per method.",
    )
    p.add_argument(
        "--methods",
        default="ANTs,FireANTs,ConvexAdam",
        help="Comma separated methods to render.",
    )
    p.add_argument("--view", default="ipl", choices=["ipl", "ras"])
    p.add_argument("--iso-mm", type=float, default=0.47)
    p.add_argument("--align-lower-to-upper", action="store_true")
    p.add_argument("--ras-rotate-180", action="store_true")
    p.add_argument("--only-first-n", type=int, default=0, help="Debug: render only first N cases (0=all).")
    args = p.parse_args()

    root = Path(args.root).resolve()
    index_csv = (root / args.index).resolve() if not Path(args.index).is_absolute() else Path(args.index).resolve()
    renderer = (root / args.renderer).resolve() if not Path(args.renderer).is_absolute() else Path(args.renderer).resolve()
    tn_seg_root = (root / args.tn_seg_root).resolve() if not Path(args.tn_seg_root).is_absolute() else Path(args.tn_seg_root).resolve()
    out_root = (root / args.out_root).resolve() if not Path(args.out_root).is_absolute() else Path(args.out_root).resolve()

    if not index_csv.exists():
        raise FileNotFoundError(f"index csv not found: {index_csv}")
    if not renderer.exists():
        raise FileNotFoundError(f"renderer script not found: {renderer}")
    if not tn_seg_root.exists():
        raise FileNotFoundError(f"tn seg root not found: {tn_seg_root}")

    case_ids = read_case_ids_from_index(index_csv)
    if args.only_first_n and args.only_first_n > 0:
        case_ids = case_ids[: args.only_first_n]

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    print(f"[INFO] root={root}")
    print(f"[INFO] cases={len(case_ids)}  methods={methods}")
    print(f"[INFO] view={args.view} iso_mm={args.iso_mm} align_lower_to_upper={args.align_lower_to_upper}")

    # 0) baseline MRI figures
    (out_root / "MRI").mkdir(parents=True, exist_ok=True)
    (out_root / "MRI_manifest").mkdir(parents=True, exist_ok=True)

    for idx, case_id in enumerate(case_ids, start=1):
        print(f"[MRI] ({idx}/{len(case_ids)}) case={case_id}")
        mri_vol = resolve_mri_wholebrain(root, case_id)
        overview_png, manifest_txt = run_renderer(
            root=root,
            renderer=renderer,
            case_id=case_id,
            tn_seg_root=tn_seg_root,
            reg_nifti=mri_vol,
            view=args.view,
            iso_mm=args.iso_mm,
            align_lower_to_upper=args.align_lower_to_upper,
            ras_rotate_180=args.ras_rotate_180,
            suffix="MRI",
        )

        dst_png = out_root / "MRI" / f"{case_id}.png"
        dst_manifest = out_root / "MRI_manifest" / f"{case_id}.txt"
        shutil.copy2(overview_png, dst_png)
        shutil.copy2(manifest_txt, dst_manifest)

    # 1) method figures
    for method in methods:
        (out_root / method).mkdir(parents=True, exist_ok=True)
        (out_root / f"{method}_manifest").mkdir(parents=True, exist_ok=True)

        for idx, case_id in enumerate(case_ids, start=1):
            print(f"[{method}] ({idx}/{len(case_ids)}) case={case_id}")
            vol = resolve_method_volume(root, method, case_id)

            overview_png, manifest_txt = run_renderer(
                root=root,
                renderer=renderer,
                case_id=case_id,
                tn_seg_root=tn_seg_root,
                reg_nifti=vol,
                view=args.view,
                iso_mm=args.iso_mm,
                align_lower_to_upper=args.align_lower_to_upper,
                ras_rotate_180=args.ras_rotate_180,
                suffix=method,
            )

            dst_png = out_root / method / f"{case_id}.png"
            dst_manifest = out_root / f"{method}_manifest" / f"{case_id}.txt"
            shutil.copy2(overview_png, dst_png)
            shutil.copy2(manifest_txt, dst_manifest)

    print("[OK] Done.")
    print(f"[OK] Figures at: {out_root}")


if __name__ == "__main__":
    main()
