#!/usr/bin/env python3
# TN-Reg-Benchmark
# Code author: Xupeng Zhang (xzhan419@jh.edu)
# Johns Hopkins University
# See README for the full author list and citation.

"""
EasyReg baseline for TN-Reg MRI-MRA registration.

Usage:
    python run_easyreg.py \
        --mri-dir data/preprocessed/mri \
        --mra-dir data/preprocessed/mra \
        --out-dir outputs/EasyReg \
        --threads 8

Requirements:
    FreeSurfer >= 7.4 (provides mri_easyreg)

References:
    Iglesias, JE. "A ready-to-use machine learning tool for symmetric
    multi-modality registration of brain MRI." Sci Rep 13, 6657 (2023).
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def find_pairs(mri_dir: Path, mra_dir: Path):
    """Match MRI-MRA pairs by case ID (filename stem before first '_' or full stem)."""
    mri_files = sorted(mri_dir.glob("*.nii.gz"))
    mra_files = {f.name: f for f in mra_dir.glob("*.nii.gz")}

    pairs = []
    for mri_path in mri_files:
        # Try exact filename match first
        if mri_path.name in mra_files:
            pairs.append((mri_path, mra_files[mri_path.name]))
            continue
        # Try matching by case ID (first part of filename)
        case_id = mri_path.name.split("_")[0]
        matched = [f for name, f in mra_files.items() if name.startswith(case_id)]
        if len(matched) == 1:
            pairs.append((mri_path, matched[0]))
        elif len(matched) > 1:
            print(f"[WARN] Multiple MRA matches for {case_id}, skipping")
        else:
            print(f"[WARN] No MRA match for {mri_path.name}")

    return pairs


def run_easyreg_single(
    mri_path: Path,
    mra_path: Path,
    out_dir: Path,
    seg_dir: Path,
    threads: int,
    affine_only: bool = False,
):
    """Run mri_easyreg on a single MRI-MRA pair."""
    case_id = mri_path.name.replace(".nii.gz", "")

    # Output paths
    warped_dir = out_dir / "warped"
    field_dir = out_dir / "fields"
    warped_dir.mkdir(parents=True, exist_ok=True)
    field_dir.mkdir(parents=True, exist_ok=True)

    warped_mra = warped_dir / f"{case_id}_reg_Warped.nii.gz"
    fwd_field = field_dir / f"{case_id}_fwd_field.nii.gz"
    bak_field = field_dir / f"{case_id}_bak_field.nii.gz"

    # SynthSeg segmentations (cached across runs)
    ref_seg = seg_dir / f"{case_id}_mri_synthseg.nii.gz"
    flo_seg = seg_dir / f"{case_id}_mra_synthseg.nii.gz"

    # Skip if already done
    if warped_mra.exists():
        print(f"[SKIP] {case_id}: {warped_mra} already exists")
        return case_id, "skipped", 0.0

    cmd = [
        "mri_easyreg",
        "--ref", str(mri_path),
        "--flo", str(mra_path),
        "--ref_seg", str(ref_seg),
        "--flo_seg", str(flo_seg),
        "--flo_reg", str(warped_mra),
        "--fwd_field", str(fwd_field),
        "--bak_field", str(bak_field),
        "--threads", str(threads),
    ]
    if affine_only:
        cmd.append("--affine_only")

    print(f"[RUN]  {case_id}: {' '.join(cmd)}")
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout per case
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"[FAIL] {case_id} ({elapsed:.1f}s):\n{result.stderr[-500:]}")
            # Save error log
            (out_dir / "logs").mkdir(exist_ok=True)
            (out_dir / "logs" / f"{case_id}_error.log").write_text(
                result.stdout + "\n" + result.stderr
            )
            return case_id, "failed", elapsed
        else:
            print(f"[OK]   {case_id} ({elapsed:.1f}s)")
            return case_id, "success", elapsed

    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {case_id}")
        return case_id, "timeout", 1800.0


def main():
    parser = argparse.ArgumentParser(description="Run EasyReg on MRI-MRA pairs")
    parser.add_argument("--mri-dir", type=Path, required=True,
                        help="Directory with preprocessed MRI volumes (.nii.gz)")
    parser.add_argument("--mra-dir", type=Path, required=True,
                        help="Directory with preprocessed MRA volumes (.nii.gz)")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/EasyReg"),
                        help="Output directory")
    parser.add_argument("--threads", type=int, default=8,
                        help="Threads per registration (passed to mri_easyreg)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of cases to run in parallel (each uses --threads)")
    parser.add_argument("--affine-only", action="store_true",
                        help="Skip nonlinear registration (affine only)")
    parser.add_argument("--cases", nargs="*", default=None,
                        help="Specific case IDs to run (default: all)")
    args = parser.parse_args()

    # Verify FreeSurfer
    fs_home = os.environ.get("FREESURFER_HOME")
    if not fs_home:
        print("[ERROR] FREESURFER_HOME not set. Please source FreeSurfer first.")
        sys.exit(1)
    easyreg_bin = Path(fs_home) / "bin" / "mri_easyreg"
    if not easyreg_bin.exists():
        # Also check PATH
        which_result = subprocess.run(["which", "mri_easyreg"], capture_output=True, text=True)
        if which_result.returncode != 0:
            print("[ERROR] mri_easyreg not found. Need FreeSurfer >= 7.4")
            sys.exit(1)

    # Find pairs
    pairs = find_pairs(args.mri_dir, args.mra_dir)
    print(f"Found {len(pairs)} MRI-MRA pairs")

    # Filter to specific cases if requested
    if args.cases:
        case_set = set(args.cases)
        pairs = [(m, a) for m, a in pairs if m.name.replace(".nii.gz", "").split("_")[0] in case_set]
        print(f"Filtered to {len(pairs)} cases")

    if not pairs:
        print("[ERROR] No pairs found")
        sys.exit(1)

    # Create dirs
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seg_dir = args.out_dir / "synthseg_cache"
    seg_dir.mkdir(exist_ok=True)

    # Run
    results = []
    if args.parallel <= 1:
        for mri_path, mra_path in pairs:
            res = run_easyreg_single(
                mri_path, mra_path, args.out_dir, seg_dir,
                args.threads, args.affine_only,
            )
            results.append(res)
    else:
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(
                    run_easyreg_single,
                    mri_path, mra_path, args.out_dir, seg_dir,
                    args.threads, args.affine_only,
                ): (mri_path, mra_path)
                for mri_path, mra_path in pairs
            }
            for future in as_completed(futures):
                results.append(future.result())

    # Summary
    success = sum(1 for _, s, _ in results if s == "success")
    skipped = sum(1 for _, s, _ in results if s == "skipped")
    failed = sum(1 for _, s, _ in results if s in ("failed", "timeout"))
    times = [t for _, s, t in results if s == "success"]
    avg_time = sum(times) / len(times) if times else 0

    print(f"\n{'='*50}")
    print(f"EasyReg Summary")
    print(f"  Success: {success}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed:  {failed}")
    if times:
        print(f"  Avg time: {avg_time:.1f}s ({avg_time/60:.1f}min)")
    print(f"  Output:  {args.out_dir / 'warped'}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()


