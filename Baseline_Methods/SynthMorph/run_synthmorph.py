#!/usr/bin/env python3
"""
SynthMorph baseline for TN-Reg MRI-MRA registration.

Usage:
    python run_synthmorph.py \
        --mri-dir data/preprocessed_cropped/mri \
        --mra-dir data/preprocessed_cropped/mra \
        --out-dir outputs/MRIfixed_MRAmoving/SynthMorph \
        --threads 8

Requirements:
    FreeSurfer >= 7.4 (provides mri_synthmorph)

python SynthMorph/run_synthmorph.py \
    --mri-dir ~/TN_Reg/data/preprocessed_cropped/mri \
    --mra-dir ~/TN_Reg/data/preprocessed_cropped/mra \
    --out-dir ~/TN_Reg/outputs/MRIfixed_MRAmoving/SynthMorph \
    --threads 8 \
    --parallel 2


References:
    Hoffmann M, et al. "Anatomy-aware and acquisition-agnostic joint
    registration with SynthMorph." Imaging Neuroscience, 2, pp 1-33, 2024.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def find_pairs(mri_dir: Path, mra_dir: Path):
    """Match MRI-MRA pairs by filename."""
    mri_files = sorted(mri_dir.glob("*.nii.gz"))
    mra_files = {f.name: f for f in mra_dir.glob("*.nii.gz")}

    pairs = []
    for mri_path in mri_files:
        if mri_path.name in mra_files:
            pairs.append((mri_path, mra_files[mri_path.name]))
            continue
        case_id = mri_path.name.split("_")[0]
        matched = [f for name, f in mra_files.items() if name.startswith(case_id)]
        if len(matched) == 1:
            pairs.append((mri_path, matched[0]))
        elif len(matched) > 1:
            print(f"[WARN] Multiple MRA matches for {case_id}, skipping")
        else:
            print(f"[WARN] No MRA match for {mri_path.name}")
    return pairs


def run_synthmorph_single(
    mri_path: Path,
    mra_path: Path,
    out_dir: Path,
    threads: int,
    reg_strength: float,
):
    """Run mri_synthmorph on a single MRI-MRA pair."""
    case_id = mri_path.name.replace(".nii.gz", "")

    warped_dir = out_dir / "warped"
    field_dir = out_dir / "fields"
    warped_dir.mkdir(parents=True, exist_ok=True)
    field_dir.mkdir(parents=True, exist_ok=True)

    warped_mra = warped_dir / f"{case_id}_reg_Warped.nii.gz"
    disp_field = field_dir / f"{case_id}_disp_field.nii.gz"

    if warped_mra.exists():
        print(f"[SKIP] {case_id}: {warped_mra} already exists")
        return case_id, "skipped", 0.0

    # mri_synthmorph register: moving=MRA, fixed=MRI
    # -o: output moved image
    # -t: output displacement field
    # -r: regularization strength (default 0.5, lower = more deformation)
    # -j: number of threads
    cmd = [
        "mri_synthmorph", "register",
        "-o", str(warped_mra),
        "-t", str(disp_field),
        "-j", str(threads),
    ]
    if reg_strength is not None:
        cmd.extend(["-r", str(reg_strength)])

    # positional args: moving fixed
    cmd.extend([str(mra_path), str(mri_path)])

    print(f"[RUN]  {case_id}: {' '.join(cmd)}")
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"[FAIL] {case_id} ({elapsed:.1f}s):\n{result.stderr[-500:]}")
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
    parser = argparse.ArgumentParser(description="Run SynthMorph on MRI-MRA pairs")
    parser.add_argument("--mri-dir", type=Path, required=True,
                        help="Directory with preprocessed MRI volumes (.nii.gz)")
    parser.add_argument("--mra-dir", type=Path, required=True,
                        help="Directory with preprocessed MRA volumes (.nii.gz)")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/MRIfixed_MRAmoving/SynthMorph"),
                        help="Output directory")
    parser.add_argument("--threads", type=int, default=8,
                        help="Threads per registration (passed to mri_synthmorph -j)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of cases to run in parallel")
    parser.add_argument("--reg-strength", type=float, default=None,
                        help="Regularization strength (0-1, lower=more deformation, default=0.5)")
    parser.add_argument("--cases", nargs="*", default=None,
                        help="Specific case IDs to run (default: all)")
    args = parser.parse_args()

    # Verify
    which_result = subprocess.run(["which", "mri_synthmorph"], capture_output=True, text=True)
    if which_result.returncode != 0:
        print("[ERROR] mri_synthmorph not found. Need FreeSurfer >= 7.4")
        sys.exit(1)

    pairs = find_pairs(args.mri_dir, args.mra_dir)
    print(f"Found {len(pairs)} MRI-MRA pairs")

    if args.cases:
        case_set = set(args.cases)
        pairs = [(m, a) for m, a in pairs
                 if m.name.replace(".nii.gz", "").split("_")[0] in case_set]
        print(f"Filtered to {len(pairs)} cases")

    if not pairs:
        print("[ERROR] No pairs found")
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    if args.parallel <= 1:
        for mri_path, mra_path in pairs:
            res = run_synthmorph_single(
                mri_path, mra_path, args.out_dir,
                args.threads, args.reg_strength,
            )
            results.append(res)
    else:
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(
                    run_synthmorph_single,
                    mri_path, mra_path, args.out_dir,
                    args.threads, args.reg_strength,
                ): (mri_path, mra_path)
                for mri_path, mra_path in pairs
            }
            for future in as_completed(futures):
                results.append(future.result())

    success = sum(1 for _, s, _ in results if s == "success")
    skipped = sum(1 for _, s, _ in results if s == "skipped")
    failed = sum(1 for _, s, _ in results if s in ("failed", "timeout"))
    times = [t for _, s, t in results if s == "success"]
    avg_time = sum(times) / len(times) if times else 0

    print(f"\n{'='*50}")
    print(f"SynthMorph Summary")
    print(f"  Success: {success}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed:  {failed}")
    if times:
        print(f"  Avg time: {avg_time:.1f}s ({avg_time/60:.1f}min)")
    print(f"  Output:  {args.out_dir / 'warped'}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()