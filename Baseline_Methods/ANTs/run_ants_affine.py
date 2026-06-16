#!/usr/bin/env python3
# TN-Reg-Benchmark
# Code author: Xupeng Zhang (xzhan419@jh.edu)
# Johns Hopkins University
# See README for the full author list and citation.

import os
import glob
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm


def case_id_from_filename(fname: str) -> str:
    """Supports CASEID.nii.gz / CASEID.nii / CASEID_xxx.nii.gz"""
    name = Path(fname).name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    return name.split("_")[0]


def resolve_nii_path(dir_path: Path, case_id: str) -> Path | None:
    """Prefer .nii.gz, fallback to .nii"""
    p = dir_path / f"{case_id}.nii.gz"
    if p.exists():
        return p
    p = dir_path / f"{case_id}.nii"
    if p.exists():
        return p
    return None


def build_cmd(fixed: Path, moving: Path, fmask: Path, mmask: Path, outpfx: Path) -> list[str]:
    """Your original ANTs affine command, only path-templated."""
    return [
        "antsRegistration",
        "--dimensionality", "3",
        "--float", "0",
        "--output", f'[{outpfx},{outpfx}Warped.nii.gz,{outpfx}InverseWarped.nii.gz]',
        "--interpolation", "Linear",
        "--use-histogram-matching", "0",
        "--initial-moving-transform", f'[{fixed},{moving},1]',
        "--transform", "Affine[0.1]",
        "--metric", f'MI[{fixed},{moving},1,32,Regular,0.25]',
        "--convergence", "[1000x500x250x100,1e-6,10]",
        "--shrink-factors", "8x4x2x1",
        "--smoothing-sigmas", "3x2x1x0vox",
        "--masks", f'[{fmask},{mmask}]',
        "--verbose", "1",
    ]


def run_one_case(args):
    """
    Worker: run antsRegistration, then move outputs into:
      warped/CASE_reg_Warped.nii.gz
      inversewarped/CASE_reg_InverseWarped.nii.gz
      mat/CASE_reg_0GenericAffine.mat
    """
    (
        cid,
        fixed,
        moving,
        fmask,
        mmask,
        tmp_dir,
        out_warped_dir,
        out_inv_dir,
        out_mat_dir,
        log_dir,
        threads_per_case,
        skip_done,
    ) = args

    fixed = Path(fixed)
    moving = Path(moving)
    fmask = Path(fmask)
    mmask = Path(mmask)

    tmp_dir = Path(tmp_dir)
    out_warped_dir = Path(out_warped_dir)
    out_inv_dir = Path(out_inv_dir)
    out_mat_dir = Path(out_mat_dir)
    log_dir = Path(log_dir)

    for d in [tmp_dir, out_warped_dir, out_inv_dir, out_mat_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Final outputs (your preferred naming)
    final_warped = out_warped_dir / f"{cid}_reg_Warped.nii.gz"
    final_inv = out_inv_dir / f"{cid}_reg_InverseWarped.nii.gz"
    final_mat = out_mat_dir / f"{cid}_reg_0GenericAffine.mat"

    if skip_done and final_warped.exists() and final_inv.exists() and final_mat.exists():
        return cid, "SKIP_DONE", str(final_warped)

    # Use a per-case temp prefix to avoid any collision
    outpfx = tmp_dir / f"{cid}_reg_"
    cmd = build_cmd(fixed, moving, fmask, mmask, outpfx)

    log_path = log_dir / f"{cid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Prevent thread oversubscription (super important with 32-way parallel)
    env = os.environ.copy()
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads_per_case)

    with open(log_path, "w") as f:
        f.write("COMMAND:\n" + " ".join(cmd) + "\n\n")
        f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)

    if p.returncode != 0:
        return cid, "FAIL", str(log_path)

    # Produced files in temp
    tmp_warped = tmp_dir / f"{cid}_reg_Warped.nii.gz"
    tmp_inv = tmp_dir / f"{cid}_reg_InverseWarped.nii.gz"
    tmp_mat = tmp_dir / f"{cid}_reg_0GenericAffine.mat"

    if not (tmp_warped.exists() and tmp_inv.exists() and tmp_mat.exists()):
        return cid, "FAIL_MISSING_OUTPUT", str(log_path)

    # Move into organized folders (overwrite if partially existed)
    if final_warped.exists():
        final_warped.unlink()
    if final_inv.exists():
        final_inv.unlink()
    if final_mat.exists():
        final_mat.unlink()

    shutil.move(str(tmp_warped), str(final_warped))
    shutil.move(str(tmp_inv), str(final_inv))
    shutil.move(str(tmp_mat), str(final_mat))

    return cid, "OK", str(final_warped)


def main():
    PROJ = Path("/path/to/TN_Reg")

    # Inputs (from your tree)
    mri_dir = PROJ / "data" / "preprocessed_cropped" / "mri"
    mra_dir = PROJ / "data" / "preprocessed_cropped" / "mra"
    mri_mask_dir = PROJ / "data" / "masks_cropped" / "mri_masks"
    mra_mask_dir = PROJ / "data" / "masks_cropped" / "mra_masks"

    # Output (your requested location)
    out_base = PROJ / "outputs" / "MRIfixed_MRAmoving" / "ANTs_result"
    out_warped_dir = out_base / "warped"
    out_inv_dir = out_base / "inversewarped"
    out_mat_dir = out_base / "mat"
    log_dir = out_base / "logs"
    tmp_dir = out_base / "_tmp"

    # Parallel settings (your server: 128 cores)
    nproc = 32
    threads_per_case = 4  # 32*4 = 128
    skip_done = True

    # Discover cases from MRI dir
    mri_files = sorted(glob.glob(str(mri_dir / "*.nii*")))
    case_ids = sorted({case_id_from_filename(p) for p in mri_files})

    jobs = []
    missing = 0

    for cid in case_ids:
        fixed = resolve_nii_path(mri_dir, cid)
        moving = resolve_nii_path(mra_dir, cid)
        fmask = mri_mask_dir / f"{cid}_mask.nii.gz"
        mmask = mra_mask_dir / f"{cid}_mask.nii.gz"

        if fixed is None or moving is None or (not fmask.exists()) or (not mmask.exists()):
            missing += 1
            continue

        jobs.append((
            cid,
            str(fixed),
            str(moving),
            str(fmask),
            str(mmask),
            str(tmp_dir),
            str(out_warped_dir),
            str(out_inv_dir),
            str(out_mat_dir),
            str(log_dir),
            threads_per_case,
            skip_done,
        ))

    out_base.mkdir(parents=True, exist_ok=True)

    print(f"Found MRI cases: {len(case_ids)}")
    print(f"Runnable paired cases (MRI+MRA+2 masks): {len(jobs)}")
    print(f"Skipped due to missing files: {missing}")
    print(f"Output base: {out_base}")
    print(f"Parallel: {nproc} processes, {threads_per_case} ITK threads per case")

    ok = fail = skip = 0

    with ProcessPoolExecutor(max_workers=nproc) as ex:
        futures = [ex.submit(run_one_case, j) for j in jobs]

        with tqdm(total=len(futures), desc="ANTs Affine (MRI fixed)", dynamic_ncols=True) as pbar:
            for fut in as_completed(futures):
                cid, status, info = fut.result()

                if status == "OK":
                    ok += 1
                    pbar.set_postfix_str(f"OK {cid}")
                elif status == "SKIP_DONE":
                    skip += 1
                    pbar.set_postfix_str(f"SKIP {cid}")
                else:
                    fail += 1
                    pbar.set_postfix_str(f"FAIL {cid}")

                pbar.update(1)

    print("\n=== SUMMARY ===")
    print("OK:", ok)
    print("SKIP_DONE:", skip)
    print("FAIL:", fail)
    print(f"Logs: {log_dir}")
    print(f"Warped: {out_warped_dir}")
    print(f"InverseWarped: {out_inv_dir}")
    print(f"Affine mats: {out_mat_dir}")
    print(f"Temp: {tmp_dir} (may contain leftovers for failed cases)")


if __name__ == "__main__":
    main()
