#!/usr/bin/env python3
# TN-Reg-Benchmark
# Code author: Xupeng Zhang (xzhan419@jh.edu)
# Johns Hopkins University
# See README for the full author list and citation.

"""
Run Batch ANTs with rigid, affine, and Syn MRI fixed & MRA moving

"""

import os
import glob
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
import argparse


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


def build_cmd(fixed: Path, moving: Path, fmask: Path, outpfx: Path) -> list[str]:
    """
    Rigid + Affine + SyN
    - fixed mask only: [fmask, NULL]
    - winsorize
    - MI for rigid/affine, CC for SyN
    """
    fixed = str(fixed)
    moving = str(moving)
    fmask = str(fmask)
    outpfx = str(outpfx)

    return [
        "antsRegistration",
        "--dimensionality", "3",
        "--float", "0",
        "--output", f"[{outpfx},{outpfx}Warped.nii.gz,{outpfx}InverseWarped.nii.gz]",
        "--interpolation", "Linear",
        "--use-histogram-matching", "0",
        "--winsorize-image-intensities", "[0.005,0.995]",
        "--initial-moving-transform", f"[{fixed},{moving},1]",
        "--masks", f"[{fmask},NULL]",

        "--transform", "Rigid[0.1]",
        "--metric", f"MI[{fixed},{moving},1,64,Regular,0.25]",
        "--convergence", "[1000x500x250x100,1e-6,10]",
        "--shrink-factors", "8x4x2x1",
        "--smoothing-sigmas", "3x2x1x0vox",

        "--transform", "Affine[0.1]",
        "--metric", f"MI[{fixed},{moving},1,64,Regular,0.25]",
        "--convergence", "[1000x500x250x100,1e-6,10]",
        "--shrink-factors", "8x4x2x1",
        "--smoothing-sigmas", "3x2x1x0vox",

        "--transform", "SyN[0.1,3,0]",
        "--metric", f"CC[{fixed},{moving},1,4]",
        "--convergence", "[100x70x50x20,1e-6,10]",
        "--shrink-factors", "8x4x2x1",
        "--smoothing-sigmas", "3x2x1x0vox",

        "--verbose", "1",
    ]


def run_one_case(args):
    (
        cid,
        fixed,
        moving,
        fmask,
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

    tmp_dir = Path(tmp_dir)
    out_warped_dir = Path(out_warped_dir)
    out_inv_dir = Path(out_inv_dir)
    out_mat_dir = Path(out_mat_dir)
    log_dir = Path(log_dir)

    for d in [tmp_dir, out_warped_dir, out_inv_dir, out_mat_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # final outputs
    final_warped = out_warped_dir / f"{cid}_reg_Warped.nii.gz"
    final_inv = out_inv_dir / f"{cid}_reg_InverseWarped.nii.gz"
    final_mat = out_mat_dir / f"{cid}_reg_0GenericAffine.mat"  # affine mat always exists

    if skip_done and final_warped.exists() and final_inv.exists() and final_mat.exists():
        return cid, "SKIP_DONE", str(final_warped)

    outpfx = tmp_dir / f"{cid}_reg_"
    cmd = build_cmd(fixed, moving, fmask, outpfx)

    log_path = log_dir / f"{cid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    env = os.environ.copy()
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads_per_case)
    env["OMP_NUM_THREADS"] = str(threads_per_case)

    with open(log_path, "w") as f:
        f.write("COMMAND:\n" + " ".join(cmd) + "\n\n")
        f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)

    if p.returncode != 0:
        return cid, "FAIL", str(log_path)

    tmp_warped = tmp_dir / f"{cid}_reg_Warped.nii.gz"
    tmp_inv = tmp_dir / f"{cid}_reg_InverseWarped.nii.gz"
    tmp_mat = tmp_dir / f"{cid}_reg_0GenericAffine.mat"

    if not (tmp_warped.exists() and tmp_inv.exists() and tmp_mat.exists()):
        return cid, "FAIL_MISSING_OUTPUT", str(log_path)

    for dst in [final_warped, final_inv, final_mat]:
        if dst.exists():
            dst.unlink()

    shutil.move(str(tmp_warped), str(final_warped))
    shutil.move(str(tmp_inv), str(final_inv))
    shutil.move(str(tmp_mat), str(final_mat))

    return cid, "OK", str(final_warped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", default="/path/to/TN_Reg")
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--only-case", default="", help="If set, run only this case id")
    ap.add_argument("--out-name", default="ANTs_result_syn_fixedmask", help="output folder name under outputs/MRIfixed_MRAmoving/")
    args = ap.parse_args()

    PROJ = Path(args.proj)

    mri_dir = PROJ / "data" / "preprocessed_cropped" / "mri"
    mra_dir = PROJ / "data" / "preprocessed_cropped" / "mra"
    mri_mask_dir = PROJ / "data" / "masks_cropped" / "mri_masks"

    out_base = PROJ / "outputs" / "MRIfixed_MRAmoving" / args.out_name
    out_warped_dir = out_base / "warped"
    out_inv_dir = out_base / "inversewarped"
    out_mat_dir = out_base / "mat"
    log_dir = out_base / "logs"
    tmp_dir = out_base / "_tmp"

    out_base.mkdir(parents=True, exist_ok=True)

    # discover cases
    mri_files = sorted(glob.glob(str(mri_dir / "*.nii*")))
    case_ids = sorted({case_id_from_filename(p) for p in mri_files})

    if args.only_case.strip():
        cid = args.only_case.strip()
        case_ids = [cid]

    jobs = []
    missing = 0
    for cid in case_ids:
        fixed = resolve_nii_path(mri_dir, cid)
        moving = resolve_nii_path(mra_dir, cid)
        fmask = mri_mask_dir / f"{cid}_mask.nii.gz"

        if fixed is None or moving is None or (not fmask.exists()):
            missing += 1
            continue

        jobs.append((
            cid,
            str(fixed),
            str(moving),
            str(fmask),
            str(tmp_dir),
            str(out_warped_dir),
            str(out_inv_dir),
            str(out_mat_dir),
            str(log_dir),
            args.threads,
            args.skip_done,
        ))

    print(f"Found MRI cases: {len(case_ids)}")
    print(f"Runnable paired cases (MRI+MRA+fixed mask): {len(jobs)}")
    print(f"Skipped due to missing files: {missing}")
    print(f"Output base: {out_base}")
    print(f"Parallel: {args.nproc} processes, {args.threads} ITK threads per case (total ~{args.nproc*args.threads} threads)")

    ok = fail = skip = 0
    with ProcessPoolExecutor(max_workers=args.nproc) as ex:
        futures = [ex.submit(run_one_case, j) for j in jobs]
        with tqdm(total=len(futures), desc="ANTs Rigid+Affine+SyN (fixedmask)", dynamic_ncols=True) as pbar:
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
