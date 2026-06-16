#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# TN-Reg-Benchmark
# Code author: Xupeng Zhang (xzhan419@jh.edu)
# Johns Hopkins University
# See README for the full author list and citation.


"""
Run Batch FireANTs MRI fixed & MRA moving

Outputs (under --out_base):
  logs/
  warped/          <case>_reg_Warped.nii.gz
  inversewarped/   <case>_reg_InverseWarped.nii.gz   (only if --do_inverse)
  mat/             transforms exported by fireants (ants-style) + npy rigid matrix

"""

import os
import sys
import glob
import time
import json
import argparse
import warnings
import logging
import traceback
import contextlib
import multiprocessing as mp
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import SimpleITK as sitk
from tqdm import tqdm

# FireANTs
warnings.filterwarnings("ignore")
# You can also reduce python warnings further if you like:
# warnings.filterwarnings("ignore", message="fireants_fused_ops not found")

from fireants.io import Image, BatchedImages
from fireants.registration.moments import MomentsRegistration
from fireants.registration.rigid import RigidRegistration
from fireants.registration.greedy import GreedyRegistration


def _silence_fireants_logging():
    # FireANTs uses python logging a lot; turn it down.
    for name in [
        "fireants",
        "fireants.utils",
        "fireants.utils.imageutils",
        "fireants.registration",
        "fireants.registration.abstract",
        "fireants.registration.optimizers",
        "fireants.registration.optimizers.adam",
    ]:
        logging.getLogger(name).setLevel(logging.ERROR)


def _ensure_3d_sitk(img: sitk.Image) -> sitk.Image:
    if img.GetDimension() == 4:
        # extract t=0
        size = img.GetSize()
        img = sitk.Extract(img, (size[0], size[1], size[2], 0), (0, 0, 0, 0))
    return img


def _load_as_fireants_image(nifti_path: str, device: str) -> Image:
    # Use SimpleITK to preserve spacing/origin/direction reliably
    sitk_img = sitk.ReadImage(nifti_path)
    sitk_img = _ensure_3d_sitk(sitk_img)
    sitk_img = sitk.Cast(sitk_img, sitk.sitkFloat32)
    return Image(sitk_img, device=device)


def _save_nifti_like_fixed(data_xyz: np.ndarray, fixed_nifti: nib.Nifti1Image, out_path: str):
    # data_xyz is expected [X,Y,Z] like nibabel get_fdata output ordering
    out_img = nib.Nifti1Image(data_xyz.astype(np.float32), fixed_nifti.affine, fixed_nifti.header)
    nib.save(out_img, out_path)


def _run_fireants_pair(
    fixed_path: str,
    moving_path: str,
    device: str,
    scales: tuple,
    iters: tuple,
    rigid_scales: tuple,
    rigid_iters: tuple,
    optimizer_lr: float,
    log_fp,
):
    """
    Returns:
      warped_xyz (np.ndarray): moving warped into fixed space, [X,Y,Z]
      rigid_matrix (np.ndarray): 4x4 (or Nx4x4) depending on fireants version
      greedy_reg: fireants GreedyRegistration object (for saving transforms)
    """
    # Prepare images
    fixed_img = _load_as_fireants_image(fixed_path, device=device)    # fixed
    moving_img = _load_as_fireants_image(moving_path, device=device)  # moving

    batch_fixed = BatchedImages([fixed_img])
    batch_moving = BatchedImages([moving_img])

    # --- Stage 1: Moments (COM align) ---
    moments_reg = MomentsRegistration(
        scale=4,
        fixed_images=batch_fixed,
        moving_images=batch_moving,
        scaling=False,
        moments=1,
    )

    # --- Stage 2: Rigid ---
    rigid_reg = RigidRegistration(
        scales=list(rigid_scales),
        iterations=list(rigid_iters),
        fixed_images=batch_fixed,
        moving_images=batch_moving,
        optimizer="Adam",
        optimizer_lr=float(optimizer_lr),
        scaling=False,
    )

    # --- Stage 3: Deformable (Greedy) ---
    greedy_reg = GreedyRegistration(
        scales=list(scales),
        iterations=list(iters),
        fixed_images=batch_fixed,
        moving_images=batch_moving,
        init_affine=None,  # filled after rigid
    )

    # FireANTs often prints per-iter; hard-suppress by redirecting stdout/stderr
    with contextlib.redirect_stdout(log_fp), contextlib.redirect_stderr(log_fp):
        moments_reg.optimize()
        rigid_reg.optimize()
        greedy_reg.init_affine = rigid_reg.get_rigid_matrix()
        greedy_reg.optimize()

        moved_tensor = greedy_reg.evaluate(batch_fixed, batch_moving)

    # Convert to numpy in nib ordering
    # FireANTs tensor is typically [B,C,Z,Y,X] -> convert to [X,Y,Z]
    moved_zyx = moved_tensor[0, 0].detach().cpu().numpy()  # [Z,Y,X]
    moved_xyz = np.transpose(moved_zyx, (2, 1, 0))

    rigid_matrix = rigid_reg.get_rigid_matrix().detach().cpu().numpy()
    return moved_xyz, rigid_matrix, greedy_reg


def _process_one_case(
    case_id: str,
    fixed_dir: str,
    moving_dir: str,
    out_base: str,
    device_id: int,
    scales: tuple,
    iters: tuple,
    rigid_scales: tuple,
    rigid_iters: tuple,
    optimizer_lr: float,
    do_inverse: bool,
):
    fixed_path = os.path.join(fixed_dir, f"{case_id}.nii.gz")
    moving_path = os.path.join(moving_dir, f"{case_id}.nii.gz")

    warped_dir = os.path.join(out_base, "warped")
    invwarped_dir = os.path.join(out_base, "inversewarped")
    mat_dir = os.path.join(out_base, "mat")
    logs_dir = os.path.join(out_base, "logs")

    os.makedirs(warped_dir, exist_ok=True)
    os.makedirs(invwarped_dir, exist_ok=True)
    os.makedirs(mat_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    out_warped = os.path.join(warped_dir, f"{case_id}_reg_Warped.nii.gz")
    out_invwarped = os.path.join(invwarped_dir, f"{case_id}_reg_InverseWarped.nii.gz")

    # Skip if done
    if os.path.exists(out_warped) and (not do_inverse or os.path.exists(out_invwarped)):
        return "SKIP_DONE", None

    if not os.path.exists(fixed_path):
        return "FAIL", f"Missing fixed: {fixed_path}"
    if not os.path.exists(moving_path):
        return "FAIL", f"Missing moving: {moving_path}"

    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"

    log_path = os.path.join(logs_dir, f"{case_id}.log")
    t0 = time.time()

    try:
        fixed_nib = nib.load(fixed_path)
        moving_nib = nib.load(moving_path)

        with open(log_path, "w") as log_fp:
            log_fp.write(f"CASE {case_id}\n")
            log_fp.write(f"fixed:  {fixed_path}\n")
            log_fp.write(f"moving: {moving_path}\n")
            log_fp.write(f"device: {device}\n")
            log_fp.write(f"scales={scales}, iters={iters}\n")
            log_fp.write(f"rigid_scales={rigid_scales}, rigid_iters={rigid_iters}, lr={optimizer_lr}\n\n")
            log_fp.flush()

            # Forward: moving -> fixed
            moved_xyz, rigid_mat, greedy_reg = _run_fireants_pair(
                fixed_path=fixed_path,
                moving_path=moving_path,
                device=device,
                scales=scales,
                iters=iters,
                rigid_scales=rigid_scales,
                rigid_iters=rigid_iters,
                optimizer_lr=optimizer_lr,
                log_fp=log_fp,
            )
            _save_nifti_like_fixed(moved_xyz, fixed_nib, out_warped)

            # Save rigid matrix
            np.save(os.path.join(mat_dir, f"{case_id}_rigid_matrix.npy"), rigid_mat)

            # Export ANTs-style transforms if available
            # Many fireants versions will generate:
            #   <prefix>0GenericAffine.mat
            #   <prefix>1Warp.nii.gz
            #   <prefix>1InverseWarp.nii.gz (sometimes)
            try:
                prefix = os.path.join(mat_dir, f"{case_id}_")
                with contextlib.redirect_stdout(log_fp), contextlib.redirect_stderr(log_fp):
                    greedy_reg.save_as_ants_transforms(prefix)
            except Exception as e:
                log_fp.write(f"\n[WARN] save_as_ants_transforms failed: {e}\n")

            # Optional inverse (fixed -> moving) by running a second registration swapped
            if do_inverse:
                inv_moved_xyz, inv_rigid_mat, inv_greedy_reg = _run_fireants_pair(
                    fixed_path=moving_path,   # swapped
                    moving_path=fixed_path,
                    device=device,
                    scales=scales,
                    iters=iters,
                    rigid_scales=rigid_scales,
                    rigid_iters=rigid_iters,
                    optimizer_lr=optimizer_lr,
                    log_fp=log_fp,
                )
                _save_nifti_like_fixed(inv_moved_xyz, moving_nib, out_invwarped)
                np.save(os.path.join(mat_dir, f"{case_id}_inv_rigid_matrix.npy"), inv_rigid_mat)
                try:
                    inv_prefix = os.path.join(mat_dir, f"{case_id}_inv_")
                    with contextlib.redirect_stdout(log_fp), contextlib.redirect_stderr(log_fp):
                        inv_greedy_reg.save_as_ants_transforms(inv_prefix)
                except Exception as e:
                    log_fp.write(f"\n[WARN] inv save_as_ants_transforms failed: {e}\n")

            dt = time.time() - t0
            log_fp.write(f"\nDONE in {dt:.2f}s\n")

        return "OK", None

    except Exception as e:
        with open(log_path, "a") as log_fp:
            log_fp.write("\n[EXCEPTION]\n")
            log_fp.write(str(e) + "\n")
            log_fp.write(traceback.format_exc() + "\n")
        return "FAIL", str(e)


def _gpu_worker(gpu_id, case_list, args, q):
    _silence_fireants_logging()
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

    for cid in case_list:
        status, msg = _process_one_case(
            case_id=cid,
            fixed_dir=args.fixed_dir,
            moving_dir=args.moving_dir,
            out_base=args.out_base,
            device_id=gpu_id,
            scales=tuple(args.scales),
            iters=tuple(args.iters),
            rigid_scales=tuple(args.rigid_scales),
            rigid_iters=tuple(args.rigid_iters),
            optimizer_lr=args.rigid_lr,
            do_inverse=args.do_inverse,
        )
        q.put((status, cid, gpu_id, msg))


def _collect_cases(fixed_dir, moving_dir):
    fixed_files = sorted(glob.glob(os.path.join(fixed_dir, "*.nii.gz")))
    case_ids = [Path(p).name.replace(".nii.gz", "") for p in fixed_files]

    valid = []
    missing = 0
    for cid in case_ids:
        if os.path.exists(os.path.join(moving_dir, f"{cid}.nii.gz")):
            valid.append(cid)
        else:
            missing += 1
    return valid, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed_dir", type=str, required=True, help="Fixed image dir (e.g. MRI)")
    parser.add_argument("--moving_dir", type=str, required=True, help="Moving image dir (e.g. MRA)")
    parser.add_argument("--out_base", type=str, required=True, help="Output base dir")
    parser.add_argument("--num_gpus", type=int, default=1)

    # FireANTs params
    parser.add_argument("--scales", type=int, nargs="+", default=[4, 2, 1])
    parser.add_argument("--iters", type=int, nargs="+", default=[200, 100, 25])
    parser.add_argument("--rigid_scales", type=int, nargs="+", default=[4, 2])
    parser.add_argument("--rigid_iters", type=int, nargs="+", default=[100, 50])
    parser.add_argument("--rigid_lr", type=float, default=0.01)

    parser.add_argument("--do_inverse", action="store_true",
                        help="Also compute inversewarped by running swapped registration (≈2x time).")

    args = parser.parse_args()

    assert len(args.scales) == len(args.iters), "--scales and --iters must have same length"
    assert len(args.rigid_scales) == len(args.rigid_iters), "--rigid_scales and --rigid_iters must match"

    os.makedirs(args.out_base, exist_ok=True)
    for sub in ["logs", "warped", "inversewarped", "mat"]:
        os.makedirs(os.path.join(args.out_base, sub), exist_ok=True)

    # Discover cases
    cases, missing = _collect_cases(args.fixed_dir, args.moving_dir)

    print(f"Found MRI cases: {len(glob.glob(os.path.join(args.fixed_dir, '*.nii.gz')))}")
    print(f"Runnable paired cases (fixed+moving): {len(cases)}")
    print(f"Skipped due to missing moving files: {missing}")

    # GPUs
    gpu_avail = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_gpus = min(args.num_gpus, gpu_avail, len(cases)) if gpu_avail > 0 else 0

    if use_gpus <= 0:
        print("ERROR: No CUDA GPUs available (or torch.cuda is not available).")
        sys.exit(1)

    print(f"GPUs available: {gpu_avail} | Using: {use_gpus}")
    print(f"Output base: {args.out_base}")
    print(f"FireANTs: scales={tuple(args.scales)}, iters={tuple(args.iters)} | "
          f"rigid_scales={tuple(args.rigid_scales)}, rigid_iters={tuple(args.rigid_iters)}, lr={args.rigid_lr}")
    if args.do_inverse:
        print("InverseWarped: ENABLED (will roughly double runtime)")
    else:
        print("InverseWarped: disabled (use --do_inverse to enable)")

    # Split cases round-robin
    chunks = [[] for _ in range(use_gpus)]
    for i, cid in enumerate(cases):
        chunks[i % use_gpus].append(cid)

    manager = mp.Manager()
    q = manager.Queue()

    procs = []
    for gid in range(use_gpus):
        p = mp.Process(target=_gpu_worker, args=(gid, chunks[gid], args, q))
        p.start()
        procs.append(p)

    ok = 0
    skip = 0
    fail = 0

    total = len(cases)
    desc = "FireANTs (MRI fixed)"
    with tqdm(total=total, desc=desc, ncols=140) as pbar:
        finished = 0
        while finished < total:
            status, cid, gid, msg = q.get()
            finished += 1
            pbar.update(1)

            if status == "OK":
                ok += 1
            elif status == "SKIP_DONE":
                skip += 1
            else:
                fail += 1
                # show one-line failure in tqdm postfix, details in logs/<case>.log
                pbar.set_postfix_str(f"FAIL {cid} (GPU{gid})")

    for p in procs:
        p.join()

    print("\n=== SUMMARY ===")
    print(f"OK: {ok}")
    print(f"SKIP_DONE: {skip}")
    print(f"FAIL: {fail}")
    print(f"Logs: {os.path.join(args.out_base, 'logs')}")
    print(f"Warped: {os.path.join(args.out_base, 'warped')}")
    print(f"InverseWarped: {os.path.join(args.out_base, 'inversewarped')}")
    print(f"Affine/warps: {os.path.join(args.out_base, 'mat')}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
