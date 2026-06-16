#!/usr/bin/env python3
# TN-Reg-Benchmark
# Code author: Xupeng Zhang (xzhan419@jh.edu)
# Johns Hopkins University
# See README for the full author list and citation.

"""
Run Batch in ConvexAdam MRI fixed & MRA Moving

"""

import os
import sys
import glob
import time
import json
import numpy as np
import nibabel as nib
import torch
import torch.multiprocessing as mp
import SimpleITK as sitk
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# ----------------------------
# 0) MIR library path anchor
# MIR is a third-party toolbox (install separately; see README).
# Set MIR_SRC to its `src` dir, or pip-install MIR so `import MIR` resolves.
# ----------------------------
LIBRARY_PATH = os.environ.get("MIR_SRC", "/path/to/MIR/src")
if LIBRARY_PATH not in sys.path:
    sys.path.insert(0, LIBRARY_PATH)

from MIR.models import convex_adam_MIND
import MIR.models.convexAdam.configs_ConvexAdam_MIND as CONFIGS_CVXAdam


def case_id_from_filename(fname: str) -> str:
    """Supports CASEID.nii.gz / CASEID.nii / CASEID_xxx.nii.gz"""
    name = Path(fname).name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    return name.split("_")[0]


def resolve_nii_path(dir_path: Path, case_id: str):
    """Prefer .nii.gz then .nii; return None if missing."""
    p = dir_path / f"{case_id}.nii.gz"
    if p.exists():
        return p
    p = dir_path / f"{case_id}.nii"
    if p.exists():
        return p
    return None


def robust_norm(vol: np.ndarray, p=99.5, eps=1e-5) -> np.ndarray:
    """Normalize to [0,1] by high percentile, robust to outliers."""
    v = vol.astype(np.float32)
    denom = np.percentile(v, p) + eps
    v = np.clip(v / denom, 0.0, 1.0)
    return v


def to_3d_if_4d(img: sitk.Image) -> sitk.Image:
    if img.GetDimension() == 4:
        img = sitk.Extract(
            img,
            (img.GetSize()[0], img.GetSize()[1], img.GetSize()[2], 0),
            (0, 0, 0, 0),
        )
    return img


def affine_align_and_resample(
    moving_path: str,
    fixed_path: str,
    moving_mask_path: str | None = None,
    fixed_mask_path: str | None = None,
    use_masks: bool = True,
):
    """
    1) SimpleITK coarse registration (MI) with GEOMETRY init
    2) Resample moving onto fixed grid
    3) Optionally resample moving mask onto fixed grid and return intersection masks
    Returns:
        m_res_np, f_np, mask_np_or_None
        plus: fixed_sitk (for geometry), final_transform
    """
    f_img = sitk.ReadImage(fixed_path)
    m_img = sitk.ReadImage(moving_path)

    f_img = to_3d_if_4d(f_img)
    m_img = to_3d_if_4d(m_img)

    f_img = sitk.Cast(f_img, sitk.sitkFloat32)
    m_img = sitk.Cast(m_img, sitk.sitkFloat32)

    # geometry-centered init
    initial_transform = sitk.CenteredTransformInitializer(
        f_img,
        m_img,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.01)
    reg.SetInterpolator(sitk.sitkLinear)

    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=30,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(initial_transform, inPlace=False)

    try:
        final_transform = reg.Execute(f_img, m_img)
    except Exception:
        final_transform = initial_transform

    # resample moving -> fixed grid
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(f_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(final_transform)
    m_res = resampler.Execute(m_img)

    m_res_np = sitk.GetArrayFromImage(m_res)  # [D,H,W]
    f_np = sitk.GetArrayFromImage(f_img)

    mask_np = None
    if use_masks and fixed_mask_path and moving_mask_path:
        try:
            fm = sitk.ReadImage(fixed_mask_path)
            mm = sitk.ReadImage(moving_mask_path)
            fm = to_3d_if_4d(fm)
            mm = to_3d_if_4d(mm)
            fm = sitk.Cast(fm, sitk.sitkUInt8)
            mm = sitk.Cast(mm, sitk.sitkUInt8)

            # resample moving mask -> fixed grid using same transform (nearest)
            res_m = sitk.ResampleImageFilter()
            res_m.SetReferenceImage(f_img)
            res_m.SetInterpolator(sitk.sitkNearestNeighbor)
            res_m.SetDefaultPixelValue(0)
            res_m.SetTransform(final_transform)
            mm_res = res_m.Execute(mm)

            fm_np = sitk.GetArrayFromImage(fm) > 0
            mm_np = sitk.GetArrayFromImage(mm_res) > 0

            mask_np = (fm_np & mm_np).astype(np.float32)  # intersection
        except Exception:
            mask_np = None

    return m_res_np, f_np, mask_np, f_img, final_transform


def save_nifti_like(ref_nii_path: str, arr: np.ndarray, out_path: str, dtype=np.float32):
    """Save numpy array with ref affine/header zooms."""
    ref = nib.load(ref_nii_path)
    img = nib.Nifti1Image(arr.astype(dtype), ref.affine, ref.header)
    img.set_data_dtype(dtype)
    nib.save(img, out_path)


def save_disp_like(ref_nii_path: str, flow_tensor: torch.Tensor, out_path: str):
    """
    flow: [1, 3, D, H, W] -> save as [D, H, W, 3]
    """
    flow_np = (
        flow_tensor.squeeze(0)
        .permute(1, 2, 3, 0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    ref = nib.load(ref_nii_path)
    img = nib.Nifti1Image(flow_np, ref.affine, ref.header)
    img.set_data_dtype(np.float32)
    nib.save(img, out_path)


def gpu_worker(
    rank: int,
    gpu_id: int,
    cases: list[dict],
    out_base: str,
    use_masks: bool,
    queue: mp.Queue,
):
    """
    One process per GPU.
    Sends one message per finished case to queue: (case_id, status, info)
    """
    try:
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        torch.backends.cudnn.benchmark = True

        config = CONFIGS_CVXAdam.get_ConvexAdam_MIND_brain_default_config()

        out_base = Path(out_base)
        disp_dir = out_base / "disp"
        pre_dir = out_base / "prealigned"
        log_dir = out_base / "logs"
        tmp_dir = out_base / "_tmp"

        for d in [disp_dir, pre_dir, log_dir, tmp_dir]:
            d.mkdir(parents=True, exist_ok=True)

        for case in cases:
            cid = case["case_id"]
            fixed = case["fixed"]
            moving = case["moving"]
            fmask = case.get("fmask")
            mmask = case.get("mmask")

            out_disp = disp_dir / f"{cid}_reg_disp.nii.gz"
            out_pre = pre_dir / f"{cid}_prealigned_mra.nii.gz"
            out_log = log_dir / f"{cid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

            if out_disp.exists() and out_pre.exists():
                queue.put((cid, "SKIP_DONE", str(out_disp)))
                continue

            t0 = time.time()
            try:
                with open(out_log, "w") as f:
                    f.write(f"GPU {gpu_id} | CASE {cid}\n")
                    f.write(f"fixed: {fixed}\n")
                    f.write(f"moving: {moving}\n")
                    f.write(f"use_masks: {use_masks}\n\n")
                    f.flush()

                    # 1) coarse affine + resample moving->fixed
                    m_np, f_np, mask_np, _, _ = affine_align_and_resample(
                        moving, fixed,
                        moving_mask_path=mmask,
                        fixed_mask_path=fmask,
                        use_masks=use_masks,
                    )

                    # 2) normalize
                    x = robust_norm(m_np, p=99.5)
                    y = robust_norm(f_np, p=99.5)

                    # 3) apply mask (if available)
                    if mask_np is not None:
                        x = x * mask_np
                        y = y * mask_np

                    # 4) save coarse prealigned moving for debugging/inspection
                    # NOTE: saved in FIXED geometry (same as fixed nifti)
                    save_nifti_like(fixed, x.transpose(2, 1, 0), str(out_pre), dtype=np.float32)

                    # 5) run ConvexAdam MIND
                    x_t = torch.from_numpy(x[None, None, ...]).to(device).float()
                    y_t = torch.from_numpy(y[None, None, ...]).to(device).float()

                    flow = convex_adam_MIND(x_t, y_t, config)

                    # 6) save displacement field in fixed space
                    save_disp_like(fixed, flow, str(out_disp))

                    # cleanup
                    del flow, x_t, y_t
                    torch.cuda.empty_cache()

                    dt = time.time() - t0
                    f.write(f"\nSUCCESS. time_sec={dt:.2f}\n")
                    f.flush()

                queue.put((cid, "OK", str(out_disp)))

            except Exception as e:
                try:
                    with open(out_log, "a") as f:
                        f.write(f"\nFAIL: {repr(e)}\n")
                except Exception:
                    pass
                torch.cuda.empty_cache()
                queue.put((cid, "FAIL", str(out_log)))

        queue.put((f"__WORKER_DONE__{rank}", "DONE", f"gpu={gpu_id}"))

    except Exception as e:
        queue.put((f"__WORKER_CRASH__{rank}", "CRASH", repr(e)))


def main():
    PROJ = Path("/path/to/TN_Reg")

    # --- ANTs-like inputs ---
    mri_dir = PROJ / "data" / "preprocessed_cropped" / "mri"   # fixed
    mra_dir = PROJ / "data" / "preprocessed_cropped" / "mra"   # moving
    mri_mask_dir = PROJ / "data" / "masks_cropped" / "mri_masks"
    mra_mask_dir = PROJ / "data" / "masks_cropped" / "mra_masks"

    # --- output base ---
    out_base = PROJ / "outputs" / "MRIfixed_MRAmoving" / "ConvexAdam_result"

    # settings
    use_masks = True
    skip_missing_masks = True  # if True: mask missing -> run without mask

    # GPUs
    n_avail = torch.cuda.device_count()
    gpus = list(range(n_avail)) if n_avail > 0 else []
    if len(gpus) == 0:
        raise RuntimeError("No CUDA GPUs detected.")

    # if you want exactly 8:
    # gpus = list(range(min(8, n_avail)))

    # discover cases from MRI folder
    mri_files = sorted(glob.glob(str(mri_dir / "*.nii*")))
    case_ids = sorted({case_id_from_filename(p) for p in mri_files})

    all_cases = []
    missing = 0

    for cid in case_ids:
        fixed = resolve_nii_path(mri_dir, cid)
        moving = resolve_nii_path(mra_dir, cid)
        if fixed is None or moving is None:
            missing += 1
            continue

        fmask = mri_mask_dir / f"{cid}_mask.nii.gz"
        mmask = mra_mask_dir / f"{cid}_mask.nii.gz"

        case = {
            "case_id": cid,
            "fixed": str(fixed),
            "moving": str(moving),
        }

        if use_masks and fmask.exists() and mmask.exists():
            case["fmask"] = str(fmask)
            case["mmask"] = str(mmask)
        else:
            if use_masks and (not skip_missing_masks):
                # require masks but missing -> skip
                missing += 1
                continue

        all_cases.append(case)

    out_base.mkdir(parents=True, exist_ok=True)
    (out_base / "disp").mkdir(parents=True, exist_ok=True)
    (out_base / "prealigned").mkdir(parents=True, exist_ok=True)
    (out_base / "logs").mkdir(parents=True, exist_ok=True)
    (out_base / "_tmp").mkdir(parents=True, exist_ok=True)

    print(f"Found MRI cases: {len(case_ids)}")
    print(f"Runnable cases (paired MRI+MRA): {len(all_cases)} | missing/skipped: {missing}")
    print(f"Output base: {out_base}")
    print(f"GPUs: {gpus} (total {len(gpus)}) | use_masks={use_masks}")

    # split cases to GPUs (round robin)
    num_gpus = len(gpus)
    chunks = [all_cases[i::num_gpus] for i in range(num_gpus)]

    mp.set_start_method("spawn", force=True)
    queue = mp.Queue()

    procs = []
    for rank, gpu_id in enumerate(gpus):
        p = mp.Process(
            target=gpu_worker,
            args=(rank, gpu_id, chunks[rank], str(out_base), use_masks, queue),
            daemon=False,
        )
        p.start()
        procs.append(p)

    # progress bar: one tick per case done/skip/fail
    total = len(all_cases)
    done_workers = 0

    ok = skip = fail = 0

    with tqdm(total=total, desc="ConvexAdam (MRI fixed)", dynamic_ncols=True) as pbar:
        while done_workers < len(procs):
            cid, status, info = queue.get()

            if status == "DONE":
                done_workers += 1
                continue
            if status == "CRASH":
                done_workers += 1
                pbar.set_postfix_str(f"WORKER_CRASH {cid}")
                continue

            if status == "OK":
                ok += 1
                pbar.set_postfix_str(f"OK={ok} SKIP={skip} FAIL={fail} | {cid}")
            elif status == "SKIP_DONE":
                skip += 1
                pbar.set_postfix_str(f"OK={ok} SKIP={skip} FAIL={fail} | {cid}")
            else:
                fail += 1
                pbar.set_postfix_str(f"OK={ok} SKIP={skip} FAIL={fail} | FAIL {cid}")

            pbar.update(1)

    for p in procs:
        p.join()

    print("\n=== SUMMARY ===")
    print("OK:", ok)
    print("SKIP_DONE:", skip)
    print("FAIL:", fail)
    print(f"disp: {out_base / 'disp'}")
    print(f"prealigned: {out_base / 'prealigned'}")
    print(f"logs: {out_base / 'logs'}")


if __name__ == "__main__":
    main()
