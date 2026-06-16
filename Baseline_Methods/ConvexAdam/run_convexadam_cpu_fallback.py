#!/usr/bin/env python3
"""
Run Batch in ConvexAdam MRI fixed & MRA Moving — CPU fallback

Uses convex_adam_pt (low-level API) so we can pass `device="cpu"`.

Author: Xupeng Zhang
Johns Hopkins University
"""

import os
import sys
import glob
import time
import numpy as np
import nibabel as nib
import torch
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

# Use convex_adam_pt directly — it accepts device= kwarg.
# The convex_adam_MIND wrapper does NOT accept device.
from MIR.models.convexAdam.convex_adam_MIND import convex_adam_pt
import MIR.models.convexAdam.configs_ConvexAdam_MIND as CONFIGS_CVXAdam


def case_id_from_filename(fname: str) -> str:
    name = Path(fname).name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    return name.split("_")[0]


def resolve_nii_path(dir_path: Path, case_id: str):
    p = dir_path / f"{case_id}.nii.gz"
    if p.exists():
        return p
    p = dir_path / f"{case_id}.nii"
    if p.exists():
        return p
    return None


def robust_norm(vol: np.ndarray, p=99.5, eps=1e-5) -> np.ndarray:
    v = vol.astype(np.float32)
    denom = np.percentile(v, p) + eps
    return np.clip(v / denom, 0.0, 1.0)


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
    f_img = sitk.ReadImage(fixed_path)
    m_img = sitk.ReadImage(moving_path)
    f_img = to_3d_if_4d(f_img)
    m_img = to_3d_if_4d(m_img)
    f_img = sitk.Cast(f_img, sitk.sitkFloat32)
    m_img = sitk.Cast(m_img, sitk.sitkFloat32)

    initial_transform = sitk.CenteredTransformInitializer(
        f_img, m_img, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.01)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=30,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(initial_transform, inPlace=False)

    try:
        final_transform = reg.Execute(f_img, m_img)
    except Exception:
        final_transform = initial_transform

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(f_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(final_transform)
    m_res = resampler.Execute(m_img)
    m_res_np = sitk.GetArrayFromImage(m_res)
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

            res_m = sitk.ResampleImageFilter()
            res_m.SetReferenceImage(f_img)
            res_m.SetInterpolator(sitk.sitkNearestNeighbor)
            res_m.SetDefaultPixelValue(0)
            res_m.SetTransform(final_transform)
            mm_res = res_m.Execute(mm)

            fm_np = sitk.GetArrayFromImage(fm) > 0
            mm_np = sitk.GetArrayFromImage(mm_res) > 0
            mask_np = (fm_np & mm_np).astype(np.float32)
        except Exception:
            mask_np = None

    return m_res_np, f_np, mask_np


def save_nifti_like(ref_nii_path: str, arr: np.ndarray, out_path: str, dtype=np.float32):
    ref = nib.load(ref_nii_path)
    img = nib.Nifti1Image(arr.astype(dtype), ref.affine, ref.header)
    img.set_data_dtype(dtype)
    nib.save(img, out_path)


def save_disp_np(ref_nii_path: str, disp, out_path: str):
    """
    disp from convex_adam_pt is a torch.Tensor of shape (1, 3, D, H, W) when save_disp=False.
    Convert to numpy (D, H, W, 3) then save as NIfTI with reference geometry.
    """
    if isinstance(disp, torch.Tensor):
        # convex_adam_pt returns disp_hr shaped (1, 3, D, H, W)
        disp_np = disp.squeeze(0).permute(1, 2, 3, 0).detach().cpu().numpy()
    else:
        disp_np = np.asarray(disp)
    ref = nib.load(ref_nii_path)
    img = nib.Nifti1Image(disp_np.astype(np.float32), ref.affine, ref.header)
    img.set_data_dtype(np.float32)
    nib.save(img, out_path)


def run_one_case(case: dict, out_base: Path, use_masks: bool, device: torch.device,
                 config) -> tuple[str, str, str]:
    """Run ConvexAdam on a single case, on given device. Returns (case_id, status, info)."""
    cid = case["case_id"]
    fixed = case["fixed"]
    moving = case["moving"]
    fmask = case.get("fmask")
    mmask = case.get("mmask")

    disp_dir = out_base / "disp"
    pre_dir = out_base / "prealigned"
    log_dir = out_base / "logs"

    out_disp = disp_dir / f"{cid}_reg_disp.nii.gz"
    out_pre = pre_dir / f"{cid}_prealigned_mra.nii.gz"
    out_log = log_dir / f"{cid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    if out_disp.exists() and out_pre.exists():
        return (cid, "SKIP_DONE", str(out_disp))

    t0 = time.time()
    try:
        with open(out_log, "w") as f:
            f.write(f"DEVICE {device} | CASE {cid}\n")
            f.write(f"fixed: {fixed}\n")
            f.write(f"moving: {moving}\n")
            f.write(f"use_masks: {use_masks}\n\n")
            f.flush()

            # 1) coarse affine + resample moving->fixed
            m_np, f_np, mask_np = affine_align_and_resample(
                moving, fixed,
                moving_mask_path=mmask,
                fixed_mask_path=fmask,
                use_masks=use_masks,
            )
            f.write(f"After affine resample: moving shape {m_np.shape}\n")
            f.flush()

            # 2) normalize
            x = robust_norm(m_np, p=99.5)
            y = robust_norm(f_np, p=99.5)

            # 3) apply mask (if available)
            if mask_np is not None:
                x = x * mask_np
                y = y * mask_np

            # 4) save prealigned moving (debug)
            save_nifti_like(fixed, x.transpose(2, 1, 0), str(out_pre), dtype=np.float32)
            f.write(f"Wrote prealigned: {out_pre}\n")
            f.flush()

            # 5) run convex_adam_pt on chosen device
            #    Accepts plain torch tensors (shape (D, H, W) float32).
            x_t = torch.from_numpy(x).float().to(device)
            y_t = torch.from_numpy(y).float().to(device)

            f.write(f"Calling convex_adam_pt on device={device}, tensor shape={x_t.shape}\n")
            f.flush()

            disp = convex_adam_pt(
                img_fixed=y.astype(np.float32),
                img_moving=x.astype(np.float32),
                mind_r=config.mind_r,
                mind_d=config.mind_d,
                lambda_weight=config.lambda_weight,
                grid_sp=config.grid_sp,
                disp_hw=config.disp_hw,
                selected_niter=config.selected_niter,
                selected_smooth=config.selected_smooth,
                grid_sp_adam=config.grid_sp_adam,
                ic=config.ic,
                use_mask=False,  # we already applied mask above
                path_fixed_mask=None,
                path_moving_mask=None,
                dtype=torch.float32,  # float16 not supported on CPU
                verbose=False,
                device=device,
                save_disp=True,
            )

            # 6) save displacement field
            # convex_adam_pt returns (D, H, W, 3) numpy when save_disp=False
            save_disp_np(fixed, disp, str(out_disp))
            f.write(f"Wrote disp: {out_disp}\n")

            dt = time.time() - t0
            f.write(f"\nSUCCESS. time_sec={dt:.2f}\n")
            f.flush()

        return (cid, "OK", str(out_disp))

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        try:
            with open(out_log, "a") as f:
                f.write(f"\nFAIL: {repr(e)}\n")
                f.write(f"\n=== TRACEBACK ===\n{tb}\n")
        except Exception:
            pass
        return (cid, "FAIL", str(out_log))


def main():
    PROJ = Path("/path/to/TN_Reg")

    # 4-case fallback input dir
    mri_dir = PROJ / "data" / "preprocessed_cropped_4cases" / "mri"
    mra_dir = PROJ / "data" / "preprocessed_cropped_4cases" / "mra"
    mri_mask_dir = PROJ / "data" / "masks_cropped" / "mri_masks"
    mra_mask_dir = PROJ / "data" / "masks_cropped" / "mra_masks"

    # Output base (same as GPU run -> goes to disp/, prealigned/, warped/ alongside)
    out_base = PROJ / "outputs" / "MRIfixed_MRAmoving" / "ConvexAdam_result"

    use_masks = True
    skip_missing_masks = True

    # Force CPU (ignore CUDA availability)
    device = torch.device("cpu")
    print(f"Running on device: {device}")

    # Discover cases
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
        case = {"case_id": cid, "fixed": str(fixed), "moving": str(moving)}
        if use_masks and fmask.exists() and mmask.exists():
            case["fmask"] = str(fmask)
            case["mmask"] = str(mmask)
        else:
            if use_masks and (not skip_missing_masks):
                missing += 1
                continue
        all_cases.append(case)

    out_base.mkdir(parents=True, exist_ok=True)
    (out_base / "disp").mkdir(parents=True, exist_ok=True)
    (out_base / "prealigned").mkdir(parents=True, exist_ok=True)
    (out_base / "logs").mkdir(parents=True, exist_ok=True)

    print(f"Found MRI cases: {len(case_ids)}")
    print(f"Runnable cases: {len(all_cases)} | missing: {missing}")
    print(f"Output base: {out_base}")
    print(f"use_masks={use_masks}")

    # Load config
    config = CONFIGS_CVXAdam.get_ConvexAdam_MIND_brain_default_config()

    # Sequential run (CPU — no need for multiprocessing)
    ok = skip = fail = 0
    with tqdm(total=len(all_cases), desc="ConvexAdam CPU (MRI fixed)", dynamic_ncols=True) as pbar:
        for case in all_cases:
            cid, status, info = run_one_case(case, out_base, use_masks, device, config)
            if status == "OK":
                ok += 1
            elif status == "SKIP_DONE":
                skip += 1
            else:
                fail += 1
            pbar.set_postfix_str(f"OK={ok} SKIP={skip} FAIL={fail} | {cid}")
            pbar.update(1)

    print("\n=== SUMMARY ===")
    print(f"OK: {ok}")
    print(f"SKIP_DONE: {skip}")
    print(f"FAIL: {fail}")
    print(f"disp: {out_base / 'disp'}")
    print(f"prealigned: {out_base / 'prealigned'}")
    print(f"logs: {out_base / 'logs'}")


if __name__ == "__main__":
    main()

