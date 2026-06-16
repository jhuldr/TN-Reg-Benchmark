#!/usr/bin/env python3
# TN-Reg-Benchmark
# Code author: Xupeng Zhang (xzhan419@jh.edu)
# Johns Hopkins University
# See README for the full author list and citation.

"""
Auto-Cropping Based on Mask with Padding
This script crops MRI/MRA images to their mask bounding box with configurable padding
Reduces memory usage and speeds up registration while preserving coordinate integrity

Key Features:
- Computes tight bounding box from mask
- Adds configurable padding (default: 10 voxels)
- Preserves physical coordinates (origin/spacing/direction)
- Crops both image and mask consistently
- Generates cropped versions for efficient registration

"""

import os
import sys
import glob
import argparse
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import logging
from datetime import datetime


def setup_logger(output_dir):
    """Setup logging configuration"""
    log_file = os.path.join(output_dir, f'autocrop_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def compute_bounding_box_from_mask(mask_img, padding=10):
    """
    Compute bounding box from binary mask with padding

    Args:
        mask_img: SimpleITK binary mask image
        padding: number of voxels to pad on each side (default: 10)

    Returns:
        bbox: dict with keys 'start' and 'size' for each dimension
    """
    mask_arr = sitk.GetArrayFromImage(mask_img)  # z, y, x

    # Find non-zero indices
    nz_indices = np.where(mask_arr > 0)

    if len(nz_indices[0]) == 0:
        raise ValueError("Mask is empty - no foreground voxels found!")

    # Compute min/max for each axis (in numpy array space: z, y, x)
    z_min, z_max = nz_indices[0].min(), nz_indices[0].max()
    y_min, y_max = nz_indices[1].min(), nz_indices[1].max()
    x_min, x_max = nz_indices[2].min(), nz_indices[2].max()

    # Add padding (clip to image boundaries)
    image_shape = mask_arr.shape  # (z, y, x)

    z_min = max(0, z_min - padding)
    z_max = min(image_shape[0] - 1, z_max + padding)
    y_min = max(0, y_min - padding)
    y_max = min(image_shape[1] - 1, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(image_shape[2] - 1, x_max + padding)

    # Convert to SimpleITK convention (x, y, z)
    # SimpleITK uses (x, y, z) order for RegionOfInterestImageFilter
    bbox = {
        'start': [int(x_min), int(y_min), int(z_min)],  # (x, y, z)
        'size': [int(x_max - x_min + 1), int(y_max - y_min + 1), int(z_max - z_min + 1)]  # (x, y, z)
    }

    return bbox


def crop_image_with_bbox(img, bbox):
    """
    Crop image using bounding box

    Args:
        img: SimpleITK image
        bbox: dict with 'start' and 'size' in (x, y, z) order

    Returns:
        cropped_img: SimpleITK image with updated origin
    """
    # Use RegionOfInterestImageFilter
    roi_filter = sitk.RegionOfInterestImageFilter()
    roi_filter.SetSize(bbox['size'])
    roi_filter.SetIndex(bbox['start'])

    cropped_img = roi_filter.Execute(img)

    return cropped_img


def process_one_case(case_id, mri_dir, mra_dir, mri_mask_dir, mra_mask_dir,
                    output_mri_dir, output_mra_dir, output_mri_mask_dir, output_mra_mask_dir,
                    padding, logger):
    """
    Process one case: crop MRI and MRA based on their masks

    Args:
        case_id: case identifier
        mri_dir: input MRI directory
        mra_dir: input MRA directory
        mri_mask_dir: input MRI mask directory
        mra_mask_dir: input MRA mask directory
        output_mri_dir: output cropped MRI directory
        output_mra_dir: output cropped MRA directory
        output_mri_mask_dir: output cropped MRI mask directory
        output_mra_mask_dir: output cropped MRA mask directory
        padding: padding in voxels
        logger: logger object

    Returns:
        result dict with statistics
    """
    try:
        logger.info(f"Processing {case_id}...")

        # Load images
        mri_path = os.path.join(mri_dir, f"{case_id}.nii.gz")
        mra_path = os.path.join(mra_dir, f"{case_id}.nii.gz")
        mri_mask_path = os.path.join(mri_mask_dir, f"{case_id}_mask.nii.gz")
        mra_mask_path = os.path.join(mra_mask_dir, f"{case_id}_mask.nii.gz")

        if not os.path.exists(mri_path):
            logger.warning(f"  SKIP {case_id}: MRI not found")
            return None
        if not os.path.exists(mra_path):
            logger.warning(f"  SKIP {case_id}: MRA not found")
            return None
        if not os.path.exists(mri_mask_path):
            logger.warning(f"  SKIP {case_id}: MRI mask not found")
            return None
        if not os.path.exists(mra_mask_path):
            logger.warning(f"  SKIP {case_id}: MRA mask not found")
            return None

        mri_img = sitk.ReadImage(mri_path)
        mra_img = sitk.ReadImage(mra_path)
        mri_mask = sitk.ReadImage(mri_mask_path)
        mra_mask = sitk.ReadImage(mra_mask_path)

        logger.info(f"  Original MRI size: {mri_img.GetSize()}")
        logger.info(f"  Original MRA size: {mra_img.GetSize()}")

        # Compute bounding boxes
        mri_bbox = compute_bounding_box_from_mask(mri_mask, padding=padding)
        mra_bbox = compute_bounding_box_from_mask(mra_mask, padding=padding)

        logger.info(f"  MRI bbox: start={mri_bbox['start']}, size={mri_bbox['size']}")
        logger.info(f"  MRA bbox: start={mra_bbox['start']}, size={mra_bbox['size']}")

        # Crop MRI and mask
        mri_cropped = crop_image_with_bbox(mri_img, mri_bbox)
        mri_mask_cropped = crop_image_with_bbox(mri_mask, mri_bbox)

        # Crop MRA and mask
        mra_cropped = crop_image_with_bbox(mra_img, mra_bbox)
        mra_mask_cropped = crop_image_with_bbox(mra_mask, mra_bbox)

        logger.info(f"  Cropped MRI size: {mri_cropped.GetSize()}")
        logger.info(f"  Cropped MRA size: {mra_cropped.GetSize()}")

        # Calculate compression ratios
        mri_original_voxels = np.prod(mri_img.GetSize())
        mri_cropped_voxels = np.prod(mri_cropped.GetSize())
        mri_compression = 100 * (1 - mri_cropped_voxels / mri_original_voxels)

        mra_original_voxels = np.prod(mra_img.GetSize())
        mra_cropped_voxels = np.prod(mra_cropped.GetSize())
        mra_compression = 100 * (1 - mra_cropped_voxels / mra_original_voxels)

        logger.info(f"  MRI compression: {mri_compression:.1f}% reduction")
        logger.info(f"  MRA compression: {mra_compression:.1f}% reduction")

        # Save cropped images
        mri_output_path = os.path.join(output_mri_dir, f"{case_id}.nii.gz")
        mra_output_path = os.path.join(output_mra_dir, f"{case_id}.nii.gz")
        mri_mask_output_path = os.path.join(output_mri_mask_dir, f"{case_id}_mask.nii.gz")
        mra_mask_output_path = os.path.join(output_mra_mask_dir, f"{case_id}_mask.nii.gz")

        sitk.WriteImage(mri_cropped, mri_output_path)
        sitk.WriteImage(mra_cropped, mra_output_path)
        sitk.WriteImage(mri_mask_cropped, mri_mask_output_path)
        sitk.WriteImage(mra_mask_cropped, mra_mask_output_path)

        logger.info(f"  ✓ Saved cropped images")
        logger.info(f"SUCCESS {case_id}\n")

        return {
            'case_id': case_id,
            'success': True,
            'mri_original_size': mri_img.GetSize(),
            'mri_cropped_size': mri_cropped.GetSize(),
            'mri_compression': mri_compression,
            'mra_original_size': mra_img.GetSize(),
            'mra_cropped_size': mra_cropped.GetSize(),
            'mra_compression': mra_compression
        }

    except Exception as e:
        logger.error(f"ERROR {case_id}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'case_id': case_id,
            'success': False,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Auto-Crop MRI/MRA Based on Masks with Padding',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Crop single test case with 10-voxel padding
  python autocrop_with_mask.py \\
      --mri_dir ../../preprocessed_ras/mri \\
      --mra_dir ../../preprocessed_ras/mra \\
      --mri_mask_dir ../../masks/mri_masks \\
      --mra_mask_dir ../../masks/mra_masks \\
      --output_mri_dir ../../preprocessed_cropped/mri \\
      --output_mra_dir ../../preprocessed_cropped/mra \\
      --output_mri_mask_dir ../../masks_cropped/mri_masks \\
      --output_mra_mask_dir ../../masks_cropped/mra_masks \\
      --padding 10 \\
      --test_cases <CASE_ID>

  # Process all cases
  python autocrop_with_mask.py \\
      --mri_dir ../../preprocessed_ras/mri \\
      --mra_dir ../../preprocessed_ras/mra \\
      --mri_mask_dir ../../masks/mri_masks \\
      --mra_mask_dir ../../masks/mra_masks \\
      --output_mri_dir ../../preprocessed_cropped/mri \\
      --output_mra_dir ../../preprocessed_cropped/mra \\
      --output_mri_mask_dir ../../masks_cropped/mri_masks \\
      --output_mra_mask_dir ../../masks_cropped/mra_masks \\
      --padding 10
        """
    )

    parser.add_argument('--mri_dir', type=str, required=True,
                        help='Input MRI directory')
    parser.add_argument('--mra_dir', type=str, required=True,
                        help='Input MRA directory')
    parser.add_argument('--mri_mask_dir', type=str, required=True,
                        help='Input MRI mask directory')
    parser.add_argument('--mra_mask_dir', type=str, required=True,
                        help='Input MRA mask directory')
    parser.add_argument('--output_mri_dir', type=str, required=True,
                        help='Output cropped MRI directory')
    parser.add_argument('--output_mra_dir', type=str, required=True,
                        help='Output cropped MRA directory')
    parser.add_argument('--output_mri_mask_dir', type=str, required=True,
                        help='Output cropped MRI mask directory')
    parser.add_argument('--output_mra_mask_dir', type=str, required=True,
                        help='Output cropped MRA mask directory')
    parser.add_argument('--padding', type=int, default=10,
                        help='Padding in voxels (default: 10)')
    parser.add_argument('--test_cases', type=str, nargs='+', default=None,
                        help='Specific case IDs to process')
    parser.add_argument('--num_cases', type=int, default=None,
                        help='Number of cases to process (for testing)')

    args = parser.parse_args()

    # Create output directories
    for dir_path in [args.output_mri_dir, args.output_mra_dir,
                     args.output_mri_mask_dir, args.output_mra_mask_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Setup logger (use first output dir for log)
    logger = setup_logger(args.output_mri_dir)

    logger.info("="*80)
    logger.info("Auto-Cropping Based on Mask with Padding")
    logger.info("="*80)
    logger.info(f"MRI input: {args.mri_dir}")
    logger.info(f"MRA input: {args.mra_dir}")
    logger.info(f"MRI mask input: {args.mri_mask_dir}")
    logger.info(f"MRA mask input: {args.mra_mask_dir}")
    logger.info(f"MRI output: {args.output_mri_dir}")
    logger.info(f"MRA output: {args.output_mra_dir}")
    logger.info(f"Padding: {args.padding} voxels")
    logger.info("="*80)

    # Get case list
    if args.test_cases:
        all_cases = args.test_cases
        logger.info(f"Processing {len(all_cases)} specified test cases\n")
    else:
        all_mri_files = sorted(glob.glob(os.path.join(args.mri_dir, "*.nii.gz")))
        all_cases = [os.path.basename(f).replace(".nii.gz", "") for f in all_mri_files]

        if args.num_cases:
            all_cases = all_cases[:args.num_cases]
            logger.info(f"Processing first {len(all_cases)} cases (test mode)\n")
        else:
            logger.info(f"Processing all {len(all_cases)} cases\n")

    # Process all cases
    results = []
    for i, case_id in enumerate(all_cases):
        logger.info(f"[{i+1}/{len(all_cases)}] " + "-"*60)
        result = process_one_case(
            case_id,
            args.mri_dir,
            args.mra_dir,
            args.mri_mask_dir,
            args.mra_mask_dir,
            args.output_mri_dir,
            args.output_mra_dir,
            args.output_mri_mask_dir,
            args.output_mra_mask_dir,
            args.padding,
            logger
        )
        if result:
            results.append(result)

    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    logger.info(f"Total processed: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")

    if successful > 0:
        # Compute average compression
        mri_compressions = [r['mri_compression'] for r in results if r['success']]
        mra_compressions = [r['mra_compression'] for r in results if r['success']]

        logger.info(f"\nAverage MRI compression: {np.mean(mri_compressions):.1f}% ± {np.std(mri_compressions):.1f}%")
        logger.info(f"Average MRA compression: {np.mean(mra_compressions):.1f}% ± {np.std(mra_compressions):.1f}%")
        logger.info(f"Max MRI compression: {np.max(mri_compressions):.1f}%")
        logger.info(f"Max MRA compression: {np.max(mra_compressions):.1f}%")

    logger.info("="*80)
    logger.info("Auto-cropping complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
