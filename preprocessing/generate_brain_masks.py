#!/usr/bin/env python3
"""
Brain Mask Generation for MRI and MRA Images
Generates binary masks to remove black background for reverse registration

Author: Xupeng Zhang
Johns Hopkins University
Date: January 27, 2026
"""

import os
import sys
import glob
import argparse
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm


def setup_logger(output_dir):
    """Setup logging configuration"""
    log_file = os.path.join(output_dir, f'generate_masks_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def generate_mri_mask(mri_image, logger):
    """
    Generate brain tissue mask for MRI (REVISED - Solid Brain Coverage)

    Strategy:
    - LOW threshold (3% of max) to include all gray/white matter
    - LARGE kernel closing (12) to connect brain parts
    - Fill holes to ensure solid interior
    - Extract largest component (exclude neck/shoulders)
    - Mild dilation to avoid edge cutoff

    Args:
        mri_image: SimpleITK image
        logger: logger object

    Returns:
        mask: SimpleITK binary mask image
        stats: dictionary with mask statistics
    """
    # Step 1: Manual low threshold (include all tissue with signal)
    img_array = sitk.GetArrayFromImage(mri_image)
    threshold_value = img_array.max() * 0.03  # 3% of maximum
    logger.info(f"  [MRI] Manual threshold: {threshold_value:.2f} (3% of max)")

    threshold_filter = sitk.BinaryThresholdImageFilter()
    threshold_filter.SetLowerThreshold(float(threshold_value))
    threshold_filter.SetUpperThreshold(float(img_array.max()))
    threshold_filter.SetInsideValue(1)
    threshold_filter.SetOutsideValue(0)
    mask = threshold_filter.Execute(mri_image)

    # Step 2: Large kernel closing (connect brain parts)
    logger.info("  [MRI] Morphological closing (kernel=12, connect brain parts)...")
    closing = sitk.BinaryMorphologicalClosingImageFilter()
    closing.SetKernelRadius(12)  # Larger kernel for aggressive connection
    closing.SetForegroundValue(1)
    mask = closing.Execute(mask)

    # Step 3: Fill holes (CRITICAL - ensure solid interior)
    logger.info("  [MRI] Filling holes (ensure solid brain interior)...")
    fill_holes = sitk.BinaryFillholeImageFilter()
    fill_holes.SetForegroundValue(1)
    mask = fill_holes.Execute(mask)

    # Step 4: Extract largest component (remove neck/shoulders)
    logger.info("  [MRI] Extracting largest connected component...")
    connected = sitk.ConnectedComponentImageFilter()
    labeled = connected.Execute(mask)

    stats_filter = sitk.LabelShapeStatisticsImageFilter()
    stats_filter.Execute(labeled)

    if stats_filter.GetNumberOfLabels() == 0:
        logger.warning("  [MRI] No foreground detected! Returning empty mask.")
        return mask, {'volume': 0, 'percentage': 0}

    largest_label = max(stats_filter.GetLabels(),
                       key=lambda l: stats_filter.GetNumberOfPixels(l))
    mask = (labeled == largest_label)

    # Step 5: Mild dilation (include cortical edges)
    logger.info("  [MRI] Mild dilation (kernel=3, avoid edge cutoff)...")
    dilate = sitk.BinaryDilateImageFilter()
    dilate.SetKernelRadius(3)  # Slightly larger than before
    dilate.SetForegroundValue(1)
    mask = dilate.Execute(mask)

    # Compute statistics
    mask_array = sitk.GetArrayFromImage(mask)
    total_voxels = mask_array.size
    foreground_voxels = np.sum(mask_array)
    percentage = (foreground_voxels / total_voxels) * 100

    stats = {
        'volume': int(foreground_voxels),
        'percentage': percentage
    }

    logger.info(f"  [MRI] Mask volume: {foreground_voxels:,} voxels ({percentage:.1f}% of FOV)")

    return mask, stats


def generate_mra_mask(mra_image, logger):
    """
    Generate vessel mask for MRA (REVISED - Vessel Network Coverage)

    Strategy:
    - MEDIUM threshold (10% of max) to include vessels and surrounding tissue
    - MEDIUM kernel closing (8) to connect vessels without over-merging
    - Fill holes to cover vessel network
    - Extract largest component (exclude noise)
    - No extra dilation (boundaries already sufficient)

    Args:
        mra_image: SimpleITK image
        logger: logger object

    Returns:
        mask: SimpleITK binary mask image
        stats: dictionary with mask statistics
    """
    # Step 1: Manual medium threshold (vessels are bright)
    img_array = sitk.GetArrayFromImage(mra_image)
    threshold_value = img_array.max() * 0.10  # 10% of maximum
    logger.info(f"  [MRA] Manual threshold: {threshold_value:.2f} (10% of max)")

    threshold_filter = sitk.BinaryThresholdImageFilter()
    threshold_filter.SetLowerThreshold(float(threshold_value))
    threshold_filter.SetUpperThreshold(float(img_array.max()))
    threshold_filter.SetInsideValue(1)
    threshold_filter.SetOutsideValue(0)
    mask = threshold_filter.Execute(mra_image)

    # Step 2: Medium kernel closing (connect vessels, avoid over-merging)
    logger.info("  [MRA] Morphological closing (kernel=8, connect vessel branches)...")
    closing = sitk.BinaryMorphologicalClosingImageFilter()
    closing.SetKernelRadius(8)  # Medium kernel to preserve vessel structure
    closing.SetForegroundValue(1)
    mask = closing.Execute(mask)

    # Step 3: Fill holes (cover vessel network, preserve structure)
    logger.info("  [MRA] Filling holes (solidify vessel network)...")
    fill_holes = sitk.BinaryFillholeImageFilter()
    fill_holes.SetForegroundValue(1)
    mask = fill_holes.Execute(mask)

    # Step 4: Extract largest component
    logger.info("  [MRA] Extracting largest connected component...")
    connected = sitk.ConnectedComponentImageFilter()
    labeled = connected.Execute(mask)

    stats_filter = sitk.LabelShapeStatisticsImageFilter()
    stats_filter.Execute(labeled)

    if stats_filter.GetNumberOfLabels() == 0:
        logger.warning("  [MRA] No foreground detected! Returning empty mask.")
        return mask, {'volume': 0, 'percentage': 0}

    largest_label = max(stats_filter.GetLabels(),
                       key=lambda l: stats_filter.GetNumberOfPixels(l))
    mask = (labeled == largest_label)

    # Step 5: No additional dilation (MRA boundaries already clear)

    # Compute statistics
    mask_array = sitk.GetArrayFromImage(mask)
    total_voxels = mask_array.size
    foreground_voxels = np.sum(mask_array)
    percentage = (foreground_voxels / total_voxels) * 100

    stats = {
        'volume': int(foreground_voxels),
        'percentage': percentage
    }

    logger.info(f"  [MRA] Mask volume: {foreground_voxels:,} voxels ({percentage:.1f}% of FOV)")

    return mask, stats


def visualize_mask_quality(image_path, mask_path, output_path, modality="Image"):
    """
    Generate 3-view visualization of mask quality

    Args:
        image_path: path to original image
        mask_path: path to mask
        output_path: path to save visualization
        modality: "MRI" or "MRA" for title
    """
    # Load data
    img = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)

    img_array = sitk.GetArrayFromImage(img)
    mask_array = sitk.GetArrayFromImage(mask)

    # Get middle slices
    D, H, W = img_array.shape
    slice_d = D // 2  # Axial
    slice_h = H // 2  # Coronal
    slice_w = W // 2  # Sagittal

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Original images
    axes[0, 0].imshow(img_array[slice_d, :, :], cmap='gray')
    axes[0, 0].set_title(f'{modality} - Axial (slice {slice_d}/{D})')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img_array[:, slice_h, :], cmap='gray')
    axes[0, 1].set_title(f'{modality} - Coronal (slice {slice_h}/{H})')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(img_array[:, :, slice_w], cmap='gray')
    axes[0, 2].set_title(f'{modality} - Sagittal (slice {slice_w}/{W})')
    axes[0, 2].axis('off')

    # Row 2: Mask overlay (red contour)
    for idx, (sl_idx, sl_name, sl_axis) in enumerate([
        (slice_d, 'Axial', (0, 1, 2)),
        (slice_h, 'Coronal', (0, 2, 1)),
        (slice_w, 'Sagittal', (1, 2, 0))
    ]):
        # Get slices
        if idx == 0:  # Axial
            img_slice = img_array[sl_idx, :, :]
            mask_slice = mask_array[sl_idx, :, :]
        elif idx == 1:  # Coronal
            img_slice = img_array[:, sl_idx, :]
            mask_slice = mask_array[:, sl_idx, :]
        else:  # Sagittal
            img_slice = img_array[:, :, sl_idx]
            mask_slice = mask_array[:, :, sl_idx]

        # Display
        axes[1, idx].imshow(img_slice, cmap='gray')

        # Overlay mask contour
        from scipy import ndimage
        mask_edge = mask_slice.astype(float) - ndimage.binary_erosion(mask_slice).astype(float)

        # Create red overlay
        overlay = np.zeros((*mask_edge.shape, 4))
        overlay[mask_edge > 0] = [1, 0, 0, 0.8]  # Red with alpha

        axes[1, idx].imshow(overlay)
        axes[1, idx].set_title(f'Mask Overlay - {sl_name}')
        axes[1, idx].axis('off')

    plt.suptitle(f'{modality} Mask Quality Check', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_one_case(case_id, mri_dir, mra_dir, output_dir, logger, visualize=True):
    """
    Generate masks for one case (MRI + MRA)

    Args:
        case_id: case identifier
        mri_dir: directory containing MRI images
        mra_dir: directory containing MRA images
        output_dir: output directory for masks
        logger: logger object
        visualize: whether to generate visualization

    Returns:
        success: True if successful
    """
    logger.info(f"Processing case: {case_id}")

    # Paths
    mri_path = os.path.join(mri_dir, f"{case_id}.nii.gz")
    mra_path = os.path.join(mra_dir, f"{case_id}.nii.gz")

    # Check if files exist
    if not os.path.exists(mri_path):
        logger.error(f"  MRI file not found: {mri_path}")
        return False
    if not os.path.exists(mra_path):
        logger.error(f"  MRA file not found: {mra_path}")
        return False

    # Load images
    try:
        mri_img = sitk.ReadImage(mri_path)
        mra_img = sitk.ReadImage(mra_path)
    except Exception as e:
        logger.error(f"  Failed to load images: {e}")
        return False

    logger.info(f"  MRI shape: {mri_img.GetSize()}, spacing: {mri_img.GetSpacing()}")
    logger.info(f"  MRA shape: {mra_img.GetSize()}, spacing: {mra_img.GetSpacing()}")

    # Generate MRI mask
    logger.info("  Generating MRI mask...")
    mri_mask, mri_stats = generate_mri_mask(mri_img, logger)

    # Generate MRA mask
    logger.info("  Generating MRA mask...")
    mra_mask, mra_stats = generate_mra_mask(mra_img, logger)

    # Check for anomalies
    if mri_stats['percentage'] < 5 or mri_stats['percentage'] > 95:
        logger.warning(f"MRI mask anomaly: {mri_stats['percentage']:.1f}% foreground")

    if mra_stats['percentage'] < 5 or mra_stats['percentage'] > 95:
        logger.warning(f"MRA mask anomaly: {mra_stats['percentage']:.1f}% foreground")

    # Save masks
    mri_mask_path = os.path.join(output_dir, "mri_masks", f"{case_id}_mask.nii.gz")
    mra_mask_path = os.path.join(output_dir, "mra_masks", f"{case_id}_mask.nii.gz")

    sitk.WriteImage(mri_mask, mri_mask_path)
    sitk.WriteImage(mra_mask, mra_mask_path)

    logger.info(f"MRI mask saved: {mri_mask_path}")
    logger.info(f"MRA mask saved: {mra_mask_path}")

    # Generate visualizations
    if visualize:
        logger.info("  Generating quality check visualizations...")

        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        mri_vis_path = os.path.join(vis_dir, f"{case_id}_mri_mask_check.png")
        mra_vis_path = os.path.join(vis_dir, f"{case_id}_mra_mask_check.png")

        try:
            visualize_mask_quality(mri_path, mri_mask_path, mri_vis_path, modality="MRI")
            logger.info(f"MRI visualization saved: {mri_vis_path}")
        except Exception as e:
            logger.warning(f"Failed to generate MRI visualization: {e}")

        try:
            visualize_mask_quality(mra_path, mra_mask_path, mra_vis_path, modality="MRA")
            logger.info(f"MRA visualization saved: {mra_vis_path}")
        except Exception as e:
            logger.warning(f"  Failed to generate MRA visualization: {e}")

    logger.info(f"SUCCESS: {case_id}\n")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate brain/vessel masks for MRI and MRA images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single test case
  python generate_brain_masks.py \\
      --mri_dir ../../nifti_mri \\
      --mra_dir ../../nifti_mra \\
      --output_dir ../../masks \\
      --test_cases <CASE_ID>

  # Process all cases without visualization
  python generate_brain_masks.py \\
      --mri_dir ../../nifti_mri \\
      --mra_dir ../../nifti_mra \\
      --output_dir ../../masks \\
      --no_visualize

  # Process specific cases with visualization
  python generate_brain_masks.py \\
      --mri_dir ../../nifti_mri \\
      --mra_dir ../../nifti_mra \\
      --output_dir ../../masks \\
      --test_cases <CASE_ID> <CASE_ID> <CASE_ID>
        """
    )

    parser.add_argument('--mri_dir', type=str, required=True,
                       help='Directory containing MRI NIfTI files')
    parser.add_argument('--mra_dir', type=str, required=True,
                       help='Directory containing MRA NIfTI files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for masks')
    parser.add_argument('--test_cases', type=str, nargs='+', default=None,
                       help='Specific case IDs to process (default: all)')
    parser.add_argument('--no_visualize', action='store_true',
                       help='Skip visualization generation')

    args = parser.parse_args()

    # Create output directories
    os.makedirs(os.path.join(args.output_dir, "mri_masks"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "mra_masks"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "visualizations"), exist_ok=True)

    # Setup logger
    logger = setup_logger(args.output_dir)

    logger.info("="*80)
    logger.info("Brain Mask Generation - Reverse Registration Pipeline")
    logger.info("="*80)
    logger.info(f"MRI directory: {args.mri_dir}")
    logger.info(f"MRA directory: {args.mra_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Visualization: {not args.no_visualize}")

    # Get case list
    if args.test_cases:
        case_ids = args.test_cases
        logger.info(f"Processing {len(case_ids)} test cases: {case_ids}")
    else:
        mri_files = sorted(glob.glob(os.path.join(args.mri_dir, "*.nii.gz")))
        case_ids = [os.path.basename(f).replace(".nii.gz", "") for f in mri_files]
        logger.info(f"Processing all {len(case_ids)} cases")

    logger.info("="*80 + "\n")

    # Process cases
    success_count = 0
    failed_cases = []

    for case_id in tqdm(case_ids, desc="Generating masks"):
        try:
            success = process_one_case(
                case_id,
                args.mri_dir,
                args.mra_dir,
                args.output_dir,
                logger,
                visualize=not args.no_visualize
            )
            if success:
                success_count += 1
            else:
                failed_cases.append(case_id)
        except Exception as e:
            logger.error(f"Unexpected error processing {case_id}: {e}")
            failed_cases.append(case_id)

    # Summary
    logger.info("="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Total cases: {len(case_ids)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {len(failed_cases)}")

    if failed_cases:
        logger.warning(f"Failed cases: {', '.join(failed_cases)}")

    logger.info("="*80)
    logger.info("Mask generation complete!")


if __name__ == "__main__":
    main()
