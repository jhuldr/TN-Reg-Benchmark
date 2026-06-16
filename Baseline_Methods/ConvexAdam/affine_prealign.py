#!/usr/bin/env python3
# TN-Reg-Benchmark
# Code author: Xupeng Zhang (xzhan419@jh.edu)
# Johns Hopkins University
# See README for the full author list and citation.

"""
Affine Pre-alignment for MRI-MRA Registration (v2)
This script performs affine registration as initialization before deformable registration
Uses SimpleITK with Mutual Information for cross-modality registration
"""

"""
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
    log_file = os.path.join(output_dir, f'affine_prealign_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def affine_registration_sitk(fixed_img, moving_img, use_masks=False, fixed_mask=None, moving_mask=None):
    """
    Perform affine registration using SimpleITK

    Key features:
    - Mutual Information similarity metric (best for MRI-MRA cross-modality)
    - Multi-resolution pyramid (3 levels: coarse to fine)
    - Center of mass initialization (MOMENTS mode - robust to padding)
    - Gradient descent optimizer

    Args:
        fixed_img: SimpleITK image (MRA - target)
        moving_img: SimpleITK image (MRI - source)
        use_masks: Whether to use masks
        fixed_mask: Optional mask for fixed image
        moving_mask: Optional mask for moving image

    Returns:
        transform: SimpleITK affine transform
        warped_img: Affine-aligned moving image
    """
    # Initialize registration method
    registration = sitk.ImageRegistrationMethod()

    # 1. Similarity Metric: Mutual Information (best for cross-modality)
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
    registration.SetMetricSamplingStrategy(registration.REGULAR)
    registration.SetMetricSamplingPercentage(0.2)  # Sample 20% of voxels for speed

    # 2. Interpolator
    registration.SetInterpolator(sitk.sitkLinear)

    # 3. Optimizer: Gradient Descent with learning rate decay
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=500,  # Increased from 200 to ensure convergence
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )
    registration.SetOptimizerScalesFromPhysicalShift()

    # 4. Multi-resolution strategy (3 levels)
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # 5. Initial transform: Center of mass alignment
    # CRITICAL: Using MOMENTS mode instead of GEOMETRY
    # - GEOMETRY: Aligns geometric centers (bad for images with large padding)
    # - MOMENTS: Aligns intensity centroids (ignores black background)
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_img,
        moving_img,
        sitk.AffineTransform(3),  # 3D affine transform (12 DOF)
        sitk.CenteredTransformInitializerFilter.MOMENTS  # Changed from GEOMETRY
    )
    registration.SetInitialTransform(initial_transform, inPlace=False)

    # 6. Optional: Use masks to focus on brain regions
    if use_masks and fixed_mask is not None and moving_mask is not None:
        registration.SetMetricFixedMask(fixed_mask)
        registration.SetMetricMovingMask(moving_mask)

    # 7. Execute registration
    final_transform = registration.Execute(
        sitk.Cast(fixed_img, sitk.sitkFloat32),
        sitk.Cast(moving_img, sitk.sitkFloat32)
    )

    # 8. Apply transform to moving image
    warped_img = sitk.Resample(
        moving_img,
        fixed_img,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving_img.GetPixelID()
    )

    return final_transform, warped_img


def compute_transform_stats(transform):
    """Extract and summarize affine transform parameters"""
    params = transform.GetParameters()

    # Affine transform has 12 parameters:
    # params[0-8]: 3x3 rotation/scale matrix (flattened row-wise)
    # params[9-11]: 3D translation vector

    matrix = np.array(params[:9]).reshape(3, 3)
    translation = np.array(params[9:12])

    # Decompose matrix into scale and rotation
    # Scale = norm of each column
    scales = np.linalg.norm(matrix, axis=0)

    # Rotation matrix = matrix / scales
    rotation = matrix / scales

    # Extract rotation angles (approximate)
    # This is simplified; full decomposition would use SVD

    stats = {
        'translation_mm': translation,
        'scales': scales,
        'scale_ratio': scales.max() / scales.min(),
        'translation_magnitude_mm': np.linalg.norm(translation),
        'determinant': np.linalg.det(matrix)
    }

    return stats


def process_one_case(case_id, mri_dir, mra_dir, output_dir, logger):
    """Process one MRI-MRA pair"""
    try:
        # Load images
        mri_path = os.path.join(mri_dir, f"{case_id}.nii.gz")
        mra_path = os.path.join(mra_dir, f"{case_id}.nii.gz")

        if not os.path.exists(mri_path) or not os.path.exists(mra_path):
            logger.warning(f"SKIP {case_id}: Missing files")
            return None

        logger.info(f"Processing {case_id}...")

        moving_img = sitk.ReadImage(mri_path, sitk.sitkFloat32)
        fixed_img = sitk.ReadImage(mra_path, sitk.sitkFloat32)

        # Log image properties
        logger.info(f"  MRI: {moving_img.GetSize()}, spacing={moving_img.GetSpacing()}")
        logger.info(f"  MRA: {fixed_img.GetSize()}, spacing={fixed_img.GetSpacing()}")

        # Perform affine registration
        logger.info(f"  Running affine registration...")
        affine_transform, warped_img = affine_registration_sitk(fixed_img, moving_img)

        # Compute statistics
        stats = compute_transform_stats(affine_transform)
        logger.info(f"  Translation: {stats['translation_magnitude_mm']:.2f} mm")
        logger.info(f"  Scale ratio: {stats['scale_ratio']:.3f}")
        logger.info(f"  Determinant: {stats['determinant']:.3f}")

        # Save affine-aligned MRI
        output_mri_path = os.path.join(output_dir, f"{case_id}_affine_aligned.nii.gz")
        sitk.WriteImage(warped_img, output_mri_path)

        # Save transform
        transform_path = os.path.join(output_dir, f"{case_id}_affine_transform.tfm")
        sitk.WriteTransform(affine_transform, transform_path)

        logger.info(f"SUCCESS {case_id}")

        return {
            'case_id': case_id,
            'success': True,
            'stats': stats
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
    # /path/to/TN_Reg
    parser = argparse.ArgumentParser(description='Affine pre-alignment for MRI-MRA registration')
    parser.add_argument('--mri_dir', type=str,
                        default='/path/to/TN_Reg/preprocessed_ras/mri',
                        help='Directory containing preprocessed MRI files')
    parser.add_argument('--mra_dir', type=str,
                        default='/path/to/TN_Reg/preprocessed_ras/mra',
                        help='Directory containing preprocessed MRA files')
    parser.add_argument('--output_dir', type=str,
                        default='/path/to/TN_Reg/affine_aligned_v2',
                        help='Output directory for affine-aligned images')
    parser.add_argument('--test_cases', type=str, nargs='+', default=None,
                        help='Specific case IDs to process (for testing). If not provided, processes all.')
    parser.add_argument('--num_cases', type=int, default=None,
                        help='Number of cases to process (for testing). If not provided, processes all.')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logger
    logger = setup_logger(args.output_dir)

    logger.info("="*60)
    logger.info("Affine Pre-alignment for MRI-MRA Registration (v2)")
    logger.info("="*60)
    logger.info(f"MRI directory: {args.mri_dir}")
    logger.info(f"MRA directory: {args.mra_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("="*60)

    # Get case list
    if args.test_cases:
        all_cases = args.test_cases
        logger.info(f"Processing {len(all_cases)} specified test cases")
    else:
        all_mri_files = sorted(glob.glob(os.path.join(args.mri_dir, "*.nii.gz")))
        all_cases = [os.path.basename(f).replace(".nii.gz", "") for f in all_mri_files]

        if args.num_cases:
            all_cases = all_cases[:args.num_cases]
            logger.info(f"Processing first {len(all_cases)} cases (test mode)")
        else:
            logger.info(f"Processing all {len(all_cases)} cases")

    # Process all cases
    results = []
    for i, case_id in enumerate(all_cases):
        logger.info(f"\n[{i+1}/{len(all_cases)}] Processing {case_id}")
        result = process_one_case(case_id, args.mri_dir, args.mra_dir, args.output_dir, logger)
        if result:
            results.append(result)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    logger.info(f"Total processed: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")

    if successful > 0:
        # Compute average statistics
        all_translations = [r['stats']['translation_magnitude_mm'] for r in results if r['success']]
        all_scales = [r['stats']['scale_ratio'] for r in results if r['success']]

        logger.info(f"\nAverage translation: {np.mean(all_translations):.2f} ± {np.std(all_translations):.2f} mm")
        logger.info(f"Average scale ratio: {np.mean(all_scales):.3f} ± {np.std(all_scales):.3f}")
        logger.info(f"Max translation: {np.max(all_translations):.2f} mm (case: {results[np.argmax(all_translations)]['case_id']})")
        logger.info(f"Max scale ratio: {np.max(all_scales):.3f} (case: {results[np.argmax(all_scales)]['case_id']})")

    logger.info("="*60)
    logger.info("Affine pre-alignment complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
