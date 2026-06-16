#!/usr/bin/env python3
"""
Lightweight Preprocessing: RAS Orientation
Only does essential preprocessing without heavy resampling.

Xupeng Zhang
Johns Hopkins University
"""

import os
import sys
import glob
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm


def setup_logger(output_dir):
    """Setup logging configuration"""
    log_file = os.path.join(output_dir, f'preprocess_ras_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def reorient_to_ras(nifti_img):
    """
    Reorient NIfTI image to RAS orientation

    Args:
        nifti_img: nibabel NIfTI image object

    Returns:
        reoriented_img: NIfTI image in RAS orientation
    """
    # Get current orientation
    current_orient = nib.aff2axcodes(nifti_img.affine)

    # Target orientation: RAS (Right-Anterior-Superior)
    target_orient = ('R', 'A', 'S')

    # If already RAS, return as-is
    if current_orient == target_orient:
        return nifti_img, False  # False = no change needed

    # Reorient to RAS
    reoriented_img = nib.as_closest_canonical(nifti_img)

    return reoriented_img, True  # True = reoriented


def robust_intensity_normalization(img_data, lower_percentile=1, upper_percentile=99):
    """
    Robust intensity normalization using percentile clipping

    Args:
        img_data: numpy array
        lower_percentile: lower percentile for clipping (default: 1)
        upper_percentile: upper percentile for clipping (default: 99)

    Returns:
        normalized_data: normalized numpy array in range [0, 1]
    """
    # Compute percentiles
    p_low = np.percentile(img_data, lower_percentile)
    p_high = np.percentile(img_data, upper_percentile)

    # Clip values
    clipped = np.clip(img_data, p_low, p_high)

    # Normalize to [0, 1]
    if p_high - p_low > 0:
        normalized = (clipped - p_low) / (p_high - p_low)
    else:
        normalized = np.zeros_like(clipped)

    return normalized.astype(np.float32)


def process_one_file(
    input_path,
    output_path,
    logger,
    modality="Image",
    normalize=False,
    lower_percentile=1,
    upper_percentile=99,
):
    """
    Process one NIfTI file: RAS reorientation (+ optional intensity normalization)

    Args:
        input_path: path to input NIfTI file
        output_path: path to output NIfTI file
        logger: logger object
        modality: "MRI" or "MRA" for logging

    Returns:
        success: True if successful, False otherwise
        stats: dictionary with processing statistics
    """
    try:
        # Load NIfTI
        img = nib.load(input_path)
        original_orient = nib.aff2axcodes(img.affine)
        original_shape = img.shape
        original_spacing = img.header.get_zooms()[:3]

        # Step 1: Reorient to RAS
        img_ras, was_reoriented = reorient_to_ras(img)

        # Step 2: Optional intensity normalization
        img_data = img_ras.get_fdata()
        if normalize:
            img_out = robust_intensity_normalization(
                img_data,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
            )
        else:
            img_out = img_data.astype(np.float32)

        # Step 3: Save
        output_img = nib.Nifti1Image(img_out, img_ras.affine, img_ras.header)
        output_img.header.set_data_dtype(np.float32)
        nib.save(output_img, output_path)

        # Statistics
        stats = {
            'original_orient': ''.join(original_orient),
            'final_orient': 'RAS',
            'reoriented': was_reoriented,
            'normalize_applied': normalize,
            'shape': original_shape,
            'spacing': original_spacing,
            'intensity_range': (img_data.min(), img_data.max()),
            'output_range': (img_out.min(), img_out.max())
        }

        return True, stats

    except Exception as e:
        logger.error(f"Error processing {os.path.basename(input_path)}: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    parser = argparse.ArgumentParser(
        description='Lightweight preprocessing: RAS orientation with optional intensity normalization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Process all MRI and MRA files
  python %(prog)s \\
      --mri_dir /path/to/data_mri \\
      --mra_dir /path/to/data_mra \\
      --output_dir /path/to/preprocessed_ras

  # RAS + normalization (only when explicitly needed)
  python %(prog)s \\
      --mri_dir /path/to/data_mri \\
      --mra_dir /path/to/data_mra \\
      --output_dir /path/to/preprocessed_ras_norm \\
      --normalize

  # Process only specific cases
  python %(prog)s \\
      --mri_dir /path/to/data_mri \\
      --mra_dir /path/to/data_mra \\
      --output_dir /path/to/preprocessed_ras \\
      --test_cases <CASE_ID> <CASE_ID>

  # Test on first 10 cases
  python %(prog)s \\
      --mri_dir /path/to/data_mri \\
      --mra_dir /path/to/data_mra \\
      --output_dir /path/to/preprocessed_ras \\
      --num_cases 10
        """
    )

    parser.add_argument('--mri_dir', type=str, required=True,
                        help='Directory containing original MRI files')
    parser.add_argument('--mra_dir', type=str, required=True,
                        help='Directory containing original MRA files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for preprocessed files')
    parser.add_argument('--test_cases', type=str, nargs='+', default=None,
                        help='Specific case IDs to process')
    parser.add_argument('--num_cases', type=int, default=None,
                        help='Number of cases to process (for testing)')
    parser.add_argument('--pattern', type=str, default='*.nii.gz',
                        help='File pattern to match (default: *.nii.gz)')
    parser.add_argument('--normalize', action='store_true',
                        help='Enable robust intensity normalization to [0,1] after RAS reorientation')
    parser.add_argument('--lower_percentile', type=float, default=1.0,
                        help='Lower percentile for normalization clipping (default: 1)')
    parser.add_argument('--upper_percentile', type=float, default=99.0,
                        help='Upper percentile for normalization clipping (default: 99)')

    args = parser.parse_args()

    # Create output directories
    output_mri_dir = os.path.join(args.output_dir, 'mri')
    output_mra_dir = os.path.join(args.output_dir, 'mra')
    os.makedirs(output_mri_dir, exist_ok=True)
    os.makedirs(output_mra_dir, exist_ok=True)

    # Setup logger
    logger = setup_logger(args.output_dir)

    logger.info("="*80)
    logger.info("Lightweight Preprocessing: RAS Orientation (+ optional normalization)")
    logger.info("="*80)
    logger.info(f"MRI input directory: {args.mri_dir}")
    logger.info(f"MRA input directory: {args.mra_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Normalization enabled: {args.normalize}")
    if args.normalize:
        logger.info(
            f"Normalization percentiles: [{args.lower_percentile}, {args.upper_percentile}]"
        )
    logger.info("="*80)

    # Get list of files
    if args.test_cases:
        # Use specified case IDs
        all_cases = args.test_cases
        logger.info(f"Processing {len(all_cases)} specified cases")
    else:
        # Find all MRI files
        mri_files = sorted(glob.glob(os.path.join(args.mri_dir, args.pattern)))

        # Extract case IDs (assuming filename format: CASEID.nii.gz or CASEID_*.nii.gz)
        all_cases = []
        for f in mri_files:
            basename = os.path.basename(f).replace('.nii.gz', '').replace('.nii', '')
            # Take first part if underscore exists (e.g., "<CASE_ID>_T2" -> "<CASE_ID>")
            case_id = basename.split('_')[0]
            all_cases.append(case_id)

        # Remove duplicates and sort
        all_cases = sorted(list(set(all_cases)))

        if args.num_cases:
            all_cases = all_cases[:args.num_cases]
            logger.info(f"Processing first {args.num_cases} cases")
        else:
            logger.info(f"Found {len(all_cases)} cases to process")

    # Process all cases
    successful_mri = 0
    successful_mra = 0
    failed_cases = []
    reoriented_count = 0

    logger.info("\nStarting preprocessing...\n")

    for i, case_id in enumerate(tqdm(all_cases, desc="Processing cases")):
        logger.info(f"[{i+1}/{len(all_cases)}] Processing case: {case_id}")

        # Find MRI and MRA files for this case
        mri_files = glob.glob(os.path.join(args.mri_dir, f"{case_id}*.nii.gz"))
        mra_files = glob.glob(os.path.join(args.mra_dir, f"{case_id}*.nii.gz"))

        if not mri_files:
            logger.warning(f"  No MRI file found for {case_id}")
            failed_cases.append(case_id)
            continue

        if not mra_files:
            logger.warning(f"  No MRA file found for {case_id}")
            failed_cases.append(case_id)
            continue

        # Take first match if multiple files found
        mri_file = mri_files[0]
        mra_file = mra_files[0]

        # Output paths
        output_mri_path = os.path.join(output_mri_dir, f"{case_id}.nii.gz")
        output_mra_path = os.path.join(output_mra_dir, f"{case_id}.nii.gz")

        # Process MRI
        success_mri, stats_mri = process_one_file(
            mri_file,
            output_mri_path,
            logger,
            "MRI",
            normalize=args.normalize,
            lower_percentile=args.lower_percentile,
            upper_percentile=args.upper_percentile,
        )
        if success_mri:
            successful_mri += 1
            if stats_mri['reoriented']:
                reoriented_count += 1
            logger.info(f"  MRI: {stats_mri['original_orient']} -> {stats_mri['final_orient']} | "
                       f"Shape: {stats_mri['shape']} | Spacing: {stats_mri['spacing']}")

        # Process MRA
        success_mra, stats_mra = process_one_file(
            mra_file,
            output_mra_path,
            logger,
            "MRA",
            normalize=args.normalize,
            lower_percentile=args.lower_percentile,
            upper_percentile=args.upper_percentile,
        )
        if success_mra:
            successful_mra += 1
            if stats_mra['reoriented']:
                reoriented_count += 1
            logger.info(f"  MRA: {stats_mra['original_orient']} -> {stats_mra['final_orient']} | "
                       f"Shape: {stats_mra['shape']} | Spacing: {stats_mra['spacing']}")

        if success_mri and success_mra:
            logger.info(f"  SUCCESS: {case_id}")
        else:
            failed_cases.append(case_id)
            logger.warning(f"  PARTIAL SUCCESS or FAILED: {case_id}")

    # Summary
    logger.info("\n" + "="*80)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("="*80)
    logger.info(f"Total cases processed: {len(all_cases)}")
    logger.info(f"Successful MRI files: {successful_mri}")
    logger.info(f"Successful MRA files: {successful_mra}")
    logger.info(f"Successfully paired: {min(successful_mri, successful_mra)}")
    logger.info(f"Failed cases: {len(failed_cases)}")
    logger.info(f"Images reoriented to RAS: {reoriented_count}")

    if failed_cases:
        logger.info(f"\nFailed case IDs: {', '.join(failed_cases)}")

    logger.info("\n" + "="*80)
    logger.info("Output structure:")
    logger.info(f"  {args.output_dir}/")
    logger.info(f"  ├── mri/")
    logger.info(f"  │   ├── {all_cases[0] if all_cases else 'CASEID'}.nii.gz")
    logger.info(f"  │   └── ... ({successful_mri} files)")
    logger.info(f"  ├── mra/")
    logger.info(f"  │   ├── {all_cases[0] if all_cases else 'CASEID'}.nii.gz")
    logger.info(f"  │   └── ... ({successful_mra} files)")
    logger.info(f"  └── preprocess_ras_*.log")
    logger.info("="*80)
    logger.info("Preprocessing complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
