# TN-Reg-Benchmark
# Code author: Xupeng Zhang (xzhan419@jh.edu)
# Johns Hopkins University
# See README for the full author list and citation.

# ============================================================
# Compute r_min FOV ratio for each case from raw NIfTI headers.
# MRN is zero-padded to 8 digits to match file naming.
# ============================================================

import os
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib

REG_ROOT = Path(os.path.expanduser('~/TN_Reg'))

mri_dir = REG_ROOT / 'data/preprocessed_cropped/mri'
mra_dir = REG_ROOT / 'data/preprocessed_cropped/mra'
print(f'Using MRI: {mri_dir}')
print(f'Using MRA: {mra_dir}')

df_b = pd.read_csv(REG_ROOT / 'eval_result/eval_trackB_all_methods.csv')
mrns = sorted(df_b['mrn'].unique())
print(f'Processing {len(mrns)} unique MRNs from eval_trackB\n')

MRN_WIDTH = 8  # files are like <CASE_ID>.nii.gz

rows = []
n_fail = 0
for mrn in mrns:
    mrn_str = str(int(mrn)).zfill(MRN_WIDTH)
    mri_path = mri_dir / f'{mrn_str}.nii.gz'
    mra_path = mra_dir / f'{mrn_str}.nii.gz'
    
    if not mri_path.exists() or not mra_path.exists():
        if n_fail < 3:
            print(f'  [missing] mrn={mrn} → {mrn_str}.nii.gz')
        n_fail += 1
        continue
    
    mri = nib.load(str(mri_path))
    mra = nib.load(str(mra_path))
    
    mri_shape   = np.array(mri.shape[:3])
    mra_shape   = np.array(mra.shape[:3])
    mri_spacing = np.array(mri.header.get_zooms()[:3])
    mra_spacing = np.array(mra.header.get_zooms()[:3])
    
    fov_mri = mri_shape * mri_spacing
    fov_mra = mra_shape * mra_spacing
    ratios  = fov_mri / fov_mra
    r_min   = ratios.min()
    
    rows.append({
        'mrn': mrn,
        'r_x': ratios[0], 'r_y': ratios[1], 'r_z': ratios[2],
        'r_min': r_min,
    })

if n_fail > 0:
    print(f'\n  [WARN] {n_fail}/{len(mrns)} cases missing')

df_fov = pd.DataFrame(rows)
print(f'\nComputed r_min for {len(df_fov)} cases')

if len(df_fov) == 0:
    print('\n[ERROR] No cases were processed. Check MRN format and file paths.')
    raise SystemExit(1)

threshold = df_fov['r_min'].quantile(0.20)
print(f'\n20th percentile of r_min: {threshold:.3f}')
print(f'Paper used threshold: 0.76')

df_fov['fov_group'] = np.where(df_fov['r_min'] <= 0.76, 'Bad', 'Good')
counts = df_fov['fov_group'].value_counts()
print(f'\nFOV groups (using paper threshold 0.76):')
for g, n in counts.items():
    print(f'  {g}: {n} patients')

print(f'\nIf both sides included (×2):')
print(f'  Good-FOV ROIs: ~{counts.get("Good", 0) * 2}')
print(f'  Bad-FOV ROIs:  ~{counts.get("Bad", 0) * 2}')

print(f'\nPaper III-G reports: 177 Good + 46 Bad ROIs = 223 paired ROIs')

out_path = REG_ROOT / 'eval_result/fov_metadata.csv'
df_fov.to_csv(out_path, index=False)
print(f'\nSaved: {out_path}')