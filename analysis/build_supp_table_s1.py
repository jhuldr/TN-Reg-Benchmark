# TN-Reg-Benchmark
# Code author: Xupeng Zhang (xzhan419@jh.edu)
# Johns Hopkins University
# See README for the full author list and citation.

# ============================================================
# Build Supplementary Table S1 — DICOMDIR-aware version.
# DICOMs here look like /raw/mri/<MRN>/<HASH1>/<HASH2>/I1000000
# (no extension). We skip VERSION/LOCKFILE/DICOMDIR metadata files
# and only read I* files.
# ============================================================
import os
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import pydicom

REG_ROOT = Path(os.path.expanduser('~/TN_Reg'))
DICOM_MRI = REG_ROOT / 'data/raw/mri'
DICOM_MRA = REG_ROOT / 'data/raw/mra'
NIFTI_MRI = REG_ROOT / 'data/preprocessed_cropped/mri'
NIFTI_MRA = REG_ROOT / 'data/preprocessed_cropped/mra'

df_a = pd.read_csv(REG_ROOT / 'eval_result/eval_trackA_all_methods.csv')
mrns = sorted(df_a['mrn'].unique())
print(f'Cohort: {len(mrns)} MRNs')

SKIP_NAMES = {'VERSION', 'LOCKFILE', 'DICOMDIR', 'README', 'README.txt'}


def find_first_real_dicom(mrn_dir):
    """Walk the tree, skip metadata files, return first DICOM with real header."""
    if not mrn_dir.exists():
        return None
    for f in mrn_dir.rglob('*'):
        if not f.is_file():
            continue
        if f.name.upper() in SKIP_NAMES:
            continue
        # Try to read; require it has a Manufacturer or other clinical attr
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            # Sanity: must look like a real image, not a metadata wrapper
            if hasattr(ds, 'Manufacturer') or hasattr(ds, 'SeriesDescription') or hasattr(ds, 'PixelSpacing'):
                return ds
        except Exception:
            continue
    return None


def extract_dicom_meta(ds):
    if ds is None:
        return {}
    def get(attr, default=None):
        try:
            v = getattr(ds, attr)
            return v if v not in (None, '') else default
        except AttributeError:
            return default
    out = {
        'manufacturer':   str(get('Manufacturer', '') or '').strip(),
        'model':          str(get('ManufacturerModelName', '') or '').strip(),
        'field_T':        None,
        'sequence':       str(get('SeriesDescription', '') or '').strip(),
        'protocol':       str(get('ProtocolName', '') or '').strip(),
        'TR_ms':          None,
        'TE_ms':          None,
        'slice_thick_mm': None,
        'study_date':     str(get('StudyDate', '') or '').strip(),
    }
    for src, dst in [('MagneticFieldStrength', 'field_T'),
                     ('RepetitionTime',         'TR_ms'),
                     ('EchoTime',               'TE_ms'),
                     ('SliceThickness',         'slice_thick_mm')]:
        try:
            out[dst] = float(getattr(ds, src))
        except Exception:
            pass
    return out


def get_nifti_meta(nifti_dir, mrn_str):
    f = nifti_dir / f'{mrn_str}.nii.gz'
    if not f.exists():
        return None
    nii = nib.load(str(f))
    return {
        'shape':   tuple(int(s) for s in nii.shape[:3]),
        'spacing': tuple(float(s) for s in nii.header.get_zooms()[:3]),
    }


def find_mrn_dir(parent, mrn):
    candidates = [
        parent / str(int(mrn)).zfill(8),
        parent / str(int(mrn)),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


# ============================================================
# Collect
# ============================================================
mri_rows, mra_rows = [], []
n_no_mri, n_no_mra = 0, 0

for i, mrn in enumerate(mrns):
    mrn_str = str(int(mrn)).zfill(8)
    
    mri_dir = find_mrn_dir(DICOM_MRI, mrn)
    ds_mri = find_first_real_dicom(mri_dir) if mri_dir else None
    if ds_mri is None: n_no_mri += 1
    nii_mri = get_nifti_meta(NIFTI_MRI, mrn_str) or {}
    mri_rows.append({'mrn': mrn, **extract_dicom_meta(ds_mri), **nii_mri})
    
    mra_dir = find_mrn_dir(DICOM_MRA, mrn)
    ds_mra = find_first_real_dicom(mra_dir) if mra_dir else None
    if ds_mra is None: n_no_mra += 1
    nii_mra = get_nifti_meta(NIFTI_MRA, mrn_str) or {}
    mra_rows.append({'mrn': mrn, **extract_dicom_meta(ds_mra), **nii_mra})
    
    if (i + 1) % 30 == 0:
        print(f'  ... processed {i+1}/{len(mrns)}')

print(f'\nDICOM read failures: MRI={n_no_mri}/{len(mrns)}, MRA={n_no_mra}/{len(mrns)}')

df_mri = pd.DataFrame(mri_rows)
df_mra = pd.DataFrame(mra_rows)


def summarize(df, label):
    n = len(df)
    print('\n' + '=' * 70)
    print(f'{label} (N={n})')
    print('=' * 70)
    
    if 'manufacturer' in df.columns:
        mans = df['manufacturer'].fillna('').replace('', 'Unknown').str.strip()
        # Normalize common spellings
        mans = mans.str.replace(r'(?i)siemens.*', 'Siemens', regex=True)
        mans = mans.str.replace(r'(?i)ge\s*medical.*|ge\s*health.*|^ge$', 'GE Healthcare', regex=True)
        mans = mans.str.replace(r'(?i)philips.*', 'Philips', regex=True)
        print('\nManufacturer:')
        for m, c in mans.value_counts().items():
            print(f'  {m:30s}  {c:3d}  ({100*c/n:.1f}%)')
    
    if 'model' in df.columns:
        models = df['model'].fillna('').replace('', 'Unknown').str.strip()
        print('\nScanner model (top 8):')
        for m, c in models.value_counts().head(8).items():
            print(f'  {m:30s}  {c:3d}  ({100*c/n:.1f}%)')
    
    if 'field_T' in df.columns:
        ft = pd.to_numeric(df['field_T'], errors='coerce').dropna()
        if len(ft):
            print(f'\nField strength (N={len(ft)} with info):')
            for thresh, label_t in [(1.5, '1.5T'), (3.0, '3T'), (7.0, '7T')]:
                c = sum(np.isclose(ft, thresh, atol=0.3))
                if c > 0:
                    print(f'  {label_t}: {c}  ({100*c/len(ft):.1f}%)')
    
    if 'sequence' in df.columns:
        seqs = df['sequence'].fillna('').replace('', 'Unknown').str.strip()
        print('\nSeries description (top 10):')
        for s, c in seqs.value_counts().head(10).items():
            print(f'  {s[:50]:52s}  {c:3d}')
    
    for col, t in [('TR_ms', 'TR (ms)'), ('TE_ms', 'TE (ms)'),
                   ('slice_thick_mm', 'Slice thickness from DICOM (mm)')]:
        v = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(v):
            print(f'\n{t}: median {v.median():.2f}, IQR [{v.quantile(0.25):.2f}, {v.quantile(0.75):.2f}], range [{v.min():.2f}, {v.max():.2f}], N={len(v)}')
    
    if 'spacing' in df.columns:
        sps = [s for s in df['spacing'] if s and len(s) == 3]
        if sps:
            sp = np.array(sps)
            inplane = sp[:, :2].mean(axis=1)
            slc = sp[:, 2]
            print(f'\nIn-plane spacing (mm) [from NIfTI]: median {np.median(inplane):.3f}, IQR [{np.quantile(inplane, 0.25):.3f}, {np.quantile(inplane, 0.75):.3f}], range [{inplane.min():.3f}, {inplane.max():.3f}]')
            print(f'Slice spacing (mm)    [from NIfTI]: median {np.median(slc):.3f}, IQR [{np.quantile(slc, 0.25):.3f}, {np.quantile(slc, 0.75):.3f}], range [{slc.min():.3f}, {slc.max():.3f}]')
    
    if 'shape' in df.columns:
        shps = [s for s in df['shape'] if s and len(s) == 3]
        if shps:
            sh = np.array(shps)
            print(f'Matrix shape (median): {tuple(int(x) for x in np.median(sh, axis=0))}')
    
    if 'study_date' in df.columns:
        dates = df['study_date'].dropna()
        dates = dates[dates.str.len() == 8]
        if len(dates):
            print(f'\nStudy date range: {dates.min()} to {dates.max()}  (N={len(dates)})')


summarize(df_mri, 'MRI (structural / CISS)')
summarize(df_mra, 'MRA (TOF)')

out_mri = REG_ROOT / 'eval_result/supp_table_s1_mri.csv'
out_mra = REG_ROOT / 'eval_result/supp_table_s1_mra.csv'
df_mri.to_csv(out_mri, index=False)
df_mra.to_csv(out_mra, index=False)
print(f'\nSaved: {out_mri}')
print(f'Saved: {out_mra}')