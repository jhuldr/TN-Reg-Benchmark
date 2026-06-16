# ============================================================
# v2: Reader study packet — ONE ROI PER PATIENT
#
# Constraint: Each patient contributes at most 1 ROI to avoid
# correlated bilateral ROIs from the same brain.
#
# If a patient has both ipsi + contra qualifying for the same stratum,
# pick the side with WORSE contrast (more diagnostic for hypothesis).
# If they qualify for different strata, prefer hypothesis-stratum.
# ============================================================
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REG_ROOT  = Path(os.path.expanduser('~/TN_Reg'))
MRI_DIR   = REG_ROOT / 'data/preprocessed_cropped/mri'
MRA_DIR   = REG_ROOT / 'data/preprocessed_cropped/mra'
SYN_DIR   = REG_ROOT / 'outputs/MRIfixed_MRAmoving/ANTs_result_syn_fixedmask/warped'

OUT_DIR   = REG_ROOT / 'eval_result/reader_study'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 404
random.seed(SEED); np.random.seed(SEED)

# ============================================================
# Build df with stratum info
# ============================================================
df_a = pd.read_csv(REG_ROOT / 'eval_result/eval_trackA_all_methods.csv')
df_b = pd.read_csv(REG_ROOT / 'eval_result/eval_trackB_all_methods.csv')
syn_a = df_a[df_a['method'] == 'ANTs_SyN'][['mrn', 'side', 'contrast_A', 'vessel_AUC']].copy()
syn_b = df_b[df_b['method'] == 'ANTs_SyN'][['mrn', 'side', 'gt2pred_mean_mm', 'pred2gt_mean_mm', 'pred_vox']].copy()
df = syn_a.merge(syn_b, on=['mrn', 'side'], how='left').dropna(subset=['contrast_A'])

# Strata
df['stratum'] = ''
m_low  = df['contrast_A'] <= 1.1
m_mid  = (df['contrast_A'] > 1.1) & (df['contrast_A'] <= 1.3)
m_high = df['contrast_A'] > 1.3
m_miss = df['gt2pred_mean_mm'].notna() & (df['gt2pred_mean_mm'] > 2.0)
m_hit  = df['gt2pred_mean_mm'].notna() & (df['gt2pred_mean_mm'] <= 2.0)
df.loc[m_low,                              'stratum'] = 'low_contrast'
df.loc[m_mid & m_miss,                     'stratum'] = 'mid_vfm_miss'
df.loc[m_high & m_hit,                     'stratum'] = 'high_control'
df.loc[m_mid & m_hit,                      'stratum'] = 'mid_high_hit_harder'

df = df[df['stratum'] != ''].copy()  # drop unstratified

# Stratum priority for tie-breaking when same patient has both sides
# in different strata: prefer hypothesis strata first
stratum_priority = {
    'low_contrast':         1,  # highest priority (main hypothesis)
    'mid_vfm_miss':         2,
    'mid_high_hit_harder':  3,
    'high_control':         4,  # lowest priority
}
df['priority'] = df['stratum'].map(stratum_priority)

# ============================================================
# ONE ROI PER PATIENT: pick highest-priority side per patient
# Tie-break by lower contrast (more diagnostic)
# ============================================================
df_sorted = df.sort_values(['mrn', 'priority', 'contrast_A'])  # ascending contrast on tie
df_one_per_patient = df_sorted.drop_duplicates(subset='mrn', keep='first').copy()

print(f'Total qualifying patient-sides: {len(df)}')
print(f'After one-per-patient deduplication: {len(df_one_per_patient)}')
print(f'\nAvailable pool by stratum (one per patient):')
print(df_one_per_patient['stratum'].value_counts())
print()

# ============================================================
# Stratified sample with patient-level uniqueness preserved
# ============================================================
targets = {
    'low_contrast':         50,
    'mid_vfm_miss':         20,
    'high_control':         20,
    'mid_high_hit_harder':  10,
}

sampled = []
for stratum, n in targets.items():
    pool = df_one_per_patient[df_one_per_patient['stratum'] == stratum]
    take = min(n, len(pool))
    if take < n:
        print(f'[WARN] stratum "{stratum}": want {n}, only {len(pool)} patients available')
    sampled.append(pool.sample(n=take, random_state=SEED))

manifest = pd.concat(sampled).reset_index(drop=True)

# ============================================================
# Verify uniqueness
# ============================================================
n_unique_patients = manifest['mrn'].nunique()
n_total_cases     = len(manifest)
assert n_unique_patients == n_total_cases, f'BUG: {n_total_cases} cases but only {n_unique_patients} unique patients!'
print(f'\nVerified: {n_total_cases} cases from {n_unique_patients} unique patients (1 ROI per patient)')

# Randomize review IDs
shuffled_idx = list(range(len(manifest)))
random.Random(SEED).shuffle(shuffled_idx)
manifest['review_id'] = [f'R{i+1:03d}' for i in shuffled_idx]
manifest = manifest.sort_values('review_id').reset_index(drop=True)

# Save
manifest_path = OUT_DIR / 'case_manifest_PRIVATE.csv'
manifest.to_csv(manifest_path, index=False)
print(f'\nManifest: {manifest_path}')
print(manifest['stratum'].value_counts())

# Reading sheet
reading = pd.DataFrame({
    'review_id':    manifest['review_id'].values,
    'evaluable':    '',
    'vessel_type':  '',
    'confidence':   '',
    'notes':        '',
})
reading_path = OUT_DIR / 'reading_sheet.csv'
reading.to_csv(reading_path, index=False)
print(f'Reading sheet: {reading_path}')


# ============================================================
# Render PDF (same as v1)
# ============================================================
def load_nii(path):
    return nib.load(str(path)).get_fdata() if Path(path).exists() else None

def get_3view(vol):
    sx, sy, sz = vol.shape[:3]
    return vol[:, :, sz // 2], vol[sx // 2, :, :], vol[:, sy // 2, :]

def show(ax, slc, title='', vrange=None):
    if slc is None:
        ax.axis('off'); ax.set_title(title + ' (missing)', fontsize=8); return
    vmin, vmax = vrange if vrange else (None, None)
    ax.imshow(slc.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])

def show_overlay(ax, mri_slc, mra_slc, title=''):
    if mri_slc is None or mra_slc is None:
        ax.axis('off'); ax.set_title(title + ' (missing)', fontsize=8); return
    if (mri_slc > 0).any():
        ax.imshow(mri_slc.T, cmap='gray', origin='lower',
                  vmin=np.percentile(mri_slc[mri_slc > 0], 1),
                  vmax=np.percentile(mri_slc[mri_slc > 0], 99))
    if (mra_slc > 0).any():
        p50 = np.percentile(mra_slc[mra_slc > 0], 50)
        p99 = np.percentile(mra_slc[mra_slc > 0], 99)
        mra_norm = np.clip((mra_slc - p50) / max(1e-6, p99 - p50), 0, 1)
        red = np.zeros((*mra_slc.shape, 4))
        red[..., 0] = 1.0; red[..., 3] = mra_norm * 0.5
        ax.imshow(np.transpose(red, (1, 0, 2)), origin='lower')
    ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])


pdf_path = OUT_DIR / 'reader_packet.pdf'
n_ok = 0

with PdfPages(pdf_path) as pdf:
    # Cover
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.7, 'TN MRI-MRA Reader Study', ha='center', fontsize=20, fontweight='bold')
    fig.text(0.5, 0.62, 'Blinded Expert Review', ha='center', fontsize=14)
    fig.text(0.1, 0.45,
             "Instructions:\n\n"
             f"\u2022 {len(manifest)} cases (1 ROI per patient), randomized order\n"
             "\u2022 For each case, please record:\n"
             "  1. Is this case EVALUABLE for offender vessel type?\n"
             "     - yes\n"
             "     - no, registration / anatomic mismatch\n"
             "     - no, inadequate MRA quality\n"
             "     - no, insufficient local anatomy / uncertain\n\n"
             "  2. If evaluable, what is the offender vessel type?\n"
             "     - likely arterial\n"
             "     - suspected venous\n"
             "     - mixed artery + vein / multiple vessels\n"
             "     - uncertain vessel type\n\n"
             "  3. Confidence: low / medium / high\n\n"
             "  4. Free-text notes (optional)\n\n"
             "Registered MRA shown is the ANTs SyN deformable output.",
             fontsize=10, family='monospace')
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
    
    for _, row in manifest.iterrows():
        rid = row['review_id']
        mrn_str = str(int(row['mrn'])).zfill(8)
        side = row['side']
        
        mri = load_nii(MRI_DIR / f'{mrn_str}.nii.gz')
        mra = load_nii(MRA_DIR / f'{mrn_str}.nii.gz')
        syn = load_nii(SYN_DIR / f'{mrn_str}_reg_Warped.nii.gz')
        
        if mri is None or mra is None:
            print(f'[skip] {rid}'); continue
        
        mri3, mra3 = get_3view(mri), get_3view(mra)
        syn3 = get_3view(syn) if syn is not None else (None, None, None)
        
        mri_v = (np.percentile(mri[mri > 0], 1), np.percentile(mri[mri > 0], 99)) if (mri > 0).any() else None
        mra_v = (np.percentile(mra[mra > 0], 1), np.percentile(mra[mra > 0], 99)) if (mra > 0).any() else None
        syn_v = (np.percentile(syn[syn > 0], 1), np.percentile(syn[syn > 0], 99)) if (syn is not None and (syn > 0).any()) else None
        
        fig, axes = plt.subplots(4, 3, figsize=(10, 13))
        fig.suptitle(f'{rid}    Side: {side}', fontsize=14, fontweight='bold', y=0.99)
        for i, view in enumerate(['Axial', 'Sagittal', 'Coronal']):
            show(axes[0, i], mri3[i], f'Structural MRI ({view})', vrange=mri_v)
            show(axes[1, i], mra3[i], f'Original MRA ({view})',   vrange=mra_v)
            show(axes[2, i], syn3[i], f'Registered MRA ({view})', vrange=syn_v)
            show_overlay(axes[3, i], mri3[i], syn3[i] if syn is not None else None,
                         f'MRI + Registered MRA overlay ({view})')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        n_ok += 1
        if n_ok % 20 == 0:
            print(f'  ... rendered {n_ok}/{len(manifest)}')

print(f'\nRendered: {n_ok}')
print(f'PDF: {pdf_path}')
print(f'\nNext: re-run organize_blindtest.py / organize_blindtest_realcopy.py to refresh blindtest folders.')
