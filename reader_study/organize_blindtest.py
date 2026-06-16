# ============================================================
# v2: per-side label (loads only ipsi or contra label per case)
#
# Source layout:
#   nii_Label/<MRN>/<MRN>_ipsi_label.nii.gz
#   nii_Label/<MRN>/<MRN>_contra_label.nii.gz
#
# Output structure (per case):
#   blindtest_copy/Rxxx/
#     mri.nii.gz
#     ants_syn_warped.nii.gz
#     synthmorph_warped.nii.gz
#     label.nii.gz       <- side-specific (only the reviewed side)
#     info.txt
# ============================================================
import os
import shutil
from pathlib import Path
import pandas as pd

REG_ROOT = Path(os.path.expanduser('~/TN_Reg'))

MRI_DIR        = REG_ROOT / 'data/preprocessed_cropped/mri'
ANTS_SYN_DIR   = REG_ROOT / 'outputs/MRIfixed_MRAmoving/ANTs_result_syn_fixedmask/warped'
SYNTHMORPH_DIR = REG_ROOT / 'outputs/MRIfixed_MRAmoving/SynthMorph/warped'
LABEL_BASE     = REG_ROOT / 'nii_Label'

OUT_BASE = REG_ROOT / 'blindtest_copy'
OUT_BASE.mkdir(parents=True, exist_ok=True)

manifest = pd.read_csv(REG_ROOT / 'eval_result/reader_study/case_manifest_PRIVATE.csv')
print(f'Loaded {len(manifest)} cases')

n_ok      = 0
n_partial = 0
missing_log = []

for i, (_, row) in enumerate(manifest.iterrows()):
    rid     = row['review_id']
    mrn_str = str(int(row['mrn'])).zfill(8)
    side    = row['side']  # 'ipsi' or 'contra'
    
    # Side-specific label filename
    label_filename = f'{mrn_str}_{side}_label.nii.gz'
    
    case_dir = OUT_BASE / rid
    case_dir.mkdir(exist_ok=True)
    
    src_map = {
        'mri.nii.gz':                MRI_DIR        / f'{mrn_str}.nii.gz',
        'ants_syn_warped.nii.gz':    ANTS_SYN_DIR   / f'{mrn_str}_reg_Warped.nii.gz',
        'synthmorph_warped.nii.gz':  SYNTHMORPH_DIR / f'{mrn_str}_reg_Warped.nii.gz',
        'label.nii.gz':              LABEL_BASE     / mrn_str / label_filename,
    }
    
    case_missing = []
    for target_name, src_path in src_map.items():
        dst = case_dir / target_name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        if src_path.exists():
            shutil.copy2(src_path, dst)
        else:
            case_missing.append(target_name)
    
    info_text = (
        f"review_id:    {rid}\n"
        f"true_mrn:     {mrn_str}\n"
        f"side:         {side}\n"
        f"label_source: {label_filename}\n"
        f"stratum:      {row['stratum']}\n"
        f"contrast:     {row['contrast_A']:.3f}\n"
        f"vessel_AUC:   {row['vessel_AUC']:.3f}\n"
    )
    if pd.notna(row.get('gt2pred_mean_mm')):
        info_text += f"gt2pred_mm:   {row['gt2pred_mean_mm']:.2f}\n"
    if pd.notna(row.get('pred_vox')):
        info_text += f"pred_vox:     {int(row['pred_vox'])}\n"
    (case_dir / 'info.txt').write_text(info_text)
    
    if case_missing:
        n_partial += 1
        missing_log.append((rid, mrn_str, side, case_missing))
    else:
        n_ok += 1
    
    if (i + 1) % 20 == 0:
        print(f'  ... copied {i + 1}/{len(manifest)}')

print(f'\nFully assembled: {n_ok}/{len(manifest)}')
print(f'Partial:          {n_partial}/{len(manifest)}')

if missing_log:
    from collections import Counter
    missing_files = Counter()
    for rid, mrn, side, miss in missing_log:
        for m in miss:
            missing_files[m] += 1
    print('\n--- Missing files ---')
    for f, c in missing_files.most_common():
        print(f'  {f}: {c} cases')
    
    print('\n--- First 5 cases with issues ---')
    for rid, mrn, side, miss in missing_log[:5]:
        print(f'  {rid} (MRN {mrn}, side={side}): missing {miss}')

# Total size
total_bytes = sum(f.stat().st_size for f in OUT_BASE.rglob('*') if f.is_file())
print(f'\nTotal size: {total_bytes / (1024**3):.2f} GB')
print(f'Output folder: {OUT_BASE}')
