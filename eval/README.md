# ROI-centered evaluation (`eval/`)

This package reproduces the benchmark's metrics, tables, and figures from
registration outputs. Everything is driven by the `REG_ROOT` environment
variable; no paths or patient identifiers are hard-coded.

## Two evaluation tracks

| Track | What | Module | Output columns |
|-------|------|--------|----------------|
| **A** — image-based | Vessel AUC, ROI NMI, local contrast inside the 48³ ROI | `metrics/track_a_intensity.py` | `vessel_AUC, roi_NMI, contrast_A, …` |
| **B** — vessel-proximity | GT→Pred / Pred→GT / symmetric surface distances, predicted volume | `metrics/track_b_geometry.py` | `gt2pred_mean_mm, pred2gt_mean_mm, pred_vox, …` |

Metric definitions follow the manuscript (Eqs. 1, 3–7).

## Expected data layout

```
$REG_ROOT/
├── data/preprocessed_cropped/{mri,mra}/<CASE_ID>.nii.gz   # cropped, RAS
├── data/raw/mri/<CASE_ID>/.../                            # DICOM (affine only)
├── 2020-22_MRIs_Cropped_and_Segmentations/
│   ├── 202022_centroids_{ipsilateral,contralateral}.csv
│   ├── 202022_{Ipsilateral,Contralateral}_Target/<CASE_ID>_mask.tiff
│   └── 202022_{Ipsilateral,Contralateral}_Input/<CASE_ID>_cropped.tiff
├── outputs/MRIfixed_MRAmoving/<method>/warped/<CASE_ID>_reg_Warped.nii.gz
└── eval_result/                                           # written here
$VFM_ROOT/ (or ~/vesselfm_results)
└── vesselfm_<method>_fixed/<CASE_ID>_vessel_mask.nii.gz   # VesselFM predictions
```

Method → warped/VesselFM subdirectory names are listed in `config.py`.

## Run order

```bash
export REG_ROOT=/path/to/TN_Reg
export VFM_ROOT=/path/to/vesselfm_results      # optional

python -m eval.compute_metrics                 # -> eval_trackA/B + skip-log CSVs
python -m eval.make_tables                     # -> Table III (CSV + LaTeX)
python -m eval.figures.overview_figures        # -> Fig. 2 / Fig. 3 + supp figs
python -m eval.figures.coupling_figures        # -> Fig. 5
python -m eval.figures.stratified_figures      # -> Fig. 6 / Fig. 7
python -m eval.stratify                        # -> bootstrap CIs (printed)
```

## Per-case figures (de-identified)

`figures/case_figures.py` and `figures/figure1_pipeline.py` render patient-anatomy
panels (manuscript Fig. 1 / Fig. 4 / Fig. 8). They are **parameterized**: the figure
title and output filename come from `--label` / `--out-name`, never the MRN. This
repository ships **no** generated patient images — run these locally on your own data.

```bash
python -m eval.figures.case_figures \
    --case-id <CASE_ID> --side contra \
    --label "Representative Case B" --out-name representative_case_B.png

python -m eval.figures.figure1_pipeline --case-id <CASE_ID> --side contra
```

## Figure ↔ filename map

The source filenames predate the final figure numbering (Fig. 1 is the schematic):

| Manuscript | File |
|-----------|------|
| Fig. 2 | `fig1_overall_intensity.png` |
| Fig. 3 | `fig2_overall_geometry.png` |
| Fig. 5 | `fig5_predvox_{gt2pred,pred2gt,symmetric}.png` |
| Fig. 6 | `fig6_contrast_tier_distribution.png`, `fig6_vesselAUC_by_tier.png` |
| Fig. 7 | `fig7_fov_paired_{connection,delta}.png` |
| Table III | `table_overall_{intensity,geometry}.{csv,tex}` |
