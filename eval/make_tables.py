"""Generate the overall metric tables (manuscript Table III) as CSV + LaTeX.

Writes table_overall_intensity.{csv,tex} and table_overall_geometry.{csv,tex}
under $REG_ROOT/eval_result/paperimage.
"""
from __future__ import annotations

import argparse

import numpy as np

from . import config
from .figures._common import load_tracks, METHOD_ORDER, ABBR


def _iqr(v):
    return np.median(v), np.percentile(v, 25), np.percentile(v, 75)


def intensity_table(df_a):
    """Return (csv_rows, latex_str) for the intensity table."""
    rows = []
    for m in METHOD_ORDER:
        sub = df_a[df_a["method"] == m]
        am, aq1, aq3 = _iqr(sub["vessel_AUC"].dropna())
        nm, nq1, nq3 = _iqr(sub["roi_NMI"].dropna())
        rows.append({
            "Method": ABBR[m],
            "Vessel AUC": f"{am:.3f} [{aq1:.3f}, {aq3:.3f}]",
            "ROI NMI": f"{nm:.3f} [{nq1:.3f}, {nq3:.3f}]",
        })
    latex = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Overall local image-based metrics across registration methods. "
        r"Values are reported as median [IQR].}",
        r"\label{tab:overall_intensity}",
        r"\begin{tabular}{lcc}", r"\toprule",
        r"\textbf{Method} & \textbf{Vessel AUC} & \textbf{ROI NMI} \\", r"\midrule",
    ]
    for r in rows:
        latex.append(f"{r['Method']:6s} & {r['Vessel AUC']} & {r['ROI NMI']} \\\\")
    latex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return rows, "\n".join(latex)


def geometry_table(df_b):
    """Return (csv_rows, latex_str) for the geometry table."""
    rows = []
    for m in METHOD_ORDER:
        sub = df_b[df_b["method"] == m]
        gm, gq1, gq3 = _iqr(sub["gt2pred_mean_mm"].dropna())
        pm, pq1, pq3 = _iqr(sub["pred2gt_mean_mm"].dropna())
        sm, sq1, sq3 = _iqr(sub["symmetric_mm"].dropna())
        vm, vq1, vq3 = _iqr(sub["pred_vox"].dropna())
        rows.append({
            "Method": ABBR[m], "N_ROIs": len(sub),
            "GT->Pred (mm)": f"{gm:.2f} [{gq1:.2f}, {gq3:.2f}]",
            "Pred->GT (mm)": f"{pm:.2f} [{pq1:.2f}, {pq3:.2f}]",
            "Symmetric (mm)": f"{sm:.2f} [{sq1:.2f}, {sq3:.2f}]",
            "Pred Volume (vox)": f"{int(vm)} [{int(vq1)}, {int(vq3)}]",
        })
    latex = [
        r"\begin{table}[h]", r"\centering",
        r"\caption{Overall geometry metrics across six registration methods (median [IQR]).}",
        r"\label{tab:overall_geometry}",
        r"\begin{tabular}{lrcccc}", r"\toprule",
        r"Method & N ROIs & GT$\to$Pred (mm) & Pred$\to$GT (mm) & Symmetric (mm) & Pred Vol (vox) \\",
        r"\midrule",
    ]
    for r in rows:
        latex.append(
            f"{r['Method']} & {r['N_ROIs']} & {r['GT->Pred (mm)']} & "
            f"{r['Pred->GT (mm)']} & {r['Symmetric (mm)']} & {r['Pred Volume (vox)']} \\\\")
    latex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return rows, "\n".join(latex)


def main():
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=None, help="default: $REG_ROOT/eval_result/paperimage")
    args = ap.parse_args()
    out_dir = config.PAPER_DIR if args.out_dir is None else __import__("pathlib").Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_a, df_b = load_tracks()

    irows, itex = intensity_table(df_a)
    pd.DataFrame(irows).to_csv(out_dir / "table_overall_intensity.csv", index=False)
    (out_dir / "table_overall_intensity.tex").write_text(itex)

    grows, gtex = geometry_table(df_b)
    pd.DataFrame(grows).to_csv(out_dir / "table_overall_geometry.csv", index=False)
    (out_dir / "table_overall_geometry.tex").write_text(gtex)

    print(f"Saved intensity + geometry tables (CSV + TeX) to {out_dir}")


if __name__ == "__main__":
    main()
