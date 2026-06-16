"""Overall per-method figures.

  fig1_overall_intensity.png  -> manuscript Fig. 2 (Vessel AUC, ROI NMI)
  fig2_overall_geometry.png   -> manuscript Fig. 3 (distances + predicted volume)
  suppfig_intensity_*.png / suppfig_geometry_*.png  -> per-metric box+violin panels

Run:  REG_ROOT=/path/to/TN_Reg python -m eval.figures.overview_figures
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

from .. import config
from ._common import apply_paper_style, load_tracks, METHOD_ORDER, ABBR, COLORS


def _median_iqr_panel(ax, data, label, baseline=None, log=False):
    """Median marker + IQR whiskers, one per method."""
    for i, (m, vals) in enumerate(zip(METHOD_ORDER, data)):
        v = np.asarray(vals, float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        med, q1, q3 = np.median(v), np.percentile(v, 25), np.percentile(v, 75)
        ax.plot([i, i], [q1, q3], color=COLORS[m], lw=2.2, alpha=0.9, zorder=2)
        ax.plot(i, med, "o", ms=9, mfc=COLORS[m], mec="black", mew=1.0, zorder=3)
    if baseline is not None:
        ax.axhline(baseline, color="red", ls="--", lw=1.2, alpha=0.7, zorder=0)
        xlim = ax.get_xlim()
        ax.text(xlim[1] - 0.02 * (xlim[1] - xlim[0]), baseline, f"Chance ({baseline})",
                color="red", fontsize=9, fontweight="bold", va="center", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="red", linewidth=0.8, alpha=0.95))
    if log:
        ax.set_yscale("log")
    ax.set_xticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels([ABBR[m] for m in METHOD_ORDER])
    ax.set_xlim(-0.5, len(METHOD_ORDER) - 0.5)
    ax.set_ylabel(label)
    ax.set_title(label, fontweight="bold", loc="left", pad=8)
    ax.grid(axis="y", alpha=0.25, lw=0.5)


def make_overall_intensity(df_a, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _median_iqr_panel(axes[0], [df_a[df_a.method == m]["vessel_AUC"].values for m in METHOD_ORDER],
                      "(A) Vessel AUC", baseline=0.5)
    _median_iqr_panel(axes[1], [df_a[df_a.method == m]["roi_NMI"].values for m in METHOD_ORDER],
                      "(B) ROI NMI")
    plt.tight_layout()
    out = out_dir / "fig1_overall_intensity.png"
    plt.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def make_overall_geometry(df_b, out_dir):
    panels = [
        ("gt2pred_mean_mm", "(A) GT→Pred (mm)", False),
        ("pred2gt_mean_mm", "(B) Pred→GT (mm)", False),
        ("symmetric_mm", "(C) Symmetric (mm)", False),
        ("pred_vox", "(D) Predicted Vessel Voxels", True),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.3))
    for ax, (col, label, log) in zip(axes, panels):
        _median_iqr_panel(ax, [df_b[df_b.method == m][col].values for m in METHOD_ORDER],
                          label, log=log)
    plt.tight_layout()
    out = out_dir / "fig2_overall_geometry.png"
    plt.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def _box_violin_panel(data, label, baseline=None, log=False):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    positions = np.arange(len(METHOD_ORDER))
    vp = ax.violinplot(data, positions=positions, widths=0.75,
                       showmeans=False, showmedians=False, showextrema=False)
    for pc, m in zip(vp["bodies"], METHOD_ORDER):
        pc.set_facecolor(COLORS[m]); pc.set_edgecolor(COLORS[m])
        pc.set_alpha(0.25); pc.set_linewidth(0)
    bp = ax.boxplot(data, positions=positions, widths=0.28, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 1.8},
                    flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.3,
                                "markerfacecolor": "gray", "markeredgecolor": "none"})
    for patch, m in zip(bp["boxes"], METHOD_ORDER):
        patch.set_facecolor(COLORS[m]); patch.set_alpha(0.85)
        patch.set_edgecolor("black"); patch.set_linewidth(0.8)
    if baseline is not None:
        ax.axhline(baseline, color="red", ls="--", lw=1.2, alpha=0.7, zorder=0)
        xlim = ax.get_xlim()
        ax.text(xlim[1] - 0.02 * (xlim[1] - xlim[0]), baseline, f"Chance ({baseline})",
                color="red", fontsize=9, fontweight="bold", va="center", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="red", linewidth=0.8, alpha=0.95))
    if log:
        ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels([ABBR[m] for m in METHOD_ORDER])
    ax.set_xlim(-0.5, len(METHOD_ORDER) - 0.5)
    ax.set_ylabel(label); ax.set_title(label, fontweight="bold", loc="left", pad=8)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    plt.tight_layout()
    return fig, ax


def make_supplementary(df_a, df_b, out_dir):
    out = []
    for col, label, base, short in [("vessel_AUC", "Vessel AUC", 0.5, "vesselAUC"),
                                    ("roi_NMI", "ROI NMI", None, "roiNMI")]:
        data = [df_a[df_a.method == m][col].dropna().values for m in METHOD_ORDER]
        fig, _ = _box_violin_panel(data, label, baseline=base)
        p = out_dir / f"suppfig_intensity_{short}.png"
        fig.savefig(str(p), dpi=300, bbox_inches="tight"); plt.close(fig); out.append(p)
    for col, label, log, short in [("gt2pred_mean_mm", "GT→Pred Distance (mm)", False, "gt2pred"),
                                   ("pred2gt_mean_mm", "Pred→GT Distance (mm)", False, "pred2gt"),
                                   ("symmetric_mm", "Symmetric Distance (mm)", False, "symmetric"),
                                   ("pred_vox", "Predicted Vessel Voxels", True, "predvox")]:
        data = [df_b[df_b.method == m][col].dropna().values for m in METHOD_ORDER]
        fig, _ = _box_violin_panel(data, label, log=log)
        p = out_dir / f"suppfig_geometry_{short}.png"
        fig.savefig(str(p), dpi=300, bbox_inches="tight"); plt.close(fig); out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out_dir = config.PAPER_DIR if args.out_dir is None else __import__("pathlib").Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_style()
    df_a, df_b = load_tracks()
    print("Saved:", make_overall_intensity(df_a, out_dir).name)
    print("Saved:", make_overall_geometry(df_b, out_dir).name)
    for p in make_supplementary(df_a, df_b, out_dir):
        print("Saved:", p.name)


if __name__ == "__main__":
    main()
