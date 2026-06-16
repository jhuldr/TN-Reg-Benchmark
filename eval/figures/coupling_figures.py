"""Figure 5: coupling between predicted vessel volume and directional distances.

Each panel is a 2D density (hexbin, log-volume x-axis) of all (method, ROI)
observations with a log-linear regression line and the Spearman correlation.
Outputs fig5_predvox_{gt2pred,pred2gt,symmetric}.png.
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, linregress

from .. import config
from ._common import apply_paper_style, load_tracks

PANELS = [
    ("gt2pred_mean_mm", "GT→Pred Distance (mm)", "A", "gt2pred"),
    ("pred2gt_mean_mm", "Pred→GT Distance (mm)", "B", "pred2gt"),
    ("symmetric_mm", "Symmetric Distance (mm)", "C", "symmetric"),
]


def make_coupling(df_b, out_dir):
    ymax = max(df_b[m].quantile(0.99) for m, *_ in PANELS) * 1.05
    saved = []
    for metric, ylabel, letter, short in PANELS:
        sub = df_b[(df_b["pred_vox"] > 0) & np.isfinite(df_b[metric])]
        x, y = sub["pred_vox"].values, sub[metric].values

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        hb = ax.hexbin(x, y, gridsize=30, xscale="log", cmap="viridis", mincnt=1, linewidths=0)
        rho, p = spearmanr(x, y)
        slope, intercept, *_ = linregress(np.log10(x), y)
        xg = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
        ax.plot(xg, slope * np.log10(xg) + intercept, color="#FF1744", lw=2.0, alpha=0.95)

        p_str = "p < 0.001" if p < 0.001 else (f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}")
        ax.set_xlabel("Predicted Vessel Voxels (log)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"({letter}) {ylabel.split(' (')[0]}   (ρ = {rho:+.2f}, {p_str})",
                     fontweight="bold", loc="left", pad=8)
        ax.set_ylim(0, ymax)
        ax.grid(alpha=0.2, lw=0.5, color="white", zorder=0)
        cb = fig.colorbar(hb, ax=ax, fraction=0.05, pad=0.02)
        cb.set_label("Count")
        plt.tight_layout()
        out = out_dir / f"fig5_predvox_{short}.png"
        plt.savefig(str(out), dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(out)
        print(f"  ({letter}) {ylabel}: Spearman ρ = {rho:+.3f}, {p_str}")
    return saved


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out_dir = config.PAPER_DIR if args.out_dir is None else __import__("pathlib").Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_style()
    _, df_b = load_tracks()
    for p in make_coupling(df_b, out_dir):
        print("Saved:", p.name)


if __name__ == "__main__":
    main()
