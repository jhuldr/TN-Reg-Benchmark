"""Figures 6 and 7: contrast- and FOV-stratified analyses.

  fig6_contrast_tier_distribution.png  -> ROI counts per Low/Mid/High tier
  fig6_vesselAUC_by_tier.png           -> Vessel AUC by tier x method
  fig7_fov_paired_connection.png       -> paired Affine->SyN by FOV group
  fig7_fov_paired_delta.png            -> distribution of paired deltas
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon

from .. import config
from ._common import (apply_paper_style, load_tracks, compute_fov,
                      assign_contrast_tier, METHOD_ORDER, ABBR, COLORS)

TIER_LABELS = config.TIER_LABELS


def make_tier_distribution(tier_counts, out_dir):
    n_total = tier_counts.sum()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    bars = ax.bar(TIER_LABELS, tier_counts.values, color=config.TIER_COLORS,
                  alpha=0.85, edgecolor="black", linewidth=0.8)
    for bar, t in zip(bars, TIER_LABELS):
        n = tier_counts[t]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + n_total * 0.01,
                f"N = {n}\n({n / n_total * 100:.0f}%)", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_xlabel("MRA Contrast Tier"); ax.set_ylabel("Count of ROIs")
    ax.set_title("(A) Contrast Tier Distribution", loc="left", fontweight="bold", pad=8)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_ylim(0, tier_counts.max() * 1.18)
    plt.tight_layout()
    out = out_dir / "fig6_contrast_tier_distribution.png"
    plt.savefig(str(out), dpi=300, bbox_inches="tight"); plt.close(fig)
    return out


def make_auc_by_tier(df_a_tier, tier_counts, out_dir):
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    n_methods = len(METHOD_ORDER)
    group_width = 0.85
    box_width = group_width / n_methods * 0.85
    x_base = np.arange(len(TIER_LABELS))
    for mi, m in enumerate(METHOD_ORDER):
        offset = (mi - n_methods / 2 + 0.5) * (group_width / n_methods)
        positions = x_base + offset
        data = [df_a_tier[(df_a_tier.method == m) & (df_a_tier.tier == t)]["vessel_AUC"].dropna().values
                for t in TIER_LABELS]
        vp = ax.violinplot(data, positions=positions, widths=box_width * 1.8,
                           showmeans=False, showmedians=False, showextrema=False)
        for pc in vp["bodies"]:
            pc.set_facecolor(COLORS[m]); pc.set_edgecolor(COLORS[m])
            pc.set_alpha(0.20); pc.set_linewidth(0)
        bp = ax.boxplot(data, positions=positions, widths=box_width, patch_artist=True,
                        medianprops={"color": "black", "linewidth": 1.4},
                        flierprops={"marker": "o", "markersize": 2, "alpha": 0.3,
                                    "markerfacecolor": "gray", "markeredgecolor": "none"})
        for patch in bp["boxes"]:
            patch.set_facecolor(COLORS[m]); patch.set_alpha(0.85)
            patch.set_edgecolor("black"); patch.set_linewidth(0.6)
    ax.axhline(0.5, color="red", ls="--", lw=1.2, alpha=0.7, zorder=0)
    xlim = ax.get_xlim()
    ax.text(xlim[1] - 0.03 * (xlim[1] - xlim[0]), 0.5, "Chance (0.5)", color="red",
            fontsize=9, fontweight="bold", va="center", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="red", linewidth=0.8, alpha=0.95))
    ax.set_xticks(x_base)
    ax.set_xticklabels([f"{t}\n(N={tier_counts[t]})" for t in TIER_LABELS])
    ax.set_ylabel("Vessel AUC"); ax.set_xlabel("Contrast Tier")
    ax.set_title("(B) Vessel AUC by Contrast Tier × Method", loc="left", fontweight="bold", pad=8)
    ax.grid(axis="y", alpha=0.25, lw=0.5); ax.set_ylim(0, 1.02)
    handles = [Patch(facecolor=COLORS[m], edgecolor="black", linewidth=0.6, alpha=0.85, label=ABBR[m])
               for m in METHOD_ORDER]
    ax.legend(handles=handles, loc="lower right", ncol=3, fontsize=8,
              frameon=True, framealpha=0.95, title="Method", title_fontsize=9)
    plt.tight_layout()
    out = out_dir / "fig6_vesselAUC_by_tier.png"
    plt.savefig(str(out), dpi=300, bbox_inches="tight"); plt.close(fig)
    return out


def _paired_affine_syn(df_b, df_fov):
    """Build the paired Affine-vs-SyN GT→Pred table with FOV group labels."""
    thresh = df_fov["min_ratio"].quantile(0.20)
    bad_mrns = set(df_fov[df_fov["min_ratio"] <= thresh]["mrn"])
    aff = df_b[df_b.method == "ANTs_Affine"][["mrn", "side", "gt2pred_mean_mm"]].rename(
        columns={"gt2pred_mean_mm": "aff"})
    syn = df_b[df_b.method == "ANTs_SyN"][["mrn", "side", "gt2pred_mean_mm"]].rename(
        columns={"gt2pred_mean_mm": "syn"})
    paired = aff.merge(syn, on=["mrn", "side"], how="inner")
    paired = paired[paired["mrn"].isin(df_fov["mrn"])].copy()
    paired["is_bad"] = paired["mrn"].isin(bad_mrns)
    paired["delta"] = paired["syn"] - paired["aff"]
    return paired, thresh


def _p_fmt(p):
    return "p < 0.001" if p < 0.001 else (f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}")


def make_fov_connection(paired, out_dir):
    bad, good = paired[paired.is_bad], paired[~paired.is_bad]
    n_bad, n_good = len(bad), len(good)
    _, p_bad = wilcoxon(bad["aff"], bad["syn"])
    _, p_good = wilcoxon(good["aff"], good["syn"])
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    pos = {"bad_aff": 0, "bad_syn": 1, "good_aff": 3, "good_syn": 4}
    rng = np.random.RandomState(0)
    jitter = 0.05
    for grp, (pa, ps, alpha) in {"bad": (0, 1, 0.45), "good": (3, 4, 0.35)}.items():
        sub = bad if grp == "bad" else good
        for _, r in sub.iterrows():
            jx1, jx2 = pa + rng.uniform(-jitter, jitter), ps + rng.uniform(-jitter, jitter)
            ax.plot([jx1, jx2], [r["aff"], r["syn"]], color="#9C9C9C", lw=0.5, alpha=alpha, zorder=1)
            ax.scatter([jx1, jx2], [r["aff"], r["syn"]], s=8,
                       color=[COLORS["ANTs_Affine"], COLORS["ANTs_SyN"]],
                       alpha=alpha + 0.1, zorder=2, edgecolors="none")

    def med(x, vals, color):
        ax.scatter([x], [np.median(vals)], s=140, marker="s", color=color,
                   edgecolors="black", linewidths=1.2, zorder=5)
    med(pos["bad_aff"], bad["aff"], COLORS["ANTs_Affine"])
    med(pos["bad_syn"], bad["syn"], COLORS["ANTs_SyN"])
    med(pos["good_aff"], good["aff"], COLORS["ANTs_Affine"])
    med(pos["good_syn"], good["syn"], COLORS["ANTs_SyN"])
    ax.plot([0, 1], [bad["aff"].median(), bad["syn"].median()], color="black", lw=2.0, zorder=4)
    ax.plot([3, 4], [good["aff"].median(), good["syn"].median()], color="black", lw=2.0, zorder=4)
    ax.set_xticks([0, 1, 3, 4]); ax.set_xticklabels(["Affine", "SyN", "Affine", "SyN"])
    ax.set_xlim(-0.8, 4.8)
    ax.text(0.5, -0.16, f"Bad FOV (N={n_bad})\nWilcoxon {_p_fmt(p_bad)}", ha="center", va="top",
            fontsize=9.5, fontweight="bold", transform=ax.get_xaxis_transform())
    ax.text(3.5, -0.16, f"Good FOV (N={n_good})\nWilcoxon {_p_fmt(p_good)}", ha="center", va="top",
            fontsize=9.5, fontweight="bold", transform=ax.get_xaxis_transform())
    ax.set_ylabel("GT→Pred Distance (mm)")
    ax.set_title("(A) Paired Affine → SyN by FOV Group", loc="left", fontweight="bold", pad=8)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    handles = [
        Line2D([0], [0], marker="s", linestyle="", markersize=9,
               markerfacecolor=COLORS["ANTs_Affine"], markeredgecolor="black", label="Affine median"),
        Line2D([0], [0], marker="s", linestyle="", markersize=9,
               markerfacecolor=COLORS["ANTs_SyN"], markeredgecolor="black", label="SyN median"),
        Line2D([0], [0], color="#B0B0B0", linewidth=1.0, label="Paired case"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True, framealpha=0.95)
    plt.tight_layout()
    out = out_dir / "fig7_fov_paired_connection.png"
    plt.savefig(str(out), dpi=300, bbox_inches="tight"); plt.close(fig)
    return out


def make_fov_delta(paired, out_dir):
    bad, good = paired[paired.is_bad], paired[~paired.is_bad]
    delta_bad, delta_good = bad["delta"].values, good["delta"].values
    n_bad, n_good = len(bad), len(good)
    lo, hi = np.percentile(np.concatenate([delta_bad, delta_good]), [1, 99])
    bins = np.linspace(lo, hi, 35)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.hist(delta_good, bins=bins, alpha=0.55, color="#54A24B", edgecolor="black",
            linewidth=0.5, label=f"Good FOV (N={n_good})", density=True)
    ax.hist(delta_bad, bins=bins, alpha=0.55, color="#E45756", edgecolor="black",
            linewidth=0.5, label=f"Bad FOV (N={n_bad})", density=True)
    ax.axvline(0, color="black", ls="--", lw=1.0, alpha=0.7, zorder=3)
    data_ymax = ax.get_ylim()[1]
    ax.set_ylim(0, data_ymax * 1.30)
    med_good, med_bad = np.median(delta_good), np.median(delta_bad)
    ax.axvline(med_good, color="#54A24B", lw=1.8, alpha=0.85, zorder=2)
    ax.axvline(med_bad, color="#E45756", lw=1.8, alpha=0.85, zorder=2)
    ax.text(lo * 0.55, data_ymax * 0.75, f"Good FOV\nmedian = {med_good:+.2f} mm", color="#54A24B",
            fontsize=9, fontweight="bold", va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#54A24B", linewidth=0.8, alpha=0.95))
    ax.text(lo * 0.55, data_ymax * 0.55, f"Bad FOV\nmedian = {med_bad:+.2f} mm", color="#E45756",
            fontsize=9, fontweight="bold", va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#E45756", linewidth=0.8, alpha=0.95))
    ax.annotate("", xy=(0.05, 0.97), xytext=(0.45, 0.97), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))
    ax.text(0.05, 0.93, "SyN better", color="gray", fontsize=9, ha="left", va="top", transform=ax.transAxes)
    ax.annotate("", xy=(0.95, 0.97), xytext=(0.55, 0.97), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))
    ax.text(0.95, 0.93, "Affine better", color="gray", fontsize=9, ha="right", va="top", transform=ax.transAxes)
    ax.set_xlabel("Δ = SyN − Affine  (mm)"); ax.set_ylabel("Density")
    ax.set_title("(B) Paired Difference Distribution", loc="left", fontweight="bold", pad=8)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.95, bbox_to_anchor=(1.0, 0.88))
    plt.tight_layout()
    out = out_dir / "fig7_fov_paired_delta.png"
    plt.savefig(str(out), dpi=300, bbox_inches="tight"); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out_dir = config.PAPER_DIR if args.out_dir is None else __import__("pathlib").Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_style()
    df_a, df_b = load_tracks()

    df_tier = assign_contrast_tier(df_a)
    df_a_tier = df_a.merge(df_tier[["mrn", "side", "tier"]], on=["mrn", "side"], how="left")
    tier_counts = df_tier["tier"].value_counts().reindex(TIER_LABELS)
    print("Saved:", make_tier_distribution(tier_counts, out_dir).name)
    print("Saved:", make_auc_by_tier(df_a_tier, tier_counts, out_dir).name)

    df_fov = compute_fov(df_a["mrn"].unique())
    paired, thresh = _paired_affine_syn(df_b, df_fov)
    print(f"FOV threshold (20th pct min_ratio): {thresh:.3f}")
    print("Saved:", make_fov_connection(paired, out_dir).name)
    print("Saved:", make_fov_delta(paired, out_dir).name)


if __name__ == "__main__":
    main()
