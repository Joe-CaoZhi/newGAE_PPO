"""
Generate Baseline Comparison Figures
=====================================

Creates publication-quality figures comparing HCGAE against published PPO baselines.
Uses real experimental results from results/BaselineComparison/baseline_comparison_summary.json.

Figures generated:
1. Bar chart: Final performance comparison across 3 environments
2. Relative improvement: % gain of HCGAE vs each baseline

Style: ICML/Nature-inspired color palette, SEM error bars, professional layout.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
RESULTS_BASE = Path("results")
BASELINE_SUMMARY = RESULTS_BASE / "BaselineComparison" / "baseline_comparison_summary.json"
MULTIENV_SUMMARY = RESULTS_BASE / "MultiEnv" / "global_summary.json"
OUT_DIR = RESULTS_BASE / "paper_figures_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Style Setup (ICML/Nature style)
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "Standard_PPO":       "#4878D0",   # Blue
    "PPO_KLPEN":          "#6ACC65",   # Green
    "PPO_Anneal":         "#D65F5F",   # Red
    "PPO_EntDecay":       "#B47CC7",   # Purple
    "PPO_VClip":          "#C4AD66",   # Gold
    "PPO_Full_Baseline":  "#77BEDB",   # Light blue
    "HCGAE_Imp12":        "#E83535",   # Vivid red (ours)
    "HCGAE_Base":         "#FF8C42",   # Orange
    "DCPPO_S":            "#2CA02C",   # Dark green
    "GAE_Lambda1":        "#9467BD",   # Purple
}

LABELS = {
    "Standard_PPO":      "PPO (Standard)",
    "PPO_KLPEN":         "PPO-KLPEN",
    "PPO_Anneal":        "PPO-Anneal",
    "PPO_EntDecay":      "PPO-EntDecay",
    "PPO_VClip":         "PPO-VClip",
    "PPO_Full_Baseline": "PPO-Full",
    "HCGAE_Imp12":       "HCGAE (Ours)",
    "HCGAE_Base":        "HCGAE-Base",
    "DCPPO_S":           "DCPPO-S (Ours)",
    "GAE_Lambda1":       "GAE (lam=1)",
}

ENVS = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]
ENV_LABELS = {"Hopper-v4": "Hopper-v4", "Walker2d-v4": "Walker2d-v4", "HalfCheetah-v4": "HalfCheetah-v4"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


def sem(vals):
    """Standard error of the mean"""
    if len(vals) <= 1:
        return 0.0
    return float(np.std(vals) / np.sqrt(len(vals)))


def load_baseline_data():
    """Load baseline comparison summary"""
    if not BASELINE_SUMMARY.exists():
        return {}
    with open(BASELINE_SUMMARY) as f:
        return json.load(f)


def load_multienv_data():
    """Load multi-environment comparison summary"""
    if not MULTIENV_SUMMARY.exists():
        return {}
    with open(MULTIENV_SUMMARY) as f:
        return json.load(f)


def get_algo_stats(data: dict, env_name: str, algo_name: str) -> tuple:
    """Returns (mean, sem) for an algorithm on an environment."""
    try:
        seeds = data.get(env_name, {}).get(algo_name, {}).get("seeds", [])
        if not seeds:
            return None, None
        # Filter out zero values (failed runs)
        valid_seeds = [s for s in seeds if s is not None and s > 0]
        if not valid_seeds:
            return None, None
        return float(np.mean(valid_seeds)), sem(valid_seeds)
    except Exception:
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Figure 10: Baseline Comparison Bar Chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig10_baseline_comparison():
    """Bar chart comparing HCGAE vs published PPO baselines across environments."""
    baseline_data = load_baseline_data()
    multienv_data = load_multienv_data()

    # Combine data sources: baseline comparison results + multienv results
    # For algorithms that may appear in both, prefer baseline_comparison
    algos_ordered = [
        "Standard_PPO",
        "PPO_KLPEN",
        "PPO_Anneal",
        "PPO_EntDecay",
        "PPO_VClip",
        "PPO_Full_Baseline",
        "HCGAE_Imp12",
    ]

    # Check which algos have data
    available_algos = []
    for algo in algos_ordered:
        has_data = False
        for env in ENVS:
            m, s = get_algo_stats(baseline_data, env, algo)
            if m is not None:
                has_data = True
                break
            # Also check multienv
            if multienv_data:
                m, _ = get_algo_stats(multienv_data.get("results", {}), env, algo)
                if m is not None:
                    has_data = True
                    break
        if has_data:
            available_algos.append(algo)

    if not available_algos:
        print("  No baseline data available yet. Figures will be generated when data is ready.")
        return None

    n_algos = len(available_algos)
    n_envs = len(ENVS)
    fig_width = max(14, n_algos * 2.5)
    fig, axes = plt.subplots(1, n_envs, figsize=(fig_width, 5))

    for ax, env_name in zip(axes, ENVS):
        means, errors, colors_list, labels_list = [], [], [], []

        for algo in available_algos:
            # Try baseline_data first, then multienv_data
            m, s = get_algo_stats(baseline_data, env_name, algo)
            if m is None and multienv_data:
                m, s = get_algo_stats(multienv_data.get("results", {}), env_name, algo)
            if m is None:
                m, s = 0.0, 0.0

            means.append(m)
            errors.append(s if s is not None else 0.0)
            colors_list.append(COLORS.get(algo, "#888888"))
            labels_list.append(LABELS.get(algo, algo))

        x = np.arange(n_algos)
        bars = ax.bar(x, means, yerr=errors, capsize=4, color=colors_list,
                      alpha=0.85, edgecolor="white", linewidth=0.5,
                      error_kw=dict(elinewidth=1.5, capthick=1.5))

        # Highlight HCGAE bar
        if "HCGAE_Imp12" in available_algos:
            hcgae_idx = available_algos.index("HCGAE_Imp12")
            bars[hcgae_idx].set_edgecolor("#E83535")
            bars[hcgae_idx].set_linewidth(2.0)
            bars[hcgae_idx].set_alpha(1.0)

        ax.set_xticks(x)
        ax.set_xticklabels(labels_list, rotation=35, ha="right", fontsize=8)
        ax.set_title(ENV_LABELS.get(env_name, env_name), fontweight="bold", fontsize=12)
        ax.set_ylabel("Mean Episode Reward" if ax == axes[0] else "", fontsize=10)
        ax.set_xlim(-0.6, n_algos - 0.4)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))

    # Add legend
    patches = []
    for algo in available_algos:
        c = COLORS.get(algo, "#888888")
        lbl = LABELS.get(algo, algo)
        style = dict(facecolor=c, edgecolor="white", linewidth=0.5)
        if algo == "HCGAE_Imp12":
            style = dict(facecolor=c, edgecolor="#E83535", linewidth=1.5)
        patches.append(mpatches.Patch(**style, label=lbl))

    fig.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, 1.05),
               ncol=min(n_algos, 4), framealpha=0.9, fontsize=8.5)

    fig.suptitle("HCGAE vs Published PPO Baselines\n(5-seed mean, SEM bars)",
                 y=1.10, fontsize=13, fontweight="bold")

    plt.tight_layout()
    out_path = OUT_DIR / "fig10_baseline_comparison.pdf"
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.savefig(str(out_path).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Figure 11: Relative Improvement over Standard PPO
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig11_relative_improvement():
    """Show percentage improvement of each method over Standard PPO."""
    baseline_data = load_baseline_data()
    multienv_data = load_multienv_data()

    algos_to_compare = [
        "PPO_KLPEN",
        "PPO_Anneal",
        "PPO_EntDecay",
        "PPO_VClip",
        "PPO_Full_Baseline",
        "HCGAE_Imp12",
    ]

    fig, axes = plt.subplots(1, len(ENVS), figsize=(14, 5))

    for ax, env_name in zip(axes, ENVS):
        # Get Standard_PPO baseline
        std_mean, _ = get_algo_stats(baseline_data, env_name, "Standard_PPO")
        if std_mean is None and multienv_data:
            std_mean, _ = get_algo_stats(
                multienv_data.get("results", {}), env_name, "Standard_PPO")
        if std_mean is None or std_mean <= 0:
            ax.set_title(f"{env_name}\n(no data)", fontsize=10)
            continue

        rel_means, rel_errors, colors_list, labels_list = [], [], [], []
        has_any = False

        for algo in algos_to_compare:
            m, s = get_algo_stats(baseline_data, env_name, algo)
            if m is None and multienv_data:
                m, s = get_algo_stats(multienv_data.get("results", {}), env_name, algo)
            if m is None:
                rel_means.append(0.0)
                rel_errors.append(0.0)
            else:
                rel_pct = (m - std_mean) / (abs(std_mean) + 1e-8) * 100
                rel_err = (s / (abs(std_mean) + 1e-8) * 100) if s else 0.0
                rel_means.append(rel_pct)
                rel_errors.append(rel_err)
                has_any = True
            colors_list.append(COLORS.get(algo, "#888888"))
            labels_list.append(LABELS.get(algo, algo))

        if not has_any:
            ax.set_title(f"{env_name}\n(pending)", fontsize=10)
            continue

        x = np.arange(len(algos_to_compare))
        bar_colors = [COLORS.get("HCGAE_Imp12", "#E83535") if a == "HCGAE_Imp12"
                      else c for a, c in zip(algos_to_compare, colors_list)]

        bars = ax.bar(x, rel_means, yerr=rel_errors, capsize=4, color=bar_colors,
                      alpha=0.85, edgecolor="white", linewidth=0.5,
                      error_kw=dict(elinewidth=1.5, capthick=1.5))

        # Positive/negative coloring
        for bar, val in zip(bars, rel_means):
            if val > 0:
                bar.set_alpha(0.85)
            else:
                bar.set_alpha(0.5)
                bar.set_hatch("///")

        ax.axhline(y=0, color="black", linewidth=1.0, linestyle="-", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_list, rotation=35, ha="right", fontsize=8)
        ax.set_title(ENV_LABELS.get(env_name, env_name), fontweight="bold", fontsize=12)
        ax.set_ylabel("% vs Standard PPO" if ax == axes[0] else "", fontsize=10)

    fig.suptitle("Relative Performance vs Standard PPO\n(positive = improvement over PPO baseline)",
                 y=1.02, fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = OUT_DIR / "fig11_relative_improvement.pdf"
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.savefig(str(out_path).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Figure 12: Summary Table (heatmap-style)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig12_comparison_table():
    """Heatmap-style table showing final performance across all methods and environments."""
    baseline_data = load_baseline_data()
    multienv_data = load_multienv_data()

    all_algos = [
        "Standard_PPO",
        "PPO_KLPEN",
        "PPO_Anneal",
        "PPO_EntDecay",
        "PPO_VClip",
        "PPO_Full_Baseline",
        "HCGAE_Base",
        "HCGAE_Imp12",
    ]

    data_matrix = np.zeros((len(all_algos), len(ENVS)))
    mask = np.ones_like(data_matrix, dtype=bool)

    for i, algo in enumerate(all_algos):
        for j, env in enumerate(ENVS):
            m, _ = get_algo_stats(baseline_data, env, algo)
            if m is None and multienv_data:
                m, _ = get_algo_stats(
                    multienv_data.get("results", {}), env, algo)
            if m is not None and m > 0:
                data_matrix[i, j] = m
                mask[i, j] = False

    fig, ax = plt.subplots(figsize=(10, 6))

    # Normalize by column (environment)
    data_norm = data_matrix.copy()
    for j in range(len(ENVS)):
        col = data_matrix[:, j]
        col_max = col.max() if col.max() > 0 else 1.0
        data_norm[:, j] = col / col_max

    im = ax.imshow(data_norm, cmap="RdYlGn", aspect="auto",
                   vmin=0, vmax=1.0, alpha=0.8)

    # Annotate with actual values
    for i in range(len(all_algos)):
        for j in range(len(ENVS)):
            if not mask[i, j]:
                val = data_matrix[i, j]
                text_color = "black" if data_norm[i, j] > 0.5 else "white"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=9, color=text_color, fontweight="bold")
            else:
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=8, color="#999999", style="italic")

    ax.set_xticks(range(len(ENVS)))
    ax.set_xticklabels([ENV_LABELS.get(e, e) for e in ENVS], fontsize=10)
    ax.set_yticks(range(len(all_algos)))
    labels_y = [LABELS.get(a, a) for a in all_algos]
    # Bold our method
    ax.set_yticklabels(labels_y, fontsize=9)
    # Mark our method row
    if "HCGAE_Imp12" in all_algos:
        hcgae_row = all_algos.index("HCGAE_Imp12")
        ax.get_yticklabels()[hcgae_row].set_fontweight("bold")
        ax.get_yticklabels()[hcgae_row].set_color(COLORS["HCGAE_Imp12"])

    plt.colorbar(im, ax=ax, label="Relative Performance (normalized per env)", shrink=0.8)
    ax.set_title("Method Performance Summary (5-seed mean, color=normalized score)",
                 fontweight="bold", fontsize=11)

    # Highlight our rows
    if "HCGAE_Imp12" in all_algos:
        hcgae_row = all_algos.index("HCGAE_Imp12")
        ax.add_patch(plt.Rectangle((-0.5, hcgae_row - 0.5), len(ENVS), 1,
                                   fill=False, edgecolor="#E83535", linewidth=2.5))

    plt.tight_layout()
    out_path = OUT_DIR / "fig12_comparison_table.pdf"
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.savefig(str(out_path).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print("  Generating Baseline Comparison Figures")
    print(f"{'='*60}")

    paths = []
    p1 = plot_fig10_baseline_comparison()
    if p1:
        paths.append(p1)

    p2 = plot_fig11_relative_improvement()
    if p2:
        paths.append(p2)

    p3 = plot_fig12_comparison_table()
    if p3:
        paths.append(p3)

    print(f"\nGenerated {len(paths)} figures:")
    for p in paths:
        print(f"  {p}")
    print(f"\n  PNG versions also saved in same directory.")


if __name__ == "__main__":
    main()

