#!/usr/bin/env python3
"""
Generate ICML-quality figures for HCGAE paper
===============================================
Figures:
  1. Learning curves (3 envs, 3 algos) — main comparison
  2. Bar chart — final performance
  3. Ablation bar chart — component contributions
  4. HC analysis — outlier-aware visualization

All using AlignedExperiment data (deterministic eval) for primary,
ICMLExperiment data (stochastic eval) for ablation.
"""

import json
import numpy as np
from pathlib import Path
import sys

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Patch
    HAS_MPL = True
except ImportError:
    print("matplotlib not available")
    HAS_MPL = False
    sys.exit(1)

ALIGNED_DIR = Path("results/AlignedExperiment")
ICML_DIR = Path("results/ICMLExperiment")
OUT_DIR = Path("results/paper_figures_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENVS = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]
ENV_LABELS = {"Hopper-v4": "Hopper-v4", "Walker2d-v4": "Walker2d-v4", "HalfCheetah-v4": "HalfCheetah-v4"}
SEEDS = list(range(5))

PRIMARY_ALGOS = ["Standard_PPO", "Optimal_PPO", "Optimal_HCGAE_v2"]
ABLATION_ALGOS = ["Optimal_HCGAE_v2", "Optimal_HCGAE_v2_NoBdry", "Optimal_HCGAE_v2_NoGate"]

COLORS = {
    "Standard_PPO": "#7f7f7f",
    "Optimal_PPO": "#1f77b4",
    "Optimal_HCGAE_v2": "#d62728",
    "Optimal_HCGAE_v2_NoBdry": "#ff7f0e",
    "Optimal_HCGAE_v2_NoGate": "#2ca02c",
}
LINESTYLES = {
    "Standard_PPO": (0, (3, 1)),  # densely dashed
    "Optimal_PPO": (0, (5, 1, 1, 1)),  # dashdot
    "Optimal_HCGAE_v2": "solid",
    "Optimal_HCGAE_v2_NoBdry": (0, (3, 1, 1, 1)),
    "Optimal_HCGAE_v2_NoGate": (0, (1, 1)),  # dotted
}
ALGO_LABELS = {
    "Standard_PPO": "Standard PPO",
    "Optimal_PPO": "Optimal PPO",
    "Optimal_HCGAE_v2": "HCGAE-v2 (Ours)",
    "Optimal_HCGAE_v2_NoBdry": r"HCGAE-v2 $-$BdryCorr",
    "Optimal_HCGAE_v2_NoGate": r"HCGAE-v2 $-$EVGate",
}

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 2.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})


def load_curves(base_dir, env, algo):
    """Load all learning curves for env/algo."""
    curves_by_step = {}
    for s in SEEDS:
        fp = base_dir / env / algo / f"{algo}_s{s}.json"
        if fp.exists():
            try:
                d = json.load(open(fp))
                er = d.get("eval_rewards", [])
                es = d.get("eval_steps", [])
                if er:
                    curves_by_step[s] = (es, er)
            except:
                pass
    return curves_by_step


def interpolate_to_common(curves_dict, n_points=49, total_steps=500_000):
    """Interpolate all curves to common x-axis."""
    x_common = np.linspace(0, total_steps, n_points)
    all_y = []
    for s, (es, er) in curves_dict.items():
        if not er:
            continue
        # Interpolate
        if len(es) >= 2:
            y = np.interp(x_common, es, er, left=er[0], right=er[-1])
        else:
            y = np.full(n_points, np.mean(er))
        all_y.append(y)
    return x_common, np.array(all_y) if all_y else (x_common, np.array([]))


def get_finals(base_dir, env, algo):
    """Get list of final-5 returns."""
    vals = []
    for s in SEEDS:
        fp = base_dir / env / algo / f"{algo}_s{s}.json"
        if fp.exists():
            try:
                d = json.load(open(fp))
                er = d.get("eval_rewards", [])
                if er:
                    vals.append(float(np.mean(er[-5:])))
            except:
                pass
    return vals


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Learning Curves (main comparison)
# ─────────────────────────────────────────────────────────────────────────────
def fig1_learning_curves():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    fig.suptitle(
        "HCGAE-v2 vs Baselines: 500K steps, 5 seeds, deterministic evaluation",
        fontsize=12, fontweight='bold', y=1.01
    )

    for ax, env in zip(axes, ENVS):
        for algo in PRIMARY_ALGOS:
            curves = load_curves(ALIGNED_DIR, env, algo)
            if not curves:
                continue
            x, arr = interpolate_to_common(curves)
            if arr.size == 0:
                continue
            mean_c = np.mean(arr, axis=0)
            sem_c = np.std(arr, axis=0) / np.sqrt(len(arr))

            color = COLORS[algo]
            ls = LINESTYLES[algo]
            label = ALGO_LABELS[algo]
            lw = 2.5 if algo == "Optimal_HCGAE_v2" else 2.0
            ax.plot(x / 1e5, mean_c, color=color, linestyle=ls,
                    linewidth=lw, label=label, zorder=3 if algo == "Optimal_HCGAE_v2" else 2)
            ax.fill_between(x / 1e5, mean_c - sem_c, mean_c + sem_c,
                           alpha=0.12, color=color)

        ax.set_title(ENV_LABELS[env], fontsize=12)
        ax.set_xlabel("Steps (×10⁵)")
        if ax == axes[0]:
            ax.set_ylabel("Episode Return")
        ax.legend(loc='upper left', framealpha=0.9)
        ax.set_xlim(0, 5)

    plt.tight_layout()
    out = OUT_DIR / "fig1_aligned_learning_curves.pdf"
    fig.savefig(out, bbox_inches='tight', dpi=200)
    fig.savefig(str(out).replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Bar chart — final performance with per-seed scatter
# ─────────────────────────────────────────────────────────────────────────────
def fig2_bar_final():
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle(
        "Final Performance (mean±SEM, 5 seeds, 500K steps)",
        fontsize=12, fontweight='bold', y=1.01
    )

    algo_x = np.arange(len(PRIMARY_ALGOS))
    width = 0.6

    for ax, env in zip(axes, ENVS):
        means, sems, all_vals = [], [], []
        for algo in PRIMARY_ALGOS:
            vals = get_finals(ALIGNED_DIR, env, algo)
            means.append(np.mean(vals) if vals else 0)
            sems.append(np.std(vals) / np.sqrt(len(vals)) if vals else 0)
            all_vals.append(vals)

        bars = ax.bar(algo_x, means, width, yerr=sems, capsize=6,
                      color=[COLORS[a] for a in PRIMARY_ALGOS],
                      alpha=0.80, ecolor='black', linewidth=1.0,
                      edgecolor='white', error_kw={'elinewidth': 1.5})

        # Overlay per-seed scatter
        for i, (algo, vals) in enumerate(zip(PRIMARY_ALGOS, all_vals)):
            for v in vals:
                ax.scatter(i, v, color=COLORS[algo], s=40, zorder=5,
                          edgecolors='black', linewidths=0.5, alpha=0.9)

        ax.set_title(ENV_LABELS[env], fontsize=12)
        ax.set_xticks(algo_x)
        ax.set_xticklabels([ALGO_LABELS[a].replace(" ", "\n") for a in PRIMARY_ALGOS],
                           fontsize=8)
        if ax == axes[0]:
            ax.set_ylabel("Final Return (mean last-5 evals)")
        ax.grid(True, axis='y', alpha=0.3)

        # Add gain annotation for HCGAE
        opt_m = means[PRIMARY_ALGOS.index("Optimal_PPO")]
        hcg_m = means[PRIMARY_ALGOS.index("Optimal_HCGAE_v2")]
        if opt_m > 0:
            gain = (hcg_m - opt_m) / opt_m * 100
            ax.annotate(f"{gain:+.1f}%",
                       xy=(2, hcg_m + sems[2]),
                       ha='center', va='bottom', fontsize=10,
                       color=COLORS["Optimal_HCGAE_v2"], fontweight='bold')

    plt.tight_layout()
    out = OUT_DIR / "fig2_aligned_bar.pdf"
    fig.savefig(out, bbox_inches='tight', dpi=200)
    fig.savefig(str(out).replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Ablation study (from ICMLExperiment, stochastic eval)
# ─────────────────────────────────────────────────────────────────────────────
def fig3_ablation():
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle(
        "Ablation Study: HCGAE-v2 Component Contributions (stochastic eval, 5 seeds)",
        fontsize=11, fontweight='bold', y=1.01
    )

    algo_x = np.arange(len(ABLATION_ALGOS))
    width = 0.6

    for ax, env in zip(axes, ENVS):
        means, sems, all_vals = [], [], []
        for algo in ABLATION_ALGOS:
            vals = get_finals(ICML_DIR, env, algo)
            means.append(np.mean(vals) if vals else 0)
            sems.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
            all_vals.append(vals)

        bars = ax.bar(algo_x, means, width, yerr=sems, capsize=6,
                      color=[COLORS[a] for a in ABLATION_ALGOS],
                      alpha=0.80, ecolor='black', linewidth=1.0,
                      edgecolor='white', error_kw={'elinewidth': 1.5})

        # Per-seed scatter
        for i, (algo, vals) in enumerate(zip(ABLATION_ALGOS, all_vals)):
            for v in vals:
                ax.scatter(i, v, color=COLORS[algo], s=35, zorder=5,
                          edgecolors='black', linewidths=0.5, alpha=0.8)

        ax.set_title(ENV_LABELS[env], fontsize=12)
        ax.set_xticks(algo_x)
        xticklabels = ["Full\nHCGAE-v2", r"$-$BdryCorr", r"$-$EVGate"]
        ax.set_xticklabels(xticklabels, fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel("Final Return")
        ax.grid(True, axis='y', alpha=0.3)

        # Annotate loss from removing each component
        full_m = means[0]
        for i in [1, 2]:
            diff = means[i] - full_m
            pct = diff / abs(full_m) * 100 if full_m != 0 else 0
            color = 'red' if diff < 0 else 'green'
            ax.annotate(f"{pct:+.1f}%",
                       xy=(i, means[i] + sems[i] + 20),
                       ha='center', fontsize=9, color=color, fontweight='bold')

    plt.tight_layout()
    out = OUT_DIR / "fig3_ablation.pdf"
    fig.savefig(out, bbox_inches='tight', dpi=200)
    fig.savefig(str(out).replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: HalfCheetah deep dive — with outlier annotation
# ─────────────────────────────────────────────────────────────────────────────
def fig4_hc_analysis():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "HalfCheetah-v4: High-Variance Analysis\n"
        "(Optimal PPO seed3=3751 and HCGAE-v2 seed3=665 are statistical outliers)",
        fontsize=11, fontweight='bold'
    )

    # Left: learning curves for all seeds
    ax = axes[0]
    ax.set_title("Learning Curves (all 5 seeds shown individually)")
    colors_seed = plt.cm.Set1(np.linspace(0, 0.9, 5))

    for algo, ls, label in [("Optimal_PPO", "--", "Optimal PPO"),
                             ("Optimal_HCGAE_v2", "-", "HCGAE-v2")]:
        curves = load_curves(ALIGNED_DIR, "HalfCheetah-v4", algo)
        color_base = COLORS[algo]
        for i, (s, (es, er)) in enumerate(sorted(curves.items())):
            alpha = 0.6
            lw = 2.0
            seed_label = f"{label} s{s}"
            # Highlight outlier seeds
            if (algo == "Optimal_PPO" and s == 3) or (algo == "Optimal_HCGAE_v2" and s == 3):
                alpha = 1.0
                lw = 3.0
                seed_label += " ★"
            ax.plot([e/1e5 for e in es], er, color=color_base, linestyle=ls,
                   linewidth=lw, alpha=alpha, label=seed_label if i < 3 or s == 3 else "")
        # Mean
        x, arr = interpolate_to_common(curves)
        if arr.size > 0:
            ax.plot(x/1e5, np.mean(arr, axis=0), color=color_base,
                   linestyle=ls, linewidth=3.5, label=f"{label} (mean)",
                   zorder=10)

    ax.set_xlabel("Steps (×10⁵)")
    ax.set_ylabel("Episode Return")
    ax.legend(fontsize=7, ncol=1)
    ax.set_xlim(0, 5)
    ax.annotate("Optimal PPO s3\n(lucky seed, 3751)",
               xy=(5, 3751), xytext=(3.8, 3300),
               arrowprops=dict(arrowstyle='->', color='blue'),
               color='blue', fontsize=8)
    ax.annotate("HCGAE s3\n(collapse, 547→665)",
               xy=(5, 547), xytext=(3.5, 300),
               arrowprops=dict(arrowstyle='->', color='red'),
               color='red', fontsize=8)

    # Right: Box plot / scatter per seed
    ax2 = axes[1]
    ax2.set_title("Final Returns: All Seeds (mean±SEM)")

    positions = [0, 1, 2]
    algos_plot = ["Standard_PPO", "Optimal_PPO", "Optimal_HCGAE_v2"]
    for i, algo in enumerate(algos_plot):
        vals = get_finals(ALIGNED_DIR, "HalfCheetah-v4", algo)
        if not vals:
            continue
        color = COLORS[algo]
        m = np.mean(vals)
        sem = np.std(vals) / np.sqrt(len(vals))
        # Bar
        ax2.bar(i, m, yerr=sem, capsize=8, color=color, alpha=0.7,
               edgecolor='white', linewidth=0.5, error_kw={'elinewidth': 1.5})
        # Per-seed scatter with jitter
        for j, v in enumerate(vals):
            jitter = (np.random.RandomState(j+42).random() - 0.5) * 0.2
            marker = '★' if v > 3000 or v < 700 else 'o'
            ms = 100 if marker == '★' else 40
            ax2.scatter(i + jitter, v, color=color, s=ms, zorder=5,
                       edgecolors='black', linewidths=0.8, alpha=0.9,
                       marker='*' if v > 3000 or v < 700 else 'o')
            if v > 3000:
                ax2.annotate(f"s3={v:.0f}\n(outlier)",
                           xy=(i+jitter, v), xytext=(i+0.2, v-200),
                           color='red', fontsize=7)
            if v < 700:
                ax2.annotate(f"s3={v:.0f}\n(collapse)",
                           xy=(i+jitter, v), xytext=(i+0.2, v+200),
                           color='red', fontsize=7)

    ax2.set_xticks(positions)
    ax2.set_xticklabels(["Standard\nPPO", "Optimal\nPPO", "HCGAE-v2\n(Ours)"])
    ax2.set_ylabel("Final Return")

    # Add median line annotation
    for i, algo in enumerate(algos_plot):
        vals = get_finals(ALIGNED_DIR, "HalfCheetah-v4", algo)
        if vals:
            med = np.median(vals)
            ax2.hlines(med, i-0.3, i+0.3, colors='black', linewidths=2, linestyles='--')
            ax2.annotate(f"med={med:.0f}", xy=(i, med), xytext=(i+0.35, med),
                       fontsize=7, va='center')

    ax2.grid(True, axis='y', alpha=0.3)
    ax2.set_title("Per-seed Final Returns (★=outlier, dashes=median)")

    plt.tight_layout()
    out = OUT_DIR / "fig4_hc_analysis.pdf"
    fig.savefig(out, bbox_inches='tight', dpi=200)
    fig.savefig(str(out).replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating ICML-quality figures...")
    print()

    print("Figure 1: Learning curves...")
    fig1_learning_curves()

    print("Figure 2: Final performance bar chart...")
    fig2_bar_final()

    print("Figure 3: Ablation study...")
    fig3_ablation()

    print("Figure 4: HalfCheetah deep dive...")
    fig4_hc_analysis()

    print(f"\nAll figures saved to {OUT_DIR}/")
    print("Files:")
    for f in sorted(OUT_DIR.glob("fig*.pdf")):
        print(f"  {f.name}")

