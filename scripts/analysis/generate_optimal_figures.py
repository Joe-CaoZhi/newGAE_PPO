#!/usr/bin/env python3
"""
Generate paper figures using Optimal PPO baseline (ICMLExperiment data).

This script generates:
  1. fig1_optimal_learning_curves.pdf/png  - Learning curves (3 envs, 4 methods)
  2. fig2_optimal_bar.pdf/png             - Bar comparison (final performance)
  3. fig3_autoscr_comparison.pdf/png      - AutoSCR vs v2 vs Optimal PPO (once data ready)
  4. fig_sample_efficiency.json           - AULC statistics

Data sources:
  - results/ICMLExperiment/               (main: 5 seeds x 500K steps, Optimal PPO baseline)
  - results/AutoSCRExperiment/            (new: AutoSCR experiment)
  - results/HighPowerExperiment/          (new: 30 seeds for power analysis)

All non-ablation/theory figures use Optimal PPO as baseline.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────
ICML_ROOT = Path("results/ICMLExperiment")
AUTOSCR_ROOT = Path("results/AutoSCRExperiment")
HIGHPOWER_ROOT = Path("results/HighPowerExperiment")
OUT_DIR = Path("results/paper_figures_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENVS = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]
ENV_SHORT = {"Hopper-v4": "Hopper-v4", "Walker2d-v4": "Walker2d-v4", "HalfCheetah-v4": "HalfCheetah-v4"}

# Main algorithms for paper figures (NON-ablation)
MAIN_ALGOS = [
    "Standard_PPO",
    "Optimal_PPO",
    "Optimal_HCGAE_v2",
    "Optimal_HCGAE_SCR",
]
ALGO_LABELS = {
    "Standard_PPO": "Standard PPO",
    "Optimal_PPO": "Optimal PPO (baseline)",
    "Optimal_HCGAE": "Optimal HCGAE v1",
    "Optimal_HCGAE_v2": "Optimal HCGAE v2 (Ours)",
    "Optimal_HCGAE_SCR": "Optimal HCGAE-SCR (Ours)",
    "Optimal_HCGAE_v2_AutoSCR": "Optimal HCGAE v2+AutoSCR (FW)",
}
ALGO_COLORS = {
    "Standard_PPO": "#6B7280",          # Gray
    "Optimal_PPO": "#F59E0B",           # Amber
    "Optimal_HCGAE": "#3B82F6",         # Blue (v1)
    "Optimal_HCGAE_v2": "#EF4444",      # Red (v2, main)
    "Optimal_HCGAE_SCR": "#8B5CF6",     # Purple (SCR)
    "Optimal_HCGAE_v2_AutoSCR": "#10B981",  # Green (AutoSCR, future)
}
ALGO_LINES = {
    "Standard_PPO": "-",
    "Optimal_PPO": "--",
    "Optimal_HCGAE": "-.",
    "Optimal_HCGAE_v2": "-",
    "Optimal_HCGAE_SCR": ":",
    "Optimal_HCGAE_v2_AutoSCR": "-.",
}

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 200,
    'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
})


# ── Data Loading ──────────────────────────────────────────────────────────
def load_icml_seeds(env_name: str, algo_name: str, root: Path = ICML_ROOT) -> list:
    """Load all seed JSON files for a given (env, algo) from ICMLExperiment."""
    algo_dir = root / env_name / algo_name
    if not algo_dir.exists():
        return []
    files = sorted(algo_dir.glob(f"{algo_name}_s*.json"))
    if not files:
        files = sorted(algo_dir.glob("*.json"))
    seeds = []
    for f in files:
        try:
            with open(f) as fp:
                seeds.append(json.load(fp))
        except Exception:
            pass
    return seeds


def get_eval_curves(seeds_data: list) -> tuple:
    """Extract (steps_list, returns_list) from seed data."""
    all_steps, all_rets = [], []
    for d in seeds_data:
        steps = d.get("eval_steps", [])
        rets = d.get("eval_rewards", [])
        if steps and rets and len(steps) == len(rets):
            all_steps.append(np.array(steps, dtype=float))
            all_rets.append(np.array(rets, dtype=float))
    return all_steps, all_rets


def interpolate_curves(all_steps, all_rets, n_points=50):
    """Interpolate all seed curves to common step grid."""
    if not all_steps:
        return None, None, None, None
    max_step = max(s[-1] for s in all_steps)
    min_step = min(s[0] for s in all_steps)
    common = np.linspace(min_step, max_step, n_points)
    interp = []
    for steps, rets in zip(all_steps, all_rets):
        interp.append(np.interp(common, steps, rets))
    mat = np.array(interp)
    return common, mat.mean(0), mat.std(0), mat


def get_final_means(seeds_data: list, n_tail=5) -> list:
    """Get final performance (mean of last n_tail evaluations) per seed."""
    finals = []
    for d in seeds_data:
        er = d.get("eval_rewards", [])
        if er:
            finals.append(float(np.mean(er[-n_tail:]) if len(er) >= n_tail else np.mean(er)))
    return finals


# ── Figure 1: Learning Curves (Main, Optimal PPO baseline) ───────────────
def plot_learning_curves_optimal(algos=None, save_prefix="fig5_optimal"):
    """3-panel learning curves using ICMLExperiment data (Optimal PPO baseline)."""
    if algos is None:
        algos = ["Standard_PPO", "Optimal_PPO", "Optimal_HCGAE_v2", "Optimal_HCGAE_SCR"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Learning Curves: Optimal PPO Baseline (5 seeds, 500K steps)", fontsize=13, y=1.02)

    for ax, env_name in zip(axes, ENVS):
        for algo in algos:
            seeds_data = load_icml_seeds(env_name, algo)
            if not seeds_data:
                continue
            all_steps, all_rets = get_eval_curves(seeds_data)
            if not all_steps:
                continue
            common, mean_r, std_r, _ = interpolate_curves(all_steps, all_rets)
            color = ALGO_COLORS.get(algo, "#333333")
            ls = ALGO_LINES.get(algo, "-")
            label = ALGO_LABELS.get(algo, algo)
            ax.plot(common / 1000, mean_r, color=color, linestyle=ls,
                    linewidth=2.0, label=label, zorder=3)
            ax.fill_between(common / 1000, mean_r - std_r, mean_r + std_r,
                            alpha=0.15, color=color, zorder=2)

        ax.set_title(env_name.split('-')[0], fontsize=12)
        ax.set_xlabel("Steps (×1000)", fontsize=11)
        ax.set_ylabel("Mean Eval Return", fontsize=11)
        ax.set_xlim(0, 510)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(algos),
               bbox_to_anchor=(0.5, -0.08), fontsize=10)
    plt.tight_layout()

    for ext in ['pdf', 'png']:
        path = OUT_DIR / f"{save_prefix}_learning_curves.{ext}"
        plt.savefig(str(path), bbox_inches='tight', dpi=200)
        print(f"  Saved: {path}")
    plt.close()


# ── Figure 2: Bar Chart (Main Results, Optimal PPO baseline) ─────────────
def plot_bar_optimal(algos=None, save_prefix="fig6_optimal"):
    """Grouped bar chart of final performance (Optimal PPO baseline)."""
    if algos is None:
        algos = ["Standard_PPO", "Optimal_PPO", "Optimal_HCGAE_v2", "Optimal_HCGAE_SCR"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Final Performance: Optimal PPO Baseline (5 seeds, 500K steps)", fontsize=13)

    x = np.arange(len(algos))
    width = 0.65

    for ax, env_name in zip(axes, ENVS):
        means, stds, colors = [], [], []
        for algo in algos:
            seeds_data = load_icml_seeds(env_name, algo)
            finals = get_final_means(seeds_data)
            means.append(float(np.mean(finals)) if finals else 0.0)
            stds.append(float(np.std(finals)) if len(finals) > 1 else 0.0)
            colors.append(ALGO_COLORS.get(algo, "#888"))

        bars = ax.bar(x, means, width, color=colors, alpha=0.85, zorder=3,
                      edgecolor='white', linewidth=0.5)
        ax.errorbar(x, means, yerr=stds, fmt='none', color='black',
                    capsize=4, linewidth=1.5, zorder=4)

        ax.set_title(env_name.split('-')[0], fontsize=12)
        ax.set_ylabel("Final Eval Return", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([ALGO_LABELS.get(a, a).replace(" ", "\n").replace("(baseline)", "").replace("(Ours)", "").strip()
                            for a in algos], fontsize=8, rotation=15, ha='right')

        # Annotate % vs Optimal PPO
        opt_mean = means[algos.index("Optimal_PPO")] if "Optimal_PPO" in algos else 0
        for i, (bar, m) in enumerate(zip(bars, means)):
            if algos[i] not in ("Optimal_PPO", "Standard_PPO") and opt_mean > 0:
                pct = (m - opt_mean) / abs(opt_mean) * 100
                ax.text(bar.get_x() + bar.get_width()/2, m + stds[i] + max(means)*0.01,
                        f"{pct:+.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold',
                        color='#22C55E' if pct >= 0 else '#EF4444')

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        path = OUT_DIR / f"{save_prefix}_bar.{ext}"
        plt.savefig(str(path), bbox_inches='tight', dpi=200)
        print(f"  Saved: {path}")
    plt.close()


# ── Figure 3: AutoSCR Comparison ─────────────────────────────────────────
def plot_autoscr_comparison():
    """Compare Optimal_PPO / HCGAE_v2 / HCGAE_v2_AutoSCR (if data exists)."""
    algos = ["Optimal_PPO", "Optimal_HCGAE_v2", "Optimal_HCGAE_v2_AutoSCR"]
    has_data = any(
        (AUTOSCR_ROOT / env / algo).exists()
        for env in ENVS for algo in algos if algo == "Optimal_HCGAE_v2_AutoSCR"
    )
    if not has_data:
        print("  [SKIP] AutoSCR data not yet available")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("AutoSCR Mode Detector vs HCGAE v2 (5 seeds, 500K steps)", fontsize=13)

    for ax, env_name in zip(axes, ENVS):
        for algo in algos:
            # Try AutoSCR dir first, then ICML dir
            seeds_data = load_icml_seeds(env_name, algo, AUTOSCR_ROOT)
            if not seeds_data:
                seeds_data = load_icml_seeds(env_name, algo, ICML_ROOT)
            if not seeds_data:
                continue

            all_steps, all_rets = get_eval_curves(seeds_data)
            if not all_steps:
                continue
            common, mean_r, std_r, _ = interpolate_curves(all_steps, all_rets)
            color = ALGO_COLORS.get(algo, "#333")
            ls = ALGO_LINES.get(algo, "-")
            label = ALGO_LABELS.get(algo, algo)
            ax.plot(common / 1000, mean_r, color=color, linestyle=ls,
                    linewidth=2.0, label=label, zorder=3)
            ax.fill_between(common / 1000, mean_r - std_r, mean_r + std_r,
                            alpha=0.15, color=color, zorder=2)

        ax.set_title(env_name.split('-')[0], fontsize=12)
        ax.set_xlabel("Steps (×1000)")
        ax.set_ylabel("Mean Eval Return")
        ax.set_xlim(0, 510)

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    fig.legend(handles, labels, loc='lower center', ncol=len(algos),
               bbox_to_anchor=(0.5, -0.08), fontsize=10)
    plt.tight_layout()

    for ext in ['pdf', 'png']:
        path = OUT_DIR / f"fig_autoscr_comparison.{ext}"
        plt.savefig(str(path), bbox_inches='tight', dpi=200)
        print(f"  Saved: {path}")
    plt.close()


# ── Figure: High-Power Boxplot ────────────────────────────────────────────
def plot_highpower_boxplot():
    """Boxplot with n=30 seeds (if data exists)."""
    if not HIGHPOWER_ROOT.exists():
        print("  [SKIP] HighPower data not yet available")
        return

    algos = ["Standard_PPO", "Optimal_PPO", "Optimal_HCGAE_v2"]
    algos_avail = [a for a in algos if (HIGHPOWER_ROOT / "Hopper-v4" / a).exists()]
    if not algos_avail:
        print("  [SKIP] HighPower runs not yet complete")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Statistical Power: n=30 seeds, 300K steps (High-Power Experiment)", fontsize=13)

    for ax, env_name in zip(axes, ENVS):
        all_data = []
        all_labels = []
        colors = []
        for algo in algos_avail:
            seeds_data = load_icml_seeds(env_name, algo, HIGHPOWER_ROOT)
            finals = get_final_means(seeds_data)
            if finals:
                all_data.append(finals)
                all_labels.append(ALGO_LABELS.get(algo, algo).replace(" ", "\n").replace("(baseline)", "").strip())
                colors.append(ALGO_COLORS.get(algo, "#888"))

        if not all_data:
            ax.set_title(f"{env_name} (no data)")
            continue

        bp = ax.boxplot(all_data, patch_artist=True, widths=0.5,
                        medianprops={'color': 'black', 'linewidth': 2})
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticklabels(all_labels, fontsize=9)
        ax.set_title(env_name.split('-')[0])
        ax.set_ylabel("Final Eval Return (300K steps)")

        # Jitter scatter overlay
        for i, (data, color) in enumerate(zip(all_data, colors)):
            jitter = np.random.default_rng(42+i).uniform(-0.15, 0.15, len(data))
            ax.scatter([i+1] * len(data) + jitter, data, color=color,
                       alpha=0.5, s=20, zorder=4)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        path = OUT_DIR / f"fig_highpower_boxplot.{ext}"
        plt.savefig(str(path), bbox_inches='tight', dpi=200)
        print(f"  Saved: {path}")
    plt.close()


# ── Sample Efficiency Statistics ──────────────────────────────────────────
def compute_sample_efficiency():
    """Compute AULC (Area Under Learning Curve) for all main algorithms."""
    algos = ["Standard_PPO", "Optimal_PPO", "Optimal_HCGAE_v2", "Optimal_HCGAE_SCR"]
    stats = {}
    for env_name in ENVS:
        stats[env_name] = {}
        for algo in algos:
            seeds_data = load_icml_seeds(env_name, algo)
            if not seeds_data:
                continue
            all_steps, all_rets = get_eval_curves(seeds_data)
            if not all_steps:
                continue
            # AULC per seed = mean of eval_rewards (uniform eval spacing)
            aulcs = []
            for rets in all_rets:
                aulcs.append(float(np.mean(rets)))
            finals = get_final_means(seeds_data)
            stats[env_name][algo] = {
                "aulc_mean": float(np.mean(aulcs)),
                "aulc_std": float(np.std(aulcs)),
                "aulc_sem": float(np.std(aulcs) / max(len(aulcs)**0.5, 1)),
                "final_mean": float(np.mean(finals)),
                "final_std": float(np.std(finals)),
                "final_sem": float(np.std(finals) / max(len(finals)**0.5, 1)),
                "n_seeds": len(finals),
                "seeds": finals,
            }
    path = OUT_DIR / "sample_efficiency_stats.json"
    with open(path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: {path}")
    return stats


# ── Print Summary Table ───────────────────────────────────────────────────
def print_summary_table():
    """Print formatted summary table of main results."""
    algos = ["Standard_PPO", "Optimal_PPO", "Optimal_HCGAE_v2", "Optimal_HCGAE_SCR"]

    print(f"\n{'Algorithm':<35}", end="")
    for env in ENVS:
        print(f"  {env.split('-')[0]:>20}", end="")
    print()
    print("─" * (35 + 22 * len(ENVS)))

    opt_means = {}
    for env_name in ENVS:
        d = load_icml_seeds(env_name, "Optimal_PPO")
        finals = get_final_means(d)
        opt_means[env_name] = float(np.mean(finals)) if finals else 1.0

    for algo in algos:
        label = ALGO_LABELS.get(algo, algo)
        print(f"  {label:<33}", end="")
        for env_name in ENVS:
            seeds_data = load_icml_seeds(env_name, algo)
            finals = get_final_means(seeds_data)
            if finals:
                m = float(np.mean(finals))
                s = float(np.std(finals))
                pct = (m - opt_means[env_name]) / abs(opt_means[env_name]) * 100
                marker = " ✅" if algo not in ("Standard_PPO", "Optimal_PPO") and pct >= 0 else ""
                print(f"  {m:7.0f} ± {s:5.0f} ({pct:+.1f}%){marker}", end="")
            else:
                print(f"  {'—':>20}", end="")
        print()
    print()


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--no-lcurves", action="store_true")
    parser.add_argument("--no-bars", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Generating Optimal PPO Baseline Figures")
    print(f"  Data: {ICML_ROOT}")
    print(f"  Output: {OUT_DIR}")
    print(f"{'='*60}\n")

    # Print summary table first
    print("  == Main Results Table ==")
    print_summary_table()

    if not args.summary_only:
        if not args.no_lcurves:
            print("  [1/5] Learning curves (Optimal PPO baseline)...")
            plot_learning_curves_optimal()

        if not args.no_bars:
            print("  [2/5] Bar chart (Optimal PPO baseline)...")
            plot_bar_optimal()

        print("  [3/5] AutoSCR comparison (if data available)...")
        plot_autoscr_comparison()

        print("  [4/5] High-power boxplot (if data available)...")
        plot_highpower_boxplot()

        print("  [5/5] Sample efficiency statistics...")
        eff = compute_sample_efficiency()

        print(f"\n  == Sample Efficiency (AULC) ==")
        for env_name in ENVS:
            print(f"  {env_name}:")
            for algo in ["Standard_PPO", "Optimal_PPO", "Optimal_HCGAE_v2", "Optimal_HCGAE_SCR"]:
                d = eff.get(env_name, {}).get(algo)
                if d:
                    print(f"    {ALGO_LABELS.get(algo, algo):<40} AULC={d['aulc_mean']:7.1f} ± {d['aulc_std']:5.1f}")

    print(f"\n  Figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()

