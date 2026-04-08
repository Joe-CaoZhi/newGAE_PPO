#!/usr/bin/env python3
"""
Generate learning curve plots for ICMLExperiment data.
Creates Figure 5: 3-panel learning curves (Hopper, Walker2d, HalfCheetah)
Also generates sensitivity analysis bar charts.

Output: results/paper_figures_final/fig5_learning_curves.png
        results/paper_figures_final/fig4_sensitivity.png (if not already exists)
"""
import json

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ──────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────
ICML_DIR = Path("results/ICMLExperiment")
SENSITIVITY_DIR = Path("results/Sensitivity")
OUTPUT_DIR = Path("results/paper_figures_final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENVS = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]
ENV_LABELS = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]
ALGOS = ["Standard_PPO", "Optimal_PPO", "Optimal_HCGAE", "Optimal_HCGAE_SCR"]
ALGO_LABELS = ["Standard PPO", "Optimal PPO", "Optimal HCGAE (Ours)", "Optimal HCGAE-SCR"]
ALGO_COLORS = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]
ALGO_LINESTYLES = ["-", "--", "-", ":"]
SEEDS = [0, 1, 2, 3, 4]
TOTAL_STEPS = 500_000
EVAL_FREQ = 10_240

# ──────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────
def load_eval_curves(env, algo, seeds=SEEDS):
    """Load eval_rewards arrays for each seed. Returns list of arrays."""
    curves = []
    for s in seeds:
        path = ICML_DIR / env / algo / f"{algo}_s{s}.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        eval_rewards = data.get("eval_rewards", [])
        eval_steps = data.get("eval_steps", [])
        curves.append((eval_steps, eval_rewards))
    return curves


def interpolate_to_common_grid(curves, n_points=50):
    """Interpolate all seed curves to a common step grid."""
    # Common x-axis
    all_steps = [np.array(s) for s, r in curves if len(s) > 0]
    if not all_steps:
        return None, None
    max_step = min(s[-1] for s in all_steps)  # conservative: use min endpoint
    x_grid = np.linspace(0, max_step, n_points)

    interpolated = []
    for steps, rewards in curves:
        if len(steps) < 2:
            continue
        steps_arr = np.array(steps, dtype=float)
        rewards_arr = np.array(rewards, dtype=float)
        y_interp = np.interp(x_grid, steps_arr, rewards_arr)
        interpolated.append(y_interp)

    if not interpolated:
        return None, None

    return x_grid, np.array(interpolated)


# ──────────────────────────────────────────────────────────────────
# Figure 5: Learning Curves
# ──────────────────────────────────────────────────────────────────
def plot_learning_curves():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Learning Curves: Mean ± 1 Std (5 seeds, 500K steps)",
                 fontsize=13, fontweight='bold', y=1.02)

    for ax_idx, (env, env_label) in enumerate(zip(ENVS, ENV_LABELS)):
        ax = axes[ax_idx]

        for algo, label, color, ls in zip(ALGOS, ALGO_LABELS, ALGO_COLORS, ALGO_LINESTYLES):
            curves = load_eval_curves(env, algo)
            if not curves:
                print(f"  [WARN] No data for {env}/{algo}")
                continue

            x_grid, interp = interpolate_to_common_grid(curves, n_points=49)
            if x_grid is None:
                continue

            mean = np.mean(interp, axis=0)
            std = np.std(interp, axis=0)

            ax.plot(x_grid / 1000, mean, color=color, linestyle=ls,
                   linewidth=2.0, label=label)
            ax.fill_between(x_grid / 1000, mean - std, mean + std,
                           color=color, alpha=0.15)

            print(f"  {env}/{algo}: n={len(curves)}, final={mean[-1]:.1f}±{std[-1]:.1f}")

        ax.set_xlabel("Environment Steps (×1000)", fontsize=11)
        ax.set_ylabel("Episode Return" if ax_idx == 0 else "", fontsize=11)
        ax.set_title(env_label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(left=0)

        # Spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Shared legend below
    handles = [mpatches.Patch(color=c, label=l)
               for c, l in zip(ALGO_COLORS, ALGO_LABELS)]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               fontsize=10, bbox_to_anchor=(0.5, -0.08),
               frameon=True, edgecolor='gray')

    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig5_learning_curves.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved learning curves: {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────
# Figure 4: Sensitivity Analysis Bar Charts
# ──────────────────────────────────────────────────────────────────
def load_sensitivity_data():
    """Load sensitivity experiment JSON files."""
    beta_results = {}
    amax_results = {}

    beta_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    for b in beta_values:
        fname = f"HCGAE_beta{int(b)}_s42.json"
        path = SENSITIVITY_DIR / fname
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            # Get final reward
            if "final_reward" in data:
                beta_results[b] = data["final_reward"]
            elif "eval_rewards" in data:
                ev = data["eval_rewards"]
                beta_results[b] = float(np.mean(ev[-5:])) if len(ev) >= 5 else float(np.mean(ev))

    amax_values = [0.3, 0.5, 0.7, 0.9]
    for a in amax_values:
        fname = f"HCGAE_amax{a}_s42.json"
        path = SENSITIVITY_DIR / fname
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            if "final_reward" in data:
                amax_results[a] = data["final_reward"]
            elif "eval_rewards" in data:
                ev = data["eval_rewards"]
                amax_results[a] = float(np.mean(ev[-5:])) if len(ev) >= 5 else float(np.mean(ev))

    return beta_results, amax_results


def plot_sensitivity():
    beta_results, amax_results = load_sensitivity_data()

    if not beta_results and not amax_results:
        print("  [WARN] No sensitivity data found. Skipping sensitivity plot.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("HCGAE Hyperparameter Sensitivity (Hopper-v4, seed=42, 300K steps)",
                 fontsize=13, fontweight='bold', y=1.02)

    # Panel 1: β sensitivity
    ax = axes[0]
    if beta_results:
        betas = sorted(beta_results.keys())
        rewards = [beta_results[b] for b in betas]
        colors = ['#d62728' if b == 3.0 else '#aec7e8' for b in betas]
        bars = ax.bar([str(b) for b in betas], rewards, color=colors,
                     edgecolor='black', linewidth=0.8)
        ax.axhline(y=beta_results.get(3.0, 0), color='red', linestyle='--',
                  alpha=0.5, linewidth=1.5)

        # Annotate best
        for bar, val in zip(bars, rewards):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                   f'{val:.0f}', ha='center', va='bottom', fontsize=9)

        ax.set_xlabel("β (sigmoid sharpness)", fontsize=11)
        ax.set_ylabel("Final Return", fontsize=11)
        ax.set_title("β Sensitivity (α_max=0.7 fixed)", fontsize=12)
        ax.set_ylim(bottom=0, top=max(rewards) * 1.15)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Legend patch
        best_patch = mpatches.Patch(color='#d62728', label='Default (β=3.0)')
        other_patch = mpatches.Patch(color='#aec7e8', label='Other values')
        ax.legend(handles=[best_patch, other_patch], fontsize=9)

    # Panel 2: α_max sensitivity
    ax = axes[1]
    if amax_results:
        amaxs = sorted(amax_results.keys())
        rewards = [amax_results[a] for a in amaxs]
        colors = ['#d62728' if a == 0.7 else '#aec7e8' for a in amaxs]
        bars = ax.bar([str(a) for a in amaxs], rewards, color=colors,
                     edgecolor='black', linewidth=0.8)
        ax.axhline(y=amax_results.get(0.7, 0), color='red', linestyle='--',
                  alpha=0.5, linewidth=1.5)

        for bar, val in zip(bars, rewards):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                   f'{val:.0f}', ha='center', va='bottom', fontsize=9)

        ax.set_xlabel("α_max (correction ceiling)", fontsize=11)
        ax.set_ylabel("Final Return", fontsize=11)
        ax.set_title("α_max Sensitivity (β=3.0 fixed)", fontsize=12)
        ax.set_ylim(bottom=0, top=max(rewards) * 1.15)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        best_patch = mpatches.Patch(color='#d62728', label='Default (α_max=0.7)')
        other_patch = mpatches.Patch(color='#aec7e8', label='Other values')
        ax.legend(handles=[best_patch, other_patch], fontsize=9)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig4_sensitivity.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved sensitivity: {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────
# Additional: Learning curve summary statistics
# ──────────────────────────────────────────────────────────────────
def compute_sample_efficiency_stats():
    """
    Compute sample efficiency metrics:
    - Steps to reach 90% of final performance
    - Area Under the Learning Curve (AULC)
    """
    print("\n=== Sample Efficiency Analysis ===")
    results = {}

    for env in ENVS:
        results[env] = {}
        final_means = {}

        # First pass: compute final performance for each algo
        for algo in ALGOS:
            curves = load_eval_curves(env, algo)
            if not curves:
                continue
            x_grid, interp = interpolate_to_common_grid(curves, n_points=49)
            if x_grid is None:
                continue
            mean = np.mean(interp, axis=0)
            final_means[algo] = mean[-1]

        # Reference: max final performance in this env
        if not final_means:
            continue
        ref_perf = max(final_means.values())

        print(f"\n  {env}:")
        for algo in ALGOS:
            curves = load_eval_curves(env, algo)
            if not curves:
                continue
            x_grid, interp = interpolate_to_common_grid(curves, n_points=49)
            if x_grid is None:
                continue
            mean = np.mean(interp, axis=0)
            std = np.std(interp, axis=0)

            # Area Under Learning Curve (normalized by max possible)
            aulc = np.trapz(mean, x_grid) / (x_grid[-1] - x_grid[0])

            # Steps to 50% of final
            target_50 = 0.5 * mean[-1]
            steps_50 = None
            for i, (step, val) in enumerate(zip(x_grid, mean)):
                if val >= target_50:
                    steps_50 = step
                    break

            results[env][algo] = {
                "final_mean": float(mean[-1]),
                "final_std": float(std[-1]),
                "aulc": float(aulc),
                "steps_to_50pct": int(steps_50) if steps_50 else None,
            }

            label = algo.replace("_", " ").replace("Optimal ", "")
            print(f"    {label:20s}: final={mean[-1]:6.0f}±{std[-1]:4.0f}, "
                  f"AULC={aulc:6.0f}, steps_50%={steps_50/1000:.0f}K"
                  if steps_50 else f"    {label:20s}: final={mean[-1]:6.0f}±{std[-1]:4.0f}, AULC={aulc:6.0f}")

    # Save
    out_path = OUTPUT_DIR / "sample_efficiency_stats.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved sample efficiency stats: {out_path}")
    return results


# ──────────────────────────────────────────────────────────────────
# Figure: EV/SNR diagnostic trajectories (from seed 0 data)
# ──────────────────────────────────────────────────────────────────
def check_data_availability():
    """Quick check of available data."""
    print("\n=== Data Availability ===")
    for env in ENVS:
        for algo in ALGOS:
            count = 0
            for s in SEEDS:
                path = ICML_DIR / env / algo / f"{algo}_s{s}.json"
                if path.exists():
                    count += 1
            print(f"  {env}/{algo}: {count}/5")

    # Check Ant-v4
    ant_dir = ICML_DIR / "Ant-v4"
    if ant_dir.exists():
        print(f"\n  Ant-v4:")
        for algo in ALGOS:
            count = sum(1 for s in SEEDS
                       if (ant_dir / algo / f"{algo}_s{s}.json").exists())
            print(f"    {algo}: {count}/5")


if __name__ == "__main__":
    print("=== Generating ICML Figures ===\n")

    # Check data
    check_data_availability()

    # Generate learning curves
    print("\n--- Generating Learning Curves (Figure 5) ---")
    plot_learning_curves()

    # Generate sensitivity plots
    print("\n--- Generating Sensitivity Plots (Figure 4) ---")
    plot_sensitivity()

    # Compute sample efficiency stats
    se_stats = compute_sample_efficiency_stats()

    print("\n=== Done ===")
    print(f"Outputs in: {OUTPUT_DIR}/")

