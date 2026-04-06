#!/usr/bin/env python3
"""
ICML-Level Analysis of Aligned Experiment Results
====================================================
Computes:
  1. Mean ± SEM for each algorithm × environment
  2. Mann-Whitney U test (pairwise)
  3. Cohen's d effect size
  4. AULC (Area Under Learning Curve, normalized)
  5. Relative gain %
  6. Generates LaTeX table and plots

Usage:
  python3 scripts/analyze_aligned_results.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

# ─────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results/AlignedExperiment")
ENVS = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]
PRIMARY_ALGOS = ["Standard_PPO", "Optimal_PPO", "Optimal_HCGAE_v2"]
ABLATION_ALGOS = ["Optimal_HCGAE_v2_NoBdry", "Optimal_HCGAE_v2_NoGate"]
SEEDS = list(range(5))

ENV_SHORT = {
    "Hopper-v4": "Hopper",
    "Walker2d-v4": "Walker2d",
    "HalfCheetah-v4": "HalfCheetah",
}

ALGO_DISPLAY = {
    "Standard_PPO": "Standard PPO",
    "Optimal_PPO": "Optimal PPO",
    "Optimal_HCGAE_v2": "HCGAE-v2 (Ours)",
    "Optimal_HCGAE_v2_NoBdry": "HCGAE-v2 –BdryCorr",
    "Optimal_HCGAE_v2_NoGate": "HCGAE-v2 –EVGate",
}


# ─────────────────────────────────────────────────────────────────────────────
def load_data(results_dir=RESULTS_DIR, algos=None, envs=ENVS, seeds=SEEDS):
    """Load all available results into nested dict: data[env][algo][seed] = dict."""
    if algos is None:
        algos = PRIMARY_ALGOS + ABLATION_ALGOS
    data = {}
    for env in envs:
        data[env] = {}
        for algo in algos:
            data[env][algo] = {}
            for seed in seeds:
                fp = results_dir / env / algo / f"{algo}_s{seed}.json"
                if fp.exists():
                    try:
                        with open(fp) as f:
                            d = json.load(f)
                        data[env][algo][seed] = d
                    except Exception as e:
                        print(f"Warning: {fp}: {e}")
    return data


def get_final_returns(data, env, algo, metric="final_reward"):
    """Return list of final returns for all available seeds."""
    returns = []
    for seed, d in data[env].get(algo, {}).items():
        er = d.get("eval_rewards", [])
        if not er:
            continue
        if metric == "final_reward":
            val = float(np.mean(er[-5:])) if len(er) >= 5 else float(np.mean(er))
        elif metric == "max_reward":
            val = float(max(er))
        elif metric == "aulc":
            val = float(np.mean(er))  # normalized AULC = mean over all checkpoints
        else:
            val = d.get(metric, float(np.mean(er[-5:])))
        returns.append(val)
    return returns


def cohen_d(a, b):
    """Cohen's d effect size."""
    a, b = np.array(a), np.array(b)
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return (np.mean(a) - np.mean(b)) / (pooled_std + 1e-8)


def mann_whitney(a, b):
    """Mann-Whitney U test (one-sided: a > b)."""
    if len(a) < 2 or len(b) < 2:
        return float('nan'), float('nan')
    stat, pval = scipy_stats.mannwhitneyu(a, b, alternative='greater')
    return float(stat), float(pval)


def compute_aulc(eval_rewards, total_steps=500_000, n_points=48):
    """Compute normalized AULC from learning curve (mean over checkpoint returns)."""
    if not eval_rewards:
        return 0.0
    return float(np.mean(eval_rewards))


# ─────────────────────────────────────────────────────────────────────────────
def full_analysis(data, algos=None, envs=ENVS):
    if algos is None:
        algos = PRIMARY_ALGOS

    results = {}
    for env in envs:
        results[env] = {}
        for algo in algos:
            final5 = get_final_returns(data, env, algo, "final_reward")
            max_r = get_final_returns(data, env, algo, "max_reward")
            aulc_list = []
            for seed, d in data[env].get(algo, {}).items():
                er = d.get("eval_rewards", [])
                if er:
                    aulc_list.append(compute_aulc(er))
            n = len(final5)
            if n == 0:
                results[env][algo] = {"n": 0, "status": "missing"}
                continue
            results[env][algo] = {
                "n": n,
                "final5_mean": float(np.mean(final5)),
                "final5_std": float(np.std(final5)),
                "final5_sem": float(np.std(final5) / np.sqrt(n)),
                "final5_seeds": final5,
                "max_mean": float(np.mean(max_r)) if max_r else None,
                "max_std": float(np.std(max_r)) if max_r else None,
                "aulc_mean": float(np.mean(aulc_list)) if aulc_list else None,
                "aulc_std": float(np.std(aulc_list)) if aulc_list else None,
            }

    # Pairwise stats: HCGAE_v2 vs Optimal_PPO and Standard_PPO
    print("\n" + "="*80)
    print("  ALIGNED EXPERIMENT: ICML-Level Statistical Analysis")
    print("  Config: 256×256 MLP, obs_norm, adv_norm, deterministic eval")
    print("="*80)

    # Table 1: Primary comparison
    print(f"\n{'Algorithm':<28}", end="")
    for env in envs:
        print(f"  {ENV_SHORT[env]:>22}", end="")
    print()
    print("─"*80)

    for algo in algos:
        display = ALGO_DISPLAY.get(algo, algo)
        marker = "  ◄ OURS" if "HCGAE_v2" == algo.replace("Optimal_", "") else ""
        print(f"{display:<28}", end="")
        for env in envs:
            r = results[env].get(algo, {})
            if r.get("n", 0) == 0:
                print(f"  {'pending':>22}", end="")
            else:
                m, se, n = r["final5_mean"], r["final5_sem"], r["n"]
                print(f"  {m:>7.0f}±{se:>4.0f}(n={n})", end="")
        print(marker)

    # Relative gains
    print("\n  Relative gain: HCGAE-v2 vs Optimal PPO (final5 mean):")
    for env in envs:
        opt = results[env].get("Optimal_PPO", {})
        hcg = results[env].get("Optimal_HCGAE_v2", {})
        if opt.get("n", 0) > 0 and hcg.get("n", 0) > 0:
            gain = (hcg["final5_mean"] - opt["final5_mean"]) / (abs(opt["final5_mean"]) + 1e-8) * 100
            print(f"    {ENV_SHORT[env]:<15}: {gain:+.1f}%")

    print("\n  Relative gain: HCGAE-v2 vs Standard PPO (final5 mean):")
    for env in envs:
        std = results[env].get("Standard_PPO", {})
        hcg = results[env].get("Optimal_HCGAE_v2", {})
        if std.get("n", 0) > 0 and hcg.get("n", 0) > 0:
            gain = (hcg["final5_mean"] - std["final5_mean"]) / (abs(std["final5_mean"]) + 1e-8) * 100
            print(f"    {ENV_SHORT[env]:<15}: {gain:+.1f}%")

    # Statistical tests
    print("\n  Statistical Tests: HCGAE-v2 vs Optimal PPO")
    print(f"  {'Environment':<15}  {'U-stat':>8}  {'p-value':>10}  {'Cohen d':>8}  {'significant':>12}")
    print("  " + "─"*60)
    for env in envs:
        opt = results[env].get("Optimal_PPO", {})
        hcg = results[env].get("Optimal_HCGAE_v2", {})
        if opt.get("n", 0) > 1 and hcg.get("n", 0) > 1:
            a = hcg["final5_seeds"]
            b = opt["final5_seeds"]
            U, p = mann_whitney(a, b)
            d = cohen_d(a, b)
            sig = "✓ (p<0.05)" if p < 0.05 else "✗" if not np.isnan(p) else "N/A"
            print(f"  {ENV_SHORT[env]:<15}  {U:>8.1f}  {p:>10.4f}  {d:>8.2f}  {sig:>12}")

    print("\n  Statistical Tests: HCGAE-v2 vs Standard PPO")
    print(f"  {'Environment':<15}  {'U-stat':>8}  {'p-value':>10}  {'Cohen d':>8}  {'significant':>12}")
    print("  " + "─"*60)
    for env in envs:
        std = results[env].get("Standard_PPO", {})
        hcg = results[env].get("Optimal_HCGAE_v2", {})
        if std.get("n", 0) > 1 and hcg.get("n", 0) > 1:
            a = hcg["final5_seeds"]
            b = std["final5_seeds"]
            U, p = mann_whitney(a, b)
            d = cohen_d(a, b)
            sig = "✓ (p<0.05)" if p < 0.05 else "✗" if not np.isnan(p) else "N/A"
            print(f"  {ENV_SHORT[env]:<15}  {U:>8.1f}  {p:>10.4f}  {d:>8.2f}  {sig:>12}")

    # AULC comparison
    print("\n  AULC (Sample Efficiency): mean normalized area under learning curve")
    print(f"  {'Algorithm':<28}", end="")
    for env in envs:
        print(f"  {ENV_SHORT[env]:>15}", end="")
    print()
    print("  " + "─"*70)
    for algo in algos:
        display = ALGO_DISPLAY.get(algo, algo)
        print(f"  {display:<28}", end="")
        for env in envs:
            r = results[env].get(algo, {})
            if r.get("aulc_mean") is not None:
                print(f"  {r['aulc_mean']:>10.1f}±{r['aulc_std']:>4.0f}", end="")
            else:
                print(f"  {'N/A':>15}", end="")
        print()

    return results


def ablation_analysis(data, envs=ENVS):
    """Analyze ablation results."""
    ablation_algos = ["Optimal_HCGAE_v2", "Optimal_HCGAE_v2_NoBdry", "Optimal_HCGAE_v2_NoGate"]

    print("\n" + "="*80)
    print("  ABLATION STUDY: Component Contributions")
    print("="*80)

    for env in envs:
        print(f"\n  {ENV_SHORT[env]}:")
        print(f"  {'Component':<30}  {'Final Mean':>12}  {'±SEM':>8}  {'n':>4}  {'vs Full':>10}")
        print("  " + "─"*70)

        full = []
        full_r = data[env].get("Optimal_HCGAE_v2", {})
        for seed, d in full_r.items():
            er = d.get("eval_rewards", [])
            if er:
                full.append(float(np.mean(er[-5:])))

        for algo in ablation_algos:
            vals = []
            for seed, d in data[env].get(algo, {}).items():
                er = d.get("eval_rewards", [])
                if er:
                    vals.append(float(np.mean(er[-5:])))

            if not vals:
                print(f"  {ALGO_DISPLAY.get(algo, algo):<30}  {'pending':>12}")
                continue

            m, se, n = np.mean(vals), np.std(vals)/np.sqrt(len(vals)), len(vals)
            vs_full = ""
            if full and algo != "Optimal_HCGAE_v2":
                pct = (m - np.mean(full)) / (abs(np.mean(full)) + 1e-8) * 100
                vs_full = f"{pct:+.1f}%"
            print(f"  {ALGO_DISPLAY.get(algo, algo):<30}  {m:>12.1f}  {se:>8.1f}  {n:>4}  {vs_full:>10}")


def generate_latex_table(results, envs=ENVS, algos=None):
    """Generate LaTeX table for paper."""
    if algos is None:
        algos = PRIMARY_ALGOS

    lines = []
    lines.append("% Auto-generated LaTeX table")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Performance comparison on MuJoCo locomotion tasks (500K steps).")
    lines.append(r"All agents use 256$\times$256 MLP with observation normalization and")
    lines.append(r"per-minibatch advantage normalization. Mean$\pm$SEM over 5 seeds,")
    lines.append(r"deterministic evaluation.}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Hopper-v4 & Walker2d-v4 & HalfCheetah-v4 \\")
    lines.append(r"\midrule")

    for algo in algos:
        display = ALGO_DISPLAY.get(algo, algo)
        row = f"{display}"
        bold = "HCGAE_v2" in algo and "NoBdry" not in algo and "NoGate" not in algo
        for env in envs:
            r = results[env].get(algo, {})
            if r.get("n", 0) == 0:
                row += " & ---"
            else:
                m, se = r["final5_mean"], r["final5_sem"]
                if bold:
                    row += f" & $\\mathbf{{{m:.0f}\\pm{se:.0f}}}$"
                else:
                    row += f" & ${m:.0f}\\pm{se:.0f}$"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_plots(data, results, envs=ENVS, algos=None):
    """Generate publication-quality plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    if algos is None:
        algos = PRIMARY_ALGOS

    COLORS = {
        "Standard_PPO": "#7f7f7f",
        "Optimal_PPO": "#1f77b4",
        "Optimal_HCGAE_v2": "#d62728",
        "Optimal_HCGAE_v2_NoBdry": "#ff7f0e",
        "Optimal_HCGAE_v2_NoGate": "#2ca02c",
    }
    LINESTYLES = {
        "Standard_PPO": "--",
        "Optimal_PPO": "-.",
        "Optimal_HCGAE_v2": "-",
        "Optimal_HCGAE_v2_NoBdry": ":",
        "Optimal_HCGAE_v2_NoGate": ":",
    }

    outdir = Path("results/paper_figures_final")
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: Learning Curves (3 envs) ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Learning Curves: Aligned Experiment (256×256, obs_norm, adv_norm)",
                 fontsize=12, fontweight='bold')

    for ax, env in zip(axes, envs):
        for algo in algos:
            all_curves = []
            for seed, d in data[env].get(algo, {}).items():
                er = d.get("eval_rewards", [])
                es = d.get("eval_steps", [])
                if er and es:
                    all_curves.append((es, er))

            if not all_curves:
                continue

            # Interpolate to common grid
            max_steps = max(len(c[1]) for c in all_curves)
            common_len = min(max_steps, 48)
            interp_curves = []
            for es, er in all_curves:
                if len(er) >= common_len:
                    interp_curves.append(er[:common_len])
                elif len(er) > 0:
                    # pad with last value
                    padded = er + [er[-1]] * (common_len - len(er))
                    interp_curves.append(padded[:common_len])

            if not interp_curves:
                continue

            arr = np.array(interp_curves)
            x = np.linspace(0, 500_000, common_len)
            mean_c = np.mean(arr, axis=0)
            sem_c = np.std(arr, axis=0) / np.sqrt(len(arr))

            color = COLORS.get(algo, "#333333")
            ls = LINESTYLES.get(algo, "-")
            label = ALGO_DISPLAY.get(algo, algo)
            ax.plot(x, mean_c, color=color, linestyle=ls, linewidth=2, label=label)
            ax.fill_between(x, mean_c - sem_c, mean_c + sem_c, alpha=0.15, color=color)

        ax.set_title(ENV_SHORT[env], fontsize=11)
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Return" if ax == axes[0] else "")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(5, 5))

    plt.tight_layout()
    fig.savefig(outdir / "fig_aligned_learning_curves.pdf", bbox_inches='tight', dpi=150)
    fig.savefig(outdir / "fig_aligned_learning_curves.png", bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {outdir}/fig_aligned_learning_curves.pdf")

    # ── Figure 2: Bar Chart ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    n_envs = len(envs)
    n_algos = len(algos)
    width = 0.2
    x = np.arange(n_envs)

    for i, algo in enumerate(algos):
        means = []
        sems = []
        for env in envs:
            r = results[env].get(algo, {})
            means.append(r.get("final5_mean", 0))
            sems.append(r.get("final5_sem", 0))

        offset = (i - n_algos/2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=sems, capsize=4,
                      color=COLORS.get(algo, "#333333"),
                      label=ALGO_DISPLAY.get(algo, algo),
                      alpha=0.85, ecolor='black', linewidth=0.5,
                      edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels([ENV_SHORT[e] for e in envs])
    ax.set_ylabel("Final Performance (5-seed mean ± SEM)")
    ax.set_title("Final Performance Comparison (500K steps, 5 seeds)")
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(outdir / "fig_aligned_bar.pdf", bbox_inches='tight', dpi=150)
    fig.savefig(outdir / "fig_aligned_bar.png", bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {outdir}/fig_aligned_bar.pdf")

    print(f"\nAll figures saved to {outdir}/")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading data from {RESULTS_DIR}...")
    all_algos = PRIMARY_ALGOS + ABLATION_ALGOS
    data = load_data(algos=all_algos)

    # Count available data
    print("\nData availability:")
    for env in ENVS:
        for algo in all_algos:
            n = len(data[env].get(algo, {}))
            if n > 0:
                print(f"  {env}/{algo}: {n} seeds")

    # Primary analysis
    results = full_analysis(data, algos=PRIMARY_ALGOS)

    # Ablation analysis
    ablation_analysis(data)

    # LaTeX table
    print("\n" + "="*80)
    print("  LATEX TABLE:")
    print("="*80)
    latex = generate_latex_table(results)
    print(latex)

    # Save results
    outdir = Path("results/paper_figures_final")
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "aligned_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {outdir}/aligned_analysis.json")

    with open(outdir / "aligned_latex_table.tex", "w") as f:
        f.write(latex)
    print(f"Saved: {outdir}/aligned_latex_table.tex")

    # Plots
    print("\nGenerating plots...")
    generate_plots(data, results)


if __name__ == "__main__":
    main()

