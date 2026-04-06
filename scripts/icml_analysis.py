#!/usr/bin/env python3
"""
ICML-Level Statistical Analysis for HCGAE v2 Aligned Experiment
================================================================

This script:
1. Reads results from results/AlignedExperiment/
2. Computes mean ± SEM across seeds for each (env, algo)
3. Runs Mann-Whitney U tests for significance
4. Computes Cohen's d effect size
5. Produces AULC (Area Under Learning Curve) comparison
6. Generates LaTeX-ready table
7. Produces publication-quality figures

Usage:
  python3 scripts/icml_analysis.py
  python3 scripts/icml_analysis.py --results results/AlignedExperiment --out results/paper_figures_final
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

ENVS = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]
ALGOS = ["Standard_PPO", "Optimal_PPO", "Optimal_HCGAE_v2"]
SEEDS = list(range(5))


def load_results(results_dir: Path):
    """Load all seed results for each (env, algo)."""
    data = {}
    for env in ENVS:
        data[env] = {}
        for algo in ALGOS:
            seeds_data = []
            for seed in SEEDS:
                fp = results_dir / env / algo / f"{algo}_s{seed}.json"
                if fp.exists():
                    with open(fp) as f:
                        d = json.load(f)
                    seeds_data.append(d)
            data[env][algo] = seeds_data
    return data


def extract_metrics(data, envs=ENVS, algos=ALGOS):
    """Extract final5, max, best10, aulc for each (env, algo, seed)."""
    metrics = {}
    for env in envs:
        metrics[env] = {}
        for algo in algos:
            seeds = data.get(env, {}).get(algo, [])
            final5, maxr, best10, aulc = [], [], [], []
            for d in seeds:
                er = d.get("eval_rewards", [])
                if not er:
                    continue
                final5.append(float(np.mean(er[-5:])) if len(er) >= 5 else float(np.mean(er)))
                maxr.append(float(max(er)))
                best10.append(float(np.mean(sorted(er)[-10:])) if len(er) >= 10 else float(np.mean(er)))
                # AULC: normalized area under learning curve (trapezoid, normalized by steps)
                es = d.get("eval_steps", list(range(len(er))))
                total_steps = d.get("total_steps", es[-1] if es else 1)
                if len(er) >= 2:
                    area = np.trapz(er, es) / (total_steps + 1e-8)
                else:
                    area = float(np.mean(er))
                aulc.append(area)
            metrics[env][algo] = {
                "final5": np.array(final5),
                "max": np.array(maxr),
                "best10": np.array(best10),
                "aulc": np.array(aulc),
                "n": len(final5),
            }
    return metrics


def cohen_d(a, b):
    """Cohen's d = (mean_a - mean_b) / pooled_std."""
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float('nan')
    s_pooled = np.sqrt(((n_a - 1) * np.std(a, ddof=1)**2 + (n_b - 1) * np.std(b, ddof=1)**2) / (n_a + n_b - 2))
    if s_pooled < 1e-10:
        return float('nan')
    return float(np.mean(a) - np.mean(b)) / s_pooled


def statistical_tests(metrics, envs=ENVS):
    """Mann-Whitney U test + Cohen's d for HCGAE_v2 vs Optimal_PPO."""
    tests = {}
    for env in envs:
        hcg = metrics[env].get("Optimal_HCGAE_v2", {})
        opt = metrics[env].get("Optimal_PPO", {})
        std = metrics[env].get("Standard_PPO", {})
        tests[env] = {}

        for metric_name in ["final5", "best10", "aulc"]:
            a = hcg.get(metric_name, np.array([]))
            b = opt.get(metric_name, np.array([]))
            b2 = std.get(metric_name, np.array([]))

            entry = {}
            if len(a) >= 3 and len(b) >= 3:
                stat, p = scipy_stats.mannwhitneyu(a, b, alternative='two-sided')
                d = cohen_d(a, b)
                gain_pct = (np.mean(a) - np.mean(b)) / (abs(np.mean(b)) + 1e-8) * 100
                entry["vs_optimal"] = {
                    "p_value": float(p), "cohen_d": float(d),
                    "gain_pct": float(gain_pct), "n_hcg": len(a), "n_opt": len(b),
                    "mean_hcg": float(np.mean(a)), "mean_opt": float(np.mean(b)),
                    "sem_hcg": float(np.std(a)/np.sqrt(len(a))),
                    "sem_opt": float(np.std(b)/np.sqrt(len(b))),
                }
            if len(a) >= 3 and len(b2) >= 3:
                stat2, p2 = scipy_stats.mannwhitneyu(a, b2, alternative='two-sided')
                d2 = cohen_d(a, b2)
                gain2 = (np.mean(a) - np.mean(b2)) / (abs(np.mean(b2)) + 1e-8) * 100
                entry["vs_standard"] = {
                    "p_value": float(p2), "cohen_d": float(d2),
                    "gain_pct": float(gain2), "n_hcg": len(a), "n_std": len(b2),
                    "mean_hcg": float(np.mean(a)), "mean_std": float(np.mean(b2)),
                }
            tests[env][metric_name] = entry
    return tests


def print_summary_table(metrics, tests, envs=ENVS):
    print("\n" + "="*100)
    print("  ALIGNED EXPERIMENT RESULTS: HCGAE v2 vs Optimal PPO vs Standard PPO")
    print("  Configuration: 256×256 MLP, obs_norm, adv_norm, lr_anneal (Optimal/HCGAE only)")
    print("  Evaluation: deterministic (policy mean), 10 episodes, every 10240 steps")
    print("="*100)

    for metric_name, metric_label in [("final5", "Final-5 Mean"), ("best10", "Best-10 Mean"), ("aulc", "AULC")]:
        print(f"\n  [{metric_label}]")
        header = f"    {'Algorithm':<25}"
        for env in envs:
            short = env.split('-')[0]
            header += f"  {short:>20}"
        print(header)
        print("    " + "─"*90)

        for algo in ALGOS:
            row = f"    {algo:<25}"
            for env in envs:
                m_data = metrics[env].get(algo, {}).get(metric_name, np.array([]))
                if len(m_data) > 0:
                    m, s, n = float(np.mean(m_data)), float(np.std(m_data)/np.sqrt(len(m_data))), len(m_data)
                    row += f"  {m:>8.1f}±{s:>5.1f}(n={n})"
                else:
                    row += f"  {'---':>20}"
            marker = " ◄ OURS" if "HCGAE" in algo else ""
            print(row + marker)

    print("\n  [STATISTICAL TESTS: HCGAE_v2 vs Optimal_PPO, metric=final5]")
    header2 = f"    {'Environment':<20}  {'Gain%':>8}  {'p-value':>10}  {'Cohen d':>9}  {'Significant':>12}"
    print(header2)
    print("    " + "─"*70)
    for env in envs:
        short = env.split('-')[0]
        t = tests.get(env, {}).get("final5", {}).get("vs_optimal", {})
        if t:
            p = t['p_value']
            d = t['cohen_d']
            gain = t['gain_pct']
            sig = "✓ (p<0.05)" if p < 0.05 else ("~ (p<0.10)" if p < 0.10 else "✗")
            print(f"    {short:<20}  {gain:>+8.1f}%  {p:>10.4f}  {d:>+9.2f}  {sig:>12}")
        else:
            print(f"    {short:<20}  (insufficient data)")

    print("\n  [AULC TESTS: HCGAE_v2 vs Optimal_PPO]")
    header3 = f"    {'Environment':<20}  {'AULC Gain%':>10}  {'p-value':>10}  {'Cohen d':>9}"
    print(header3)
    print("    " + "─"*60)
    for env in envs:
        short = env.split('-')[0]
        t = tests.get(env, {}).get("aulc", {}).get("vs_optimal", {})
        if t:
            print(f"    {short:<20}  {t['gain_pct']:>+10.1f}%  {t['p_value']:>10.4f}  {t['cohen_d']:>+9.2f}")
        else:
            print(f"    {short:<20}  (insufficient data)")


def generate_latex_table(metrics, tests, envs=ENVS):
    """Generate LaTeX-formatted performance table."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{HCGAE-v2 vs Optimal PPO vs Standard PPO on aligned configuration.")
    lines.append(r"         All agents: 256×256 MLP, obs-norm, adv-norm; Optimal/HCGAE additionally use LR annealing.")
    lines.append(r"         Scores = mean±SEM over 5 seeds, eval with deterministic policy mean.}")
    lines.append(r"\label{tab:aligned_results}")
    lines.append(r"\begin{tabular}{l" + "c" * len(envs) + "}")
    lines.append(r"\toprule")

    env_header = " & ".join(e.split('-')[0] for e in envs)
    lines.append(f"Algorithm & {env_header} \\\\")
    lines.append(r"\midrule")

    algo_display = {
        "Standard_PPO": r"\textsc{Standard-PPO}",
        "Optimal_PPO": r"\textsc{Optimal-PPO}",
        "Optimal_HCGAE_v2": r"\textsc{HCGAE-v2} (ours)",
    }

    for algo in ALGOS:
        row_parts = [algo_display.get(algo, algo)]
        for env in envs:
            m_data = metrics[env].get(algo, {}).get("final5", np.array([]))
            if len(m_data) > 0:
                m = float(np.mean(m_data))
                s = float(np.std(m_data) / np.sqrt(len(m_data)))
                row_parts.append(f"${m:.0f}\\pm{s:.0f}$")
            else:
                row_parts.append("---")
        lines.append(" & ".join(row_parts) + r" \\")

    lines.append(r"\midrule")

    # Gain row
    gain_parts = [r"\% gain (HCGAE↑Opt.)"]
    for env in envs:
        t = tests.get(env, {}).get("final5", {}).get("vs_optimal", {})
        if t:
            gain = t['gain_pct']
            p = t['p_value']
            sig = r"$^{*}$" if p < 0.05 else ""
            gain_parts.append(f"${gain:+.1f}\\%${sig}")
        else:
            gain_parts.append("---")
    lines.append(" & ".join(gain_parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_figures(data, metrics, out_dir: Path):
    """Generate publication-quality learning curve and bar chart figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        ALGO_STYLE = {
            "Standard_PPO": dict(color="#6B6B6B", linestyle="--", linewidth=1.5, label="Standard-PPO"),
            "Optimal_PPO": dict(color="#2196F3", linestyle="-.", linewidth=1.5, label="Optimal-PPO"),
            "Optimal_HCGAE_v2": dict(color="#F44336", linestyle="-", linewidth=2.0, label="HCGAE-v2 (ours)"),
        }

        envs = ENVS
        n_envs = len(envs)
        fig, axes = plt.subplots(1, n_envs, figsize=(5*n_envs, 4))
        if n_envs == 1:
            axes = [axes]

        for ax, env in zip(axes, envs):
            for algo in ALGOS:
                seeds = data.get(env, {}).get(algo, [])
                if not seeds:
                    continue
                # Interpolate to common step grid
                all_steps, all_rewards = [], []
                for d in seeds:
                    es = d.get("eval_steps", [])
                    er = d.get("eval_rewards", [])
                    if es and er:
                        all_steps.append(np.array(es))
                        all_rewards.append(np.array(er))
                if not all_steps:
                    continue
                max_len = min(len(s) for s in all_steps)
                arr = np.array([r[:max_len] for r in all_rewards])
                steps_common = all_steps[0][:max_len]
                mean_r = arr.mean(0)
                sem_r = arr.std(0) / np.sqrt(len(arr))

                style = ALGO_STYLE.get(algo, {})
                ax.plot(steps_common / 1e3, mean_r, **style)
                ax.fill_between(steps_common / 1e3, mean_r - sem_r, mean_r + sem_r,
                                color=style.get('color', 'gray'), alpha=0.15)

            env_short = env.split('-')[0]
            ax.set_title(env_short, fontsize=12, fontweight='bold')
            ax.set_xlabel("Steps (×10³)", fontsize=10)
            ax.set_ylabel("Episode Return", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, linestyle=':')
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

        plt.suptitle("Aligned Experiment: HCGAE-v2 vs Baselines\n(256×256 MLP, obs-norm, adv-norm, det. eval)",
                     fontsize=11, y=1.02)
        plt.tight_layout()
        out_lc = out_dir / "aligned_learning_curves.png"
        plt.savefig(out_lc, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_lc}")

        # Bar chart: final5 mean ± SEM
        fig2, axes2 = plt.subplots(1, n_envs, figsize=(4*n_envs, 4))
        if n_envs == 1:
            axes2 = [axes2]
        bar_colors = {"Standard_PPO": "#6B6B6B", "Optimal_PPO": "#2196F3", "Optimal_HCGAE_v2": "#F44336"}
        bar_labels = {"Standard_PPO": "Standard-PPO", "Optimal_PPO": "Optimal-PPO", "Optimal_HCGAE_v2": "HCGAE-v2"}
        x = np.arange(len(ALGOS))
        width = 0.6

        for ax, env in zip(axes2, envs):
            vals, errs = [], []
            for algo in ALGOS:
                m_data = metrics[env].get(algo, {}).get("final5", np.array([]))
                if len(m_data) > 0:
                    vals.append(float(np.mean(m_data)))
                    errs.append(float(np.std(m_data)/np.sqrt(len(m_data))))
                else:
                    vals.append(0.0)
                    errs.append(0.0)

            bars = ax.bar(x, vals, width, yerr=errs, capsize=4,
                         color=[bar_colors[a] for a in ALGOS], alpha=0.85, edgecolor='black', linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels([bar_labels[a] for a in ALGOS], rotation=20, ha='right', fontsize=8)
            ax.set_title(env.split('-')[0], fontsize=12, fontweight='bold')
            ax.set_ylabel("Final-5 Return", fontsize=10)
            ax.grid(True, alpha=0.3, axis='y', linestyle=':')

        plt.suptitle("Aligned Experiment: Final Performance\n(mean±SEM, 5 seeds)", fontsize=11, y=1.02)
        plt.tight_layout()
        out_bar = out_dir / "aligned_bar_chart.png"
        plt.savefig(out_bar, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_bar}")

    except ImportError as e:
        print(f"  [WARN] matplotlib not available: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/AlignedExperiment", type=Path)
    parser.add_argument("--out", default="results/paper_figures_final", type=Path)
    parser.add_argument("--latex", action="store_true", help="Print LaTeX table")
    args = parser.parse_args()

    results_dir = Path(args.results)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    data = load_results(results_dir)

    # Count available runs
    total_available = sum(
        len(data[env].get(algo, []))
        for env in ENVS for algo in ALGOS
    )
    print(f"\n[INFO] Found {total_available} / {len(ENVS)*len(ALGOS)*len(SEEDS)} runs completed")

    # Extract metrics
    metrics = extract_metrics(data)

    # Statistical tests
    tests = statistical_tests(metrics)

    # Print summary
    print_summary_table(metrics, tests)

    # Save
    out = {"metrics": {}, "tests": {}}
    for env in ENVS:
        out["metrics"][env] = {}
        for algo in ALGOS:
            m = metrics[env].get(algo, {})
            out["metrics"][env][algo] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in m.items()
            }
    out["tests"] = tests

    sp = out_dir / "aligned_analysis.json"
    with open(sp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved analysis: {sp}")

    # LaTeX table
    if args.latex:
        print("\n" + "="*80 + "\n  LaTeX Table:\n" + "="*80)
        print(generate_latex_table(metrics, tests))

    # Figures
    generate_figures(data, metrics, out_dir)


if __name__ == "__main__":
    main()

