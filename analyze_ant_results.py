#!/usr/bin/env python3
"""
Analyze Ant-v4 experiment results and update paper draft.

This script:
1. Reads all completed Ant-v4 JSON results
2. Computes mean ± std for each algorithm
3. Runs Mann-Whitney U tests
4. Computes Cohen's d effect sizes
5. Outputs formatted text to paste into paper_draft.md
"""
import json
import os
from pathlib import Path
import numpy as np

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not available, skipping statistical tests")


RESULTS_DIR = Path("results/ICMLExperiment/Ant-v4")
ALGORITHMS = ["Standard_PPO", "Optimal_PPO", "Optimal_HCGAE", "Optimal_HCGAE_SCR"]

def cohen_d(a, b):
    """Compute Cohen's d between two groups."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float('nan')
    pooled_std = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    if pooled_std < 1e-10:
        return float('nan')
    return (np.mean(a) - np.mean(b)) / pooled_std


def effect_label(d):
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    else:
        return "large"


def load_algo_results(algo_name):
    algo_dir = RESULTS_DIR / algo_name
    if not algo_dir.exists():
        return []
    results = []
    for f in sorted(algo_dir.glob("*.json")):
        with open(f) as fp:
            d = json.load(fp)
        final = d.get("final_reward", float("nan"))
        results.append(final)
    return results


def main():
    print("=" * 70)
    print("Ant-v4 Experiment Results Analysis")
    print("=" * 70)

    data = {}
    for algo in ALGORITHMS:
        rewards = load_algo_results(algo)
        data[algo] = rewards
        n = len(rewards)
        if n > 0:
            mean = np.mean(rewards)
            std = np.std(rewards, ddof=1) if n > 1 else 0.0
            print(f"  {algo}: n={n}, mean={mean:.1f}, std={std:.1f}")
            if n < 5:
                print(f"    WARNING: Only {n}/5 seeds complete!")
        else:
            print(f"  {algo}: NO DATA")

    print()

    # Check completeness
    all_complete = all(len(data[a]) == 5 for a in ALGORITHMS)
    if not all_complete:
        print("⚠️  NOT ALL EXPERIMENTS COMPLETE:")
        for algo in ALGORITHMS:
            n = len(data[algo])
            status = "✓" if n == 5 else f"⚠ {n}/5"
            print(f"   {status}  {algo}")
        print()
        print("Showing partial results (may be updated when all experiments complete)")
        print()
    else:
        print("✓ All experiments complete (5 seeds each)")
        print()

    # Summary table
    print("=" * 70)
    print("Performance Summary Table")
    print("=" * 70)
    print(f"{'Algorithm':<24} | {'Mean':>8} | {'Std':>8} | {'n':>3}")
    print("-" * 50)
    for algo in ALGORITHMS:
        rewards = data[algo]
        if not rewards:
            print(f"{algo:<24} | {'N/A':>8} | {'N/A':>8} | {0:>3}")
            continue
        n = len(rewards)
        mean = np.mean(rewards)
        std = np.std(rewards, ddof=1) if n > 1 else 0.0
        print(f"{algo:<24} | {mean:>8.1f} | {std:>8.1f} | {n:>3}")

    # Statistical comparisons
    if HAS_SCIPY and len(data.get("Optimal_HCGAE", [])) >= 2 and len(data.get("Optimal_PPO", [])) >= 2:
        print()
        print("=" * 70)
        print("Statistical Comparisons (Mann-Whitney U, two-sided)")
        print("=" * 70)

        comparisons = [
            ("Optimal_HCGAE", "Optimal_PPO", "HCGAE vs Optimal PPO"),
            ("Optimal_HCGAE", "Standard_PPO", "HCGAE vs Standard PPO"),
            ("Optimal_PPO", "Standard_PPO", "Optimal PPO vs Standard PPO"),
            ("Optimal_HCGAE_SCR", "Optimal_PPO", "HCGAE-SCR vs Optimal PPO"),
        ]

        for algo_a, algo_b, label in comparisons:
            a = data.get(algo_a, [])
            b = data.get(algo_b, [])
            if len(a) < 2 or len(b) < 2:
                print(f"  {label}: insufficient data (n_a={len(a)}, n_b={len(b)})")
                continue

            mean_a, mean_b = np.mean(a), np.mean(b)
            delta_pct = 100 * (mean_a - mean_b) / (abs(mean_b) + 1e-8)
            u_stat, p_val = stats.mannwhitneyu(a, b, alternative='two-sided')
            d = cohen_d(a, b)

            sig = "**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s.")
            print(f"\n  {label}:")
            print(f"    Δ% = {delta_pct:+.1f}%  |  U={u_stat:.0f}  |  p={p_val:.3f} {sig}  |  d={d:+.2f} ({effect_label(d)})")

    # Markdown table for paper
    print()
    print("=" * 70)
    print("PAPER-READY MARKDOWN (paste into Table 1 or dedicated Ant-v4 table)")
    print("=" * 70)

    print()
    print("**Table: Ant-v4 Performance (5 seeds, 500K steps)**")
    print()
    print("| Method | Ant-v4 |")
    print("|---|:---:|")
    for algo in ALGORITHMS:
        rewards = data[algo]
        if not rewards:
            print(f"| {algo} | N/A |")
            continue
        n = len(rewards)
        mean = np.mean(rewards)
        std = np.std(rewards, ddof=1) if n > 1 else 0.0
        note = f" (n={n})" if n < 5 else ""
        print(f"| {algo} | {mean:.0f} ± {std:.0f}{note} |")

    print()

    # Interpretation
    print("=" * 70)
    print("INTERPRETATION FOR PAPER")
    print("=" * 70)

    hcgae_r = data.get("Optimal_HCGAE", [])
    opt_r = data.get("Optimal_PPO", [])
    std_r = data.get("Standard_PPO", [])

    if hcgae_r and opt_r:
        mean_h = np.mean(hcgae_r)
        mean_o = np.mean(opt_r)
        delta = 100 * (mean_h - mean_o) / (abs(mean_o) + 1e-8)
        sign = "positive" if delta > 0 else "NEGATIVE"

        print(f"\nHCGAE vs Optimal PPO on Ant-v4:")
        print(f"  Δ = {delta:+.1f}% ({sign})")

        # Compare with HalfCheetah pattern
        print()
        if delta < -5:
            print("  → Pattern: NEGATIVE (similar to HalfCheetah)")
            print("  → Interpretation: Ant-v4 has dense rewards (similar to HalfCheetah),")
            print("    confirming the SCR < 1 regime for dense-reward high-dimensional tasks.")
            print()
            print("  SUGGESTED TEXT FOR PAPER ABSTRACT:")
            print(f"  - **Ant-v4**: HCGAE achieves {mean_h:.0f}±{np.std(hcgae_r, ddof=1):.0f} vs Optimal PPO's")
            print(f"    {mean_o:.0f}±{np.std(opt_r, ddof=1):.0f} ({delta:+.1f}%), consistent with the dense-reward")
            print("    negative effect observed on HalfCheetah-v4.")
        elif delta > 5:
            print("  → Pattern: POSITIVE (similar to Hopper/Walker2d)")
            print("  → Interpretation: Ant-v4 may have episodic failure modes where")
            print("    Critic initialization bias is dominant. Interesting!")
            print()
            print("  SUGGESTED TEXT FOR PAPER ABSTRACT:")
            print(f"  - **Ant-v4**: HCGAE achieves {mean_h:.0f}±{np.std(hcgae_r, ddof=1):.0f} vs Optimal PPO's")
            print(f"    {mean_o:.0f}±{np.std(opt_r, ddof=1):.0f} ({delta:+.1f}%), showing HCGAE benefits extend")
            print("    to high-dimensional (8D, 27-obs) locomotion tasks.")
        else:
            print("  → Pattern: NEUTRAL (near zero)")
            print("  → Interpretation: Ant-v4 is in the boundary regime (SCR ≈ 1).")

    print()
    print("Script complete. Update paper_draft.md with the results above.")
    print("Specifically update:")
    print("  1. Abstract: Replace '(Results pending...)' with actual numbers")
    print("  2. §4.1 Setup: Update Ant-v4 description")
    print("  3. Table 1 (or add Table for Ant-v4)")
    print("  4. §4.3 Key Findings: Add Ant-v4 finding")
    print("  5. §7 Limitations: Update point 1 with Ant-v4 result")
    print("  6. §8 Conclusion: Update Ant-v4 mention")


if __name__ == "__main__":
    main()

