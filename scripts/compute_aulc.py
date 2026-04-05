#!/usr/bin/env python3
"""计算 ICMLExperiment 数据的 AULC 和 Steps-to-50%"""
import glob
import json

import numpy as np
from scipy import stats

ICML_ROOT = "results/ICMLExperiment"
ENVS = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]
ALGOS = ["Standard_PPO", "Optimal_PPO", "Optimal_HCGAE_v2", "Optimal_HCGAE_v2_NoBdry", "Optimal_HCGAE_v2_NoGate"]

def load_and_aulc(env, algo):
    """计算 AULC（时间均值回报）和 steps-to-50%"""
    files = sorted(glob.glob(f"{ICML_ROOT}/{env}/{algo}/{algo}_s*.json"))
    if not files:
        return None, None, None

    aulcs = []
    steps_to_50 = []
    final_perf = []

    for f in files:
        d = json.load(open(f))
        steps = np.array(d.get("eval_steps", []))
        rets  = np.array(d.get("eval_rewards", []))
        if len(steps) < 2:
            continue

        # AULC = trapezoidal integral / total steps
        aulc = np.trapz(rets, steps) / (steps[-1] - steps[0])
        aulcs.append(aulc)

        # final performance (last 5 evals)
        fp = np.mean(rets[-5:])
        final_perf.append(fp)

        # steps to 50% of final performance
        target = fp * 0.5
        reached = steps[rets >= target]
        if len(reached) > 0:
            steps_to_50.append(reached[0])
        else:
            steps_to_50.append(steps[-1])

    aulcs = np.array(aulcs)
    steps_to_50 = np.array(steps_to_50)
    final_perf = np.array(final_perf)

    return aulcs, steps_to_50, final_perf


def mann_whitney(a, b):
    if len(a) < 2 or len(b) < 2:
        return 1.0, 0.0
    _, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    d = (np.mean(a) - np.mean(b)) / (pooled_std + 1e-8)
    return float(p), float(d)


print("=" * 80)
print("AULC and Sample Efficiency Statistics (ICMLExperiment, n=5 seeds, 500K steps)")
print("=" * 80)

results = {}
for env in ENVS:
    print(f"\n{'=' * 40} {env} {'=' * 40}")
    env_data = {}
    for algo in ALGOS:
        aulcs, s50, fp = load_and_aulc(env, algo)
        if aulcs is None:
            continue
        env_data[algo] = {
            "aulc_mean": float(np.mean(aulcs)),
            "aulc_std": float(np.std(aulcs, ddof=1)) if len(aulcs) > 1 else 0.0,
            "aulc_sem": float(np.std(aulcs, ddof=1) / np.sqrt(len(aulcs))) if len(aulcs) > 1 else 0.0,
            "steps_to_50_mean": float(np.mean(s50)),
            "steps_to_50_all": [float(x) for x in s50],
            "final_mean": float(np.mean(fp)),
            "final_std": float(np.std(fp, ddof=1)) if len(fp) > 1 else 0.0,
        }
        sem = np.std(aulcs, ddof=1) / np.sqrt(len(aulcs)) if len(aulcs) > 1 else 0.0
        print(f"  {algo:35s}: AULC={np.mean(aulcs):.1f} ± {sem:.1f}  "
              f"Final={np.mean(fp):.1f} ± {np.std(fp, ddof=1):.1f}  "
              f"~Steps50%={int(np.mean(s50)/1000)}K")

    results[env] = env_data

    # Statistical tests vs Optimal PPO
    if "Optimal_PPO" in env_data and "Optimal_HCGAE_v2" in env_data:
        files_v2  = sorted(glob.glob(f"{ICML_ROOT}/{env}/Optimal_HCGAE_v2/Optimal_HCGAE_v2_s*.json"))
        files_opt = sorted(glob.glob(f"{ICML_ROOT}/{env}/Optimal_PPO/Optimal_PPO_s*.json"))

        fp_v2  = np.array([np.mean(json.load(open(f)).get("eval_rewards", [])[-5:]) for f in files_v2])
        fp_opt = np.array([np.mean(json.load(open(f)).get("eval_rewards", [])[-5:]) for f in files_opt])

        p, d = mann_whitney(fp_v2, fp_opt)
        imp = (np.mean(fp_v2) - np.mean(fp_opt)) / (np.mean(fp_opt) + 1e-8) * 100
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "n.s.")
        print(f"\n  HCGAE-v2 vs Optimal PPO: {imp:+.1f}% | p={p:.3f} {sig} | d={d:.3f}")

# Summary table for paper
print("\n\n" + "=" * 80)
print("TABLE 7 (for paper): AULC Summary")
print("=" * 80)
print(f"{'Method':<35} {'Hopper AULC':>14} {'Walker AULC':>14} {'HalfCheetah AULC':>18} {'Hopper Steps→50%':>18}")
print("-" * 100)
for algo in ALGOS:
    row = f"{algo:<35}"
    for env in ENVS:
        if env in results and algo in results[env]:
            v = results[env][algo]
            row += f"  {v['aulc_mean']:.0f} ± {v['aulc_sem']:.0f}"
        else:
            row += "  N/A"
    # Hopper steps to 50%
    if "Hopper-v4" in results and algo in results["Hopper-v4"]:
        s50 = results["Hopper-v4"][algo]["steps_to_50_mean"]
        row += f"  ~{int(s50/1000)}K"
    print(row)

# Save to JSON
import json as json_mod
with open("results/paper_figures_final/sample_efficiency_stats.json", "w") as f:
    json_mod.dump(results, f, indent=2)
print("\nSaved to results/paper_figures_final/sample_efficiency_stats.json")

