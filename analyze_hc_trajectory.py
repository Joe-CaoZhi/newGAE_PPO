#!/usr/bin/env python3
"""分析HalfCheetah学习曲线，理解HCGAE负面效果的阶段性"""
import json
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("results/ICMLExperiment")

for env in ["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"]:
    print(f"\n=== {env} ===")
    for algo in ["Standard_PPO", "Optimal_PPO", "Optimal_HCGAE"]:
        algo_dir = RESULTS_DIR / env / algo
        if not algo_dir.exists():
            continue
        files = sorted(algo_dir.glob("*.json"))
        if not files:
            continue
        all_rewards = []
        for f in files:
            d = json.load(open(f))
            all_rewards.append(d["eval_rewards"])

        r = np.array(all_rewards)  # shape: (n_seeds, n_evals)
        n_evals = r.shape[1]
        # 评估间隔 = 10240步，每个checkpt代表10240步
        step_per_eval = 10240
        # 分成4个阶段
        q1 = n_evals // 4
        q2 = n_evals // 2
        q3 = 3 * n_evals // 4

        mean_early = np.mean(r[:, :q1])
        mean_mid = np.mean(r[:, q1:q2])
        mean_late_early = np.mean(r[:, q2:q3])
        mean_late = np.mean(r[:, q3:])
        mean_final = np.mean([x[-5:] for x in all_rewards])
        std_final = np.std([np.mean(x[-5:]) for x in all_rewards])

        print(f"  {algo} (n={len(files)}):")
        print(f"    Phase 1 (0-{q1*step_per_eval/1000:.0f}K): {mean_early:.1f}")
        print(f"    Phase 2 ({q1*step_per_eval/1000:.0f}K-{q2*step_per_eval/1000:.0f}K): {mean_mid:.1f}")
        print(f"    Phase 3 ({q2*step_per_eval/1000:.0f}K-{q3*step_per_eval/1000:.0f}K): {mean_late_early:.1f}")
        print(f"    Phase 4 ({q3*step_per_eval/1000:.0f}K-500K): {mean_late:.1f}")
        print(f"    Final (last 5): {mean_final:.1f} ± {std_final:.1f}")

# 还要分析AULC
print("\n=== AULC Comparison ===")
for env in ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]:
    print(f"\n{env}:")
    for algo in ["Standard_PPO", "Optimal_PPO", "Optimal_HCGAE", "Optimal_HCGAE_SCR"]:
        algo_dir = RESULTS_DIR / env / algo
        if not algo_dir.exists():
            continue
        files = sorted(algo_dir.glob("*.json"))
        if not files:
            continue
        aulcs = []
        for f in files:
            d = json.load(open(f))
            aulcs.append(np.mean(d["eval_rewards"]))
        print(f"  {algo}: AULC = {np.mean(aulcs):.1f} ± {np.std(aulcs):.1f}")

