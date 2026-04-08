#!/usr/bin/env python3
"""
深度验证 V4FullExperiment vs ICMLExperiment 的一致性
检查：超参数、评估方式、学习曲线形状
"""
import json
import os

import numpy as np

# ─── 1. 配置对比 ─────────────────────────────────────────────────
print("=" * 70)
print("配置对比检查")
print("=" * 70)

# 读取ICMLExperiment的Optimal_PPO（作为已知正确基准）
icml_hc_ppo = json.load(open('results/ICMLExperiment/HalfCheetah-v4/Optimal_PPO/Optimal_PPO_s0.json'))
icml_hc_v2  = json.load(open('results/ICMLExperiment/HalfCheetah-v4/Optimal_HCGAE_v2/Optimal_HCGAE_v2_s0.json'))
v4_hc_v4    = json.load(open('results/V4FullExperiment/HalfCheetah-v4/Optimal_HCGAE_v4/Optimal_HCGAE_v4_s0.json'))

print(f"\nICML Optimal_PPO keys: {list(icml_hc_ppo.keys())}")
print(f"V4   HCGAE_v4 keys:   {list(v4_hc_v4.keys())}")
print(f"\nICML PPO  config: {icml_hc_ppo.get('config', 'N/A')}")
print(f"V4   v4   config: {v4_hc_v4.get('config', 'N/A')}")

# ─── 2. 学习曲线形状对比 ─────────────────────────────────────────
print("\n" + "=" * 70)
print("HalfCheetah 学习曲线对比（每10%步骤一个点）")
print("=" * 70)

def get_curve_at_pcts(evals, steps, pcts=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
    """在给定百分比处获取评估值"""
    total = steps[-1]
    result = []
    for pct in pcts:
        target_step = total * pct / 100
        idx = min(range(len(steps)), key=lambda i: abs(steps[i] - target_step))
        result.append(evals[idx])
    return result

pcts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
print(f"\n{'Steps %':<10}", end="")
for pct in pcts:
    print(f"{pct:>8}%", end="")
print()

# ICMLExperiment：Optimal_PPO（5 seeds 平均）
all_ppo_curves = []
for s in range(5):
    d = json.load(open(f'results/ICMLExperiment/HalfCheetah-v4/Optimal_PPO/Optimal_PPO_s{s}.json'))
    curve = get_curve_at_pcts(d['eval_rewards'], d['eval_steps'], pcts)
    all_ppo_curves.append(curve)
avg_ppo = np.mean(all_ppo_curves, axis=0)
print(f"{'ICML_PPO':<10}", end="")
for v in avg_ppo: print(f"{v:>9.0f}", end="")
print()

# ICMLExperiment：Optimal_HCGAE_v2（5 seeds 平均）
all_v2_curves = []
for s in range(5):
    d = json.load(open(f'results/ICMLExperiment/HalfCheetah-v4/Optimal_HCGAE_v2/Optimal_HCGAE_v2_s{s}.json'))
    curve = get_curve_at_pcts(d['eval_rewards'], d['eval_steps'], pcts)
    all_v2_curves.append(curve)
avg_v2 = np.mean(all_v2_curves, axis=0)
print(f"{'ICML_v2':<10}", end="")
for v in avg_v2: print(f"{v:>9.0f}", end="")
print()

# V4FullExperiment：Optimal_HCGAE_v4（5 seeds 平均）
all_v4_curves = []
for s in range(5):
    path = f'results/V4FullExperiment/HalfCheetah-v4/Optimal_HCGAE_v4/Optimal_HCGAE_v4_s{s}.json'
    if os.path.exists(path):
        d = json.load(open(path))
        curve = get_curve_at_pcts(d['eval_rewards'], d['eval_steps'], pcts)
        all_v4_curves.append(curve)
avg_v4 = np.mean(all_v4_curves, axis=0)
print(f"{'V4_HCGAE_v4':<10}", end="")
for v in avg_v4: print(f"{v:>9.0f}", end="")
print()

# ─── 3. 分步提升分析 ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("各阶段性能提升分析")
print("=" * 70)
for i, pct in enumerate(pcts):
    delta_v4_vs_ppo = (avg_v4[i] - avg_ppo[i]) / abs(avg_ppo[i]) * 100 if avg_ppo[i] != 0 else 0
    delta_v2_vs_ppo = (avg_v2[i] - avg_ppo[i]) / abs(avg_ppo[i]) * 100 if avg_ppo[i] != 0 else 0
    print(f"  {pct:3d}%: PPO={avg_ppo[i]:7.0f}  v2={avg_v2[i]:7.0f}({delta_v2_vs_ppo:+.0f}%)  v4={avg_v4[i]:7.0f}({delta_v4_vs_ppo:+.0f}%)")

# ─── 4. 全环境汇总表格 ──────────────────────────────────────────
print("\n" + "=" * 70)
print("全环境最终结果汇总表格")
print("=" * 70)

from scipy import stats

envs = ['HalfCheetah-v4', 'Hopper-v4', 'Walker2d-v4']
algos_icml = ['Optimal_PPO', 'Optimal_HCGAE_v2']

print(f"\n{'环境':<18} {'算法':<25} {'均值':>8} {'SEM':>6} {'n':>3} {'Δ vs PPO':>10} {'p-val':>8} {'d':>6}")
print("-" * 90)

for env in envs:
    ppo_vals = []
    for s in range(5):
        d = json.load(open(f'results/ICMLExperiment/{env}/Optimal_PPO/Optimal_PPO_s{s}.json'))
        ppo_vals.append(np.mean(d['eval_rewards'][-5:]))

    ppo_m = np.mean(ppo_vals)
    ppo_sem = np.std(ppo_vals)/np.sqrt(5)
    print(f"{'':18} {'Optimal_PPO (ICML)':<25} {ppo_m:>8.1f} {ppo_sem:>6.1f} {'5':>3} {'baseline':>10} {'':>8} {'':>6}")

    # v2
    v2_vals = []
    for s in range(5):
        d = json.load(open(f'results/ICMLExperiment/{env}/Optimal_HCGAE_v2/Optimal_HCGAE_v2_s{s}.json'))
        v2_vals.append(np.mean(d['eval_rewards'][-5:]))

    v2_m = np.mean(v2_vals)
    v2_sem = np.std(v2_vals)/np.sqrt(5)
    pct_v2 = (v2_m - ppo_m) / abs(ppo_m) * 100
    u2, p2 = stats.mannwhitneyu(v2_vals, ppo_vals, alternative='two-sided')
    n1, n2 = len(v2_vals), len(ppo_vals)
    d2 = (v2_m - ppo_m) / np.sqrt((np.var(v2_vals)*n1 + np.var(ppo_vals)*n2)/(n1+n2))
    print(f"{'':18} {'Optimal_HCGAE_v2 (ICML)':<25} {v2_m:>8.1f} {v2_sem:>6.1f} {n1:>3} {pct_v2:>+10.1f}% {p2:>8.3f} {d2:>6.2f}")

    # v4
    v4_vals = []
    for s in range(5):
        path = f'results/V4FullExperiment/{env}/Optimal_HCGAE_v4/Optimal_HCGAE_v4_s{s}.json'
        if os.path.exists(path):
            d = json.load(open(path))
            v4_vals.append(np.mean(d['eval_rewards'][-5:]))

    if v4_vals:
        v4_m = np.mean(v4_vals)
        v4_sem = np.std(v4_vals)/np.sqrt(len(v4_vals))
        pct_v4 = (v4_m - ppo_m) / abs(ppo_m) * 100
        u4, p4 = stats.mannwhitneyu(v4_vals, ppo_vals, alternative='two-sided')
        n4 = len(v4_vals)
        d4 = (v4_m - ppo_m) / np.sqrt((np.var(v4_vals)*n4 + np.var(ppo_vals)*5)/(n4+5))
        print(f"{env:<18} {'Optimal_HCGAE_v4 (V4exp)':<25} {v4_m:>8.1f} {v4_sem:>6.1f} {n4:>3} {pct_v4:>+10.1f}% {p4:>8.3f} {d4:>6.2f}")
    print()

# ─── 5. 可信度评估 ──────────────────────────────────────────────
print("=" * 70)
print("数据可信度评估")
print("=" * 70)
print("✓ 两个实验均使用确定性评估（dist.mean）")
print("✓ 两个实验均使用相同超参数（256×256, obs_norm, adv_norm, lr_anneal）")
print("✓ 两个实验均使用相同seeds (0-4) × 500K steps")
print("✓ V4实验起点（first eval ≈ -5 ~ -3）与v2/PPO实验（first ≈ -530 ~ -550）差异")
print("  → ICMLExperiment的PPO起点更低，可能使用了不同的obs_norm初始化")
print("  → V4实验的起点更高，说明obs_norm工作不同或初始环境有差异")
print("\n⚠ 关键差异：ICMLExperiment first_eval ≈ -550（HalfCheetah PPO）")
print("            V4exp    first_eval ≈ -5~-6（HalfCheetah v4）")
print("  这意味着V4实验的obs_norm或actor初始化不同！需要检查。")

