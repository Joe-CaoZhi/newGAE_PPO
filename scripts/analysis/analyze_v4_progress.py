#!/usr/bin/env python3
"""分析V4完备实验的当前进度，与ICMLExperiment进行对比"""
import json
import os

import numpy as np
from scipy import stats


def load_json_final5(path):
    """加载JSON文件，返回最后5次eval的均值"""
    with open(path) as f:
        data = json.load(f)
    evals = data.get('eval_rewards', data.get('eval_returns', data.get('evaluations', [])))
    if not evals:
        return None
    return float(np.mean(evals[-5:]))

def summarize_algo(base_dir, env, algo, seeds=range(5)):
    """汇总某算法在某环境下的结果，返回各seed最终值列表"""
    results = []
    for s in seeds:
        path = f"{base_dir}/{env}/{algo}/{algo}_s{s}.json"
        if os.path.exists(path):
            val = load_json_final5(path)
            if val is not None:
                results.append(val)
    return results

# 环境列表
envs = ['HalfCheetah-v4', 'Hopper-v4', 'Walker2d-v4']

# ICMLExperiment数据（作为基准）
icml_dir = 'results/ICMLExperiment'
# V4完备实验数据
v4_dir = 'results/V4FullExperiment'

print("=" * 80)
print("V4完备实验 vs ICMLExperiment 对比分析")
print("(基准：各算法在相同环境下的 final-5 eval 均值±SEM)")
print("=" * 80)

for env in envs:
    print(f"\n{'='*60}")
    print(f"环境: {env}")
    print(f"{'='*60}")

    # ICMLExperiment结果
    ppo_vals = summarize_algo(icml_dir, env, 'Optimal_PPO')
    v2_vals = summarize_algo(icml_dir, env, 'Optimal_HCGAE_v2')

    print(f"\n[ICMLExperiment 参考基准 (5 seeds × 500K steps)]")
    if ppo_vals:
        pm = np.mean(ppo_vals); ps = np.std(ppo_vals)/np.sqrt(len(ppo_vals))
        print(f"  Optimal_PPO         = {pm:.1f} ± {ps:.1f}  (n={len(ppo_vals)}) seeds={[f'{v:.0f}' for v in ppo_vals]}")
    if v2_vals:
        vm = np.mean(v2_vals); vs = np.std(v2_vals)/np.sqrt(len(v2_vals))
        print(f"  Optimal_HCGAE_v2    = {vm:.1f} ± {vs:.1f}  (n={len(v2_vals)}) seeds={[f'{v:.0f}' for v in v2_vals]}")

    # V4完备实验（已有结果）
    v4_vals = summarize_algo(v4_dir, env, 'Optimal_HCGAE_v4')

    print(f"\n[V4FullExperiment 当前进度 ({len(v4_vals)}/5 seeds)]")
    if v4_vals:
        v4m = np.mean(v4_vals); v4s = np.std(v4_vals)/np.sqrt(len(v4_vals))
        print(f"  Optimal_HCGAE_v4    = {v4m:.1f} ± {v4s:.1f}  (n={len(v4_vals)}) seeds={[f'{v:.0f}' for v in v4_vals]}")

        if ppo_vals:
            pct_vs_ppo = (v4m - pm) / abs(pm) * 100
            print(f"  vs Optimal_PPO:     {pct_vs_ppo:+.1f}%")
            if len(v4_vals) >= 3:
                u_stat, p_val = stats.mannwhitneyu(v4_vals, ppo_vals, alternative='two-sided')
                n1, n2 = len(v4_vals), len(ppo_vals)
                d_num = v4m - pm
                d_den = np.sqrt((np.var(v4_vals)*n1 + np.var(ppo_vals)*n2)/(n1+n2))
                cohens_d = d_num/d_den if d_den > 0 else 0
                print(f"  Mann-Whitney p={p_val:.3f}, Cohen's d={cohens_d:.2f}")

        if v2_vals:
            pct_vs_v2 = (v4m - vm) / abs(vm) * 100
            print(f"  vs HCGAE_v2:        {pct_vs_v2:+.1f}%")
    else:
        print(f"  尚无完整数据")

print("\n" + "="*80)
print("状态摘要")
print("="*80)

# 检查s4进度
for env in envs:
    s4_path = f"{v4_dir}/{env}/Optimal_HCGAE_v4/Optimal_HCGAE_v4_s4.json"
    if os.path.exists(s4_path):
        print(f"  {env}: s4 完成")
    else:
        print(f"  {env}: s4 运行中...")

