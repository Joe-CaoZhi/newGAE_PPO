#!/usr/bin/env python3
"""诊断两组实验的离群点问题和条件差异"""
import json
import numpy as np
from pathlib import Path

from scipy import stats

print("="*65)
print("诊断：ICMLExperiment vs AlignedExperiment 离群点问题")
print("="*65)

# 1. 对比两组实验的 per-seed 数据
for env in ['HalfCheetah-v4', 'Hopper-v4', 'Walker2d-v4']:
    print(f"\n{'='*55}")
    print(f"环境: {env}")
    print(f"{'='*55}")
    for algo in ['Optimal_PPO', 'Optimal_HCGAE_v2']:
        for exp in ['ICMLExperiment', 'AlignedExperiment']:
            seeds_data = []
            for s in range(5):
                p = Path(f'results/{exp}/{env}/{algo}/{algo}_s{s}.json')
                if p.exists():
                    d = json.load(open(p))
                    evals = d.get('eval_rewards', [])
                    if evals:
                        final = np.mean(evals[-5:])
                        seeds_data.append((s, final))
            if seeds_data:
                vals = [v for _, v in seeds_data]
                mean = np.mean(vals)
                std = np.std(vals)
                # 离群点检测: z-score > 2.0
                zscores = np.abs(stats.zscore(vals)) if len(vals) >= 3 else [0]*len(vals)
                outliers = [(seeds_data[i][0], vals[i]) for i, z in enumerate(zscores) if z > 2.0]
                print(f"  [{exp[:15]:15s}] {algo:25s}: {mean:7.1f} ± {std/max(np.sqrt(len(vals)),1):5.1f}  seeds={[round(v) for v in vals]}")
                if outliers:
                    print(f"    ⚠️  离群点(z>2): seed{outliers}")

# 2. 检查实验配置差异
print("\n\n" + "="*65)
print("配置对比：两组实验的关键差异")
print("="*65)

for exp in ['ICMLExperiment', 'AlignedExperiment']:
    p = Path(f'results/{exp}/HalfCheetah-v4/Optimal_PPO/Optimal_PPO_s0.json')
    if p.exists():
        d = json.load(open(p))
        cfg = d.get('config', {})
        print(f"\n{exp}:")
        print(f"  config: {cfg}")
        print(f"  total_steps: {d.get('total_steps')}")
        evals = d.get('eval_rewards', [])
        print(f"  num_evals: {len(evals)}")
        print(f"  eval_steps: {d.get('eval_steps', [])[:3]} ... {d.get('eval_steps', [])[-3:]}" if d.get('eval_steps') else "  eval_steps: N/A")

# 3. 统计检验：排除离群点后的对比
print("\n\n" + "="*65)
print("排除离群点(z>2)后的结果对比")
print("="*65)

for exp in ['ICMLExperiment', 'AlignedExperiment']:
    print(f"\n{exp}:")
    for env in ['HalfCheetah-v4', 'Hopper-v4', 'Walker2d-v4']:
        ppo_vals = []
        hcgae_vals = []
        for s in range(5):
            for algo, lst in [('Optimal_PPO', ppo_vals), ('Optimal_HCGAE_v2', hcgae_vals)]:
                p = Path(f'results/{exp}/{env}/{algo}/{algo}_s{s}.json')
                if p.exists():
                    d = json.load(open(p))
                    evals = d.get('eval_rewards', [])
                    if evals:
                        lst.append(np.mean(evals[-5:]))
        if ppo_vals and hcgae_vals:
            # 排除离群点
            def remove_outliers(vals, threshold=2.0):
                if len(vals) < 3:
                    return vals
                z = np.abs(stats.zscore(vals))
                return [v for v, zi in zip(vals, z) if zi <= threshold]
            ppo_clean = remove_outliers(ppo_vals)
            hcgae_clean = remove_outliers(hcgae_vals)
            # 原始和清洗后对比
            delta_raw = (np.mean(hcgae_vals) - np.mean(ppo_vals)) / max(np.mean(ppo_vals), 1) * 100
            delta_clean = (np.mean(hcgae_clean) - np.mean(ppo_clean)) / max(np.mean(ppo_clean), 1) * 100
            print(f"  {env:20s}: raw Δ={delta_raw:+.1f}%  (n={len(ppo_vals)})  |  clean Δ={delta_clean:+.1f}%  (n={len(ppo_clean)}/{len(hcgae_clean)})")

