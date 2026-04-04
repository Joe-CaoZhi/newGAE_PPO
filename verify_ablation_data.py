#!/usr/bin/env python3
"""
验证HCGAE消融实验数据与论文Table 3的一致性
"""

import json
import numpy as np
from pathlib import Path

ABLATION_DIR = Path("/Users/joe-caozhi/newGAE_ppo/results/Hopper-v4-Ablation-MultiSeed")

def load_json(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return None

def get_final_reward(data):
    if data is None:
        return None

    if 'final_reward' in data:
        return data['final_reward']
    elif 'eval_rewards' in data and isinstance(data['eval_rewards'], list):
        return np.mean(data['eval_rewards'][-10:])
    elif 'episode_rewards' in data and isinstance(data['episode_rewards'], list):
        return np.mean(data['episode_rewards'][-10:])
    else:
        return None

print("="*80)
print("验证 HCGAE 消融实验数据")
print("="*80)

variants = {
    'HCGAE_Base': 'HCGAE_Base (无改进)',
    'HCGAE_Imp1': 'HCGAE_Imp1 (仅改进I)',
    'HCGAE_Imp2': 'HCGAE_Imp2 (仅改进II)',
    'HCGAE_Imp12': 'HCGAE_Imp12 (组合改进)'
}

results = {}

for variant_key, variant_name in variants.items():
    print(f"\n### {variant_name} ###")

    # 收集所有种子的最终奖励
    rewards = []
    for seed_file in sorted(ABLATION_DIR.glob(f"{variant_key}_s*.json")):
        data = load_json(seed_file)
        reward = get_final_reward(data)
        if reward is not None:
            rewards.append(reward)
            print(f"  {seed_file.name}: {reward:.1f}")

    if rewards:
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards, ddof=1)
        print(f"\n  总结: {mean_reward:.0f} ± {std_reward:.0f} (n={len(rewards)})")
        results[variant_key] = {
            'mean': mean_reward,
            'std': std_reward,
            'n': len(rewards),
            'rewards': rewards
        }
    else:
        print("  ❌ 无法提取奖励数据")

# 计算协同效应
print("\n" + "="*80)
print("协同效应分析")
print("="*80)

if all(k in results for k in ['HCGAE_Base', 'HCGAE_Imp1', 'HCGAE_Imp2', 'HCGAE_Imp12']):
    base = results['HCGAE_Base']['mean']
    imp1 = results['HCGAE_Imp1']['mean']
    imp2 = results['HCGAE_Imp2']['mean']
    imp12 = results['HCGAE_Imp12']['mean']

    delta_imp1 = imp1 - base
    delta_imp2 = imp2 - base
    delta_imp12 = imp12 - base

    additive_pred = delta_imp1 + delta_imp2
    actual_gain = delta_imp12
    synergy = actual_gain - additive_pred

    print(f"\n基线 (Base): {base:.0f}")
    print(f"改进I单独: {imp1:.0f} (Δ = {delta_imp1:+.0f})")
    print(f"改进II单独: {imp2:.0f} (Δ = {delta_imp2:+.0f})")
    print(f"改进I+II组合: {imp12:.0f} (Δ = {delta_imp12:+.0f})")
    print(f"\n加性预测: {delta_imp1:+.0f} + {delta_imp2:+.0f} = {additive_pred:+.0f}")
    print(f"实际增益: {actual_gain:+.0f}")
    print(f"协同效应: {synergy:+.0f} (实际 - 预测)")

    print("\n论文Table 3中的数据:")
    print("  HCGAE_Base: 2653 ± 627")
    print("  +Imp-I only: 2406 ± 787 (vs Base: -247)")
    print("  +Imp-II only: 2425 ± 615 (vs Base: -228)")
    print("  +Imp-I+II: 2839 ± 543 (vs Base: +186)")
    print("  Synergy: +661")

    print("\n实际数据与论文对比:")
    print(f"  HCGAE_Base: {base:.0f} ± {results['HCGAE_Base']['std']:.0f} vs 论文: 2653 ± 627")
    print(f"  +Imp-I: {imp1:.0f} ± {results['HCGAE_Imp1']['std']:.0f} vs 论文: 2406 ± 787 (Δ={delta_imp1:+.0f} vs 论文: -247)")
    print(f"  +Imp-II: {imp2:.0f} ± {results['HCGAE_Imp2']['std']:.0f} vs 论文: 2425 ± 615 (Δ={delta_imp2:+.0f} vs 论文: -228)")
    print(f"  +Imp-I+II: {imp12:.0f} ± {results['HCGAE_Imp12']['std']:.0f} vs 论文: 2839 ± 543 (Δ={delta_imp12:+.0f} vs 论文: +186)")
    print(f"  Synergy: {synergy:+.0f} vs 论文: +661")

# 保存验证结果
with open('/Users/joe-caozhi/newGAE_ppo/results/ablation_verification.json', 'w') as f:
    json.dump({
        'results': {k: {'mean': v['mean'], 'std': v['std'], 'n': v['n']}
                    for k, v in results.items()},
        'synergy_analysis': {
            'delta_imp1': delta_imp1 if 'delta_imp1' in dir() else None,
            'delta_imp2': delta_imp2 if 'delta_imp2' in dir() else None,
            'delta_imp12': delta_imp12 if 'delta_imp12' in dir() else None,
            'synergy': synergy if 'synergy' in dir() else None
        }
    }, f, indent=2)

print("\n验证结果已保存至: results/ablation_verification.json")

