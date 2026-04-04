#!/usr/bin/env python3
"""
对比论文中的数据与实际JSON数据
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats

RESULTS_DIR = Path("/Users/joe-caozhi/newGAE_ppo/results")

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

# 论文Table 1中的数据 (从paper_draft.md提取)
paper_table1 = {
    'Hopper-v4': {
        'Standard_PPO': {'mean': 1804, 'std': 69},
        'Optimal_PPO': {'mean': 1598, 'std': 149},
        'Optimal_HCGAE': {'mean': 1752, 'std': 81},
        'Optimal_HCGAE_SCR': {'mean': 1366, 'std': 133}
    },
    'Walker2d-v4': {
        'Standard_PPO': {'mean': 1425, 'std': 223},
        'Optimal_PPO': {'mean': 1596, 'std': 417},
        'Optimal_HCGAE': {'mean': 1872, 'std': 547},
        'Optimal_HCGAE_SCR': {'mean': 1896, 'std': 682}
    },
    'HalfCheetah-v4': {
        'Standard_PPO': {'mean': 1051, 'std': 134},
        'Optimal_PPO': {'mean': 1487, 'std': 61},
        'Optimal_HCGAE': {'mean': 1250, 'std': 53},
        'Optimal_HCGAE_SCR': {'mean': 1254, 'std': 78}
    }
}

# 论文统计表中的数据
paper_stats = {
    'HCGAE vs Optimal PPO': {
        'Hopper': {'p': 0.222, 'd': 1.28},
        'Walker2d': {'p': 0.841, 'd': 0.57},
        'HalfCheetah': {'p': 0.008, 'd': -4.14}
    }
}

print("="*80)
print("论文数据验证报告")
print("="*80)

# 加载实际数据
icml_dir = RESULTS_DIR / "ICMLExperiment"
actual_data = {}

for env in ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']:
    actual_data[env] = {}

    for algo in ['Standard_PPO', 'Optimal_PPO', 'Optimal_HCGAE', 'Optimal_HCGAE_SCR']:
        algo_dir = icml_dir / env / algo
        if algo_dir.exists():
            rewards = []
            for seed_file in sorted(algo_dir.glob(f"{algo}_s*.json")):
                data = load_json(seed_file)
                reward = get_final_reward(data)
                if reward is not None:
                    rewards.append(reward)

            if rewards:
                actual_data[env][algo] = {
                    'mean': np.mean(rewards),
                    'std': np.std(rewards, ddof=1),
                    'rewards': rewards
                }

# 对比Table 1
print("\n### Table 1 数据对比 ###\n")

all_match = True
for env in ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']:
    print(f"#### {env} ####")

    for algo in ['Standard_PPO', 'Optimal_PPO', 'Optimal_HCGAE', 'Optimal_HCGAE_SCR']:
        if algo in actual_data[env] and algo in paper_table1[env]:
            actual = actual_data[env][algo]
            paper = paper_table1[env][algo]

            mean_diff = actual['mean'] - paper['mean']
            std_diff = actual['std'] - paper['std']

            mean_match = abs(mean_diff) < 1.0
            std_match = abs(std_diff) < 10.0

            if mean_match and std_match:
                status = "✓"
            else:
                status = "⚠"
                all_match = False

            print(f"  {status} {algo}:")
            print(f"     论文: {paper['mean']:.0f} ± {paper['std']:.0f}")
            print(f"     实际: {actual['mean']:.1f} ± {actual['std']:.1f}")
            print(f"     差异: Δ_mean={mean_diff:+.1f}, Δ_std={std_diff:+.1f}")

    print()

# 验证统计比较
print("### 统计比较验证 (HCGAE vs Optimal PPO) ###\n")

for env in ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']:
    env_short = env.split('-')[0]

    if 'Optimal_HCGAE' in actual_data[env] and 'Optimal_PPO' in actual_data[env]:
        hcgae = actual_data[env]['Optimal_HCGAE']['rewards']
        ppo = actual_data[env]['Optimal_PPO']['rewards']

        if len(hcgae) >= 2 and len(ppo) >= 2:
            stat, p_value = stats.mannwhitneyu(hcgae, ppo, alternative='two-sided')
            cohens_d = (np.mean(hcgae) - np.mean(ppo)) / np.sqrt(
                (np.std(hcgae, ddof=1)**2 + np.std(ppo, ddof=1)**2) / 2
            )

            paper_p = paper_stats['HCGAE vs Optimal PPO'][env_short]['p']
            paper_d = paper_stats['HCGAE vs Optimal PPO'][env_short]['d']

            p_match = abs(p_value - paper_p) < 0.01
            d_match = abs(cohens_d - paper_d) < 0.1

            if p_match and d_match:
                status = "✓"
            else:
                status = "⚠"

            print(f"{status} {env}:")
            print(f"   论文: p={paper_p:.3f}, d={paper_d:.2f}")
            print(f"   实际: p={p_value:.3f}, d={cohens_d:.2f}")
            print()

# 总结
print("="*80)
if all_match:
    print("✓ 所有Table 1数据验证通过!")
else:
    print("⚠ 存在数据不一致，需要检查")

print("="*80)

