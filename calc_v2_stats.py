import json
import os

import numpy as np

# 读取 ICMLExperiment 中的 Optimal_HCGAE_v2 数据
envs = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']
for env in envs:
    v2_dir = f'results/ICMLExperiment/{env}/Optimal_HCGAE_v2'
    opt_ppo_dir = f'results/ICMLExperiment/{env}/Optimal_PPO'
    std_ppo_dir = f'results/ICMLExperiment/{env}/Standard_PPO'
    v1_dir = f'results/ICMLExperiment/{env}/Optimal_HCGAE'

    v2_rewards, opt_ppo_rewards, std_ppo_rewards, v1_rewards = [], [], [], []

    for fname in sorted(os.listdir(v2_dir)):
        if fname.endswith('.json'):
            with open(f'{v2_dir}/{fname}') as f:
                data = json.load(f)
            final = np.mean(data['eval_rewards'][-5:])
            v2_rewards.append(final)

    for fname in sorted(os.listdir(opt_ppo_dir)):
        if fname.endswith('.json'):
            with open(f'{opt_ppo_dir}/{fname}') as f:
                data = json.load(f)
            final = np.mean(data['eval_rewards'][-5:])
            opt_ppo_rewards.append(final)

    for fname in sorted(os.listdir(std_ppo_dir)):
        if fname.endswith('.json'):
            with open(f'{std_ppo_dir}/{fname}') as f:
                data = json.load(f)
            final = np.mean(data['eval_rewards'][-5:])
            std_ppo_rewards.append(final)

    for fname in sorted(os.listdir(v1_dir)):
        if fname.endswith('.json'):
            with open(f'{v1_dir}/{fname}') as f:
                data = json.load(f)
            final = np.mean(data['eval_rewards'][-5:])
            v1_rewards.append(final)

    v2_mean = np.mean(v2_rewards)
    v2_std = np.std(v2_rewards)
    opt_mean = np.mean(opt_ppo_rewards)
    opt_std = np.std(opt_ppo_rewards)
    std_mean = np.mean(std_ppo_rewards)
    v1_mean = np.mean(v1_rewards)
    v1_std = np.std(v1_rewards)

    pct_vs_opt = (v2_mean - opt_mean) / opt_mean * 100
    pct_vs_std = (v2_mean - std_mean) / std_mean * 100
    pct_v2_vs_v1 = (v2_mean - v1_mean) / v1_mean * 100

    print(f'{env}:')
    print(f'  Optimal_HCGAE_v2: {v2_mean:.1f} +/- {v2_std:.1f} seeds: {[round(r) for r in v2_rewards]}')
    print(f'  Optimal_HCGAE_v1: {v1_mean:.1f} +/- {v1_std:.1f} seeds: {[round(r) for r in v1_rewards]}')
    print(f'  Optimal_PPO:      {opt_mean:.1f} +/- {opt_std:.1f} seeds: {[round(r) for r in opt_ppo_rewards]}')
    print(f'  Standard_PPO:     {std_mean:.1f} +/- {np.std(std_ppo_rewards):.1f}')
    print(f'  v2 vs Opt_PPO: {pct_vs_opt:+.1f}%')
    print(f'  v2 vs Std_PPO: {pct_vs_std:+.1f}%')
    print(f'  v2 vs v1:      {pct_v2_vs_v1:+.1f}%')
    print()

