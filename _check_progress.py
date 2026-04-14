#!/usr/bin/env python3
"""检查各数据集完整度和效果"""
import json
from pathlib import Path
import numpy as np

ENVS = ['HalfCheetah-v4','Hopper-v4','Walker2d-v4','Ant-v4']

def load_scores(base_dir, env, algo, min_steps=0, last_n=20):
    d = Path(base_dir) / env / algo
    if not d.exists():
        return []
    scores = []
    for f in sorted(d.glob('*.json')):
        try:
            data = json.load(open(f))
            if data.get('total_steps', 0) < min_steps:
                continue
            ep = data.get('eval_episode_rewards') or data.get('episode_rewards', [])
            if ep:
                scores.append(float(np.mean(ep[-last_n:])))
        except Exception:
            pass
    return scores

print("\n" + "="*65)
print("  FinalOptimal PPO 数据（1.5M steps, 15 seeds目标）")
print("="*65)
for env in ENVS:
    ppo   = load_scores('results/FinalOptimal', env, 'Optimal_PPO', min_steps=1_400_000)
    hcgae = load_scores('results/FinalOptimal', env, 'Optimal_HCGAE_Optimal', min_steps=1_400_000)
    print(f"  {env}:")
    print(f"    Optimal_PPO:         {len(ppo):2d} seeds  mean={np.mean(ppo):7.0f}±{np.std(ppo):.0f}" if ppo else f"    Optimal_PPO:          0 seeds")
    print(f"    Optimal_HCGAE:       {len(hcgae):2d} seeds  mean={np.mean(hcgae):7.0f}±{np.std(hcgae):.0f}" if hcgae else f"    Optimal_HCGAE:         0 seeds")
    if ppo and hcgae:
        imp = (np.mean(hcgae) - np.mean(ppo)) / abs(np.mean(ppo)) * 100
        print(f"    提升: {imp:+.1f}%")

print("\n" + "="*65)
print("  GRPO 数据（1.5M steps, 15 seeds目标）")
print("="*65)
for env in ENVS:
    std_grpo  = load_scores('results/GRPO', env, 'Optimal_GRPO', min_steps=1_400_000)
    hcgae_grpo = load_scores('results/GRPO', env, 'HCGAE_Optimal_GRPO', min_steps=1_400_000)
    print(f"  {env}:")
    print(f"    Optimal_GRPO:       {len(std_grpo):2d}/15 seeds  mean={np.mean(std_grpo):7.0f}" if std_grpo else f"    Optimal_GRPO:         0/15 seeds")
    print(f"    HCGAE_Optimal_GRPO: {len(hcgae_grpo):2d}/15 seeds  mean={np.mean(hcgae_grpo):7.0f}" if hcgae_grpo else f"    HCGAE_Optimal_GRPO:   0/15 seeds")
    if std_grpo and hcgae_grpo:
        imp = (np.mean(hcgae_grpo) - np.mean(std_grpo)) / abs(np.mean(std_grpo)) * 100
        print(f"    GRPO提升: {imp:+.1f}%")

print("\n" + "="*65)
print("  PPO 消融数据（1M steps, 8 seeds目标）")
print("="*65)
ABLATION_ALGOS = ['Optimal_HCGAE_Optimal','Optimal_HCGAE_NoFixSCR','Optimal_HCGAE_NoMCSigmoid','Optimal_HCGAE_NoEVGate']
for env in ENVS:
    print(f"  {env}:")
    for algo in ABLATION_ALGOS:
        scores = load_scores('results/Ablation', env, algo, min_steps=950_000)
        tag = "✅" if len(scores) >= 8 else f"🔄{len(scores)}/8"
        print(f"    {algo:<38} {tag}  mean={np.mean(scores):7.0f}" if scores else f"    {algo:<38} {tag}")

print()

