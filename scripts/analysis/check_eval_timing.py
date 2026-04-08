#!/usr/bin/env python3
"""检查两个实验的eval时机差异"""
import json

print("=== HalfCheetah eval_steps 对比 ===")

# ICMLExperiment PPO
d = json.load(open('results/ICMLExperiment/HalfCheetah-v4/Optimal_PPO/Optimal_PPO_s0.json'))
steps_icml = d['eval_steps']
evals_icml = d['eval_rewards']
print(f"\nICML Optimal_PPO s0:")
print(f"  eval_steps 前5: {steps_icml[:5]}")
print(f"  eval_rewards 前5: {[f'{v:.1f}' for v in evals_icml[:5]]}")
print(f"  total steps: {d['total_steps']}")

# V4 HCGAE_v4
d = json.load(open('results/V4FullExperiment/HalfCheetah-v4/Optimal_HCGAE_v4/Optimal_HCGAE_v4_s0.json'))
steps_v4 = d['eval_steps']
evals_v4 = d['eval_rewards']
print(f"\nV4 HCGAE_v4 s0:")
print(f"  eval_steps 前5: {steps_v4[:5]}")
print(f"  eval_rewards 前5: {[f'{v:.1f}' for v in evals_v4[:5]]}")
print(f"  total steps: {d['total_steps']}")

print("\n=== ICMLExperiment HCGAE_v2 ===")
d = json.load(open('results/ICMLExperiment/HalfCheetah-v4/Optimal_HCGAE_v2/Optimal_HCGAE_v2_s0.json'))
steps_v2 = d['eval_steps']
evals_v2 = d['eval_rewards']
print(f"  eval_steps 前5: {steps_v2[:5]}")
print(f"  eval_rewards 前5: {[f'{v:.1f}' for v in evals_v2[:5]]}")

print("\n=== 关键发现 ===")
print(f"ICMLExperiment: 首次eval在step={steps_icml[0]}, eval={evals_icml[0]:.1f}")
print(f"V4FullExperiment: 首次eval在step={steps_v4[0]}, eval={evals_v4[0]:.1f}")
print(f"ICML eval_freq: {steps_icml[1]-steps_icml[0]} steps/eval（第1->第2间隔）")
print(f"V4 eval_freq: {steps_v4[1]-steps_v4[0]} steps/eval（第1->第2间隔）")

print("\n=== HalfCheetah ICMLExperiment 前10步骤 ===")
for s in range(5):
    d = json.load(open(f'results/ICMLExperiment/HalfCheetah-v4/Optimal_PPO/Optimal_PPO_s{s}.json'))
    print(f"s{s}: steps={d['eval_steps'][:3]}, rewards={[f'{v:.1f}' for v in d['eval_rewards'][:3]]}")

