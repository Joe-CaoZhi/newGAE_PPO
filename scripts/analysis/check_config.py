import glob
import json

# 检查典型配置
f = 'results/LargeScaleExperiment/Hopper-v4/Optimal_PPO/Optimal_PPO_s0.json'
d = json.load(open(f))
print("=== 实验配置 ===")
for k, v in d['config'].items():
    print(f"  {k}: {v}")
print(f"  total_steps: {d['total_steps']:,}")
print(f"  elapsed_s: {d['elapsed_s']/60:.1f} min")
print(f"  n_eval_points: {len(d['eval_steps'])}")

print("\n=== 各环境final_reward随训练步数的变化 ===")
print("(Hopper PPO s0 — 每个eval点的得分)")
steps = d['eval_steps']
rewards = d['eval_rewards']
for i, (s, r) in enumerate(zip(steps, rewards)):
    if i % 5 == 0 or i >= len(steps) - 5:
        print(f"  step {s//1000:>4}K: {r:.0f}")

print("\n=== Hopper各seed的500K时刻得分 vs 1M时刻得分 ===")
files = sorted(glob.glob('results/LargeScaleExperiment/Hopper-v4/Optimal_PPO/*.json'))
print(f"{'seed':>5}  {'@500K':>8}  {'@1M':>8}  {'change':>8}")
for f in files:
    d = json.load(open(f))
    steps = d['eval_steps']
    rewards = d['eval_rewards']
    # 找最接近500K的eval点
    idx_500k = min(range(len(steps)), key=lambda i: abs(steps[i] - 500000))
    r_500k = rewards[idx_500k]
    r_1m = d['final_reward']
    change = r_1m - r_500k
    seed = d['seed']
    print(f"  s{seed:>2}  {r_500k:>8.0f}  {r_1m:>8.0f}  {change:>+8.0f}")

print("\n=== HalfCheetah各seed的500K vs 1M ===")
files = sorted(glob.glob('results/LargeScaleExperiment/HalfCheetah-v4/Optimal_PPO/*.json'))
print(f"{'seed':>5}  {'@500K':>8}  {'@1M':>8}  {'change':>8}  {'稳定?':>8}")
for f in files:
    d = json.load(open(f))
    steps = d['eval_steps']
    rewards = d['eval_rewards']
    idx_500k = min(range(len(steps)), key=lambda i: abs(steps[i] - 500000))
    r_500k = rewards[idx_500k]
    r_1m = d['final_reward']
    change = r_1m - r_500k
    seed = d['seed']
    stable = "稳定" if abs(change) < 500 else ("继续涨" if change > 500 else "后期崩")
    print(f"  s{seed:>2}  {r_500k:>8.0f}  {r_1m:>8.0f}  {change:>+8.0f}  {stable:>8}")

