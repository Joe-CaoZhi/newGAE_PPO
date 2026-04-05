"""
Ant-v4 深度日志分析 - 正确字段版本
通过 eval_rewards, episode_rewards, episode_lengths 分析训练过程
"""
import json
import os

import numpy as np


def read_seeds(base_path, algo, n=5):
    data = []
    for i in range(n):
        p = os.path.join(base_path, algo, f'{algo}_s{i}.json')
        if os.path.exists(p):
            with open(p) as f:
                d = json.load(f)
            data.append(d)
    return data

base = 'results/ICMLExperiment/Ant-v4/'
hopper_base = 'results/ICMLExperiment/Hopper-v4/'

algos = ['Standard_PPO', 'Optimal_PPO', 'Optimal_HCGAE', 'Optimal_HCGAE_v2']

print("=" * 80)
print("ANT-v4 训练轨迹分析")
print("=" * 80)

for algo in algos:
    seeds = read_seeds(base, algo)
    if not seeds:
        print(f"[{algo}] No data found!")
        continue

    print(f"\n{'='*60}")
    print(f"[{algo}]  n={len(seeds)} seeds")

    # Final reward
    final_rewards = [d['final_reward'] for d in seeds]
    print(f"  Final reward: mean={np.mean(final_rewards):.1f} ± {np.std(final_rewards):.1f}")
    print(f"  Per seed: {[round(r,1) for r in final_rewards]}")

    # Eval reward trajectory
    eval_rewards_all = []
    eval_steps_all = []
    for d in seeds:
        if d['eval_rewards']:
            eval_rewards_all.append(d['eval_rewards'])
            eval_steps_all.append(d['eval_steps'])

    if eval_rewards_all:
        # 找公共长度
        min_len = min(len(e) for e in eval_rewards_all)
        eval_arr = np.array([e[:min_len] for e in eval_rewards_all])  # (seeds, evals)
        steps_arr = np.array(eval_steps_all[0][:min_len])  # 假设steps一致

        print(f"\n  Eval reward trajectory (mean ± std across seeds):")
        checkpoints = [0, 4, 9, 19, min_len-1]
        checkpoints = sorted(set([c for c in checkpoints if c < min_len]))
        for c in checkpoints:
            step = steps_arr[c]
            vals = eval_arr[:, c]
            print(f"    Step {step:>7d}: {vals.mean():8.1f} ± {vals.std():6.1f}")

        # 初始奖励（前5个eval）
        init_reward = eval_arr[:, :5].mean()
        # 末期奖励（最后10个eval）
        final_avg = eval_arr[:, -10:].mean()
        print(f"\n  初始阶段 eval reward (前5次): {init_reward:.1f}")
        print(f"  末期阶段 eval reward (后10次): {final_avg:.1f}")

    # Episode reward statistics
    episode_rewards_all = []
    episode_lengths_all = []
    for d in seeds:
        episode_rewards_all.extend(d['episode_rewards'])
        episode_lengths_all.extend(d['episode_lengths'])

    all_ep_r = np.array(episode_rewards_all)
    all_ep_l = np.array(episode_lengths_all)

    print(f"\n  Episode statistics (across all seeds):")
    print(f"    Episode reward: mean={all_ep_r.mean():.1f}, std={all_ep_r.std():.1f}")
    print(f"    Episode length: mean={all_ep_l.mean():.1f}, std={all_ep_l.std():.1f}")
    print(f"    N episodes total: {len(all_ep_r)}")

    # 奖励分布分析
    neg_pct = (all_ep_r < 0).mean() * 100
    pos_pct = (all_ep_r > 100).mean() * 100
    print(f"    Negative reward episodes: {neg_pct:.1f}%")
    print(f"    Reward > 100 episodes:    {pos_pct:.1f}%")


print("\n" + "=" * 80)
print("ANT-v4 vs HOPPER-v4：环境特性对比 (Optimal_HCGAE_v2)")
print("=" * 80)

for env_name, env_base in [('Hopper-v4', hopper_base), ('Ant-v4', base)]:
    seeds = read_seeds(env_base, 'Optimal_HCGAE_v2')
    if not seeds:
        continue

    print(f"\n[{env_name}]")

    final_rewards = [d['final_reward'] for d in seeds]
    print(f"  Final: {np.mean(final_rewards):.1f} ± {np.std(final_rewards):.1f}")

    # Eval trajectory（前期 vs 后期）
    eval_all = [d['eval_rewards'] for d in seeds if d['eval_rewards']]
    if eval_all:
        min_len = min(len(e) for e in eval_all)
        eval_arr = np.array([e[:min_len] for e in eval_all])
        print(f"  Eval - 前5次: {eval_arr[:, :5].mean():.1f}")
        print(f"  Eval - 后10次: {eval_arr[:, -10:].mean():.1f}")
        print(f"  Eval - 最大值: {eval_arr.max():.1f}")

    # Episode 分析
    ep_all = []
    for d in seeds:
        ep_all.extend(d['episode_rewards'])
    ep_all = np.array(ep_all)
    print(f"  Episode mean: {ep_all.mean():.1f}, std: {ep_all.std():.1f}")
    print(f"  Neg episodes: {(ep_all < 0).mean()*100:.1f}%")


print("\n" + "=" * 80)
print("关键发现：Ant-v4 的 HCGAE 混合强度分析")
print("=" * 80)
print("""
通过 eval_steps 和 eval_rewards 可以重建训练曲线。
关键问题：为何 HCGAE v2 在 Ant 上表现不佳？

以下是通过对比 Standard_PPO vs Optimal_HCGAE_v2 的训练动态来分析：
""")

# 计算 "学习速度"：eval reward 从初始到最终的提升率
for algo in ['Standard_PPO', 'Optimal_PPO', 'Optimal_HCGAE', 'Optimal_HCGAE_v2']:
    seeds = read_seeds(base, algo)
    if not seeds:
        continue

    early_rewards = []
    late_rewards = []
    peak_rewards = []
    recovery_ratios = []

    for d in seeds:
        ev = d['eval_rewards']
        if len(ev) < 20:
            continue
        early = np.mean(ev[:5])
        late = np.mean(ev[-10:])
        peak = max(ev)
        early_rewards.append(early)
        late_rewards.append(late)
        peak_rewards.append(peak)
        if abs(early) > 1:
            recovery_ratios.append(late / abs(early))

    print(f"{algo}:")
    print(f"  Early eval: {np.mean(early_rewards):.1f} ± {np.std(early_rewards):.1f}")
    print(f"  Late eval:  {np.mean(late_rewards):.1f} ± {np.std(late_rewards):.1f}")
    print(f"  Peak eval:  {np.mean(peak_rewards):.1f} ± {np.std(peak_rewards):.1f}")
    print(f"  早期负奖励比例: {np.mean([e < 0 for e in early_rewards])*100:.0f}%")
    print()

