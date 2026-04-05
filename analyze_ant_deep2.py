"""
Ant-v4 深度日志分析 - 修正版
分析各算法的 eval reward 轨迹，找出HCGAE失效的关键证据
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
    print(f"  Final reward: mean={np.mean(final_rewards):.1f} +- {np.std(final_rewards):.1f}")
    print(f"  Per seed: {[round(r,1) for r in final_rewards]}")

    # Eval reward trajectory - 统一对齐
    eval_arrs = []
    for d in seeds:
        ev = d['eval_rewards']
        if ev:
            eval_arrs.append(ev)

    if eval_arrs:
        min_len = min(len(e) for e in eval_arrs)
        eval_mat = np.array([e[:min_len] for e in eval_arrs])  # (n_seeds, n_evals)
        n_evals = eval_mat.shape[1]

        print(f"\n  Eval trajectory (n_evals={n_evals}):")
        checkpoints = [0, 4, 9, 19, 29, n_evals-1]
        checkpoints = sorted(set([c for c in checkpoints if c < n_evals]))
        for c in checkpoints:
            vals = eval_mat[:, c]
            pct = int(100 * c / n_evals)
            print(f"    Eval {c+1:2d} ({pct:3d}% steps): {vals.mean():8.1f} +- {vals.std():.1f}")

        # 关键统计
        print(f"\n  早期阶段 (前10% evals):")
        early_n = max(1, n_evals // 10)
        early = eval_mat[:, :early_n]
        print(f"    Mean: {early.mean():.1f},  Negative: {(early < 0).mean()*100:.0f}%")

        print(f"  中期阶段 (25%-75% evals):")
        mid_start = n_evals // 4
        mid_end = 3 * n_evals // 4
        mid = eval_mat[:, mid_start:mid_end]
        print(f"    Mean: {mid.mean():.1f},  Std: {mid.std():.1f}")

        print(f"  末期阶段 (后20% evals):")
        late_n = max(1, n_evals // 5)
        late = eval_mat[:, -late_n:]
        print(f"    Mean: {late.mean():.1f},  Std: {late.std():.1f}")

        # Peak performance
        peak_per_seed = eval_mat.max(axis=1)
        print(f"\n  Peak reward per seed: {[round(p,1) for p in peak_per_seed]}")
        print(f"  Mean peak: {peak_per_seed.mean():.1f}")


print("\n" + "=" * 80)
print("ANT-v4 vs HOPPER-v4: 环境特性深度对比")
print("=" * 80)

for env_name, env_base in [('Hopper-v4', hopper_base), ('Ant-v4', base)]:
    for algo in ['Optimal_PPO', 'Optimal_HCGAE_v2']:
        seeds = read_seeds(env_base, algo)
        if not seeds:
            continue

        eval_arrs = []
        for d in seeds:
            if d['eval_rewards']:
                eval_arrs.append(d['eval_rewards'])

        if not eval_arrs:
            continue

        min_len = min(len(e) for e in eval_arrs)
        eval_mat = np.array([e[:min_len] for e in eval_arrs])

        final_rewards = [d['final_reward'] for d in seeds]

        # 计算"预热时间"：从开始到首次突破0的eval步数
        warmup_evals = []
        for seed_evals in eval_mat:
            cross_zero = np.where(seed_evals > 0)[0]
            warmup_evals.append(cross_zero[0] if len(cross_zero) > 0 else min_len)

        # 计算最终性能相对于Optimal_PPO的比率
        print(f"\n[{env_name} | {algo}]")
        print(f"  Final: {np.mean(final_rewards):.1f} +- {np.std(final_rewards):.1f}")
        print(f"  Warmup to positive reward: {np.mean(warmup_evals):.1f} evals (max={max(warmup_evals)})")
        print(f"  Early (10%): {eval_mat[:, :min_len//10].mean():.1f}")
        print(f"  Late (80-100%): {eval_mat[:, 4*min_len//5:].mean():.1f}")


print("\n" + "=" * 80)
print("关键诊断：奖励尺度与方差对比")
print("=" * 80)

print("\nAnt-v4 episode 奖励统计:")
for algo in algos:
    seeds = read_seeds(base, algo)
    if not seeds:
        continue

    all_ep = []
    for d in seeds:
        all_ep.extend(d['episode_rewards'])
    ep = np.array(all_ep)

    print(f"\n  {algo}:")
    print(f"    Episode reward: mean={ep.mean():.1f}, std={ep.std():.1f}")
    print(f"    Range: [{ep.min():.1f}, {ep.max():.1f}]")
    print(f"    Negative episodes: {(ep < 0).mean()*100:.1f}%")
    print(f"    Coef. Variation: {ep.std() / (abs(ep.mean()) + 1e-8):.2f}")

    # 分析前1/3 vs 后1/3的差异
    n = len(ep)
    ep_early = ep[:n//3]
    ep_late = ep[2*n//3:]
    print(f"    Early 1/3 mean: {ep_early.mean():.1f}")
    print(f"    Late 1/3 mean:  {ep_late.mean():.1f}")
    print(f"    Improvement: {ep_late.mean() - ep_early.mean():.1f}")


print("\n" + "=" * 80)
print("Ant-v4 vs Hopper-v4: 环境奖励尺度对比")
print("=" * 80)
print("\n  注意：Ant-v4 的奖励结构与 Hopper 有本质不同！")

for env_name, env_base in [('Hopper-v4', hopper_base), ('Ant-v4', base)]:
    algo = 'Standard_PPO'
    seeds = read_seeds(env_base, algo)
    if not seeds:
        continue

    all_ep = []
    for d in seeds:
        all_ep.extend(d['episode_rewards'])
    ep = np.array(all_ep)

    # 各阶段
    n = len(ep)
    thirds = [ep[:n//3], ep[n//3:2*n//3], ep[2*n//3:]]

    print(f"\n[{env_name}] Standard_PPO:")
    print(f"  Episode: mean={ep.mean():.1f}, std={ep.std():.1f}")
    print(f"  Std/Mean ratio (var coef): {ep.std()/max(abs(ep.mean()),1e-8):.2f}")
    print(f"  Early | Mid | Late: {thirds[0].mean():.1f} | {thirds[1].mean():.1f} | {thirds[2].mean():.1f}")
    print(f"  Negative rate: {(ep < 0).mean()*100:.1f}%")
    all_ep_len = []
    for d in seeds:
        all_ep_len.extend(d['episode_lengths'])
    ep_len = np.array(all_ep_len)
    print(f"  Episode length: mean={ep_len.mean():.1f}, std={ep_len.std():.1f}")

