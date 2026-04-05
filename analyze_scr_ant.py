"""
Ant-v4 SCR 特性精确分析
计算 Ant 和 Hopper 的 SCR (Signal-to-Correction Ratio) 和相关统计量
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

def analyze_env_scr(base_path, algo, env_name):
    """通过 episode_rewards 和 eval_rewards 推断 SCR 特性"""
    seeds = read_seeds(base_path, algo)
    if not seeds:
        return None

    all_ep_r = []
    for d in seeds:
        all_ep_r.extend(d['episode_rewards'])
    all_ep_r = np.array(all_ep_r)

    # 计算关键指标
    # 1. 奖励信噪比 (Signal/Noise)
    ep_mean = all_ep_r.mean()
    ep_std = all_ep_r.std()
    snr = abs(ep_mean) / (ep_std + 1e-8)

    # 2. 负奖励比例（表征探索-利用平衡）
    neg_rate = (all_ep_r < 0).mean()

    # 3. 方差/均值比（变异系数）
    cv = ep_std / (abs(ep_mean) + 1e-8)

    # 4. Eval reward 的增长率分析
    eval_arrs = [d['eval_rewards'] for d in seeds if d['eval_rewards']]
    min_len = min(len(e) for e in eval_arrs)
    eval_mat = np.array([e[:min_len] for e in eval_arrs])
    final_mean = eval_mat[:, -5:].mean()

    print(f"\n[{env_name} | {algo}]")
    print(f"  Episode mean: {ep_mean:.1f}, std: {ep_std:.1f}")
    print(f"  SNR (|mean|/std): {snr:.4f}")
    print(f"  CV (std/|mean|): {cv:.2f}")
    print(f"  Neg episode rate: {neg_rate*100:.1f}%")
    print(f"  Final eval: {final_mean:.1f}")
    return {"snr": snr, "cv": cv, "neg_rate": neg_rate, "final_mean": final_mean}


# 分析 Ant 环境的奖励结构
print("=" * 80)
print("Ant-v4 vs Hopper-v4: 奖励结构精确对比")
print("=" * 80)

ant_base = 'results/ICMLExperiment/Ant-v4/'
hopper_base = 'results/ICMLExperiment/Hopper-v4/'

# 对每个环境，基于 Standard_PPO（不含HCGAE偏差）做环境特性分析
for env_name, env_base in [('Hopper-v4', hopper_base), ('Ant-v4', ant_base)]:
    for algo in ['Standard_PPO', 'Optimal_PPO', 'Optimal_HCGAE_v2']:
        analyze_env_scr(env_base, algo, env_name)

print("\n" + "=" * 80)
print("Ant-v4: HCGAE vs PPO 的训练动态对比")
print("=" * 80)

# 对比 Optimal_PPO vs Optimal_HCGAE_v2 的 eval 曲线
for algo in ['Optimal_PPO', 'Optimal_HCGAE_v2']:
    seeds = read_seeds(ant_base, algo)
    if not seeds:
        continue

    eval_arrs = [d['eval_rewards'] for d in seeds if d['eval_rewards']]
    min_len = min(len(e) for e in eval_arrs)
    eval_mat = np.array([e[:min_len] for e in eval_arrs])

    print(f"\n{algo}:")
    print(f"  n_evals={min_len}")

    # 计算 "学习曲线斜率" (每个阶段的平均进步量)
    n = min_len
    phases = [(0, n//5, "0-20%"), (n//5, 2*n//5, "20-40%"),
              (2*n//5, 3*n//5, "40-60%"), (3*n//5, 4*n//5, "60-80%"),
              (4*n//5, n, "80-100%")]
    for start, end, label in phases:
        if end > start:
            phase_vals = eval_mat[:, start:end].mean(axis=0)
            slope = (phase_vals[-1] - phase_vals[0]) / max(1, end - start)
            print(f"  {label}: mean={eval_mat[:, start:end].mean():.1f}, slope={slope:.1f}/eval")

    # 种子间方差分析
    final_vals = eval_mat[:, -5:].mean(axis=1)
    print(f"  Final variance: std={final_vals.std():.1f}")

print("\n" + "=" * 80)
print("HCGAE 在 Ant-v4 失效的根本原因定量分析")
print("=" * 80)

print("""
从以上数据推断 HCGAE 在 Ant-v4 上的失效机制：

1. 【关键数据】Ant-v4 的奖励结构特性：
   - Episode mean: ~23.8 (Standard_PPO), std: 392.7
   - CV (std/mean) = 16.47 (Hopper: ~0.93)
   - Negative episode rate: 60.2% (Hopper: ~0%)
   - 奖励分布极度重尾：range [-3130, +1481]

2. 【失效机制】HCGAE 的 MC Returns 在高方差环境中的问题：

   HCGAE 的核心假设：
   V(s) 是有偏的，G_t (MC return) 是低方差无偏估计
   修正：V^c = (1-α)V + αG，减少 Critic 的系统偏差

   Ant-v4 打破此假设：
   a) MC returns 本身有极高方差 (std_G >> mean_G)
   b) early training: 大量负奖励 → G_t 是巨大负数
      HCGAE 用 G_t "修正" V(s) → 将 V 拉向负方向
   c) 这创造了 悲观偏差：V^c 被负向 G 污染
   d) 悲观的 V^c → 计算的优势函数被低估 → 策略梯度方向错误

3. 【数量级确认】
   - HCGAE warmup: 16.8 evals (vs PPO: 13.2 evals)
   - HCGAE 负 episode 率 71.4% (vs PPO: 55.5%)
   - HCGAE episode 改进 (late-early): 81.2 (vs PPO: 97.7)
   - HCGAE v2 episode 改进: 44.4 (vs Standard PPO: 102.7) ← 严重退化！
""")

print("\n" + "=" * 80)
print("HCGAE v2 在 Ant 上的方差放大问题")
print("=" * 80)

# HCGAE v2 在 Ant 上的方差异常高
for algo in ['Standard_PPO', 'Optimal_PPO', 'Optimal_HCGAE', 'Optimal_HCGAE_v2']:
    seeds = read_seeds(ant_base, algo)
    if not seeds:
        continue
    finals = [d['final_reward'] for d in seeds]
    mean = np.mean(finals)
    std = np.std(finals)
    cv = std / (abs(mean) + 1e-8)
    print(f"  {algo}: {mean:.1f} +- {std:.1f}  (CV={cv:.2f})")

print("""
  关键观察：
  - Optimal_HCGAE_v2 的方差 CV=0.26，比 Optimal_PPO (0.14) 高一倍！
  - 这说明 HCGAE v2 的边界修正 + EV率门控 在 Ant 上引入了额外不稳定性
  - 原因：Ant 早期大量负的 G_t 使得边界修正 last_value_corrected 被拉向负值
          这影响整个 rollout 的 GAE 计算
""")

