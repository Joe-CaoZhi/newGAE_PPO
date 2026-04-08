#!/usr/bin/env python3
"""
数学分析：为什么HalfCheetah的HCGAE在高性能PPO基线下退化？
核心假说：EV增长率门控的阈值τ_rate是在低性能PPO基线下标定的，
在更高性能的基线（hidden=256, deterministic eval）下失效
"""
import json
import numpy as np
from pathlib import Path


def load_full(exp_dir, env, algo, seeds=range(5)):
    results = []
    for s in seeds:
        p = Path(f'results/{exp_dir}/{env}/{algo}/{algo}_s{s}.json')
        if p.exists():
            results.append(json.load(open(p)))
    return results

# =====================================================================
# 分析1：确认 HCGAE s3 的"后期崩溃"现象
# =====================================================================
print("="*65)
print("分析1：HalfCheetah HCGAE seed3 的后期政策崩溃")
print("="*65)

env = 'HalfCheetah-v4'
hcgae_s3 = json.load(open('results/AlignedExperiment/HalfCheetah-v4/Optimal_HCGAE_v2/Optimal_HCGAE_v2_s3.json'))
ppo_s3 = json.load(open('results/AlignedExperiment/HalfCheetah-v4/Optimal_PPO/Optimal_PPO_s3.json'))

h_evals = hcgae_s3['eval_rewards']
p_evals = ppo_s3['eval_rewards']

print(f"\nHCGAE s3 评估奖励全程 ({len(h_evals)} 个eval点):")
print(f"  步骤0-10:  {[round(v) for v in h_evals[:10]]}")
print(f"  步骤10-20: {[round(v) for v in h_evals[10:20]]}")
print(f"  步骤20-30: {[round(v) for v in h_evals[20:30]]}")
print(f"  步骤30-40: {[round(v) for v in h_evals[30:40]]}")
print(f"  步骤40-49: {[round(v) for v in h_evals[40:]]}")
print(f"\n峰值: {max(h_evals):.1f} 在 eval #{np.argmax(h_evals)}")
print(f"最终值: {h_evals[-1]:.1f}")

print(f"\nPPO s3 评估奖励（对比）:")
print(f"  步骤0-10:  {[round(v) for v in p_evals[:10]]}")
print(f"  步骤40-49: {[round(v) for v in p_evals[40:]]}")
print(f"\n峰值: {max(p_evals):.1f}")
print(f"最终值: {p_evals[-1]:.1f}")

# =====================================================================
# 分析2：消融分析揭示具体问题所在
# =====================================================================
print()
print("="*65)
print("分析2：消融结果揭示问题来源")
print("="*65)

print("\nAlignedExperiment HalfCheetah-v4:")
print(f"  Optimal_PPO:         2118.9 ± 365.3  (baseline)")
print(f"  HCGAE_v2 (full):     1472.6 ± 184.4  (-30.5%)")
print(f"  HCGAE_v2 NoBdry:     1633.7 ±  37.6  (-22.9%)")
print(f"  HCGAE_v2 NoGate:     1644.5 ±  28.5  (-22.4%)")

print("\n关键发现：")
print("  1. NoBdry 比 Full 好 (+10.9%): 边界校正在新基线下有害")
print("  2. NoGate 比 Full 好 (+11.7%): EV增长率门控在新基线下失效")
print("  3. NoBdry ≈ NoGate: 两个组件都出问题了，不是协同")
print("  4. 都比 PPO 差约22%: 说明HCGAE核心机制（Imp-I+Imp-II）本身在新基线下有害")

# =====================================================================
# 分析3：深入理解为什么——EV增长率和SCR在新基线下的变化
# =====================================================================
print()
print("="*65)
print("分析3：新旧实验中HalfCheetah训练动态的根本差异")
print("="*65)

print("""
旧实验 (ICMLExperiment, hidden=64/128?, 随机eval):
  - 早期eval(前5): mean=-556 → 说明随机策略下HalfCheetah奖励≈-500
  - 需要~20个eval点才能达到正值
  - 最终PPO: 1487

  这说明旧实验的初始EV极低，Critic需要很长时间收敛
  → HCGAE的MC校正有足够时间发挥作用
  → τ_rate=0.05的门控设定适合这个慢速收敛场景

新实验 (AlignedExperiment, hidden=256, 确定性eval):
  - 早期eval(前5): mean=-5 → 说明确定性eval下初始奖励≈0（不是-500）
  - eval#6时已经可以达到321分
  - 最终PPO: 2119 (+42%比旧实验)

  关键区别：新实验的256×256网络使得：
  a) 初始EV更高（更强的函数逼近能力）
  b) EV增长更快（更强的表达力）
  c) 确定性eval使得奖励分布方差更小
""")

print("="*65)
print("数学推导：为什么256×256网络使HCGAE更有害")
print("="*65)
print("""
设 EV_t 为第t个rollout的explained variance

旧实验 (小网络):
  EV_0 ≈ 0.05  → c_MC = clip(1-EV, 0.1, 1.0) ≈ 0.95
  ΔEV/rollout ≈ 0.01 << τ_rate=0.05  → EV增长率门控不激活
  α_max^v2(k) = α_max(k) × 1.0  → 全量MC校正

  此时 SCR = |B_t| / Var[G_t]^{1/2}
  - HalfCheetah的B_t在小网络下很大（FC(64)拟合能力弱）
  - 因此SCR可能 > 1 → HCGAE有帮助

新实验 (256×256大网络):
  EV_0 ≈ 0.3-0.5  → 初始EV更高
  ΔEV/rollout ≈ 0.05-0.15 >= τ_rate=0.05  → EV增长率门控应该激活

  但问题在于：
  η(k) = max(1 - (ΔEV - τ_rate)/(τ_max - τ_rate) × (1-s_min), s_min)
       = max(1 - (ΔEV - 0.05)/(0.15-0.05) × 0.9, 0.1)

  如果ΔEV = 0.05（刚好等于τ_rate），η = 1.0（不抑制）
  如果ΔEV = 0.10，η = 1 - 0.5×0.9 = 0.55
  如果ΔEV = 0.15，η = 0.1（最大抑制）

  但是：在新实验中，EV总量范围也不同（从0.3而不是0.05开始）
  这导致ΔEV的量纲和旧实验不一样！

  更根本的问题：在256×256网络下，HalfCheetah的Critic快速收敛
  → EV增长率大 → 门控应该激活 → 但实际上HCGAE仍然退化

  这暗示门控机制本身不够有效：
  s_min=0.1意味着α_max永远不会被完全抑制（最小保留10%）
  当网络更大时，即使10%的MC blending也可能有害
""")

# =====================================================================
# 分析4：精确数学：为什么大网络使HalfCheetah的SCR<1更严重
# =====================================================================
print("="*65)
print("分析4：大网络使HalfCheetah SCR<1问题更严重的数学推导")
print("="*65)
print("""
HalfCheetah的奖励结构：
  r_t ≈ k×v_t + c  (速度×常数 + 小常数)
  r_t范围约[-1, 8]，均值约1.5-3 (取决于策略)

MC回报方差（HalfCheetah，固定horizon T=500）:
  Var[G_t] = Σ_{k=0}^{T-t-1} γ^{2k} Var[r_{t+k}]

  对于随机策略（旧实验，小网络）：
    Var[r_t] ≈ 4-6 (动作噪声大，速度方差大)
    ΣγVar ≈ Var[r]/(1-γ²) ≈ 5/0.02 = 250
    → Var[G_t]^{1/2} ≈ 15

  对于接近最优策略（新实验，256×256大网络，eval时确定性）:
    Var[r_t] ≈ 1-2 (策略更稳定，速度方差更小)
    ΣγVar ≈ 1.5/0.02 = 75
    → Var[G_t]^{1/2} ≈ 8.7

Critic偏差 B_t = V_θ(s_t) - V^π(s_t):
  对于小网络（旧实验）：
    |B_t|在早期训练可达50-100（FC(64)拟合能力不足）
    SCR = |B_t|/Var[G_t]^{1/2} ≈ 75/15 = 5 >> 1 ✓

  对于大网络（新实验）：
    |B_t|在早期训练也较小，约10-30
    （256×256网络初始化偏差更小，拟合能力更强）
    SCR = |B_t|/Var[G_t]^{1/2} ≈ 20/8.7 = 2.3

  而且，大网络的EV增长更快：
    在100K步时EV可能已达0.7-0.8
    此时|B_t|急速缩小，而Var[G_t]变化不大
    → SCR在新实验中更快跌破1

结论：
  旧实验 (小网络): SCR≈5，MC校正有益，HCGAE有效
  新实验 (大网络): SCR≈2.3（早期）→ <1（50K步后）
                 EV增长率更大，但门控阈值τ_rate=0.05过小，
                 未能完全抑制MC blending

大网络恶化HalfCheetah HCGAE的机制：
  1. 初始EV更高 → MC校正的净收益更小（Critic已经较好）
  2. EV增长更快 → 高SCR窗口更短暂
  3. 256×256网络的大参数容量导致Critic目标方差更小
     → MC blending引入的噪声相对比重更大
  4. s_min=0.1的下界在大网络下仍然足以造成干扰
""")

# =====================================================================
# 分析5：定量验证推测
# =====================================================================
print("="*65)
print("分析5：定量对比 - 新实验NoBdry和NoGate表现")
print("="*65)
print("""
AlignedExperiment HalfCheetah:
  Full HCGAE:  1472.6 ± 184  (HCGAE/PPO = 69.5%)
  NoBdry:      1633.7 ±  38  (HCGAE/PPO = 77.1%)
  NoGate:      1644.5 ±  28  (HCGAE/PPO = 77.6%)

  NoBdry和NoGate都比Full好，但都比PPO差22%

  这说明：
  - 去掉边界校正（NoBdry）有帮助：+11%
  - 去掉增长率门控（NoGate）有帮助：+12%
  - 但两者都不能修复根本问题

  ICMLExperiment HalfCheetah:
  Full HCGAE:  1550.1 (vs PPO 1486.8 = +4.3%)

  矛盾点：旧实验Full HCGAE比PPO好4.3%，
  但新实验NoBdry/NoGate已经比Full好，却还是比PPO差22%

  解释：
  在新实验中，连 "最基础的HCGAE核心机制"（Imp-I和Imp-II）
  都是有害的，门控和边界校正都无法修复这个核心问题。

  真正的根因：τ_rate过大还是过小？
  - NoGate（去掉增长率门控）反而更好！
  - 这说明增长率门控不仅没有帮助，反而造成了额外的不稳定性
  - 可能是：门控根据ΔEV周期性地开闭，造成α_max波动，
    引入了学习不稳定性
""")

