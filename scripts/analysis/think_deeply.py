"""
深度剖析 GAE 的根本缺陷，为革新性改进提供理论依据

GAE 的核心公式：
  A_t^GAE = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
  δ_t = r_t + γV(s_{t+1}) - V(s_t)

根本缺陷分析：
1. 时序同质假设：所有时刻的 δ 用相同的衰减权重 (γλ)^l
   → 忽略了不同时刻 Critic 精度不同、环境动态不同
   → λ 是全局的，不能响应局部的状态分布变化

2. 单向时序依赖：只向后展开
   → 对于长轨迹，早期动作的优势估计方差极大
   → 但实际上后续状态的信息对早期动作是有用的

3. Critic 误差的传播机制有缺陷：
   A_t = δ_t + γλ δ_{t+1} + (γλ)^2 δ_{t+2} + ...
   当 Critic 系统性高估时（训练早期常见），每个 δ_t 都是负的
   → 优势整体偏低，策略更新过于保守

4. 归一化的时机问题：
   标准 PPO 在 update 时归一化优势 (A-μ)/σ
   → 这丢失了优势的绝对大小信息
   → 不同 rollout 间的优势可比性消失

5. 奖励尺度盲区：
   GAE 对奖励没有归一化
   → 不同环境/阶段的奖励尺度不同，但 λ 相同
   → 奖励大的环境需要更小的 λ（更多 TD），奖励小的需要更大的 λ（更多 MC）
"""

import json
import os

import numpy as np


def analyze_gae_flaws(results_dir):
    print("=== GAE 根本缺陷诊断 ===\n")

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith('.json'):
            continue
        d = json.load(open(f'{results_dir}/{fname}'))
        name = d.get('agent_name', fname)

        vlosses = np.array(d.get('value_losses', []))
        evs = np.array(d.get('explained_variances', []))
        kls = np.array(d.get('approx_kls', []))
        evals = np.array(d.get('eval_rewards', []))

        if len(vlosses) < 5:
            continue

        n = len(vlosses)
        print(f"--- {name} ---")

        # 1. Critic 精度的时间演化
        ev_curve = [evs[int(p*n)-1] for p in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0] if int(p*n) > 0]
        print(f"  EV曲线(10%-100%): {[f'{e:.3f}' for e in ev_curve]}")

        # 2. KL 发散模式（探索行为）
        kl_curve = [kls[int(p*n)-1] for p in [0.1, 0.3, 0.5, 0.7, 1.0] if int(p*n) > 0]
        print(f"  KL曲线(10%-100%): {[f'{k:.4f}' for k in kl_curve]}")

        # 3. VLoss 收敛速度
        vl_early = vlosses[:n//5].mean()
        vl_late = vlosses[-n//5:].mean()
        compression = vl_early / (vl_late + 1e-8)
        print(f"  VLoss压缩比(early/late): {vl_early:.2f}/{vl_late:.2f} = {compression:.1f}x")

        # 4. 评估奖励的方差（稳定性）
        if len(evals) > 10:
            ev_std = evals[-10:].std()
            print(f"  最后10次eval std: {ev_std:.2f}")

        print()

if __name__ == '__main__':
    import sys
    env = sys.argv[1] if len(sys.argv) > 1 else 'Acrobot-v1'
    analyze_gae_flaws(f'results/{env}')

    print("\n" + "="*60)
    print("革新性改进方向思考:")
    print("""
核心洞察：GAE 的本质是对 δ 序列的加权平均，但权重是固定的几何衰减。
真正的革新在于：让权重的形状本身成为可学习的，并且考虑到：

1. 【方向一】因果结构感知优势估计（Causal Advantage）
   - GAE 假设因果链是单向时序的，但实际上同一 rollout 内的状态是相关的
   - 改进：用 attention 机制捕捉 rollout 内的非局部相关
   - A_t = Σ_j α(s_t, s_j) * δ_j   （softmax attention over 整个 rollout）
   - 优点：δ_j 大时不仅影响 A_j，还通过 attention 影响 A_t
   - 因果约束：α(s_t, s_j) = 0 when j < t（未来不影响过去）

2. 【方向二】奖励规模自适应 λ（Scale-Adaptive GAE）
   - 真正的问题不是 λ 固定，而是优势估计对奖励尺度没有不变性
   - 改进：对奖励进行在线归一化，然后用固定 λ
     r_normalized = (r - μ_r) / (σ_r + ε)
     这使得 GAE 对奖励尺度变换不变
   - 这是 MuZero/R2D2 等大规模系统的标准实践

3. 【方向三】Hindsight 优势估计（Hindsight Advantage）
   - 标准 GAE 是因果的（只用过去信息），但 returns 是回望的（用了未来）
   - 真正的革新：用 rollout 完成后的全局信息重新估计 δ
     δ_t^hindsight = r_t + γV_hindsight(s_{t+1}) - V(s_t)
     V_hindsight(s) = 用整个 rollout 的 MC return 修正后的 V
   - 本质：用更准确的 V 来减小 δ 的方差

4. 【方向四】多时间尺度 GAE（Multi-Scale GAE）
   - GAE 只有一个 λ，控制一个时间尺度的偏差-方差权衡
   - 改进：同时维护多个 λ 尺度的 GAE，然后加权组合
     A_t = Σ_k w_k * A_t^{λ_k}   其中 {λ_k} = {0, 0.5, 0.9, 0.99}
     权重 w_k 可以是固定的（集成）或可学习的
   - 这类似于多尺度特征提取：不同 λ 捕捉不同时间跨度的信息

5. 【方向五】Return 分布感知优势（Distributional-Aware GAE）
   - 标准 GAE 用均值 V(s)，但忽略了 return 的分布形状
   - 改进：用分位数回归估计 V 的分布
     Q_τ(s) = τ-quantile of return distribution
     A_t^dist = r_t + γQ_median(s') - Q_median(s) + 风险调整项
   - 这让优势估计对高方差的回报更鲁棒
    """)

