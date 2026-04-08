#!/usr/bin/env python3
"""
HCGAE v4 设计分析：
核心问题是在高性能基线(256×256 MLP, deterministic eval)下：
1. SCR在训练早期就跌破1，导致MC校正有害
2. EV增长率门控存在但s_min=0.1仍然导致残余干扰
3. 高方差seed导致统计上的不稳定

HCGAE v4 的核心改进设计：
"""

print("="*70)
print("HCGAE v4 设计：核心问题和解决方案")
print("="*70)

print("""
【核心问题1】：大网络(256×256)下的SCR动态
─────────────────────────────────────────────
现有v2的EV增长率门控假设：
  τ_rate = 0.05 (触发阈值)
  τ_max = 0.15  (最大抑制阈值)
  s_min = 0.1   (最小保留比例)

在大网络下：
  小网络(旧): EV₀≈0.05, ΔEV/rollout≈0.01, 门控很少激活
  大网络(新): EV₀≈0.3-0.5, ΔEV/rollout≈0.05-0.15, 门控应该激活

但即使门控激活并达到s_min=0.1, 对于α_max=0.7:
  α_max_k_effective = 0.1 * (0.1 + (0.7-0.1) * cos_factor * ev_gate)
  在训练早期 (k小, cos≈1, ev_gate≈0.8):
  α_max_k ≈ 0.1 + 0.6 * 1.0 * 0.8 = 0.58
  有门控后: 0.58 * 0.1 = 0.058  ← 仍然不为0

  SCR<1时, 即使α=0.058的校正也是有害的！
""")

print("""
【核心问题2】：SCR是训练动态信号，不是静态参数
─────────────────────────────────────────────
SCR = |Bias[V]| / Std[G]

  旧实验(小网络, 随机eval):
    早期: Bias[V]≈50-100 (小网络初始化偏差大)
          Std[G]≈15 (随机策略高方差)
          SCR ≈ 5 >> 1  → MC校正有益

  新实验(大网络, 确定性eval):
    早期: Bias[V]≈10-30 (大网络初始化偏差小)
          Std[G]≈8-12 (更稳定的策略eval)
          SCR ≈ 2-3

    50K步后: Bias[V]迅速缩小 (大网络学习快)
             SCR → < 1
             → 此时MC校正有害

  → 需要实时估计SCR并动态控制校正强度！
""")

print("""
【HCGAE v4 核心设计：动态SCR自适应门控】
─────────────────────────────────────────────
核心思想：
  将SCR的实时估计与EV增长率门控联合使用
  允许在SCR>1时使用MC校正
  当SCR→1时平滑切换到纯GAE

【三项核心改进】:
──────────────────────────────────────────

① 零均值偏差校正（Zero-Mean Bias Correction）
   问题: δ = G_t - V(s_t) 的均值不为0，说明存在系统性偏差
   解法: 使用 δ_centered = G_t - V(s_t) - E[G_t - V(s_t)] 来更准确地判断
         即在计算 error e_t 时，减去当前 rollout 的均值偏差
   数学: e_t^centered = |G_t - V(s_t) - mean(G-V)| + bias_sign * |mean(G-V)|
         这样既保留了相对误差信息，又不丢失系统性偏差信息

② 实时 SCR 估计门控（Real-time SCR Gate）
   问题: 当前 v2 仅用 EV 增长率判断，没有直接测量 SCR
   解法: 每个 rollout 实时估计 SCR_t = |mean(G-V)| / std(G)
         当 SCR_ema < scr_threshold(=1.0) 时，将 α_max 缩放至 scr_min_scale
   数学公式:
     scr_t = |mean(G-V)| / (std(G) + ε)
     scr_ema = (1-ρ_scr) * scr_ema + ρ_scr * scr_t
     scr_scale = sigmoid(γ_scr * (scr_ema - scr_threshold))
               ∈ [scr_min_scale, 1.0]  (平滑 soft gate)

③ 动态底边界（Dynamic Floor）
   问题: s_min=0.1 是静态的，在 SCR 极低时仍然会引入噪声
   解法: 当 SCR_ema < scr_lo_threshold 时，将 s_min 动态降低至 0
         s_min_dynamic = s_min * clip(scr_ema / scr_lo_threshold, 0, 1)
   效果: SCR很低时, s_min → 0, α_max可以完全被压制
         SCR较高时, s_min = s_min_default (保留最小校正以避免完全零化)
""")

print("""
【数学推导：HCGAE v4 的最优校正强度】
─────────────────────────────────────────────
目标: 最小化 E[(V^c(s_t) - V^π(s_t))²]

V^c = (1-α)V + α*G
MSE(α) = E[(V^c - V^π)²]
       = E[((1-α)(V-V^π) + α(G-V^π))²]
       = (1-α)² B² + α² Var[G-V^π]
       = (1-α)² B² + α² σ_G²  (assuming V^π known)

dMSE/dα = -2(1-α)B² + 2α σ_G² = 0
→ α* = B² / (B² + σ_G²) = 1 / (1 + σ_G²/B²)
     = 1 / (1 + 1/SCR²)

当 SCR = 1: α* = 0.5
当 SCR = 0.5: α* = 0.2
当 SCR = 0.1: α* = 0.0099 ≈ 0.01
当 SCR = 2: α* = 0.8
当 SCR = 5: α* = 0.96

→ 这就是为什么需要实时追踪SCR：
  在新实验中SCR快速下降到<1，最优α*也迅速减小！
""")

print("""
【v4 实现的 SCR-Adaptive α*】
─────────────────────────────────────────────
实际实现中我们无法精确知道 B_t，但可以估计：
  bias_est = |mean(G-V)| ← 系统性偏差（均值偏差）
  noise_est = std(G)      ← MC回报的标准差
  SCR_hat = bias_est / (noise_est + ε)

然后最优 α 上界：
  α_max_scr = 1 / (1 + 1/SCR_hat²) = SCR_hat² / (1 + SCR_hat²)

这给了我们一个原则性的上界！

实际中结合cosine annealing 和 EV gate：
  α_max_v4(k) = min(α_max_v2(k), α_max_scr)

这样在 SCR_hat < 1 时，α_max_scr < 0.5，自动约束校正强度。
""")

# 数值验证
print("【数值验证：不同SCR下的最优α*】")
print(f"{'SCR':>6} | {'α*(理论)':>10} | {'当前v2 α_max约':>15} | {'问题':>20}")
print("-"*60)
for scr in [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]:
    alpha_opt = scr**2 / (1 + scr**2)
    # v2的alpha_max在早期(k=1, cos=1, ev_gate=0.8, ev_rate_scale=1):
    alpha_v2 = 0.1 + (0.7-0.1) * 1.0 * 0.8  # = 0.58
    issue = "⚠️ v2过度校正" if alpha_v2 > alpha_opt * 1.5 else "✓ 合理"
    print(f"{scr:6.1f} | {alpha_opt:10.3f} | {alpha_v2:15.3f} | {issue:>20}")

print("""
结论：当SCR<1时，v2的α_max=0.58远超最优α*，导致校正有害。
HCGAE v4通过实时估计SCR并约束α_max上界解决这个问题。
""")

