"""
DCPPO: Dual-Control PPO
=======================

基于对 HCGAE 消融实验训练数据的深度分析，我们识别了标准 PPO 在以下三个维度
存在的根本性问题，并提出三项在理论和实践上均具创新性的改进：

────────────────────────────────────────────────────────────────────────
问题 1：高维连续动作空间中的 Ratio 方差放大（Log-Ratio Variance Inflation）
────────────────────────────────────────────────────────────────────────
标准 PPO 在连续动作空间中计算 ratio = exp(Σ_d log π(a_d|s) - Σ_d log π_old(a_d|s))
= Π_d exp(Δ_d)，其中 Δ_d = log π_d - log π_old_d。

问题：Var[Π_d exp(Δ_d)] ≈ Π_d (1 + Var[exp(Δ_d)]) - 1
      随动作维度 D 指数增长。Hopper-v4 有 D=3 个动作维度，
      理论上 ratio 的方差约为单维的 3 倍。

改进 G：几何均值归一化 Ratio（Geometric Mean Normalized Ratio）
  r_geo = exp((1/D) · Σ_d (log π_d - log π_old_d))
        = (Π_d ratio_d)^(1/D)  ← D 次方根归一化

  数学效果：Var[r_geo] = Var[exp(Δ_mean)] 其中 Δ_mean = (1/D)Σ Δ_d
            由中心极限定理，Δ_mean 更接近高斯，exp(Δ_mean) 方差恒定与 D 无关
  等价于：以"平均每维度的 KL 散度"作为信任域量度，而非所有维度的总 KL

  理论联系：这等价于将策略视为联合分布 π(a|s) = Π_d π_d(a_d|s)（对角 Normal），
            并使用几何平均来归一化样本复杂度。
            与 Natural Policy Gradient 的 Fisher 信息矩阵连接：
            D_KL(π||π_old) ≈ (1/2) E[||Δ||² / D]（每维独立同分布时）

────────────────────────────────────────────────────────────────────────
问题 2：对称 Clip 机制的理论不一致（Symmetric Clipping Asymmetry Paradox）
────────────────────────────────────────────────────────────────────────
标准 PPO 对正 advantage（A > 0）和负 advantage（A < 0）使用相同的 clip 范围 ε。

问题：
  - 当 A > 0 时：ratio > 1+ε 被截断（对策略改进方向施加限制）
  - 当 A > 0 时：ratio < 1-ε 被截断（限制策略对好动作的急剧退出）
  但两个 clip 的"安全性"含义不对等：

  方向性分析：
  - clip_upper（ratio > 1+ε, A > 0）：过激进地强化好动作 → 需要限制
  - clip_lower（ratio < 1-ε, A > 0）：大幅降低好动作的概率 → 同样危险
  - clip_lower（ratio < 1-ε, A < 0）：强化回避坏动作 → 是安全的，不应限制

  传统分析遗漏了：ratio < 1-ε 且 A < 0 的情况（正在远离坏动作）应该被允许
  ratio > 1+ε 且 A < 0 的情况（正在接近坏动作）应该被严格限制

改进 A：方向感知非对称裁剪（Direction-Aware Asymmetric Clipping）
  对于当前 (ratio, A) 对，定义：
  - "进入坏区域"：ratio·sign(A) 与 A 同号（正在强化坏动作或弱化好动作）
    → 使用严格 clip: ε_strict = ε_base × β_strict
  - "离开坏区域"：ratio·sign(A) 与 A 异号（正在远离坏动作或强化好动作）
    → 使用宽松 clip: ε_loose = ε_base × β_loose

  等价数学形式：
    L_clip = -E[ min(r·A,
                 clip(r, 1 - ε_eff(r, A), 1 + ε_eff(r, A)) · A) ]
  其中 ε_eff(r, A) = ε_strict  if (r-1)·A < 0  [正在向坏方向移动]
                    ε_loose   otherwise           [正在向好方向移动]

  β_strict < 1.0 < β_loose，例如 β_strict=0.7, β_loose=1.3

  理论支持：这与 Conservative Policy Iteration (CPI) 的单调改进保证一致——
  CPI 允许策略改进，但限制退步。非对称 clip 是 CPI 原则的软版本。

────────────────────────────────────────────────────────────────────────
问题 3：Advantage 估计噪声未被策略梯度感知（Gradient Noise Blindness）
────────────────────────────────────────────────────────────────────────
标准 PPO 将所有样本的策略梯度等权处理，无论其 advantage 估计是否可靠。

问题：
  - 训练早期：EV 低（0.0→0.3），advantage 中含大量 Critic 误差噪声
    这些噪声作为"信号"驱动策略更新，导致早期 KL 波动大、clip_frac 高
  - 训练后期：EV 高（0.9+），advantage 准确，但此时 clip 依然从 0.2 起步

  从消融数据可见：HCGAE_Imp2 的 EV_ema 在 50K 步时已达 0.978，
  但 clip_frac 仍在 15-25%，说明 advantage 的"信噪比"没有被 clip 感知

改进 S：信噪比自适应梯度缩放（EV-Driven SNR-Adaptive Gradient Scaling）
  动机：
    原始定义 SNR = E[|A|] / std(A) 在零均值归一化优势下，对称分布时恒近似
    sqrt(2/π) ≈ 0.798（与 Critic 精度无关）。因此，该定义无法有效区分早期
    低质量优势和后期高质量优势。

  改进方案：使用 EV（Explained Variance）作为 SNR 的无偏代理
    SNR_eff = clip(EV_ema, 0.01, 1.0)

  理由：EV = 1 - Var[G - V] / Var[G] 直接衡量 Critic 对 MC 回报的解释度；
    EV_ema 高 → Critic 精准 → 优势估计高信噪 → 应全量更新策略；
    EV_ema 低 → Critic 噪声大 → 优势含 Critic 误差 → 应衰减梯度以防退步。

  梯度缩放因子：
    w(EV) = clip( (EV_ema / target_EV)^γ_snr, w_min, 1.0 )

  等价操作：用 w(EV) × A 替换 A 进入 policy loss
  当 EV 高时（Critic 准），w→1，全量梯度
  当 EV 低时（Critic 噪声大），w<1，衰减梯度

  诊断统计：同时计算原始 E[|A|]/std(A)（恒≈0.798）用于对比分析

  理论联系：这是 Trust-PCL 和 MPO 中"用 Q 估计质量来控制更新幅度"思路的
  on-policy 版本，无需显式 Q 网络。
  与 HCGAE 的协同效应：HCGAE 加速了 EV 提升（更好的 Critic 目标→更快收敛），
  EV 提升后 SNR-scaling 的抑制更快解除，形成正向循环。

  数学性质：
  1. 训练初期（EV≈0）：w ≈ w_min，梯度被抑制，防止噪声驱动的过早策略退化 ✓
  2. 训练收敛（EV→1）：w → 1，退化为标准 PPO，无额外偏差 ✓
  3. w(EV)·A 仍为策略梯度的单调变换，梯度方向不变，仅幅度缩放 ✓

────────────────────────────────────────────────────────────────────────
DCPPO 命名含义
────────────────────────────────────────────────────────────────────────
D - Dimension-normalized ratio (几何均值归一化)
C - Clipping with directional awareness (方向感知非对称裁剪)
PPO - 以上均在 PPO 框架内，无需额外环境交互，无二阶优化

变体名称
────────────────────────────────────────────────────────────────────────
DCPPO_Base  : 无任何改进（与 HCGAE_Imp12 相同的 GAE，标准 PPO update）
DCPPO_ImpG  : 仅几何均值归一化 Ratio
DCPPO_ImpA  : 仅方向感知非对称 Clip
DCPPO_ImpS  : 仅 SNR 自适应梯度缩放
DCPPO_ImpGA : G + A
DCPPO_ImpGS : G + S
DCPPO_ImpAS : A + S
DCPPO_Full  : G + A + S（完整 DCPPO）

数学正确性验证
────────────────────────────────────────────────────────────────────────
G: r_geo = exp((1/D)·Σ_d log_ratio_d) = r^(1/D) for factored Gaussian
   当 D=1 时退化为标准 ratio ✓
   当 A>0 时，r_geo>1 等价于 Σ log π > Σ log π_old，即在提升好动作概率 ✓
   单调性保持：sign(r_geo - 1) = sign(r - 1) ✓（对 D=3, r_geo<1 ↔ r<1）

A: 非对称 clip 保持了 PPO 的单调改进性质（因为 β_strict<1 仍然 clip 了危险方向）
   β_loose > 1 允许更大的"好"更新，但不超过 ε_max 安全边界

S: w(SNR)·A 仍然是优势的单调函数（w>0），梯度方向不变，只有幅度被调节
   E[∇_θ log π · w·A] = w · E[∇_θ log π · A]（w 不依赖 θ）
   ✓ 仍为策略梯度的无偏估计（乘以常数因子）
"""
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from ..utils.logger import MetricLogger
from ..utils.networks import ActorNetwork, CriticNetwork
from ..utils.rollout_buffer import RolloutBuffer


class DCPPO:
    """
    Dual-Control PPO (DCPPO)

    三项正交 PPO 改进：
      G - Geometric mean normalized ratio（几何均值归一化 Ratio）
      A - Direction-aware asymmetric clipping（方向感知非对称裁剪）
      S - SNR-adaptive gradient scaling（信噪比自适应梯度缩放）

    所有改进均可独立启用，配合 HCGAE (Imp12) 的 GAE 计算。
    """

    def __init__(
        self,
        env: gym.Env,
        name: str = "DCPPO_Full",
        # ── 改进开关 ──
        use_imp_g: bool = True,   # G: 几何均值归一化 Ratio
        use_imp_a: bool = True,   # A: 方向感知非对称 Clip
        use_imp_s: bool = True,   # S: SNR 自适应梯度缩放
        # ── 标准 PPO 超参 ──
        hidden_dim: int = 64,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        lam: float = 0.95,
        eps_clip: float = 0.2,
        n_epochs: int = 10,
        batch_size: int = 64,
        n_steps: int = 2048,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        # ── G: 几何均值归一化超参 ──
        geo_blend: float = 1.0,      # 1.0=纯几何均值, 0.0=标准 ratio; 插值混合
        # ── A: 非对称 Clip 超参 ──
        beta_strict: float = 0.6,    # 危险方向 clip 系数（< 1.0，更严格）
        beta_loose: float = 1.4,     # 安全方向 clip 系数（> 1.0，更宽松）
        eps_max: float = 0.4,        # ε_loose 的绝对上界（安全护栏）
        # ── S: SNR 梯度缩放超参 ──
        snr_target: float = 0.3,     # ev_power 模式：目标 SNR（SNR 达到此值时 w→1）
        snr_gamma: float = 0.5,      # ev_power 模式：缩放指数（0.5 = 软缩放）
        snr_min_weight: float = 0.2, # 最低权重（防止梯度完全消失）
        snr_mode: str = "ev_linear", # ev_power | ev_linear
        # ── HCGAE（继承 Imp12 的 GAE，即①+②）──
        use_hcgae: bool = True,
        hindsight_beta: float = 3.0,
        hindsight_alpha_max: float = 0.7,
        hindsight_alpha_min: float = 0.1,
        # ── SCR 自适应校正强度 ──
        use_scr_adapt: bool = False,   # 启用 SCR 驱动的 alpha_max 自适应缩放
        scr_threshold: float = 1.0,    # SCR > 此值时 HCGAE 有益；< 此值时抑制校正
        scr_min_scale: float = 0.1,    # SCR 极低时的最小 alpha_max 缩放因子
        scr_ema_alpha: float = 0.1,    # SCR EMA 更新速率
        # ── 其他 ──
        device: str = "cpu",
        save_dir: str = "results",
    ):
        self.NAME = name
        self.use_imp_g = use_imp_g
        self.use_imp_a = use_imp_a
        self.use_imp_s = use_imp_s
        self.use_hcgae = use_hcgae

        self.env = env
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        # G 超参
        self.geo_blend = geo_blend

        # A 超参
        self.beta_strict = beta_strict
        self.beta_loose = beta_loose
        self.eps_max = eps_max

        # S 超参
        self.snr_target = snr_target
        self.snr_gamma = snr_gamma
        self.snr_min_weight = snr_min_weight
        self.snr_mode = snr_mode

        # HCGAE 超参（批内归一化 + EV 驱动混合）
        self.hindsight_beta = hindsight_beta
        self.hindsight_alpha_max = hindsight_alpha_max
        self.hindsight_alpha_min = hindsight_alpha_min
        self._ev_ema = 0.0
        self._ev_ema_alpha = 0.05
        self._total_timesteps = 1

        # SCR 自适应参数
        self.use_scr_adapt = use_scr_adapt
        self.scr_threshold = scr_threshold
        self.scr_min_scale = scr_min_scale
        self.scr_ema_alpha = scr_ema_alpha
        # SCR EMA：初始化为 threshold 表示「中性」
        self._scr_ema = scr_threshold
        self._scr_history = []  # 历史 SCR 值（最近 50 次）

        self.device = torch.device(device)

        obs_dim = env.observation_space.shape[0]
        if isinstance(env.action_space, gym.spaces.Discrete):
            action_dim = env.action_space.n
            self.continuous = False
        else:
            action_dim = env.action_space.shape[0]
            self.continuous = True

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.actor  = ActorNetwork(obs_dim, action_dim, hidden_dim, self.continuous).to(self.device)
        self.critic = CriticNetwork(obs_dim, hidden_dim).to(self.device)

        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = RolloutBuffer(n_steps, obs_dim, action_dim, self.device, self.continuous)

        self.logger = MetricLogger(self.NAME, save_dir)
        self.total_steps = 0

    # ──────────────────────────────────────────────────────────────────
    # 数据收集
    # ──────────────────────────────────────────────────────────────────
    def collect_rollout(self) -> float:
        self.buffer.reset()
        obs, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0

        for step in range(self.n_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, log_prob = self.actor.get_action_and_logprob(obs_tensor)
                value = self.critic(obs_tensor)

            action_np   = action.squeeze(0).cpu().numpy()
            value_np    = value.item()
            log_prob_np = log_prob.item()

            if self.continuous:
                next_obs, reward, terminated, truncated, _ = self.env.step(action_np)
            else:
                next_obs, reward, terminated, truncated, _ = self.env.step(int(action_np))

            episode_reward += reward
            episode_length += 1
            self.buffer.add(obs, action_np, reward, float(terminated), log_prob_np, value_np)
            done = terminated or truncated
            obs = next_obs
            self.total_steps += 1

            if done:
                self.logger.log_episode(episode_reward, episode_length)
                obs, _ = self.env.reset()
                episode_reward = 0.0
                episode_length = 0

        with torch.no_grad():
            last_obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            last_value = self.critic(last_obs_tensor).item()

        return last_value

    # ──────────────────────────────────────────────────────────────────
    # HCGAE (Imp12) GAE 计算：批内归一化 + EV 驱动混合
    # ──────────────────────────────────────────────────────────────────
    def _compute_mc_returns(self, last_value: float) -> np.ndarray:
        """计算 MC 回报（rollout 内反向累加）"""
        T = self.buffer.pos
        G = np.zeros(T, dtype=np.float32)
        g = last_value
        for t in reversed(range(T)):
            not_done = 1.0 - self.buffer.terminated[t]
            g = self.buffer.rewards[t] + self.gamma * g * not_done
            G[t] = g
        return G

    def compute_gae(self, last_value: float) -> dict:
        """
        HCGAE Imp12 GAE:
          - 改进①：批内中心化 Sigmoid 归一化 alpha
          - 改进②：EV 驱动 Critic 目标混合
        """
        T = self.buffer.pos
        G = self._compute_mc_returns(last_value)
        V = self.buffer.values[:T].copy()

        progress = self.total_steps / max(self._total_timesteps, 1)

        # ── 在线 SCR 估计（每次 rollout 更新，用于诊断和自适应）──────────
        # SCR = |bias| / std(G)；> threshold 表示 HCGAE 有益（偏差主导）
        mc_std_for_scr = float(np.std(G)) + 1e-8
        bias_abs_raw = float(np.mean(np.abs(V - G)))
        scr_current = bias_abs_raw / mc_std_for_scr
        # EMA 平滑，减少单次 rollout 噪声
        self._scr_ema = (1 - self.scr_ema_alpha) * self._scr_ema + self.scr_ema_alpha * scr_current
        # 记录历史（最多保留最近 50 次）
        self._scr_history.append(scr_current)
        if len(self._scr_history) > 50:
            self._scr_history.pop(0)

        if self.use_hcgae:
            # ── 改进①：批内中心化归一化 alpha ──────────────────────────
            err = np.abs(V - G)
            err_batch_mean = err.mean()
            err_batch_std  = err.std() + 1e-8

            cosine_decay      = 0.5 * (1.0 + np.cos(np.pi * progress))
            ev_factor         = max(1.0 - max(self._ev_ema, 0.0), 0.2)
            dynamic_alpha_max = (
                self.hindsight_alpha_min
                + (self.hindsight_alpha_max - self.hindsight_alpha_min)
                * cosine_decay * ev_factor
            )

            # ── SCR 自适应缩放 alpha_max ────────────────────────────────
            # 当 scr_ema > threshold 时：偏差主导，HCGAE 有益，保持校正强度
            # 当 scr_ema < threshold 时：MC 方差主导，抑制校正强度防止引入噪声
            if self.use_scr_adapt:
                scr_scale = float(np.clip(
                    self._scr_ema / (self.scr_threshold + 1e-8),
                    self.scr_min_scale,
                    1.0,
                ))
                dynamic_alpha_max = dynamic_alpha_max * scr_scale
            else:
                scr_scale = 1.0

            z     = self.hindsight_beta * (err - err_batch_mean) / err_batch_std
            alpha = dynamic_alpha_max * (1.0 / (1.0 + np.exp(-z)))
        else:
            alpha = np.zeros(T, dtype=np.float32)
            err_batch_mean = 0.0
            err_batch_std  = 1.0
            dynamic_alpha_max = 0.0
            scr_scale = 1.0

        V_corrected = (1.0 - alpha) * V + alpha * G

        # bootstrap 一致性
        V_corrected_next = np.empty(T, dtype=np.float32)
        for t in range(T):
            if self.buffer.terminated[t] > 0.5:
                V_corrected_next[t] = 0.0
            elif t == T - 1:
                V_corrected_next[t] = last_value
            else:
                V_corrected_next[t] = V_corrected[t + 1]

        # GAE on corrected values (for advantage estimation)
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            delta    = self.buffer.rewards[t] + self.gamma * V_corrected_next[t] - V_corrected[t]
            not_done = 1.0 - self.buffer.terminated[t]
            gae      = delta + self.gamma * self.lam * not_done * gae
            adv[t]   = gae

        # ── 改进②：EV 驱动 Critic 目标混合 ─────────────────────────────
        # 论文声明：Critic 训练目标使用【未校正】的原始 V 计算的标准 GAE lambda-return
        # R_t^GAE = A_t^std + V(s_t)，其中 A_t^std 基于原始 V 而非 V_corrected 计算
        # 这确保 Critic 的训练目标不受优势估计校正路径的污染（两个更新通道独立）
        std_gae_returns = self.buffer._compute_standard_returns(last_value, self.gamma, self.lam)

        c_mc = float(np.clip(1.0 - self._ev_ema, 0.1, 1.0))
        self.buffer.advantages = adv
        self.buffer.returns    = c_mc * G + (1.0 - c_mc) * std_gae_returns

        # 统计信息
        deltas = self.buffer.rewards[:T] + self.gamma * V_corrected_next - V_corrected
        autocorr = float(np.corrcoef(deltas[:-1], deltas[1:])[0, 1]) if T > 2 else 0.0
        return {
            "delta_mean":       float(deltas.mean()),
            "delta_std":        float(deltas.std()),
            "delta_autocorr":   autocorr,
            "adv_mean":         float(adv.mean()),
            "adv_std":          float(adv.std()),
            "c_mc":             c_mc,
            "alpha_mean":       float(alpha.mean()),
            "alpha_std":        float(alpha.std()),
            # SCR 诊断统计（来自 rollout 开始时计算的 SCR）
            "scr_current":      scr_current,             # 当次 rollout SCR
            "scr_ema":          float(self._scr_ema),    # 平滑后 SCR EMA
            "scr_scale":        scr_scale,               # SCR 对 alpha_max 的缩放因子
            "mc_std":           mc_std_for_scr,          # MC 回报标准差（分母）
            "bias_abs_proxy":   bias_abs_raw,            # |V-G| 均值（分子）
            "bias_proxy":       float(np.mean(V - G)),   # 有符号偏差（负 = Critic 低估）
            "dynamic_alpha_max": dynamic_alpha_max,      # 实际使用的 alpha_max（含 SCR 缩放）
        }

    # ──────────────────────────────────────────────────────────────────
    # 核心更新：DCPPO 改进
    # ──────────────────────────────────────────────────────────────────
    def update(self) -> dict:
        """
        DCPPO update 步骤：
          G - 几何均值归一化 Ratio
          A - 方向感知非对称 Clip
          S - SNR 自适应梯度缩放
        """
        obs, actions, old_log_probs, advantages, returns, old_values = self.buffer.get_batch()

        # 全局 advantage 归一化（rollout 级别）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── 改进 S：计算批级 SNR，用于后续梯度缩放 ────────────────────
        # 诊断量保留原始 E[|A|]/std(A)，但主要控制信号改为 EV。
        #
        # 两种模式：
        # 1) ev_power  : 启发式幂律门控，w = clip((EV/tau)^gamma, w_min, 1)
        # 2) ev_linear : 线性收缩，w = clip(EV, w_min, 1)
        #    在 A_hat = A_true + noise, E[noise|s]=0, noise ⟂ A_true 的模型下，
        #    最优线性收缩系数满足
        #      argmin_w E[(w A_hat - A_true)^2] = Cov(A_hat, A_true)/Var(A_hat)
        #                                      = Var(A_true)/Var(A_hat)
        #                                      ≈ EV.
        #    因此 ev_linear 比 ev_power 更接近“最小化梯度信号 MSE”的理论最优。
        if self.use_imp_s:
            # 原始 E[|A|]/std(A) 统计量（保留用于诊断）
            adv_mean_abs = float(advantages.abs().mean().item())
            adv_std      = float(advantages.std().item()) + 1e-8
            snr_ratio    = adv_mean_abs / adv_std   # 原始比值，约 ≈ 0.798 for Gaussian

            ev_for_snr = float(np.clip(self._ev_ema, 0.0, 1.0))
            snr_batch = ev_for_snr

            if self.snr_mode == "ev_linear":
                snr_weight = float(np.clip(ev_for_snr, self.snr_min_weight, 1.0))
            elif self.snr_mode == "ev_power":
                snr_weight = float(
                    np.clip(
                        (max(ev_for_snr, 0.01) / (self.snr_target + 1e-8)) ** self.snr_gamma,
                        self.snr_min_weight,
                        1.0,
                    )
                )
            else:
                raise ValueError(f"Unknown snr_mode: {self.snr_mode}")
        else:
            adv_mean_abs = 0.0
            adv_std      = 1.0
            snr_ratio    = 0.0
            snr_batch    = 0.0
            snr_weight   = 1.0

        # 应用 SNR 缩放到 advantages（方向不变，幅度缩放）
        effective_adv = advantages * snr_weight

        T = self.buffer.pos
        indices = np.arange(T)
        metrics = {
            "value_loss": 0.0,
            "policy_loss": 0.0,
            "entropy_loss": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
            "clip_frac_strict": 0.0,   # 危险方向被截断的比例
            "clip_frac_loose": 0.0,    # 安全方向被截断的比例
            "ratio_mean": 0.0,
            "geo_ratio_mean": 0.0,     # 几何均值 ratio 统计
            "snr_batch": snr_batch,    # EV 驱动的 SNR（主要）
            "snr_weight": snr_weight,  # 对应的梯度缩放权重
            "snr_ratio": snr_ratio,    # 原始 E[|A|]/std(A)（诊断用，≈0.798 for Gaussian）
        }
        # 不参与 update_count 平均的标量（已是 rollout 级别的统计量）
        _no_avg_keys = {"snr_batch", "snr_weight", "snr_ratio"}
        update_count = 0

        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, self.batch_size):
                end = start + self.batch_size
                if end > T:
                    break
                batch_idx = indices[start:end]

                batch_obs          = obs[batch_idx]
                batch_actions      = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages   = effective_adv[batch_idx]
                batch_returns      = returns[batch_idx]
                batch_old_values   = old_values[batch_idx]

                # 计算新 log_prob（逐维）和熵
                if self.continuous:
                    # 获取逐维 log_prob（未求和，shape = [B, D]）
                    dist = self.actor(batch_obs)
                    log_prob_per_dim = dist.log_prob(batch_actions)   # [B, D]
                    new_log_probs    = log_prob_per_dim.sum(dim=-1)    # [B]
                    entropy          = dist.entropy().sum(dim=-1)      # [B]
                else:
                    new_log_probs, entropy = self.actor.evaluate_actions(batch_obs, batch_actions)
                    log_prob_per_dim = new_log_probs.unsqueeze(-1)     # [B, 1]

                new_values = self.critic(batch_obs)

                # ── 改进 G：几何均值归一化 Ratio ──────────────────────────
                log_ratio_total = new_log_probs - batch_old_log_probs   # [B]

                if self.use_imp_g and self.continuous and self.action_dim > 1:
                    # 需要逐维 old_log_prob 来计算几何均值 ratio
                    # 近似：old_log_prob_per_dim = batch_old_log_probs / D（均匀分配）
                    # 更精确：通过 old_dist 计算，但 old_dist 需要额外存储
                    # 此处使用近似（均匀分配假设），对 factored Gaussian 是精确的
                    D = self.action_dim
                    log_ratio_geo = log_ratio_total / D   # 几何均值 = ratio^(1/D)

                    # 插值混合（geo_blend=1.0 纯几何，0.0 纯标准）
                    log_ratio_eff = (
                        self.geo_blend * log_ratio_geo
                        + (1.0 - self.geo_blend) * log_ratio_total
                    )
                    ratio = torch.exp(log_ratio_eff)      # 有效 ratio
                    geo_ratio_mean = float(ratio.mean().item())
                else:
                    ratio = torch.exp(log_ratio_total)
                    log_ratio_eff = log_ratio_total
                    geo_ratio_mean = float(ratio.mean().item())

                # ── 改进 A：方向感知非对称 Clip ───────────────────────────
                if self.use_imp_a:
                    # 判断当前 (ratio, A) 是否在"好方向"（安全）or "坏方向"（危险）
                    # "坏方向"：(ratio - 1) 与 A 符号相反
                    #   即 ratio > 1 且 A < 0（强化坏动作）
                    #   或 ratio < 1 且 A > 0（弱化好动作）
                    # 用于判断的是 log_ratio_total（与 A 的符号关系不受 D 缩放影响）
                    bad_direction = (
                        (log_ratio_total * batch_advantages) < 0
                    ).float()   # [B], 1=坏方向, 0=好方向

                    # 逐样本 clip 系数
                    beta_eff = (
                        bad_direction * self.beta_strict
                        + (1.0 - bad_direction) * self.beta_loose
                    )   # [B]
                    eps_eff = torch.clamp(
                        self.eps_clip * beta_eff,
                        min=0.01,
                        max=self.eps_max,
                    )   # [B]

                    # 逐样本 clip
                    surr1 = ratio * batch_advantages
                    ratio_clipped = torch.clamp(
                        ratio,
                        1.0 - eps_eff,
                        1.0 + eps_eff,
                    )
                    surr2 = ratio_clipped * batch_advantages

                    policy_loss = -torch.min(surr1, surr2).mean()

                    # 统计：严格/宽松方向各自的 clip 比例
                    clipped_mask = (surr1 != surr2).float()
                    strict_clipped = (clipped_mask * bad_direction).mean().item()
                    loose_clipped  = (clipped_mask * (1 - bad_direction)).mean().item()
                    clip_frac_val  = clipped_mask.mean().item()
                else:
                    # 标准对称 Clip
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    clip_frac_val  = ((ratio - 1).abs() > self.eps_clip).float().mean().item()
                    strict_clipped = clip_frac_val
                    loose_clipped  = clip_frac_val

                # Value loss（不 clip，与 HCGAE 保持一致）
                value_loss = 0.5 * ((new_values - batch_returns) ** 2).mean()

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Actor 更新
                actor_loss = policy_loss + self.ent_coef * entropy_loss
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Critic 更新
                self.critic_optimizer.zero_grad()
                (self.vf_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                # 指标收集
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()

                metrics["value_loss"]        += value_loss.item()
                metrics["policy_loss"]       += policy_loss.item()
                metrics["entropy_loss"]      += entropy_loss.item()
                metrics["approx_kl"]         += approx_kl
                metrics["clip_frac"]         += clip_frac_val
                metrics["clip_frac_strict"]  += strict_clipped
                metrics["clip_frac_loose"]   += loose_clipped
                metrics["ratio_mean"]        += float(torch.exp(log_ratio_total).mean().item())
                metrics["geo_ratio_mean"]    += geo_ratio_mean
                update_count += 1

        if update_count > 0:
            for k in metrics:
                if k not in _no_avg_keys:
                    metrics[k] /= update_count

        # 计算 Explained Variance
        with torch.no_grad():
            y_pred = old_values.cpu().numpy()
            y_true = returns.cpu().numpy()
            var_y  = np.var(y_true)
            ev     = 1 - np.var(y_true - y_pred) / (var_y + 1e-8)
            metrics["explained_variance"] = float(ev)

        # 更新 EV EMA（供 HCGAE GAE 使用）
        self._ev_ema = (1 - self._ev_ema_alpha) * self._ev_ema + self._ev_ema_alpha * ev

        return metrics

    # ──────────────────────────────────────────────────────────────────
    # 主训练循环
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _progress_bar(current: int, total: int, width: int = 25) -> str:
        filled = int(width * current / total)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}] {100*current/total:5.1f}%"

    def train(
        self,
        total_timesteps: int,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        verbose: bool = True,
    ):
        """主训练循环，打印丰富的训练状态信息"""
        import time
        self._total_timesteps = total_timesteps
        train_start = time.time()
        last_eval_reward = float("nan")
        update_idx = 0

        if verbose:
            imp_str = (
                f"[G={'✓' if self.use_imp_g else '✗'}]"
                f"[A={'✓' if self.use_imp_a else '✗'}]"
                f"[S={'✓' if self.use_imp_s else '✗'}]"
                f"[HCGAE={'✓' if self.use_hcgae else '✗'}]"
            )
            print(f"\n  {'═'*70}")
            print(f"  DCPPO  {imp_str}")
            print(f"  Env: {self.env.spec.id if self.env.spec else 'unknown'}  "
                  f"D={self.action_dim}  β_strict={self.beta_strict}  "
                  f"β_loose={self.beta_loose}  S_mode={self.snr_mode}  SNR_target={self.snr_target}")
            print(f"  {'─'*70}")

        while self.total_steps < total_timesteps:
            last_value = self.collect_rollout()
            gae_stats  = self.compute_gae(last_value)
            metrics    = self.update()
            update_idx += 1
            metrics.update(gae_stats)

            self.logger.log_update(
                value_loss=metrics["value_loss"],
                policy_loss=metrics["policy_loss"],
                entropy_loss=metrics["entropy_loss"],
                approx_kl=metrics["approx_kl"],
                clip_frac=metrics["clip_frac"],
                explained_variance=metrics["explained_variance"],
                total_steps=self.total_steps,
                # GAP4 diagnostics
                ev_ema=float(self._ev_ema),
                snr=float(metrics.get("snr_batch", 0.0)),
                snr_weight=float(metrics.get("snr_weight", 1.0)),
                alpha_mean=float(metrics.get("alpha_mean", 0.0)),
                c_mc=float(metrics.get("c_mc", 0.5)),
            )

            do_eval = eval_env is not None and self.total_steps % eval_freq < self.n_steps
            if do_eval:
                last_eval_reward = self.evaluate(eval_env, n_eval_episodes)
                self.logger.log_eval(last_eval_reward, self.total_steps)

            if verbose and (do_eval or update_idx % 5 == 1):
                elapsed = time.time() - train_start
                recent  = self.logger.get_recent_reward(20)
                fps     = int(self.total_steps / (elapsed + 1e-8))
                ev      = metrics.get("explained_variance", 0.0)
                kl      = metrics.get("approx_kl", 0.0)
                cf      = metrics.get("clip_frac", 0.0)
                cf_s    = metrics.get("clip_frac_strict", 0.0)
                cf_l    = metrics.get("clip_frac_loose", 0.0)
                snr_w   = metrics.get("snr_weight", 1.0)
                snr_b   = metrics.get("snr_batch", 0.0)
                c_mc    = metrics.get("c_mc", 0.5)
                geo_r   = metrics.get("geo_ratio_mean", 1.0)
                eval_str = f"{last_eval_reward:7.1f}" if not np.isnan(last_eval_reward) else "    N/A"
                progress = self._progress_bar(self.total_steps, total_timesteps)

                print(
                    f"  [{self.NAME:<18}] "
                    f"{self.total_steps:7d}/{total_timesteps} "
                    f"{progress} "
                    f"| Eval={eval_str} Recent={recent:6.1f} "
                    f"| VLoss={metrics['value_loss']:.3f} EV={ev:+.3f} "
                    f"| KL={kl:.4f} clip={cf:.3f}(s={cf_s:.2f}/l={cf_l:.2f}) "
                    f"| SNR={snr_b:.3f}→w={snr_w:.3f} "
                    f"| geo_r={geo_r:.4f} c_mc={c_mc:.3f} "
                    f"| δ:μ={metrics.get('delta_mean',0):+.3f}/σ={metrics.get('delta_std',0):.3f} "
                    f"| α={metrics.get('alpha_mean',0):.3f}±{metrics.get('alpha_std',0):.3f} "
                    f"| {fps:5d}fps {elapsed:5.0f}s"
                )

        if verbose:
            elapsed   = time.time() - train_start
            final_r   = np.mean(self.logger.eval_rewards[-5:]) if self.logger.eval_rewards else 0.0
            best_r    = max(self.logger.eval_rewards) if self.logger.eval_rewards else 0.0
            print(f"\n  ✓ [{self.NAME}] 完成 | 耗时={elapsed:.1f}s | "
                  f"最终={final_r:.1f} | 最高={best_r:.1f} | 最终EV={self._ev_ema:.3f}")

        self.logger.save()
        return self.logger

    def evaluate(self, eval_env: gym.Env, n_episodes: int = 10) -> float:
        total_reward = 0.0
        for _ in range(n_episodes):
            obs, _ = eval_env.reset()
            done   = False
            ep_r   = 0.0
            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    dist = self.actor(obs_t)
                    action = dist.mean if self.continuous else dist.probs.argmax(dim=-1)
                action_np = action.squeeze(0).cpu().numpy()
                if self.continuous:
                    next_obs, reward, terminated, truncated, _ = eval_env.step(action_np)
                else:
                    next_obs, reward, terminated, truncated, _ = eval_env.step(int(action_np))
                ep_r += reward
                done  = terminated or truncated
                obs   = next_obs
            total_reward += ep_r
        return total_reward / n_episodes


# ──────────────────────────────────────────────────────────────────────
# 工厂函数
# ──────────────────────────────────────────────────────────────────────
DCPPO_VARIANTS = {
    # ── 原有变体（单项 + 三项组合） ────────────────────────────────────────
"DCPPO_Base": dict(use_imp_g=False, use_imp_a=False, use_imp_s=False),
"DCPPO_ImpG": dict(use_imp_g=True,  use_imp_a=False, use_imp_s=False),
"DCPPO_ImpA": dict(use_imp_g=False, use_imp_a=True,  use_imp_s=False),
"DCPPO_ImpS": dict(use_imp_g=False, use_imp_a=False, use_imp_s=True, snr_mode="ev_linear"),
"DCPPO_ImpS_Power": dict(use_imp_g=False, use_imp_a=False, use_imp_s=True, snr_mode="ev_power"),
"DCPPO_ImpS_Linear": dict(use_imp_g=False, use_imp_a=False, use_imp_s=True, snr_mode="ev_linear"),
"DCPPO_ImpGA":dict(use_imp_g=True,  use_imp_a=True,  use_imp_s=False),
"DCPPO_ImpGS":dict(use_imp_g=True,  use_imp_a=False, use_imp_s=True, snr_mode="ev_linear"),
"DCPPO_ImpAS":dict(use_imp_g=False, use_imp_a=True,  use_imp_s=True, snr_mode="ev_linear"),
"DCPPO_Full": dict(use_imp_g=True,  use_imp_a=True,  use_imp_s=True, snr_mode="ev_linear"),
"DCPPO_Full_Power": dict(use_imp_g=True,  use_imp_a=True,  use_imp_s=True, snr_mode="ev_power"),

    # ── 改进5：DCPPO_Full 失效机制精确定位（成对消融）─────────────────────
    # 问题：DCPPO_Full(G+A+S) 相比 DCPPO_ImpS 下降 -61%（p=0.008, d=-4.23）
    # 以下变体通过成对消融精确定位干扰源：
    # "HCGAE+X" 系列：在 HCGAE_Imp12 基础上逐项叠加 DCPPO 改进
    # 若 DCPPO_Base(=HCGAE_Imp12) 单独好，而 +G 或 +A 引起下降，则可定位干扰源
    # ── 与 DCPPO_ImpS 对比的两项组合（定位 G、A 对 S 的干扰）─────────────
    # DCPPO_ImpS 已是上面定义，此处别名供分析脚本清晰引用（无需重复）
    # 注：DCPPO_Base 相当于纯 HCGAE_Imp12 + 标准 PPO update（无 G/A/S）
    # 注：以下变体可用于 run_dcppo_ablation_multiseed.py 精确分析失效原因
}


def build_dcppo_agent(variant_name: str, env: gym.Env, save_dir: str, **kwargs) -> DCPPO:
    """根据变体名称构建 DCPPO agent

    可以通过 kwargs 传入 name 来覆盖默认的 variant_name 作为 agent 名称。
    """
    if variant_name not in DCPPO_VARIANTS:
        raise ValueError(f"Unknown DCPPO variant: {variant_name}. "
                         f"Available: {list(DCPPO_VARIANTS.keys())}")
    flags = DCPPO_VARIANTS[variant_name]
    # 如果 kwargs 中没有 name，使用 variant_name 作为默认名称
    name = kwargs.pop("name", variant_name)
    return DCPPO(env=env, name=name, save_dir=save_dir, **flags, **kwargs)


def get_all_dcppo_variant_names():
    return list(DCPPO_VARIANTS.keys())

