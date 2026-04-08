"""
HCGAE 消融实验：逐项/组合验证四项 v2 改进的贡献量

改进列表
─────────────────────────────────────────────────────────────────
①  批内中心化 Sigmoid 归一化（替代慢速 EMA 分母）
②  EV 驱动的 Critic 目标混合系数（替代固定 50-50）
③  末端 Bootstrap 一致性修正
④  优势归一化统计量冻结（在 compute_gae 阶段计算，update 全程复用）
─────────────────────────────────────────────────────────────────

变体矩阵
─────────────────────────────────────────────────────────────────
名称            ①   ②   ③   ④   说明
HCGAE_Base      ✗   ✗   ✗   ✗   v1 风格基线（EMA 归一化 + 固定混合）
HCGAE_Imp1      ✓   ✗   ✗   ✗   仅批内中心化归一化
HCGAE_Imp2      ✗   ✓   ✗   ✗   仅 EV 驱动混合
HCGAE_Imp3      ✗   ✗   ✓   ✗   仅末端 Bootstrap 修正
HCGAE_Imp4      ✗   ✗   ✗   ✓   仅冻结优势统计量
HCGAE_Imp12     ✓   ✓   ✗   ✗   ①+② 组合
HCGAE_Imp124    ✓   ✓   ✗   ✓   ①+②+④（不含末端修正）
HCGAE_Full      ✓   ✓   ✓   ✓   全量 v2（= 正式版本）
─────────────────────────────────────────────────────────────────

SCR (Signal-to-Correction Ratio) 在线估计器
─────────────────────────────────────────────────────────────────
SCR = |Bias_t| / Var[G_t]^{1/2}

诊断逻辑：
  SCR > scr_threshold (默认 1.0)：Critic 偏差 > MC 方差 → HCGAE 有益
  SCR < scr_threshold：MC 方差主导 → 校正反而引入噪声 → 适当抑制校正

实现方式：
  - 使用滑动平均 SCR EMA（alpha=0.1）避免单次 rollout 噪声干扰
  - 当 use_scr_adapt=True 时，alpha_max 会根据 SCR EMA 动态缩放
  - scr_scale_factor = clip(scr_ema / scr_threshold, scr_min_scale, 1.0)
  - 这将 HCGAE 从「只对某些环境有益」→「自适应所有环境安全部署」
─────────────────────────────────────────────────────────────────
"""
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from .optimal_ppo import RunningMeanStd
from ..utils.logger import MetricLogger
from ..utils.networks import ActorNetwork, CriticNetwork
from ..utils.rollout_buffer import RolloutBuffer


class HindsightAblation:
    """
    HCGAE 消融实验基类：通过布尔开关控制各项改进的启用/禁用。
    所有变体共享完全相同的代码路径，仅通过开关决定行为。
    """

    def __init__(
        self,
        env: gym.Env,
        name: str,
        # 改进开关
        use_imp1: bool = False,   # ① 批内中心化归一化
        use_imp2: bool = False,   # ② EV 驱动混合（Critic 目标用原始 V 的标准 GAE，与优势估计路径完全分离）
        use_imp3: bool = False,   # ③ 末端 Bootstrap 修正
        use_imp4: bool = False,   # ④ 冻结优势统计量
        # SCR 自适应校正强度（⑤ 在线 SCR 估计 + 自动环境适配）
        use_scr_adapt: bool = False,   # ⑤ SCR 驱动的校正强度自适应
        scr_threshold: float = 1.0,    # SCR > 此值时 HCGAE 有益；< 此值时抑制校正
        scr_min_scale: float = 0.1,    # SCR 极低时的最小 alpha_max 缩放因子
        scr_ema_alpha: float = 0.1,    # SCR EMA 更新速率
        # 自适应 β（⑥ 基于 err 分布峰度的自动 β 调整）
        use_adapt_beta: bool = False,  # ⑥ 自适应 β：根据 err 峰度自动调整 sigmoid 陡峭度
        adapt_beta_gamma: float = 0.5, # β_t = β₀ / max(Kurt[e]^γ, 1.0)，γ 控制调整幅度
        # 标准化开关（与文献对齐）
        use_obs_norm: bool = True,     # ★ Obs normalization (running mean/std)
        use_adv_norm: bool = True,     # ★ Per-minibatch advantage normalization
        # 标准 PPO 超参
        hidden_dim: int = 256,
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
        # HCGAE 超参
        hindsight_beta: float = 3.0,
        hindsight_alpha_max: float = 0.7,
        hindsight_alpha_min: float = 0.1,
        device: str = "cpu",
        save_dir: str = "results",
    ):
        self.NAME = name
        self.use_imp1 = use_imp1
        self.use_imp2 = use_imp2
        self.use_imp3 = use_imp3
        self.use_imp4 = use_imp4
        # SCR 自适应参数
        self.use_scr_adapt = use_scr_adapt
        self.scr_threshold = scr_threshold
        self.scr_min_scale = scr_min_scale
        self.scr_ema_alpha = scr_ema_alpha
        # 自适应 β 参数
        self.use_adapt_beta = use_adapt_beta
        self.adapt_beta_gamma = adapt_beta_gamma

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
        self.hindsight_beta = hindsight_beta
        self.hindsight_alpha_max = hindsight_alpha_max
        self.hindsight_alpha_min = hindsight_alpha_min
        self.use_obs_norm = use_obs_norm
        self.use_adv_norm = use_adv_norm
        self.device = torch.device(device)
        self._total_timesteps = 1

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

        # Running stats for observation normalization
        if use_obs_norm:
            self.obs_rms = RunningMeanStd(shape=(obs_dim,))
        else:
            self.obs_rms = None

        # 慢速 EMA（v1 基线使用）
        self._err_ema = 1.0
        self._err_ema_alpha = 0.05
        # EV EMA（② 使用）
        self._ev_ema = 0.0
        self._ev_ema_alpha = 0.1
        # 冻结统计量（④ 使用）
        self._adv_mean_frozen = 0.0
        self._adv_std_frozen  = 1.0
        # SCR EMA（⑤ 在线 SCR 估计，用于诊断和自适应校正强度）
        # SCR = bias_proxy / mc_std；初始化为 threshold，表示「中性」
        self._scr_ema = scr_threshold
        self._scr_history = []   # 最近 N 次 rollout 的 SCR 值，用于统计分析

        self.logger = MetricLogger(self.NAME, save_dir)
        self.total_steps = 0

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation using running stats."""
        if self.obs_rms is None:
            return obs
        return self.obs_rms.normalize(obs).astype(np.float32)

    def update_obs_rms(self, obs: np.ndarray):
        """Update running stats for observation normalization."""
        if self.obs_rms is not None:
            self.obs_rms.update(obs.reshape(1, -1) if obs.ndim == 1 else obs)

    # ──────────────────────────────────────────────────────────────
    # 数据收集
    # ──────────────────────────────────────────────────────────────
    def collect_rollout(self) -> float:
        self.buffer.reset()
        obs, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0

        for step in range(self.n_steps):
            # Obs normalization: update stats & normalize
            self.update_obs_rms(obs)
            obs_normalized = self.normalize_obs(obs)

            obs_tensor = torch.FloatTensor(obs_normalized).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, log_prob = self.actor.get_action_and_logprob(obs_tensor)
                value = self.critic(obs_tensor)

            action_np = action.squeeze(0).cpu().numpy()
            value_np  = value.item()
            log_prob_np = log_prob.item()

            if self.continuous:
                next_obs, reward, terminated, truncated, _ = self.env.step(action_np)
            else:
                next_obs, reward, terminated, truncated, _ = self.env.step(int(action_np))

            episode_reward += reward
            episode_length += 1
            self.buffer.add(obs_normalized, action_np, reward, float(terminated), log_prob_np, value_np)
            done = terminated or truncated
            obs = next_obs
            self.total_steps += 1

            if done:
                self.logger.log_episode(episode_reward, episode_length)
                obs, _ = self.env.reset()
                episode_reward = 0.0
                episode_length = 0

        with torch.no_grad():
            last_obs_normalized = self.normalize_obs(obs)
            last_obs_tensor = torch.FloatTensor(last_obs_normalized).unsqueeze(0).to(self.device)
            last_value = self.critic(last_obs_tensor).item()

        return last_value

    # ──────────────────────────────────────────────────────────────
    # MC Returns（共用）
    # ──────────────────────────────────────────────────────────────
    def _compute_mc_returns(self, last_value: float) -> np.ndarray:
        T = self.buffer.pos
        G = np.zeros(T, dtype=np.float32)
        g = last_value
        for t in reversed(range(T)):
            not_done = 1.0 - self.buffer.terminated[t]
            g = self.buffer.rewards[t] + self.gamma * g * not_done
            G[t] = g
        return G

    # ──────────────────────────────────────────────────────────────
    # GAE 计算（核心逻辑，通过开关分叉）
    # ──────────────────────────────────────────────────────────────
    def compute_gae(self, last_value: float) -> dict:
        T   = self.buffer.pos
        buf = self.buffer

        # ── MC Returns（所有变体都需要，用于计算误差 / 改进②的目标混合）
        G = self._compute_mc_returns(last_value)
        V = buf.values[:T]
        err = np.abs(V - G)

        # ── ⑤ 在线 SCR 估计（每次 rollout 更新，诊断 HCGAE 适用性）──────
        # SCR = |bias| / std(G)；> threshold 表示 HCGAE 有益（偏差主导）
        # < threshold 表示 MC 方差主导（校正可能引入噪声）
        mc_std = float(np.std(G)) + 1e-8
        bias_proxy_raw = float(np.mean(np.abs(V - G)))   # |V - G| 均值作为偏差代理
        scr_current = bias_proxy_raw / mc_std
        # EMA 平滑，减少单次 rollout 噪声
        self._scr_ema = (1 - self.scr_ema_alpha) * self._scr_ema + self.scr_ema_alpha * scr_current
        # 记录历史（最多保留最近 50 次）
        self._scr_history.append(scr_current)
        if len(self._scr_history) > 50:
            self._scr_history.pop(0)

        # ── 改进① / v1 EMA 归一化 的分叉 ──────────────────────────
        # 首先计算批内统计量（两个分支均需要）
        err_batch_mean = float(err.mean())
        err_batch_std  = float(err.std()) + 1e-8
        self._err_ema  = (1 - self._err_ema_alpha) * self._err_ema + self._err_ema_alpha * err_batch_mean

        # ── 改进⑥：自适应 β（根据 err 分布的峰度动态调整 sigmoid 陡峭度）──
        # 当 err 分布扁平（高峰度）时，降低 β 防止 sigmoid 过陡导致二值化；
        # 分布集中（低峰度）时，允许较高 β 使选择性更锐利。
        # β_t = β₀ / max(Kurt[e]^γ, 1.0)，其中 Kurt[e] = E[(e-μ)⁴]/σ⁴
        # 正态分布 Kurt=3，故 β 退化因子约为 3^γ（γ=0.5→≈1.73）
        if self.use_adapt_beta:
            err_centered = err - err_batch_mean
            err_var = float(np.var(err_centered)) + 1e-8
            # 超额峰度（excess kurtosis）= Kurt[e] - 3，正值=胖尾，负值=细尾
            kurtosis_raw = float(np.mean(err_centered ** 4)) / (err_var ** 2)
            # 使用原始峰度（≥1 保证分母有效），不减3
            beta_factor = max(kurtosis_raw ** self.adapt_beta_gamma, 1.0)
            beta_effective = self.hindsight_beta / beta_factor
        else:
            beta_effective = self.hindsight_beta

        if self.use_imp1:
            # 改进①：批内中心化归一化
            z = beta_effective * (err - err_batch_mean) / err_batch_std
        else:
            # v1 基线：慢速 EMA 归一化（分母为历史均值，无中心化）
            # v1：z = β * err / err_ema（无中心化，用慢速 EMA 作分母）
            z = beta_effective * err / (self._err_ema + 1e-8)

        # ── 自适应 α_max（EV 门控 + 余弦退火）──────────────────────
        progress = min(self.total_steps / max(self._total_timesteps, 1), 1.0)
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        ev_factor = max(1.0 - max(self._ev_ema, 0.0), 0.2)
        dynamic_alpha_max = (
            self.hindsight_alpha_min
            + (self.hindsight_alpha_max - self.hindsight_alpha_min)
            * cosine_decay * ev_factor
        )

        # ── ⑤ SCR 自适应缩放 alpha_max ─────────────────────────────
        # 当 scr_ema > threshold 时：Critic 偏差主导，HCGAE 有益，保持/放大校正
        # 当 scr_ema < threshold 时：MC 方差主导，校正引入噪声，抑制校正强度
        if self.use_scr_adapt:
            # scr_scale ∈ [scr_min_scale, 1.0]
            # 线性缩放：scr_ema / threshold，饱和于 1.0
            scr_scale = float(np.clip(
                self._scr_ema / (self.scr_threshold + 1e-8),
                self.scr_min_scale,
                1.0,
            ))
            dynamic_alpha_max = dynamic_alpha_max * scr_scale
        else:
            scr_scale = 1.0

        alpha = dynamic_alpha_max * (1.0 / (1.0 + np.exp(-z)))
        alpha = np.clip(alpha, 0.0, dynamic_alpha_max)

        # ── Hindsight 价值修正 ──────────────────────────────────────
        V_corrected = (1.0 - alpha) * V + alpha * G

        # ── 改进③ 末端 Bootstrap 修正分叉 ─────────────────────────
        if self.use_imp3:
            tail_n = min(10, T)
            approx_err_last = float(err[-tail_n:].mean())
            alpha_last = dynamic_alpha_max * (1.0 / (1.0 + np.exp(
                -self.hindsight_beta * (approx_err_last - err_batch_mean) / err_batch_std
            )))
            approx_G_last = G[-1]
            last_value_corrected = (1.0 - alpha_last) * last_value + alpha_last * approx_G_last
        else:
            # v1：直接使用未修正的 last_value
            last_value_corrected = last_value
            alpha_last = 0.0

        # ── 构建 V_corrected_next（含末端）─────────────────────────
        V_corrected_next = np.empty(T, dtype=np.float32)
        for t in range(T):
            if buf.terminated[t] > 0.5:
                V_corrected_next[t] = 0.0
            elif t == T - 1:
                V_corrected_next[t] = last_value_corrected
            else:
                V_corrected_next[t] = V_corrected[t + 1]

        # ── 标准 GAE 展开（用修正后的 V）──────────────────────────
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            delta_corrected = (
                buf.rewards[t]
                + self.gamma * V_corrected_next[t]
                - V_corrected[t]
            )
            not_done = 1.0 - buf.terminated[t]
            gae    = delta_corrected + self.gamma * self.lam * not_done * gae
            adv[t] = gae

        buf.advantages = adv

        # ── 改进② EV 驱动混合分叉 ─────────────────────────────────
        # 关键设计：Critic 目标必须使用【未校正原始 V】计算的标准 GAE returns
        # R_t^{Critic} = A_t^{std}(V_orig) + V_orig(s_t)
        # 不能使用 adv（基于 V_corrected 计算）+ V_orig，那样数学上不一致：
        #   adv + V_orig ≠ A^{std}(V_orig) + V_orig（因为 adv 的 δ 使用了 V_corrected）
        # _compute_standard_returns 内部使用 buf.values（原始 V），确保两条路径完全分离。
        std_gae_returns = buf._compute_standard_returns(last_value, self.gamma, self.lam)
        if self.use_imp2:
            ev_current = max(0.0, min(1.0, self._ev_ema))
            c_mc = float(np.clip(1.0 - ev_current, 0.1, 1.0))
            buf.returns = c_mc * G + (1.0 - c_mc) * std_gae_returns
        else:
            # v1：固定 50-50 混合（同样使用原始 V 的标准 GAE，保持一致性）
            c_mc = 0.5
            buf.returns = 0.5 * G + 0.5 * std_gae_returns

        # ── 改进④ 冻结统计量分叉 ──────────────────────────────────
        if self.use_imp4:
            self._adv_mean_frozen = float(adv.mean())
            self._adv_std_frozen  = float(adv.std()) + 1e-8
        # (若不冻结，update() 内会当场计算)

        # ── 统计信息（用于打印和分析）──────────────────────────────
        raw_deltas = buf.rewards[:T] + self.gamma * buf._next_values(last_value) - V
        autocorr = float(np.corrcoef(raw_deltas[:-1], raw_deltas[1:])[0, 1]) if T > 2 else 0.0

        # 偏差-方差分解
        # bias_t = E[A_GAE_k(s,a)] - A_MC(s,a) ≈ err 的均值（Critic 系统性误差）
        bias_proxy = float(np.mean(V - G))          # Critic 系统性偏差代理（负→低估）
        variance_proxy = float(np.var(adv))          # 优势方差
        snr = float(np.abs(adv.mean()) / (np.std(adv) + 1e-8))  # 信噪比

        return {
            "delta_mean"        : float(raw_deltas.mean()),
            "delta_std"         : float(raw_deltas.std()),
            "delta_autocorr"    : autocorr,
            "mean_alpha"        : float(alpha.mean()),
            "alpha_std"         : float(alpha.std()),
            "dynamic_alpha_max" : float(dynamic_alpha_max),
            "err_batch_mean"    : err_batch_mean,
            "err_batch_std"     : err_batch_std,
            "err_ema"           : float(self._err_ema),
            "beta_effective"    : float(beta_effective),  # ⑥ 实际使用的 β（自适应时与 hindsight_beta 不同）
            "c_mc"              : c_mc,
            "alpha_last"        : float(alpha_last),
            # 数学量
            "bias_proxy"        : bias_proxy,
            "variance_proxy"    : variance_proxy,
            "adv_snr"           : snr,
            "V_correction_norm" : float(np.mean(np.abs(V_corrected - V))),   # ||ΔV||_1
            # 修正后的诊断：使用 std_gae_returns（原始V路径）而非 adv+V（V_corrected路径）
            "mc_gae_diff"       : float(np.mean(np.abs(G - std_gae_returns))), # MC vs Critic目标差异
            # SCR 诊断
            "scr_current"       : scr_current,           # 当次 rollout SCR
            "scr_ema"           : float(self._scr_ema),  # 平滑后 SCR EMA
            "scr_scale"         : scr_scale,             # SCR 对 alpha_max 的缩放因子
            "mc_std"            : mc_std,                # MC 回报标准差（分母）
            "bias_abs_proxy"    : bias_proxy_raw,        # |V-G| 均值（分子）
        }

    # ──────────────────────────────────────────────────────────────
    # PPO 更新
    # ──────────────────────────────────────────────────────────────
    def update(self) -> dict:
        obs, actions, old_log_probs, advantages, returns, old_values = self.buffer.get_batch()

        if self.use_imp4:
            # 改进④：使用预先冻结的统计量
            advantages = (advantages - self._adv_mean_frozen) / self._adv_std_frozen
        elif not self.use_adv_norm:
            # 全局归一化（仅在 use_adv_norm=False 时使用）
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # 否则：per-minibatch normalization 在 minibatch 循环中处理

        T = self.buffer.pos
        indices = np.arange(T)
        metrics = {"value_loss": 0.0, "policy_loss": 0.0, "entropy_loss": 0.0,
                   "approx_kl": 0.0, "clip_frac": 0.0}
        update_count = 0

        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, self.batch_size):
                end = start + self.batch_size
                if end > T:
                    break
                batch_idx = indices[start:end]

                batch_obs        = obs[batch_idx]
                batch_actions    = actions[batch_idx]
                batch_old_lp     = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns    = returns[batch_idx]
                batch_old_values = old_values[batch_idx]

                # Per-minibatch advantage normalization (★ key trick, when not use_imp4)
                if self.use_adv_norm and not self.use_imp4:
                    batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

                new_log_probs, entropy = self.actor.evaluate_actions(batch_obs, batch_actions)
                new_values = self.critic(batch_obs)

                ratio  = torch.exp(new_log_probs - batch_old_lp)
                surr1  = ratio * batch_advantages
                surr2  = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                policy_loss  = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()
                value_loss   = 0.5 * ((new_values - batch_returns) ** 2).mean()

                self.actor_optimizer.zero_grad()
                (policy_loss + self.ent_coef * entropy_loss).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                (self.vf_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    clip_frac = ((ratio - 1).abs() > self.eps_clip).float().mean().item()

                metrics["value_loss"]   += value_loss.item()
                metrics["policy_loss"]  += policy_loss.item()
                metrics["entropy_loss"] += entropy_loss.item()
                metrics["approx_kl"]    += approx_kl
                metrics["clip_frac"]    += clip_frac
                update_count += 1

        if update_count > 0:
            for k in metrics:
                metrics[k] /= update_count

        with torch.no_grad():
            y_pred = old_values.cpu().numpy()
            y_true = returns.cpu().numpy()
            var_y  = np.var(y_true)
            ev = 1 - np.var(y_true - y_pred) / (var_y + 1e-8)
            metrics["explained_variance"] = float(ev)

        return metrics

    # ──────────────────────────────────────────────────────────────
    # 训练循环（含丰富状态打印）
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _bar(cur, tot, w=20):
        f = int(w * cur / tot)
        return f"[{'█'*f}{'░'*(w-f)}] {100*cur/tot:5.1f}%"

    def train(
        self,
        total_timesteps: int,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 5000,
        n_eval_episodes: int = 5,
        verbose: bool = True,
        print_interval: int = 5,   # 每隔多少次 update 打印一次（非 eval 时）
    ):
        import time
        self._total_timesteps = total_timesteps
        train_start = time.time()
        last_eval_reward = float("nan")
        update_idx = 0

        while self.total_steps < total_timesteps:
            last_value = self.collect_rollout()
            gae_stats  = self.compute_gae(last_value)
            metrics    = self.update()
            update_idx += 1
            metrics.update(gae_stats)

            ev_val = metrics.get("explained_variance", 0.0)
            self._ev_ema = (1 - self._ev_ema_alpha) * self._ev_ema + self._ev_ema_alpha * ev_val

            self.logger.log_update(
                value_loss=metrics["value_loss"],
                policy_loss=metrics["policy_loss"],
                entropy_loss=metrics["entropy_loss"],
                approx_kl=metrics["approx_kl"],
                clip_frac=metrics["clip_frac"],
                explained_variance=metrics["explained_variance"],
                ev_ema=float(self._ev_ema),
                alpha_mean=float(metrics.get("mean_alpha", 0.0)),
                c_mc=float(metrics.get("c_mc", 0.5)),
                total_steps=self.total_steps,
            )

            do_eval = eval_env is not None and self.total_steps % eval_freq < self.n_steps
            if do_eval:
                last_eval_reward = self.evaluate(eval_env, n_eval_episodes)
                self.logger.log_eval(last_eval_reward, self.total_steps)

            if verbose and (do_eval or update_idx % print_interval == 1):
                elapsed = time.time() - train_start
                recent  = self.logger.get_recent_reward(20)
                fps     = int(self.total_steps / (elapsed + 1e-8))
                eval_str = f"{last_eval_reward:7.1f}" if not np.isnan(last_eval_reward) else "    N/A"

                # ── 丰富的数学量打印 ───────────────────────────────
                # 标识符：显示哪些改进已启用
                imp_flags = (
                    f"①{'✓' if self.use_imp1 else '✗'}"
                    f"②{'✓' if self.use_imp2 else '✗'}"
                    f"③{'✓' if self.use_imp3 else '✗'}"
                    f"④{'✓' if self.use_imp4 else '✗'}"
                    f"⑤{'✓' if self.use_scr_adapt else '✗'}"
                )
                scr_str = (
                    f"SCR={metrics.get('scr_ema', 0):.3f}(×{metrics.get('scr_scale',1):.2f}) "
                    if self.use_scr_adapt else
                    f"scr={metrics.get('scr_current', 0):.3f} "
                )
                print(
                    f"  [{self.NAME:<18}|{imp_flags}] "
                    f"{self.total_steps:7d}/{total_timesteps} {self._bar(self.total_steps, total_timesteps)} "
                    f"| Eval={eval_str} Rec={recent:6.1f} "
                    f"| VL={metrics['value_loss']:.3f} EV={metrics['explained_variance']:+.3f} "
                    f"| ᾱ={metrics.get('mean_alpha',0):.3f}(±{metrics.get('alpha_std',0):.3f}) "
                    f"αmax={metrics.get('dynamic_alpha_max',0):.2f} "
                    f"| c_mc={metrics.get('c_mc',0):.2f} "
                    f"| bias={metrics.get('bias_proxy',0):+.3f} "
                    f"var={metrics.get('variance_proxy',0):.3f} "
                    f"SNR={metrics.get('adv_snr',0):.3f} "
                    f"| {scr_str}"
                    f"ΔV={metrics.get('V_correction_norm',0):.3f} "
                    f"err_μ={metrics.get('err_batch_mean',0):.3f} "
                    f"err_ema={metrics.get('err_ema',0):.3f} "
                    f"| δ:μ={metrics.get('delta_mean',0):+.3f} "
                    f"σ={metrics.get('delta_std',0):.3f} "
                    f"r1={metrics.get('delta_autocorr',0):+.2f} "
                    f"| KL={metrics['approx_kl']:.4f} "
                    f"clip={metrics['clip_frac']:.2f} "
                    f"| {fps:5d}fps {elapsed:5.0f}s"
                )

        if verbose:
            elapsed = time.time() - train_start
            final_r = np.mean(self.logger.eval_rewards[-5:]) if self.logger.eval_rewards else 0.0
            best_r  = max(self.logger.eval_rewards) if self.logger.eval_rewards else 0.0
            print(f"  ✓ [{self.NAME}] 完成 | 耗时={elapsed:.1f}s | 最终={final_r:.1f} | 最高={best_r:.1f}")

        self.logger.save()
        return self.logger

    def evaluate(self, eval_env: gym.Env, n_episodes: int = 5) -> float:
        total_reward = 0.0
        for _ in range(n_episodes):
            obs, _ = eval_env.reset()
            done   = False
            ep_r   = 0.0
            while not done:
                obs_normalized = self.normalize_obs(obs)
                obs_t = torch.FloatTensor(obs_normalized).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    dist = self.actor(obs_t)
                    action = dist.mean if self.continuous else dist.probs.argmax(dim=-1)
                a = action.squeeze(0).cpu().numpy()
                if self.continuous:
                    next_obs, r, terminated, truncated, _ = eval_env.step(a)
                else:
                    next_obs, r, terminated, truncated, _ = eval_env.step(int(a))
                ep_r += r
                done  = terminated or truncated
                obs   = next_obs
            total_reward += ep_r
        return total_reward / n_episodes


# ════════════════════════════════════════════════════════════════════
# 工厂函数：根据名称构建各消融变体
# ════════════════════════════════════════════════════════════════════

_ABLATION_CONFIGS = {
    # name             imp1   imp2   imp3   imp4
    "HCGAE_Base":  (False, False, False, False),  # v1 风格基线
    "HCGAE_Imp1":  (True,  False, False, False),  # 仅批内归一化
    "HCGAE_Imp2":  (False, True,  False, False),  # 仅 EV 驱动混合
    "HCGAE_Imp3":  (False, False, True,  False),  # 仅末端 Bootstrap
    "HCGAE_Imp4":  (False, False, False, True ),  # 仅冻结统计量
    "HCGAE_Imp12": (True,  True,  False, False),  # ①+②
    "HCGAE_Imp14": (True,  False, False, True ),  # ①+④
    "HCGAE_Imp24": (False, True,  False, True ),  # ②+④
    "HCGAE_Imp124":(True,  True,  False, True ),  # ①+②+④（不含末端）
    "HCGAE_Full":  (True,  True,  True,  True ),  # 全量 v2
    # SCR 自适应变体（⑤）
    "HCGAE_Imp12_SCR":      (True,  True,  False, False),  # ①+②+⑤(SCR 自适应)
    "HCGAE_Full_SCR":       (True,  True,  True,  True ),  # 全量 v2 + SCR 自适应
    # 自适应 β 变体（⑥）—— 与ICML 改进4 对应
    "HCGAE_Imp12_AdaptBeta":(True,  True,  False, False),  # ①+②+⑥(自适应β)
    "HCGAE_Full_AdaptBeta": (True,  True,  True,  True ),  # 全量 v2 + 自适应β
}

# SCR 自适应开关（仅 SCR 变体启用）
_SCR_ADAPT_VARIANTS = {"HCGAE_Imp12_SCR", "HCGAE_Full_SCR"}
# 自适应 β 开关（仅 AdaptBeta 变体启用）
_ADAPT_BETA_VARIANTS = {"HCGAE_Imp12_AdaptBeta", "HCGAE_Full_AdaptBeta"}


def build_ablation_agent(
    variant_name: str,
    env: gym.Env,
    **kwargs
) -> HindsightAblation:
    """
    工厂函数：根据变体名称构建消融实验 Agent。
    kwargs 传入 HindsightAblation.__init__ 的所有其他参数。
    """
    if variant_name not in _ABLATION_CONFIGS:
        raise ValueError(
            f"未知变体 '{variant_name}'。"
            f"可用变体：{list(_ABLATION_CONFIGS.keys())}"
        )
    imp1, imp2, imp3, imp4 = _ABLATION_CONFIGS[variant_name]
    # 如果 kwargs 里没有 name，使用 variant_name 作为默认名称
    kwargs.setdefault("name", variant_name)
    # SCR 自适应变体自动启用 use_scr_adapt
    if variant_name in _SCR_ADAPT_VARIANTS:
        kwargs.setdefault("use_scr_adapt", True)
    # 自适应 β 变体自动启用 use_adapt_beta
    if variant_name in _ADAPT_BETA_VARIANTS:
        kwargs.setdefault("use_adapt_beta", True)
    return HindsightAblation(
        env=env,
        use_imp1=imp1,
        use_imp2=imp2,
        use_imp3=imp3,
        use_imp4=imp4,
        **kwargs,
    )


def get_all_variant_names():
    return list(_ABLATION_CONFIGS.keys())

