"""
革新方法二：Multi-Scale GAE（多尺度 GAE，MSGAE）

核心思想：
─────────────────────────────────────────────────────────────────────
标准 GAE 的根本局限：
  单一 λ ∈ (0,1) 固定了「时间视野」。
  λ 大 → 依赖远期奖励（低偏差、高方差，像 MC）
  λ 小 → 仅依赖单步 TD（高偏差、低方差，像 TD）

真正的问题：
  不同状态对「视野长度」的需求不同：
  - 高风险/分叉状态（状态转移随机大）→ 需要短视野（小 λ）
  - 稳定过渡状态（状态转移确定）→ 需要长视野（大 λ）

  但单一 λ 无法区分这两种情况。

Multi-Scale 洞察：
  不同 λ 的 GAE 就像不同频率的滤波器：
    λ=0   → 只看当前步的 δ（最短时间尺度，高频）
    λ=0.5 → 展开约 2 步
    λ=0.9 → 展开约 10 步
    λ=0.99→ 展开约 100 步（接近 MC，低频）

  MSGAE 的核心：同时计算所有尺度，然后「让数据说话」——
  用一个轻量的权重网络，根据当前状态的特征，学习最优的尺度组合。

MSGAE 公式：
─────────────────────────────────────────────────────────────────────
  1. 计算 K 个尺度的 GAE：
     A_t^{(k)} = GAE(δ, λ_k)   for k = 1,...,K

  2. 用状态特征预测混合权重：
     w_t = softmax( MLP(V(s_t), δ_t, std_δ_local) )   [K维]

  3. 加权组合：
     A_t^MSGAE = Σ_k w_t[k] * A_t^{(k)}

  4. 权重网络的训练目标：
     最小化 |A_t^MSGAE * std_adv - true_adv|
     用「哪个尺度最接近事后 MC advantage」来监督

     其中 true_adv ≈ G_t - V(s_t)（MC advantage，训练后期逐渐可靠）

关键优势：
  - 理论保证：固定权重 w_k = 1/K 退化为多尺度集成（至少不差）
  - 适应性强：权重网络可以学习状态相关的最优尺度分配
  - 端到端：与 PPO 一同训练，无需额外数据

实现细节：
  - 选用 K=4 个 λ 尺度：{0.0, 0.5, 0.9, 0.97}
    代表四种不同的时间视野
  - 权重网络输入：当前 δ_t 和局部 δ 的标准差（5步窗口）
    捕捉局部的 Critic 不确定性
  - 初始化：均匀权重（公平起点）
  - 正则化：权重熵正则，鼓励多样性而非退化为单一尺度

与 TD(λ) 的关系：
  TD(λ) = Σ_{n=1}^∞ (1-λ)λ^{n-1} n-step TD return
  MSGAE = 数据驱动的 λ 混合，不假设任何固定的几何混合权重
─────────────────────────────────────────────────────────────────────
"""
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.logger import MetricLogger
from ..utils.networks import ActorNetwork, CriticNetwork
from ..utils.rollout_buffer import RolloutBuffer


class ScaleWeightNetwork(nn.Module):
    """
    多尺度权重网络：根据局部特征预测每个 λ 尺度的权重

    输入特征：
      - δ_t（当前步 TD 残差）
      - local_std（局部 5 步窗口内 δ 的标准差）
      - V(s_t)（归一化的价值估计，提供状态信息）

    输出：K 维 softmax 权重（K = lambda 尺度数量）
    """

    def __init__(self, input_dim: int = 3, hidden_dim: int = 32, n_scales: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_scales),
        )
        # 初始化为均匀权重（零输出 → softmax → 均匀）
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回 softmax 权重，shape (batch, n_scales)"""
        logits = self.net(x)
        return F.softmax(logits, dim=-1)


class MultiScalePPO:
    """
    PPO + Multi-Scale GAE（MSGAE）

    同时计算 K 个不同 λ 尺度的 GAE，
    用状态相关的权重网络自适应组合。
    """

    NAME = "MultiScale_GAE"

    # 6 个 λ 尺度：TD(0), 短程, 中程, 长程，超长程
    # 增加 0.7 和 0.99 两个尺度：
    # λ=0.7：介于 TD 和标准 GAE 之间，方差/偏差平衡点
    # λ=0.99：接近 MC return，适合 Critic 精度高的后期
    LAMBDA_SCALES = [0.0, 0.5, 0.7, 0.9, 0.95, 0.99]

    def __init__(
        self,
        env: gym.Env,
        hidden_dim: int = 64,
        scale_hidden_dim: int = 32,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        lr_scale: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,             # 用于 returns 计算（保持与基线一致）
        eps_clip: float = 0.2,
        n_epochs: int = 10,
        batch_size: int = 64,
        n_steps: int = 2048,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        # MSGAE 超参数
        scale_entropy_coef: float = 0.01,  # 权重熵正则系数
        device: str = "cpu",
        save_dir: str = "results",
    ):
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
        self.scale_entropy_coef = scale_entropy_coef
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
        self.n_scales = len(self.LAMBDA_SCALES)

        # 主网络
        self.actor  = ActorNetwork(obs_dim, action_dim, hidden_dim, self.continuous).to(self.device)
        self.critic = CriticNetwork(obs_dim, hidden_dim).to(self.device)

        # 多尺度权重网络（输入：δ_t_norm, local_std, V_normalized, snr）
        # 新增信噪比 SNR = |δ| / local_std 作为第 4 个特征
        self.scale_net = ScaleWeightNetwork(
            input_dim=4, hidden_dim=scale_hidden_dim, n_scales=self.n_scales
        ).to(self.device)

        # 优化器
        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),     lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),    lr=lr_critic)
        self.scale_optimizer  = torch.optim.Adam(self.scale_net.parameters(), lr=lr_scale)

        # Rollout Buffer
        self.buffer = RolloutBuffer(n_steps, obs_dim, action_dim, self.device, self.continuous)

        # 存储权重序列（用于监控）
        self._last_weights = None   # shape (T, n_scales)
        self._v_running_std = 1.0   # V 的运行标准差（用于归一化）
        self._v_ema_alpha = 0.05

        self.logger = MetricLogger(self.NAME, save_dir)
        self.total_steps = 0

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

            action_np = action.squeeze(0).cpu().numpy()
            value_np = value.item()
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

    def _compute_gae_single_scale(self, nv: np.ndarray, lam_val: float) -> np.ndarray:
        """计算单个 λ 尺度的 GAE，返回优势数组"""
        T = self.buffer.pos
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            delta    = self.buffer.rewards[t] + self.gamma * nv[t] - self.buffer.values[t]
            not_done = 1.0 - self.buffer.terminated[t]
            gae      = delta + self.gamma * lam_val * not_done * gae
            adv[t]   = gae
        return adv

    def compute_gae(self, last_value: float) -> dict:
        """
        Multi-Scale GAE：
        1. 计算 K 个尺度的 GAE
        2. 计算局部特征（δ, local_std, V_normalized）
        3. 用权重网络预测混合权重
        4. 加权组合 → 最终优势
        """
        T   = self.buffer.pos
        nv  = self.buffer._next_values(last_value)
        buf = self.buffer

        # Step 1: 计算所有尺度的 GAE
        all_adv = np.stack([
            self._compute_gae_single_scale(nv, lam_k)
            for lam_k in self.LAMBDA_SCALES
        ], axis=-1)  # shape (T, K)

        # Step 2: 计算局部特征
        raw_deltas = buf.rewards[:T] + self.gamma * nv - buf.values[:T]  # shape (T,)

        # 局部 δ 标准差（5步滑动窗口）
        local_std = np.zeros(T, dtype=np.float32)
        win = 5
        for t in range(T):
            lo = max(0, t - win // 2)
            hi = min(T, t + win // 2 + 1)
            local_std[t] = raw_deltas[lo:hi].std() + 1e-8

        # 价值归一化（用运行标准差）
        v_std = max(float(buf.values[:T].std()), 1e-8)
        self._v_running_std = (1 - self._v_ema_alpha) * self._v_running_std + self._v_ema_alpha * v_std
        V_normalized = buf.values[:T] / max(self._v_running_std, 1e-8)

        # Step 3: 用权重网络计算混合权重
        # 特征矩阵：(T, 4)  [δ_t_norm, local_std_t, V_normalized_t, SNR_t]
        delta_normalized = raw_deltas / (raw_deltas.std() + 1e-8)  # 全局归一化的 δ
        snr = np.abs(raw_deltas) / (local_std + 1e-8)              # 信噪比 |δ|/σ_local
        snr_normalized = snr / (snr.mean() + 1e-8)                 # 归一化 SNR
        features = np.stack([
            delta_normalized,                   # 全局归一化的 δ
            local_std / (local_std.mean() + 1e-8),  # 归一化局部不确定性
            V_normalized,                       # 状态值
            snr_normalized,                     # 信噪比（新增）
        ], axis=-1)  # shape (T, 4)

        feat_tensor = torch.FloatTensor(features).to(self.device)
        with torch.no_grad():
            weights = self.scale_net(feat_tensor).cpu().numpy()  # shape (T, K)

        self._last_weights = weights   # 存储用于监控和训练

        # Step 4: 加权组合
        # A_t^MSGAE = Σ_k w_t[k] * A_t^{(k)}
        adv_combined = (all_adv * weights).sum(axis=-1)  # shape (T,)

        buf.advantages = adv_combined
        # returns = 标准 GAE returns（基础 λ=self.lam），给 Critic 稳定目标
        buf.returns = buf._compute_standard_returns(last_value, self.gamma, self.lam)

        # 统计
        autocorr = float(np.corrcoef(raw_deltas[:-1], raw_deltas[1:])[0, 1]) if T > 2 else 0.0
        weights_mean = weights.mean(axis=0)  # 每个尺度的平均权重

        stats = {
            "delta_mean"    : float(raw_deltas.mean()),
            "delta_std"     : float(raw_deltas.std()),
            "delta_autocorr": autocorr,
        }
        for k, (lam_k, wm) in enumerate(zip(self.LAMBDA_SCALES, weights_mean)):
            stats[f"w_lambda_{lam_k}"] = float(wm)

        return stats

    def update_scale_network(
        self,
        features: torch.Tensor,
        mc_returns: torch.Tensor,
        all_adv_tensor: torch.Tensor,   # shape (batch, K)
        values: torch.Tensor,
    ) -> float:
        """
        训练权重网络 - 改进版本：

        ★ 优化1: 双目标混合损失
        -------------------------------------------------------
        (a) MC 对齐损失：||Σ_k w_k * A^{(k)} - (G_t - V)|| → 最小
            驱动网络学会在 Critic 准时选高λ，不准时选低λ
        (b) 每尺度方差惩罚：高方差的尺度应得低权重
            对于 λ 大的尺度（长视野），方差天然更大；让网络学会权衡

        ★ 优化2: 大幅增强熵正则
        -------------------------------------------------------
        原先 scale_entropy_coef=0.01 太小，导致权重坍缩。
        动态调整熵系数：(K-1)/K * coef，确保 K 个尺度都被利用。
        """
        mc_adv = mc_returns - values.detach()  # shape (batch,)

        # 预测权重
        weights = self.scale_net(features)  # shape (batch, K)

        # (a) MC 对齐损失
        combined_adv = (weights * all_adv_tensor).sum(dim=-1)  # shape (batch,)
        mc_align_loss = ((combined_adv - mc_adv) ** 2).mean()

        # (b) 方差惩罚：让权重倾向于低方差尺度
        # 计算每个尺度在当前 batch 的方差
        adv_var = all_adv_tensor.var(dim=0) + 1e-8  # shape (K,)
        adv_var_norm = adv_var / adv_var.sum()       # 归一化
        # 高方差尺度被高权重时的惩罚 = Σ_k w_k * var_k_normalized
        var_penalty = (weights * adv_var_norm.unsqueeze(0)).sum(dim=-1).mean()

        # 熵正则（鼓励权重多样性，防止坍缩）
        # 提高系数：K 越多，熵正则越重要
        K = weights.shape[-1]
        dynamic_entropy_coef = self.scale_entropy_coef * (K / 4.0)  # 随尺度数量缩放
        entropy_reg = -(weights * torch.log(weights + 1e-9)).sum(dim=-1).mean()

        loss = mc_align_loss + 0.1 * var_penalty - dynamic_entropy_coef * entropy_reg

        self.scale_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.scale_net.parameters(), self.max_grad_norm)
        self.scale_optimizer.step()

        return mc_align_loss.item()

    def update(self) -> dict:
        obs, actions, old_log_probs, advantages, returns, old_values = self.buffer.get_batch()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        T = self.buffer.pos
        indices = np.arange(T)
        metrics = {"value_loss": 0.0, "policy_loss": 0.0, "entropy_loss": 0.0,
                   "approx_kl": 0.0, "clip_frac": 0.0, "scale_loss": 0.0}
        update_count = 0

        # 预计算所有尺度的 GAE（用于权重网络训练）
        nv_np = self.buffer._next_values(
            old_values[-1].item()   # 用最后一个 old_value 近似 last_value
        )
        all_adv_np = np.stack([
            self._compute_gae_single_scale(nv_np, lam_k)
            for lam_k in self.LAMBDA_SCALES
        ], axis=-1)  # (T, K)
        all_adv_tensor_full = torch.FloatTensor(all_adv_np).to(self.device)

        # 重构特征（用于权重网络训练）
        raw_deltas_np = (
            self.buffer.rewards[:T]
            + self.gamma * nv_np
            - self.buffer.values[:T]
        )
        local_std_np = np.zeros(T, dtype=np.float32)
        win = 5
        for t in range(T):
            lo = max(0, t - win // 2)
            hi = min(T, t + win // 2 + 1)
            local_std_np[t] = raw_deltas_np[lo:hi].std() + 1e-8
        V_norm_np = self.buffer.values[:T] / max(self._v_running_std, 1e-8)

        # 与 compute_gae 保持一致：4维特征
        delta_norm_np  = raw_deltas_np / (raw_deltas_np.std() + 1e-8)
        snr_np         = np.abs(raw_deltas_np) / (local_std_np + 1e-8)
        snr_norm_np    = snr_np / (snr_np.mean() + 1e-8)
        lstd_norm_np   = local_std_np / (local_std_np.mean() + 1e-8)
        features_np = np.stack([
            delta_norm_np,
            lstd_norm_np,
            V_norm_np,
            snr_norm_np,
        ], axis=-1)
        features_full = torch.FloatTensor(features_np).to(self.device)

        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, self.batch_size):
                end = start + self.batch_size
                if end > T:
                    break
                batch_idx = indices[start:end]

                batch_obs         = obs[batch_idx]
                batch_actions     = actions[batch_idx]
                batch_old_lp      = old_log_probs[batch_idx]
                batch_advantages  = advantages[batch_idx]
                batch_returns     = returns[batch_idx]
                batch_old_values  = old_values[batch_idx]

                new_log_probs, entropy = self.actor.evaluate_actions(batch_obs, batch_actions)
                new_values = self.critic(batch_obs)

                ratio = torch.exp(new_log_probs - batch_old_lp)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                policy_loss  = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()

                v_loss_unclipped = (new_values - batch_returns) ** 2
                v_clipped = batch_old_values + torch.clamp(
                    new_values - batch_old_values, -self.eps_clip, self.eps_clip
                )
                v_loss_clipped = (v_clipped - batch_returns) ** 2
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                self.actor_optimizer.zero_grad()
                (policy_loss + self.ent_coef * entropy_loss).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                (self.vf_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                # 训练权重网络（每 batch 更新一次）
                batch_feat    = features_full[batch_idx]
                batch_all_adv = all_adv_tensor_full[batch_idx]
                batch_old_v   = old_values[batch_idx]
                scale_loss_v  = self.update_scale_network(
                    batch_feat, batch_returns,
                    batch_all_adv, batch_old_v
                )
                metrics["scale_loss"] += scale_loss_v

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
            var_y = np.var(y_true)
            ev = 1 - np.var(y_true - y_pred) / (var_y + 1e-8)
            metrics["explained_variance"] = float(ev)

        return metrics

    @staticmethod
    def _progress_bar(current: int, total: int, width: int = 25) -> str:
        filled = int(width * current / total)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}] {100*current/total:5.1f}%"

    def train(
        self,
        total_timesteps: int,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 5000,
        n_eval_episodes: int = 5,
        verbose: bool = True,
    ):
        import time
        train_start = time.time()
        last_eval_reward = float("nan")
        update_idx = 0

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
            )

            do_eval = eval_env is not None and self.total_steps % eval_freq < self.n_steps
            if do_eval:
                last_eval_reward = self.evaluate(eval_env, n_eval_episodes)
                self.logger.log_eval(last_eval_reward, self.total_steps)

            if verbose and (do_eval or update_idx % 5 == 1):
                elapsed  = time.time() - train_start
                recent   = self.logger.get_recent_reward(20)
                progress = self._progress_bar(self.total_steps, total_timesteps)
                fps      = int(self.total_steps / (elapsed + 1e-8))
                eval_str = f"{last_eval_reward:7.1f}" if not np.isnan(last_eval_reward) else "    N/A"
                # 显示各尺度权重
                w_strs = " ".join(
                    f"w{lam_k}={metrics.get(f'w_lambda_{lam_k}', 0):.2f}"
                    for lam_k in self.LAMBDA_SCALES
                )
                delta_str = (
                    f"δ:μ={metrics.get('delta_mean',0):+.3f}"
                    f"/σ={metrics.get('delta_std',0):.3f}"
                )
                print(
                    f"  [{self.NAME:<22}] "
                    f"{self.total_steps:7d}/{total_timesteps} "
                    f"{progress} "
                    f"| Eval={eval_str} Recent={recent:6.1f} "
                    f"| VLoss={metrics['value_loss']:.3f} EV={metrics['explained_variance']:+.2f} "
                    f"| {delta_str}"
                    f"| {w_strs}"
                    f"| {fps:5d}fps {elapsed:6.0f}s"
                )

        if verbose:
            elapsed = time.time() - train_start
            final_r = np.mean(self.logger.eval_rewards[-5:]) if self.logger.eval_rewards else 0.0
            best_r  = max(self.logger.eval_rewards) if self.logger.eval_rewards else 0.0
            print(f"\n  ✓ [{self.NAME}] 完成 | 总耗时={elapsed:.1f}s "
                  f"| 最终评估={final_r:.1f} | 最高评估={best_r:.1f}")

        self.logger.save()
        return self.logger

    def evaluate(self, eval_env: gym.Env, n_episodes: int = 5) -> float:
        total_reward = 0.0
        for _ in range(n_episodes):
            obs, _ = eval_env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    dist = self.actor(obs_tensor)
                    action = dist.mean if self.continuous else dist.probs.argmax(dim=-1)
                action_np = action.squeeze(0).cpu().numpy()
                if self.continuous:
                    next_obs, reward, terminated, truncated, _ = eval_env.step(action_np)
                else:
                    next_obs, reward, terminated, truncated, _ = eval_env.step(int(action_np))
                ep_reward += reward
                done = terminated or truncated
                obs = next_obs
            total_reward += ep_reward
        return total_reward / n_episodes

