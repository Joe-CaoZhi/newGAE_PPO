"""
革新方法二：Multi-Scale GAE（多尺度 GAE，MSGAE）v2

改进清单（v2）：
─────────────────────────────────────────────────────────────────────
① 权重网络输入特征增强（加入 Critic 中间层特征）
   旧：仅 (δ_t, local_std, V) 三个标量特征，不含状态语义信息
   新：从 Critic 网络中提取第二隐藏层 h_c ∈ R^{hidden_dim}，
       与 (δ_norm, local_std_norm, snr) 拼接 → scale_net 输入
       物理含义：h_c 包含价值函数学到的状态表征，让 scale_net 真正感知状态的
       "动力学复杂度"，而非仅依赖局部 δ 统计量

② 权重网络损失修正：用"遗憾值"替代方差惩罚
   旧：方差惩罚 = Σ_k w_k * var_k（在偏差-方差权衡中偏袒小λ，忽略偏差代价）
   新：Regret Loss = Σ_k w_k * |A^{(k)} - G_t + V|
       对每个尺度计算与 MC advantage 的绝对误差，用加权平均作为损失
       物理含义：最优权重应分配给误差最小的尺度，而非方差最小的尺度
       数学上这等价于最优决策问题（Bayesian model averaging）

③ 高噪声截断：SNR 极低时强制选择短视野
   旧：无保护，噪声大时仍可能选高λ（长视野），引入大量无效信息
   新：当 SNR = |δ| / σ_local < snr_threshold（默认 0.5）时，
       对高λ尺度的权重 logit 加入负向偏置，实现软截断
       物理含义：信号远弱于噪声时，长程展开等于放大随机性

④ 归一化统计量冻结（与 HCGAE v2 一致）
   在 compute_gae 阶段冻结 (adv_mean, adv_std)，update 内使用冻结值
─────────────────────────────────────────────────────────────────────
"""
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.logger import MetricLogger
from ..utils.networks import ActorNetwork
from ..utils.rollout_buffer import RolloutBuffer


# ──────────────────────────────────────────────────────────────────────────────
# 改进①：带中间层特征提取接口的 Critic
# ──────────────────────────────────────────────────────────────────────────────
class CriticWithFeature(nn.Module):
    """
    带特征提取接口的 Critic 网络：
    - 前向传播同标准 Critic
    - 额外提供 get_features() 接口，输出最后一隐藏层的激活
      供 ScaleWeightNetwork 使用，注入状态语义信息
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.layer1 = nn.Linear(obs_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        # 正交初始化
        for layer in [self.layer1, self.layer2]:
            nn.init.orthogonal_(layer.weight, np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.value_head.weight, 1.0)
        nn.init.constant_(self.value_head.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.layer1(x))
        h = torch.tanh(self.layer2(h))
        return self.value_head(h).squeeze(-1)

    def get_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (value, hidden_features)"""
        h = torch.tanh(self.layer1(x))
        h = torch.tanh(self.layer2(h))
        v = self.value_head(h).squeeze(-1)
        return v, h  # h.shape = (batch, hidden_dim)


# ──────────────────────────────────────────────────────────────────────────────
# 改进①：支持 Critic 特征输入的 ScaleWeightNetwork
# ──────────────────────────────────────────────────────────────────────────────
class ScaleWeightNetwork(nn.Module):
    """
    多尺度权重网络 v2：
    输入 = [δ_norm(1) | local_std_norm(1) | snr_norm(1) | h_critic(hidden_dim)]
    共 3 + hidden_dim 维
    输出：K 维 softmax 权重

    初始化为零输出（softmax → 均匀权重），保证训练初期等同于均匀组合
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, n_scales: int = 6):
        super().__init__()
        self.n_scales = n_scales
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_scales),
        )
        # 初始化为均匀权重
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, snr_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        snr_mask: shape (batch, n_scales)，低 SNR 状态对高λ尺度施加软惩罚
                  值为负偏置（如 -10 表示强截断，-2 表示软截断）
        """
        logits = self.net(x)
        if snr_mask is not None:
            logits = logits + snr_mask
        return F.softmax(logits, dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# MultiScalePPO v2
# ──────────────────────────────────────────────────────────────────────────────
class MultiScalePPO:
    """PPO + Multi-Scale GAE（MSGAE）v2"""

    NAME = "MultiScale_GAE"

    # 6 个 λ 尺度：TD(0), 短程, 中程, 长程，超长程
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
        lam: float = 0.95,
        eps_clip: float = 0.2,
        n_epochs: int = 10,
        batch_size: int = 64,
        n_steps: int = 2048,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        scale_entropy_coef: float = 0.01,
        snr_threshold: float = 0.5,      # ③ 低 SNR 截断阈值
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
        self.snr_threshold = snr_threshold
        self.device = torch.device(device)
        self.hidden_dim = hidden_dim

        obs_dim = env.observation_space.shape[0]
        if isinstance(env.action_space, gym.spaces.Discrete):
            action_dim = env.action_space.n
            self.continuous = False
        else:
            action_dim = env.action_space.shape[0]
            self.continuous = True

        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.n_scales   = len(self.LAMBDA_SCALES)

        # 改进①：使用带特征提取接口的 Critic
        self.actor  = ActorNetwork(obs_dim, action_dim, hidden_dim, self.continuous).to(self.device)
        self.critic = CriticWithFeature(obs_dim, hidden_dim).to(self.device)

        # 改进①：scale_net 输入 = 3个标量 + hidden_dim 维 Critic 特征
        scale_input_dim = 3 + hidden_dim
        self.scale_net = ScaleWeightNetwork(
            input_dim=scale_input_dim,
            hidden_dim=scale_hidden_dim,
            n_scales=self.n_scales
        ).to(self.device)

        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),     lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),    lr=lr_critic)
        self.scale_optimizer  = torch.optim.Adam(self.scale_net.parameters(), lr=lr_scale)

        self.buffer = RolloutBuffer(n_steps, obs_dim, action_dim, self.device, self.continuous)

        self._last_weights   = None
        self._v_running_std  = 1.0
        self._v_ema_alpha    = 0.05
        # ④ 冻结归一化统计量
        self._adv_mean_frozen = 0.0
        self._adv_std_frozen  = 1.0

        self.logger = MetricLogger(self.NAME, save_dir)
        self.total_steps = 0

    def collect_rollout(self) -> float:
        self.buffer.reset()
        obs, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0

        for _ in range(self.n_steps):
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
            obs  = next_obs
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
        T = self.buffer.pos
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            delta    = self.buffer.rewards[t] + self.gamma * nv[t] - self.buffer.values[t]
            not_done = 1.0 - self.buffer.terminated[t]
            gae      = delta + self.gamma * lam_val * not_done * gae
            adv[t]   = gae
        return adv

    def _build_features_and_mask(
        self,
        raw_deltas: np.ndarray,
        obs_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建 scale_net 输入特征和低 SNR 软截断掩码。
        返回：(features_tensor(T, 3+hidden_dim), snr_mask_tensor(T, K))
        """
        T = raw_deltas.shape[0]
        win = 5

        # 局部标准差
        local_std = np.zeros(T, dtype=np.float32)
        for t in range(T):
            lo = max(0, t - win // 2)
            hi = min(T, t + win // 2 + 1)
            local_std[t] = raw_deltas[lo:hi].std() + 1e-8

        # 标量特征归一化
        delta_norm = raw_deltas / (raw_deltas.std() + 1e-8)
        snr        = np.abs(raw_deltas) / (local_std + 1e-8)
        snr_norm   = snr / (snr.mean() + 1e-8)
        lstd_norm  = local_std / (local_std.mean() + 1e-8)

        scalar_feats = np.stack([delta_norm, lstd_norm, snr_norm], axis=-1)  # (T, 3)

        # ① 提取 Critic 中间层特征 h_c: (T, hidden_dim)
        with torch.no_grad():
            _, h_critic = self.critic.get_features(obs_tensor)  # (T, hidden_dim)
        h_critic_np = h_critic.cpu().numpy()

        # 拼接
        features_np = np.concatenate([scalar_feats, h_critic_np], axis=-1)  # (T, 3+hidden_dim)
        features_t  = torch.FloatTensor(features_np).to(self.device)

        # ③ 低 SNR 软截断掩码：SNR < threshold 时对高λ尺度施加 -3 偏置
        # 尺度越高（λ越大），惩罚越重
        snr_low_mask = (snr < self.snr_threshold).astype(np.float32)  # (T,)
        # 尺度惩罚强度：λ 越大（索引越大）惩罚越重
        scale_penalties = np.array(
            [0.0, -1.0, -2.0, -3.0, -4.0, -5.0], dtype=np.float32
        )  # 对应 LAMBDA_SCALES
        # snr_mask[t, k] = snr_low_mask[t] * scale_penalties[k]
        snr_mask_np = snr_low_mask[:, None] * scale_penalties[None, :]  # (T, K)
        snr_mask_t  = torch.FloatTensor(snr_mask_np).to(self.device)

        return features_t, snr_mask_t

    def compute_gae(self, last_value: float) -> dict:
        T   = self.buffer.pos
        buf = self.buffer
        nv  = buf._next_values(last_value)

        # Step 1: 计算所有尺度的 GAE
        all_adv = np.stack([
            self._compute_gae_single_scale(nv, lam_k)
            for lam_k in self.LAMBDA_SCALES
        ], axis=-1)  # (T, K)

        raw_deltas = buf.rewards[:T] + self.gamma * nv - buf.values[:T]

        # 价值运行标准差更新
        v_std = max(float(buf.values[:T].std()), 1e-8)
        self._v_running_std = (
            (1 - self._v_ema_alpha) * self._v_running_std
            + self._v_ema_alpha * v_std
        )

        # ① 构建特征（含 Critic 中间层）和低 SNR 掩码
        obs_tensor = torch.FloatTensor(buf.observations[:T]).to(self.device)
        features_t, snr_mask_t = self._build_features_and_mask(raw_deltas, obs_tensor)

        # Step 2: 权重预测（加入 SNR 掩码）
        with torch.no_grad():
            weights_t = self.scale_net(features_t, snr_mask_t)  # (T, K)
        weights_np = weights_t.cpu().numpy()
        self._last_weights = weights_np

        # Step 3: 加权组合优势
        adv_combined = (all_adv * weights_np).sum(axis=-1)  # (T,)
        buf.advantages = adv_combined
        buf.returns    = buf._compute_standard_returns(last_value, self.gamma, self.lam)

        # ④ 冻结归一化统计量
        self._adv_mean_frozen = float(adv_combined.mean())
        self._adv_std_frozen  = float(adv_combined.std()) + 1e-8

        # 统计
        autocorr = float(np.corrcoef(raw_deltas[:-1], raw_deltas[1:])[0, 1]) if T > 2 else 0.0
        weights_mean = weights_np.mean(axis=0)
        stats = {
            "delta_mean"    : float(raw_deltas.mean()),
            "delta_std"     : float(raw_deltas.std()),
            "delta_autocorr": autocorr,
        }
        for lam_k, wm in zip(self.LAMBDA_SCALES, weights_mean):
            stats[f"w_lambda_{lam_k}"] = float(wm)
        return stats

    def _update_scale_network(
        self,
        features: torch.Tensor,       # (batch, 3+hidden_dim)
        snr_mask: torch.Tensor,        # (batch, K)
        all_adv: torch.Tensor,         # (batch, K)
        mc_returns: torch.Tensor,      # (batch,)
        values: torch.Tensor,          # (batch,)
    ) -> float:
        """
        改进②：遗憾值损失（Regret Loss）替代方差惩罚

        Regret_t(k) = |A^{(k)}_t - (G_t - V_t)|
        Loss = Σ_t Σ_k w_t(k) * Regret_t(k) - λ_ent * H(w_t) + λ_snr * ...

        直觉：最优权重应全部分配给 Regret 最小的尺度（贝叶斯模型平均）
        熵正则防止坍缩到单一尺度
        """
        mc_adv = (mc_returns - values.detach())  # (batch,)

        weights = self.scale_net(features, snr_mask)  # (batch, K)

        # ② Regret Loss：每个尺度与 MC advantage 的绝对误差，加权平均
        # regret[i, k] = |all_adv[i,k] - mc_adv[i]|
        regret = (all_adv - mc_adv.unsqueeze(-1)).abs()  # (batch, K)
        regret_loss = (weights * regret).sum(dim=-1).mean()  # 加权平均遗憾

        # 熵正则
        K = weights.shape[-1]
        dynamic_ent_coef = self.scale_entropy_coef * (K / 4.0)
        entropy_reg = -(weights * torch.log(weights + 1e-9)).sum(dim=-1).mean()

        loss = regret_loss - dynamic_ent_coef * entropy_reg

        self.scale_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.scale_net.parameters(), self.max_grad_norm)
        self.scale_optimizer.step()

        return regret_loss.item()

    def update(self) -> dict:
        obs, actions, old_log_probs, advantages, returns, old_values = self.buffer.get_batch()

        # ④ 使用冻结统计量归一化
        advantages = (advantages - self._adv_mean_frozen) / self._adv_std_frozen

        T = self.buffer.pos
        indices = np.arange(T)
        metrics = {
            "value_loss": 0.0, "policy_loss": 0.0, "entropy_loss": 0.0,
            "approx_kl": 0.0, "clip_frac": 0.0, "scale_loss": 0.0
        }
        update_count = 0

        # 预计算 update 阶段用的特征和所有 GAE（保持一致性）
        nv_np = self.buffer._next_values(old_values[-1].item())
        all_adv_np = np.stack([
            self._compute_gae_single_scale(nv_np, lam_k)
            for lam_k in self.LAMBDA_SCALES
        ], axis=-1)
        raw_deltas_np = self.buffer.rewards[:T] + self.gamma * nv_np - self.buffer.values[:T]

        features_full, snr_mask_full = self._build_features_and_mask(raw_deltas_np, obs)
        all_adv_tensor_full = torch.FloatTensor(all_adv_np).to(self.device)

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, self.batch_size):
                end = start + self.batch_size
                if end > T:
                    break
                bidx = indices[start:end]

                b_obs      = obs[bidx]
                b_act      = actions[bidx]
                b_old_lp   = old_log_probs[bidx]
                b_adv      = advantages[bidx]
                b_ret      = returns[bidx]
                b_old_v    = old_values[bidx]

                new_log_probs, entropy = self.actor.evaluate_actions(b_obs, b_act)
                new_values = self.critic(b_obs)

                ratio = torch.exp(new_log_probs - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * b_adv
                policy_loss  = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()

                v_loss_unclipped = (new_values - b_ret) ** 2
                v_clipped = b_old_v + torch.clamp(
                    new_values - b_old_v, -self.eps_clip, self.eps_clip
                )
                value_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - b_ret) ** 2).mean()

                self.actor_optimizer.zero_grad()
                (policy_loss + self.ent_coef * entropy_loss).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                (self.vf_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                # 权重网络更新（遗憾值损失）
                b_feat     = features_full[bidx]
                b_snr_mask = snr_mask_full[bidx]
                b_all_adv  = all_adv_tensor_full[bidx]
                scale_loss = self._update_scale_network(
                    b_feat, b_snr_mask, b_all_adv, b_ret, b_old_v
                )
                metrics["scale_loss"] += scale_loss

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
            ev     = 1 - np.var(y_true - y_pred) / (var_y + 1e-8)
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
                eval_str = (
                    f"{last_eval_reward:7.1f}" if not np.isnan(last_eval_reward) else "    N/A"
                )
                w_strs = " ".join(
                    f"λ{lam_k}={metrics.get(f'w_lambda_{lam_k}', 0):.2f}"
                    for lam_k in self.LAMBDA_SCALES
                )
                print(
                    f"  [{self.NAME:<22}] "
                    f"{self.total_steps:7d}/{total_timesteps} {progress} "
                    f"| Eval={eval_str} Recent={recent:6.1f} "
                    f"| VLoss={metrics['value_loss']:.3f} EV={metrics['explained_variance']:+.2f} "
                    f"| KL={metrics['approx_kl']:.4f} "
                    f"| Regret={metrics['scale_loss']:.4f} "
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
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    dist   = self.actor(obs_t)
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

