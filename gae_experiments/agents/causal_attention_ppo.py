"""
革新方法三：Causal Attention GAE（因果注意力 GAE，CAGAE）

核心思想：
─────────────────────────────────────────────────────────────────────
标准 GAE 的权重结构是固定的几何衰减：
  A_t = Σ_{l≥0} (γλ)^l δ_{t+l}

这个权重假设 δ_{t+1} 对 A_t 的贡献恰好是 δ_t 的 γλ 倍。
但这是一个强假设，现实中并非如此：

  - 某些 δ_{t+k} 可能与 s_t 高度相关（动作的延迟效果），应给更大权重
  - 某些 δ_{t+k} 可能是随机噪声（纯粹由环境随机性引起），应给更小权重
  - episode 边界应该是硬截断，但对于 n_steps rollout，边界内部是连续的

CAGAE 的革新：
  用可学习的因果注意力权重替代固定的几何衰减。

  A_t^CA = Σ_{j≥t} α(s_t, s_j, j-t) * δ_j   [只看未来，保持因果性]

  其中 α 是注意力权重，满足：
    1. 因果约束：α(s_t, s_j, ...) = 0  when j < t
    2. 衰减约束：α 随 j-t 递减（确保收敛）
    3. 归一化：Σ_{j≥t} α(s_t, s_j, j-t) = 1（权重是分布）

CAGAE 设计：
─────────────────────────────────────────────────────────────────────
注意力权重由两部分构成：
  α(s_t, s_j, h) = softmax_over_j[ score(s_t, δ_j, h) + position_bias(h) ]

  score(s_t, δ_j, h) = f(δ_j, h)  [轻量：只用 δ 值和步距，不用 s 特征]
  position_bias(h) = -β * h        [衰减偏置，确保远处权重衰减]

具体实现（轻量版）：
  1. 计算原始注意力 logit：
     logit(t, j) = δ_j_clipped * gate_j - decay_coef * (j - t)

     其中 gate_j = sigmoid(W * [δ_j, local_std_j]) 是「δ 质量门控」
     - gate_j 接近 1 → δ_j 可信（用于计算优势）
     - gate_j 接近 0 → δ_j 不可信（来自高随机性区域）

  2. 因果掩码（只看当前步之后）+ episode 边界截断

  3. softmax 得到注意力权重 α_j（对每个 t，归一化到 1）

  4. A_t^CA = Σ_j α(t,j) * δ_j

训练信号：
  gate 网络不需要额外监督——它与主网络一起通过策略梯度自然优化。
  但我们加一个辅助损失：
    gate 应该更关注「当前 Critic 已经精确预测」的步骤
    辅助损失：gate_j 应该与 (1 - |δ_j| / err_scale) 相关

与 Transformer 的关系：
  CAGAE 是「单层因果 Transformer over rollout」的优势估计器。
  但不用 query/key/value 结构（太重），而是用更轻量的「门控衰减注意力」。

与 GAE 的关系（退化情形）：
  当 gate_j = 1（均匀置信）且 score = -γλ*(j-t) 时，
  CAGAE 退化为标准 GAE（softmax 退化为几何分布）。
─────────────────────────────────────────────────────────────────────
"""
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from ..utils.logger import MetricLogger
from ..utils.networks import ActorNetwork, CriticNetwork
from ..utils.rollout_buffer import RolloutBuffer


class GateNetwork(nn.Module):
    """
    轻量 δ 质量门控网络
    输入：[δ_t, local_std_t]（标量特征）
    输出：gate_t ∈ (0, 1)，代表 δ_t 的可信度
    """

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        # 初始化为中性（gate ≈ 0.5）
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: shape (T, 2)，返回 gate: shape (T,)"""
        return torch.sigmoid(self.net(x)).squeeze(-1)


class CausalAttentionPPO:
    """
    PPO + Causal Attention GAE（CAGAE）

    用因果注意力机制替代固定几何衰减权重，
    让模型自适应地关注 rollout 中质量高的 δ。
    """

    NAME = "CausalAttn_GAE"

    def __init__(
        self,
        env: gym.Env,
        hidden_dim: int = 64,
        gate_hidden_dim: int = 16,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        lr_gate: float = 1e-3,
        gamma: float = 0.99,
        lam: float = 0.95,           # 用于 returns 计算和位置偏置初始化
        eps_clip: float = 0.2,
        n_epochs: int = 10,
        batch_size: int = 64,
        n_steps: int = 2048,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        # CAGAE 超参数
        attn_horizon: int = 64,          # 注意力最大视野（步数）
        decay_init: float = 0.05,        # 位置偏置初始衰减（对应λ≈exp(-decay)≈0.95）
        gate_aux_coef: float = 0.1,      # gate 辅助损失系数
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
        self.attn_horizon = attn_horizon
        self.decay_init = decay_init
        self.gate_aux_coef = gate_aux_coef
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

        # 主网络
        self.actor  = ActorNetwork(obs_dim, action_dim, hidden_dim, self.continuous).to(self.device)
        self.critic = CriticNetwork(obs_dim, hidden_dim).to(self.device)

        # 门控网络（轻量）
        self.gate_net = GateNetwork(hidden_dim=gate_hidden_dim).to(self.device)

        # ★ 可学习的衰减系数（初始化为对应 λ=0.95 的值）
        # 我们用 log_decay 参数，令 decay = exp(log_decay) > 0
        # 初始 decay = -log(γλ)，使得 exp(-decay * h) = (γλ)^h
        # 因此 log_decay_init = log(-log(γλ))
        decay_init_val = -np.log(max(gamma * lam, 1e-8))   # 正数，约 0.051（γλ=0.9405）
        self.log_decay = nn.Parameter(
            torch.tensor(np.log(max(decay_init_val, 1e-4)), dtype=torch.float32),
        )
        self.log_decay = self.log_decay.to(self.device)

        # 优化器
        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),    lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),   lr=lr_critic)
        # gate 和 decay 一起优化
        self.gate_optimizer   = torch.optim.Adam(
            list(self.gate_net.parameters()) + [self.log_decay],
            lr=lr_gate
        )

        self.buffer = RolloutBuffer(n_steps, obs_dim, action_dim, self.device, self.continuous)

        # 运行统计
        self._delta_ema = 0.0
        self._delta_std_ema = 1.0
        self._ema_alpha = 0.05

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

    def compute_gae(self, last_value: float) -> dict:
        """
        Causal Attention GAE：

        A_t^CA = Σ_{j=t}^{t+H} α(t,j) * δ_j

        α(t,j) = softmax_j[ gate_j * δ_j_norm - decay * (j-t) ]
              （因果：j≥t；episode 边界截断）
        """
        T   = self.buffer.pos
        buf = self.buffer
        nv  = buf._next_values(last_value)

        # 计算原始 δ
        raw_deltas = buf.rewards[:T] + self.gamma * nv - buf.values[:T]

        # 更新 δ 的运行统计
        batch_mean = float(raw_deltas.mean())
        batch_std  = float(raw_deltas.std()) + 1e-8
        self._delta_ema     = (1 - self._ema_alpha) * self._delta_ema + self._ema_alpha * batch_mean
        self._delta_std_ema = (1 - self._ema_alpha) * self._delta_std_ema + self._ema_alpha * batch_std

        # 归一化 δ（供门控网络使用）
        delta_norm = (raw_deltas - self._delta_ema) / (self._delta_std_ema + 1e-8)

        # 局部 δ 标准差（5步窗口）
        local_std = np.zeros(T, dtype=np.float32)
        win = 5
        for t in range(T):
            lo = max(0, t - win // 2)
            hi = min(T, t + win // 2 + 1)
            local_std[t] = raw_deltas[lo:hi].std() + 1e-8

        # 用门控网络计算每个 δ 的质量分数
        gate_input = np.stack([delta_norm, local_std / (self._delta_std_ema + 1e-8)], axis=-1)
        gate_tensor = torch.FloatTensor(gate_input).to(self.device)
        with torch.no_grad():
            gate = self.gate_net(gate_tensor).cpu().numpy()  # shape (T,)
            decay = float(torch.exp(self.log_decay).item())  # 当前衰减系数

        # 计算因果注意力优势
        H = min(self.attn_horizon, T)
        adv = np.zeros(T, dtype=np.float32)

        for t in range(T):
            logits = []
            delta_vals = []
            valid_end = min(t + H, T)

            for j in range(t, valid_end):
                # 检查 episode 边界（如果 j-1 是 terminated，则 j 是新 episode 的开始）
                if j > t and buf.terminated[j - 1] > 0.5:
                    break  # episode 边界截断
                h = j - t
                # logit = gate_j (质量加分) - decay * h (距离惩罚)
                logit = gate[j] - decay * h
                logits.append(logit)
                delta_vals.append(raw_deltas[j])

            if not logits:
                adv[t] = raw_deltas[t]
                continue

            logits_arr = np.array(logits, dtype=np.float32)
            delta_arr  = np.array(delta_vals, dtype=np.float32)

            # 数值稳定的 softmax
            logits_arr -= logits_arr.max()
            w = np.exp(logits_arr)
            w /= w.sum() + 1e-9

            adv[t] = (w * delta_arr).sum()

        buf.advantages = adv
        # returns = 标准 GAE returns（Critic 的稳定目标）
        buf.returns = buf._compute_standard_returns(last_value, self.gamma, self.lam)

        # 统计
        autocorr = float(np.corrcoef(raw_deltas[:-1], raw_deltas[1:])[0, 1]) if T > 2 else 0.0
        return {
            "delta_mean"    : float(raw_deltas.mean()),
            "delta_std"     : float(raw_deltas.std()),
            "delta_autocorr": autocorr,
            "mean_gate"     : float(gate.mean()),
            "decay_coef"    : decay,
            "eff_lambda"    : float(np.exp(-decay)),  # 等效 λ = exp(-decay)
        }

    def update_gate_network(
        self,
        gate_feat_full: torch.Tensor,   # shape (T, 2)
        raw_delta_full: torch.Tensor,   # shape (T,)
        batch_idx: np.ndarray,
        err_scale: float,
    ) -> float:
        """
        训练门控网络 - 改进版本：

        ★ 优化1: TV-Consistency 辅助损失
        -----------------------------------------
        直觉：gate 是 δ 的「可信度平滑指示器」。
        相邻步的 δ 变化不大时，它们的 gate 也应相近（平滑性）。
        当相邻 δ 差距大时，允许 gate 差距大（允许区分高低质量）。

        L_tv = Σ_t  max(0, |gate_t - gate_{t-1}| - ε * |δ_t - δ_{t-1}|)
        ε：容忍比例，允许 gate 随 δ 变化适度改变

        ★ 优化2: 熵正则防止 gate 坍缩
        -----------------------------------------
        gate ∈ (0,1) 的熵 = -gate*log(gate) - (1-gate)*log(1-gate)
        最大化熵使 gate 保持多样性，不向 0 或 1 坍缩。

        ★ 优化3: 移除强制方向性的 target
        -----------------------------------------
        旧版：target = sigmoid(-2|δ|/err_scale) → 强制 gate 趋近 0，破坏区分能力
        新版：只用 TV 和熵，让 gate 自由学习，不硬编码方向
        """
        T = gate_feat_full.shape[0]
        gate_full = self.gate_net(gate_feat_full)  # shape (T,)

        # 优化1: TV-Consistency（全序列，取相邻帧）
        if T > 1:
            gate_diff  = (gate_full[1:] - gate_full[:-1]).abs()
            delta_diff = (raw_delta_full[1:] - raw_delta_full[:-1]).abs()
            eps_tv = 0.5  # 容忍比例
            tv_loss = torch.clamp(gate_diff - eps_tv * delta_diff / (err_scale + 1e-8), min=0.0).mean()
        else:
            tv_loss = gate_full.new_zeros(1).squeeze()

        # 优化2: 基于 δ 一致性的软目标（给 gate 方向性信号）
        # 思路：相邻 δ 方向一致（同号）时，说明此处的 TD 误差是可信的信号，gate 应高
        # 相邻 δ 方向不一致（异号），说明随机性主导，gate 应低
        if T > 2:
            delta_sign_consistency = torch.zeros(T, device=gate_feat_full.device)
            # 与左右邻居的符号一致性
            sign_agree_left  = (raw_delta_full[1:].sign() == raw_delta_full[:-1].sign()).float()
            sign_agree_right = (raw_delta_full[:-1].sign() == raw_delta_full[1:].sign()).float()
            # 中间点：与左右都一致
            delta_sign_consistency[1:-1] = 0.5 * (sign_agree_left[:-1] + sign_agree_right[1:])
            delta_sign_consistency[0] = sign_agree_left[0]   # 首端
            delta_sign_consistency[-1] = sign_agree_right[-1] # 末端
            # 软目标 = 一致性比例（范围 [0,1]）
            soft_target = delta_sign_consistency.detach()
            # 允许 gate 有分化空间：不强制等于 soft_target，只是轻微引导
            direction_loss = ((gate_full - soft_target) ** 2).mean() * 0.5
        else:
            direction_loss = gate_full.new_zeros(1).squeeze()

        # 熵正则（防止 gate 坍缩到同一值，配合方向性信号使用）
        eps_ent = 1e-6
        g = gate_full.clamp(eps_ent, 1.0 - eps_ent)
        entropy_gate = -(g * g.log() + (1 - g) * (1 - g).log()).mean()
        # 目标：最大化熵，但方向性信号优先
        entropy_loss = -entropy_gate * 0.05  # 降低熵权重，让方向信号主导

        aux_loss = tv_loss + direction_loss + entropy_loss

        self.gate_optimizer.zero_grad()
        (self.gate_aux_coef * aux_loss).backward()
        nn.utils.clip_grad_norm_(
            list(self.gate_net.parameters()) + [self.log_decay],
            self.max_grad_norm
        )
        self.gate_optimizer.step()

        return aux_loss.item()

    def update(self) -> dict:
        obs, actions, old_log_probs, advantages, returns, old_values = self.buffer.get_batch()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        T = self.buffer.pos
        indices = np.arange(T)
        metrics = {"value_loss": 0.0, "policy_loss": 0.0, "entropy_loss": 0.0,
                   "approx_kl": 0.0, "clip_frac": 0.0, "gate_loss": 0.0}
        update_count = 0

        # 预计算 gate 特征（用于训练）
        nv_np = self.buffer._next_values(old_values[-1].item())
        raw_delta_np = (
            self.buffer.rewards[:T] + self.gamma * nv_np - self.buffer.values[:T]
        )
        delta_norm_np = (raw_delta_np - self._delta_ema) / (self._delta_std_ema + 1e-8)
        local_std_np = np.zeros(T, dtype=np.float32)
        for t in range(T):
            lo = max(0, t - 2); hi = min(T, t + 3)
            local_std_np[t] = raw_delta_np[lo:hi].std() + 1e-8
        gate_feat_np = np.stack([delta_norm_np, local_std_np / (self._delta_std_ema + 1e-8)], axis=-1)
        gate_feat_full = torch.FloatTensor(gate_feat_np).to(self.device)
        raw_delta_full = torch.FloatTensor(raw_delta_np).to(self.device)
        err_scale = float(np.abs(raw_delta_np).mean()) + 1e-8

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

                # 训练门控网络（新版：全序列TV+熵正则，每轮epoch只更新一次以全序列为单位）
                # 注意：此处 batch_idx 只是标记，gate 训练在 epoch==0 时使用全序列
                gate_loss_v = self.update_gate_network(
                    gate_feat_full,   # 传全序列，以计算TV-consistency
                    raw_delta_full,
                    batch_idx,
                    err_scale,
                )
                metrics["gate_loss"] += gate_loss_v

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
                gate_m   = metrics.get('mean_gate', 0)
                decay_c  = metrics.get('decay_coef', 0)
                eff_lam  = metrics.get('eff_lambda', 0)
                delta_str = (
                    f"δ:μ={metrics.get('delta_mean',0):+.3f}"
                    f"/σ={metrics.get('delta_std',0):.3f}"
                    f"/r1={metrics.get('delta_autocorr',0):+.2f}"
                )
                print(
                    f"  [{self.NAME:<22}] "
                    f"{self.total_steps:7d}/{total_timesteps} "
                    f"{progress} "
                    f"| Eval={eval_str} Recent={recent:6.1f} "
                    f"| VLoss={metrics['value_loss']:.3f} EV={metrics['explained_variance']:+.2f} "
                    f"| {delta_str}"
                    f"| gate={gate_m:.3f} decay={decay_c:.3f} eff_λ={eff_lam:.3f}"
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

