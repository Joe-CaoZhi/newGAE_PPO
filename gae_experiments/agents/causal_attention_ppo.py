"""
革新方法三：Causal Attention GAE（因果注意力 GAE，CAGAE）v2

改进清单（v2）：
─────────────────────────────────────────────────────────────────────
① 向量化注意力实现（消除 O(T²) Python 循环）
   旧：Python 双重 for 循环，T=2048 时约 4M 次迭代，严重拖慢训练
   新：构建 (T, H) 滑动窗口矩阵（gather 操作），纯 NumPy/Torch 向量化
       时间复杂度保持 O(T*H)，但常数因子极大降低（向量化 vs. 解释器循环）
       H = attn_horizon = 64，实际可以每步最多看 64 步

② 有界衰减参数（稳定训练）
   旧：decay = exp(log_decay)，log_decay 无界，训练早期可能 decay→∞（注意力坍缩到 δ_t 自身）
       或 decay→0（注意力均匀分配，等同于平均），失去位置先验
   新：decay = sigmoid(raw_decay) * (decay_max - decay_min) + decay_min
       即强制 decay ∈ [0.01, 0.5]，对应 eff_λ ∈ [exp(-0.5), exp(-0.01)] ≈ [0.61, 0.99]
       物理含义：保证有一定衰减（不退化为均匀注意力），也不会衰减过快（不退化为 TD(0)）

③ 余弦相似度监督信号（替代符号一致性）
   旧：gate 训练用符号一致性（±1 离散信号），信息量少且对噪声敏感
   新：用「δ_t 与局部 δ 滑动平均的余弦相似度」作为软目标
       cosine_t = (δ_t · μ_local) / (|δ_t| * |μ_local| + ε)
       cosine ∈ [-1, 1]，映射到 gate_target = (cosine + 1) / 2 ∈ [0, 1]
       物理含义：δ_t 与局部趋势一致时（余弦>0），该步信号可信（gate 应高）；
       δ_t 与局部趋势相反时（余弦<0），该步可能是噪声（gate 应低）

④ 归一化统计量冻结（与 HCGAE/MSGAE v2 一致）
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
    轻量 δ 质量门控网络 v2
    输入：[δ_norm, local_std_norm]（标量特征）
    输出：gate ∈ (0, 1)，代表 δ 的可信度
    """

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


class CausalAttentionPPO:
    """PPO + Causal Attention GAE（CAGAE）v2"""

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
        lam: float = 0.95,
        eps_clip: float = 0.2,
        n_epochs: int = 10,
        batch_size: int = 64,
        n_steps: int = 2048,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        attn_horizon: int = 64,
        gate_aux_coef: float = 0.1,
        # ② 有界衰减参数范围
        decay_min: float = 0.01,
        decay_max: float = 0.5,
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
        self.gate_aux_coef = gate_aux_coef
        self.decay_min = decay_min
        self.decay_max = decay_max
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

        self.actor    = ActorNetwork(obs_dim, action_dim, hidden_dim, self.continuous).to(self.device)
        self.critic   = CriticNetwork(obs_dim, hidden_dim).to(self.device)
        self.gate_net = GateNetwork(hidden_dim=gate_hidden_dim).to(self.device)

        # ② 有界衰减参数：raw_decay 无界，通过 sigmoid 映射到 [decay_min, decay_max]
        # 初始化使得 decay 对应 γλ 衰减
        # decay_init = -log(γλ) ≈ 0.051（γ=0.99,λ=0.95）
        target_decay = -np.log(max(gamma * lam, 1e-8))
        target_decay = float(np.clip(target_decay, decay_min, decay_max))
        # 反向求解 sigmoid 的初始值
        sigma = (target_decay - decay_min) / (decay_max - decay_min + 1e-8)
        sigma = float(np.clip(sigma, 1e-6, 1 - 1e-6))
        raw_decay_init = float(np.log(sigma / (1 - sigma)))  # sigmoid 的 logit
        self.raw_decay = nn.Parameter(
            torch.tensor(raw_decay_init, dtype=torch.float32)
        ).to(self.device)

        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),    lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),   lr=lr_critic)
        self.gate_optimizer   = torch.optim.Adam(
            list(self.gate_net.parameters()) + [self.raw_decay], lr=lr_gate
        )

        self.buffer = RolloutBuffer(n_steps, obs_dim, action_dim, self.device, self.continuous)

        self._delta_ema     = 0.0
        self._delta_std_ema = 1.0
        self._ema_alpha     = 0.05
        # ④ 冻结归一化统计量
        self._adv_mean_frozen = 0.0
        self._adv_std_frozen  = 1.0

        self.logger = MetricLogger(self.NAME, save_dir)
        self.total_steps = 0

    def _get_decay(self) -> float:
        """② 有界衰减：decay ∈ [decay_min, decay_max]"""
        return float(
            torch.sigmoid(self.raw_decay) * (self.decay_max - self.decay_min) + self.decay_min
        )

    def collect_rollout(self) -> float:
        self.buffer.reset()
        obs, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0

        for _ in range(self.n_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, log_prob = self.actor.get_action_and_logprob(obs_t)
                value = self.critic(obs_t)

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
            last_obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            last_value = self.critic(last_obs_t).item()
        return last_value

    def compute_gae(self, last_value: float) -> dict:
        """
        ① 向量化因果注意力 GAE

        对每个时刻 t，视野 [t, t+H-1]：
        - 构建 (T, H) 的 delta_window 和 gate_window 矩阵
        - logit[t, h] = gate[t+h] - decay * h  (h=0..H-1)
        - 对超出 rollout 范围 或 episode 边界的位置施加 -inf 掩码
        - softmax over h → attention weights
        - adv[t] = sum_h weight[t,h] * delta[t+h]
        """
        T   = self.buffer.pos
        buf = self.buffer
        nv  = buf._next_values(last_value)

        raw_deltas = buf.rewards[:T] + self.gamma * nv - buf.values[:T]

        # 更新运行统计
        batch_mean = float(raw_deltas.mean())
        batch_std  = float(raw_deltas.std()) + 1e-8
        self._delta_ema     = (1 - self._ema_alpha) * self._delta_ema + self._ema_alpha * batch_mean
        self._delta_std_ema = (1 - self._ema_alpha) * self._delta_std_ema + self._ema_alpha * batch_std

        delta_norm = (raw_deltas - self._delta_ema) / (self._delta_std_ema + 1e-8)

        # 局部 δ 标准差（5步窗口）
        local_std = np.zeros(T, dtype=np.float32)
        win = 5
        for t in range(T):
            lo = max(0, t - win // 2)
            hi = min(T, t + win // 2 + 1)
            local_std[t] = raw_deltas[lo:hi].std() + 1e-8

        # 门控网络
        gate_input = np.stack([delta_norm, local_std / (self._delta_std_ema + 1e-8)], axis=-1)
        gate_t     = torch.FloatTensor(gate_input).to(self.device)
        with torch.no_grad():
            gate   = self.gate_net(gate_t).cpu().numpy()   # (T,)
            decay  = self._get_decay()

        # ① 向量化窗口构建
        H = min(self.attn_horizon, T)

        # 构建 (T, H) 索引矩阵，超出范围用 T-1 填充（之后掩码置 -inf）
        t_idx = np.arange(T, dtype=np.int64)[:, None]     # (T, 1)
        h_idx = np.arange(H, dtype=np.int64)[None, :]     # (1, H)
        j_idx = np.clip(t_idx + h_idx, 0, T - 1)          # (T, H)

        # delta_window[t, h] = raw_deltas[t+h]（超出范围位置值无所谓，掩码会屏蔽）
        delta_window = raw_deltas[j_idx]   # (T, H)
        gate_window  = gate[j_idx]         # (T, H)

        # 位置惩罚矩阵
        h_mat = h_idx.astype(np.float32)   # (1, H)

        # logit[t, h] = gate[t+h] - decay * h
        logits = gate_window - decay * h_mat   # (T, H)

        # 构建掩码：
        # (a) 超出 rollout 末端
        out_of_range = (t_idx + h_idx) >= T   # (T, H)
        # (b) episode 边界：j 和 t 不在同一 episode
        # terminated[j-1]=1 意味着 j 是新 episode 的第一步
        # 构建 cumsum 方式的 episode ID
        episode_id = np.zeros(T, dtype=np.int32)
        for i in range(1, T):
            episode_id[i] = episode_id[i - 1] + int(buf.terminated[i - 1] > 0.5)
        episode_window = episode_id[j_idx]        # (T, H)
        cross_episode  = (episode_window != episode_id[:, None])   # (T, H)

        mask = out_of_range | cross_episode
        logits[mask] = -1e9

        # softmax（数值稳定）
        logits -= logits.max(axis=-1, keepdims=True)
        w  = np.exp(logits)
        w[mask] = 0.0  # 确保掩码位置权重为 0
        w_sum = w.sum(axis=-1, keepdims=True) + 1e-9
        weights = w / w_sum   # (T, H)

        # 加权求和
        adv = (weights * delta_window).sum(axis=-1)   # (T,)

        buf.advantages = adv
        buf.returns    = buf._compute_standard_returns(last_value, self.gamma, self.lam)

        # ④ 冻结归一化统计量
        self._adv_mean_frozen = float(adv.mean())
        self._adv_std_frozen  = float(adv.std()) + 1e-8

        autocorr = float(np.corrcoef(raw_deltas[:-1], raw_deltas[1:])[0, 1]) if T > 2 else 0.0
        eff_lambda = float(np.exp(-decay))
        return {
            "delta_mean"    : float(raw_deltas.mean()),
            "delta_std"     : float(raw_deltas.std()),
            "delta_autocorr": autocorr,
            "mean_gate"     : float(gate.mean()),
            "decay_coef"    : decay,
            "eff_lambda"    : eff_lambda,
        }

    def _update_gate(
        self,
        gate_feat_full: torch.Tensor,    # (T, 2)
        raw_delta_full: torch.Tensor,    # (T,)
        err_scale: float,
    ) -> float:
        """
        ③ 余弦相似度监督信号替代符号一致性

        local_mu[t] = 加权局部 δ 均值（5步窗口）
        cosine_t    = dot(δ_t, local_mu_t) / (|δ_t| * |local_mu_t| + ε)
        gate_target = (cosine_t + 1) / 2 ∈ [0, 1]

        物理含义：
          - δ_t 与局部趋势同向 → cosine > 0 → gate_target > 0.5 → gate 高（可信）
          - δ_t 与局部趋势反向 → cosine < 0 → gate_target < 0.5 → gate 低（噪声）
        """
        T = gate_feat_full.shape[0]
        gate_full = self.gate_net(gate_feat_full)   # (T,)

        # ③ 余弦相似度软目标
        win = 5
        local_mu = torch.zeros(T, device=self.device)
        for t in range(T):
            lo = max(0, t - win // 2)
            hi = min(T, t + win // 2 + 1)
            local_mu[t] = raw_delta_full[lo:hi].mean()

        # 余弦相似度（1D 情形等于符号+幅度信息）
        cos_sim = (raw_delta_full * local_mu) / (
            raw_delta_full.abs() * local_mu.abs() + 1e-8
        )   # (T,) ∈ [-1, 1]
        gate_target = ((cos_sim + 1.0) / 2.0).detach()   # ∈ [0, 1]

        # 监督损失（BCE 形式，防止硬 0/1）
        eps_bce = 0.05
        gate_target_clamped = gate_target.clamp(eps_bce, 1.0 - eps_bce)
        direction_loss = nn.functional.binary_cross_entropy(
            gate_full, gate_target_clamped
        )

        # TV 一致性（平滑性约束）
        if T > 1:
            gate_diff  = (gate_full[1:] - gate_full[:-1]).abs()
            delta_diff = (raw_delta_full[1:] - raw_delta_full[:-1]).abs()
            eps_tv = 0.5
            tv_loss = torch.clamp(
                gate_diff - eps_tv * delta_diff / (err_scale + 1e-8), min=0.0
            ).mean()
        else:
            tv_loss = gate_full.new_zeros(1).squeeze()

        aux_loss = direction_loss + 0.1 * tv_loss

        self.gate_optimizer.zero_grad()
        (self.gate_aux_coef * aux_loss).backward()
        nn.utils.clip_grad_norm_(
            list(self.gate_net.parameters()) + [self.raw_decay],
            self.max_grad_norm
        )
        self.gate_optimizer.step()

        return aux_loss.item()

    def update(self) -> dict:
        obs, actions, old_log_probs, advantages, returns, old_values = self.buffer.get_batch()

        # ④ 使用冻结统计量归一化
        advantages = (advantages - self._adv_mean_frozen) / self._adv_std_frozen

        T = self.buffer.pos
        indices = np.arange(T)
        metrics = {
            "value_loss": 0.0, "policy_loss": 0.0, "entropy_loss": 0.0,
            "approx_kl": 0.0, "clip_frac": 0.0, "gate_loss": 0.0
        }
        update_count = 0

        # 预计算 gate 特征
        nv_np = self.buffer._next_values(old_values[-1].item())
        raw_delta_np = (
            self.buffer.rewards[:T] + self.gamma * nv_np - self.buffer.values[:T]
        )
        delta_norm_np = (raw_delta_np - self._delta_ema) / (self._delta_std_ema + 1e-8)
        local_std_np  = np.zeros(T, dtype=np.float32)
        for t in range(T):
            lo = max(0, t - 2); hi = min(T, t + 3)
            local_std_np[t] = raw_delta_np[lo:hi].std() + 1e-8
        gate_feat_np = np.stack(
            [delta_norm_np, local_std_np / (self._delta_std_ema + 1e-8)], axis=-1
        )
        gate_feat_full = torch.FloatTensor(gate_feat_np).to(self.device)
        raw_delta_full = torch.FloatTensor(raw_delta_np).to(self.device)
        err_scale      = float(np.abs(raw_delta_np).mean()) + 1e-8

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, self.batch_size):
                end = start + self.batch_size
                if end > T:
                    break
                bidx = indices[start:end]

                b_obs     = obs[bidx]
                b_act     = actions[bidx]
                b_old_lp  = old_log_probs[bidx]
                b_adv     = advantages[bidx]
                b_ret     = returns[bidx]
                b_old_v   = old_values[bidx]

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

                # gate 网络：每 n_epochs 第一次 minibatch 用全序列更新一次
                gate_loss_v = self._update_gate(gate_feat_full, raw_delta_full, err_scale)
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
                elapsed   = time.time() - train_start
                recent    = self.logger.get_recent_reward(20)
                progress  = self._progress_bar(self.total_steps, total_timesteps)
                fps       = int(self.total_steps / (elapsed + 1e-8))
                eval_str  = (
                    f"{last_eval_reward:7.1f}" if not np.isnan(last_eval_reward) else "    N/A"
                )
                gate_m    = metrics.get('mean_gate', 0)
                decay_c   = metrics.get('decay_coef', 0)
                eff_lam   = metrics.get('eff_lambda', 0)
                delta_str = (
                    f"δ:μ={metrics.get('delta_mean',0):+.3f}"
                    f"/σ={metrics.get('delta_std',0):.3f}"
                    f"/r1={metrics.get('delta_autocorr',0):+.2f}"
                )
                print(
                    f"  [{self.NAME:<22}] "
                    f"{self.total_steps:7d}/{total_timesteps} {progress} "
                    f"| Eval={eval_str} Recent={recent:6.1f} "
                    f"| VLoss={metrics['value_loss']:.3f} EV={metrics['explained_variance']:+.2f} "
                    f"| {delta_str} "
                    f"| gate={gate_m:.3f} decay={decay_c:.4f} eff_λ={eff_lam:.3f}"
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

