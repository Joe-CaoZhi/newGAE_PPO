"""
革新方法一：Hindsight-Corrected GAE（HCGAE）

核心思想：
─────────────────────────────────────────────────────────────────────
标准 GAE 的根本局限：

  δ_t = r_t + γV(s_{t+1}) - V(s_t)

这里 V(s) 是「事前」估计，在训练早期精度很低（EV 仅 0.1~0.5）。
Critic 的系统性误差会通过 GAE 展开传播，每步都带有偏差。

Hindsight 洞察：
  rollout 结束后，我们「已经知道」每个 episode 的 MC return G_t。
  这是比 V(s_t) 更准确的价值估计（对当前策略来说是无偏的）。

  传统做法：把 G_t 作为 Critic 的训练目标（returns）。
  我们的做法：用 G_t 构造事后修正的 Critic，再用修正后的 V 重新计算 δ。

HCGAE 公式：
─────────────────────────────────────────────────────────────────────
  1. 计算 MC returns（从 rollout 末尾反向展开）：
     G_t = r_t + γ G_{t+1}   （G_{T-1} = r_{T-1} + γ V(last_state)）

  2. 构造修正价值（线性插值）：
     V_corrected(s_t) = (1-α_t) V(s_t) + α_t G_t

     其中修正系数 α_t ∈ [0,1] 由 Critic 的局部误差决定：
     err_t = |V(s_t) - G_t|
     α_t = σ(β * err_t / (running_scale + ε))   （Sigmoid 软门控）

     - Critic 准确时（err≈0）→ α≈0，用原始 V（低方差）
     - Critic 不准时（err大）→ α→1，用 MC return（低偏差）

  3. 用修正价值重新计算 δ：
     δ_t^corrected = r_t + γ V_corrected(s_{t+1}) - V_corrected(s_t)
     A_t = GAE(δ_corrected, λ)

  4. Critic 目标：用原始 MC returns（标准做法，不受修正污染）

关键优势：
  - 自适应偏差-方差权衡：不依赖手动调 λ，而是根据实际 Critic 误差自动选择
  - 无额外网络参数：α_t 完全由误差决定，不需要额外神经网络
  - 理论一致性：训练后期 Critic 精确时，HCGAE 退化为标准 GAE（α→0）

与 TD(λ) 的关系：
  TD(λ) 通过 λ 加权不同步数的 TD，HCGAE 通过误差门控在 TD 和 MC 之间切换。
  这是更精细的控制：不是全局调整，而是逐步骤自适应。

与 V-trace / Retrace 的关系：
  V-trace 用重要性采样修正 off-policy 数据，HCGAE 用 hindsight 误差修正 on-policy 偏差。
  方向不同，可以组合。
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


class HindsightPPO:
    """
    PPO + Hindsight-Corrected GAE（HCGAE）

    用事后 MC return 自适应修正 Critic 偏差，从而减小 δ_t 的方差。
    修正强度由 Critic 的局部误差自动控制。
    """

    NAME = "Hindsight_GAE"

    def __init__(
        self,
        env: gym.Env,
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
        # HCGAE 超参数
        hindsight_beta: float = 3.0,      # Sigmoid 门控的陡峭度（越大则切换越陡）
        hindsight_alpha_max: float = 0.7,  # 最大修正系数（训练初期上限，后期衰减）
        hindsight_alpha_min: float = 0.1,  # 最小修正系数（训练后期Critic精确时的下限）
        mc_critic_coef: float = 0.5,       # Critic 目标 = mc_critic_coef*MC + (1-coef)*GAE_returns
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
        self.hindsight_beta = hindsight_beta
        self.hindsight_alpha_max = hindsight_alpha_max
        self.hindsight_alpha_min = hindsight_alpha_min
        self.mc_critic_coef = mc_critic_coef
        self.device = torch.device(device)
        # 训练进度追踪（用于 α 退火）
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

        # 在线估计 |V-G| 的滑动均值（用于归一化误差）
        self._err_ema = 1.0          # EMA of |V - G|，初始化为 1
        self._err_ema_alpha = 0.05   # EMA 更新系数（慢速追踪）
        # EV（explained variance）的 EMA，用于估计 Critic 当前精度
        self._ev_ema = 0.0
        self._ev_ema_alpha = 0.1

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

    def _compute_mc_returns(self, last_value: float) -> np.ndarray:
        """
        计算 MC returns（从轨迹末尾反向展开）
        G_T = last_value
        G_t = r_t + γ * G_{t+1} * (1 - terminated[t])
        """
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
        Hindsight-Corrected GAE：

        步骤：
        1. 计算 MC returns G_t
        2. 计算逐步误差 err_t = |V(s_t) - G_t|
        3. 门控系数 α_t = α_max * sigmoid(β * err_t / err_scale)
        4. 修正价值 V_corrected(s_t) = (1-α_t) * V(s_t) + α_t * G_t
        5. 用修正价值重新计算 δ_t，然后标准 GAE 展开
        6. returns = G_t（原始 MC，给 Critic 干净目标）
        """
        T   = self.buffer.pos
        buf = self.buffer

        # Step 1: MC returns
        G = self._compute_mc_returns(last_value)

        # Step 2: 逐步误差
        V = buf.values[:T]
        err = np.abs(V - G)

        # Step 3: 更新误差的 EMA（在线估计当前误差规模）
        batch_err_mean = float(err.mean())
        self._err_ema = (1 - self._err_ema_alpha) * self._err_ema + self._err_ema_alpha * batch_err_mean
        err_scale = max(self._err_ema, 1e-8)

        # ★ 自适应 α_max：随训练进度和 Critic 精度退火
        # 训练初期 Critic 不准 → α_max 大（多用MC）
        # 训练后期 Critic 精确 → α_max 小（更信任Critic，减少MC高方差）
        # 用 EV 代理 Critic 精度：EV 越高 → 越依赖 Critic
        progress = min(self.total_steps / max(self._total_timesteps, 1), 1.0)
        # 余弦退火：从 alpha_max 衰减到 alpha_min
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        # EV 自适应：EV 高时进一步降低 α（Critic 已经很准，MC 修正意义减小）
        ev_factor = max(1.0 - max(self._ev_ema, 0.0), 0.2)   # EV=1→factor=0.2; EV=0→factor=1.0
        dynamic_alpha_max = (
            self.hindsight_alpha_min
            + (self.hindsight_alpha_max - self.hindsight_alpha_min)
            * cosine_decay * ev_factor
        )

        # 门控：err 大 → α 大 → 更多用 MC；err 小 → α 小 → 更多用 V
        alpha = dynamic_alpha_max / (1.0 + np.exp(-self.hindsight_beta * (err / err_scale - 1.0)))

        # Step 4: 修正价值
        V_corrected = (1.0 - alpha) * V + alpha * G  # shape (T,)

        # 修正后的 next_value
        # 对于 t < T-1：next_value = V_corrected[t+1]
        # 对于 t = T-1：next_value = last_value（bootstrap；我们没有 last_state 的 MC return，不修正）
        # 对于 terminated[t]=1：next_value = 0
        V_corrected_next = np.empty(T, dtype=np.float32)
        for t in range(T):
            if buf.terminated[t] > 0.5:
                V_corrected_next[t] = 0.0
            elif t == T - 1:
                # rollout 末端：直接用 last_value 作为 bootstrap（无对应 MC 误差可修正）
                V_corrected_next[t] = last_value
            else:
                V_corrected_next[t] = V_corrected[t + 1]

        # Step 5: 用修正后的 V 重新计算 δ，然后标准 GAE 展开
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            # 修正后的 TD 残差
            delta_corrected = (
                buf.rewards[t]
                + self.gamma * V_corrected_next[t]
                - V_corrected[t]
            )
            not_done = 1.0 - buf.terminated[t]
            gae      = delta_corrected + self.gamma * self.lam * not_done * gae
            adv[t]   = gae

        buf.advantages = adv
        # Step 6: Critic 目标 = MC_returns 和标准 GAE_returns 的混合
        # 训练初期：MC 更准确（Critic偏差大），多用 MC
        # 训练后期：GAE_returns 方差更低（Critic已经较准），多用 GAE_returns
        # mc_critic_coef 控制混合比例（固定值，简单有效）
        gae_returns = buf._compute_standard_returns(last_value, self.gamma, self.lam)
        buf.returns = self.mc_critic_coef * G + (1.0 - self.mc_critic_coef) * gae_returns

        # 统计
        raw_deltas = buf.rewards[:T] + self.gamma * buf._next_values(last_value) - V
        autocorr = float(np.corrcoef(raw_deltas[:-1], raw_deltas[1:])[0, 1]) if T > 2 else 0.0
        return {
            "delta_mean"         : float(raw_deltas.mean()),
            "delta_std"          : float(raw_deltas.std()),
            "delta_autocorr"     : autocorr,
            "mean_alpha"         : float(alpha.mean()),
            "dynamic_alpha_max"  : float(dynamic_alpha_max),
            "err_scale"          : err_scale,
            "mean_hindsight_err" : batch_err_mean,
        }

    def update(self) -> dict:
        obs, actions, old_log_probs, advantages, returns, old_values = self.buffer.get_batch()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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

                # Critic 直接拟合 MC returns（无 clip，因为目标是 MC 而非 A+V）
                value_loss = 0.5 * ((new_values - batch_returns) ** 2).mean()

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

        self._total_timesteps = total_timesteps  # 供 compute_gae 退火用
        while self.total_steps < total_timesteps:
            last_value = self.collect_rollout()
            gae_stats  = self.compute_gae(last_value)
            metrics    = self.update()
            update_idx += 1
            metrics.update(gae_stats)
            # 更新 EV 的 EMA（用于自适应 α_max）
            ev_val = metrics.get('explained_variance', 0.0)
            self._ev_ema = (1 - self._ev_ema_alpha) * self._ev_ema + self._ev_ema_alpha * ev_val

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
                alpha_m    = metrics.get('mean_alpha', 0)
                alpha_max  = metrics.get('dynamic_alpha_max', 0)
                err_sc     = metrics.get('err_scale', 0)
                delta_str = (
                    f"δ:μ={metrics.get('delta_mean',0):+.3f}"
                    f"/σ={metrics.get('delta_std',0):.3f}"
                    f"/r1={metrics.get('delta_autocorr',0):+.2f}"
                )
                print(
                    f"  [{self.NAME:<24}] "
                    f"{self.total_steps:7d}/{total_timesteps} "
                    f"{progress} "
                    f"| Eval={eval_str} Recent={recent:6.1f} "
                    f"| VLoss={metrics['value_loss']:.3f} EV={metrics['explained_variance']:+.2f} "
                    f"| KL={metrics['approx_kl']:.4f} clip={metrics['clip_frac']:.2f}"
                    f"| {delta_str}"
                    f"| α={alpha_m:.3f}(αmax={alpha_max:.2f}) err_scale={err_sc:.3f}"
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

