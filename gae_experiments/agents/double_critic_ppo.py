"""
改进一：保守 Bootstrap GAE（Conservative Bootstrap GAE）

诊断发现（v2 修复）：
  - 原版 bias_coef 固定为 0.2，训练后期 EMA 已经与在线 Critic 高度一致，
    |V - V_ema| 趋近 0，保守惩罚消失 → 退化为标准 GAE
  - 但训练后期如果奖励分布突变（exploration 减少），bias 再次出现，
    导致 δ 突变 → 性能崩溃

v2 改进：
  1. 自适应 bias_coef 调度：初期大（保守防过估计），后期小（精确追踪）
     bias_coef(t) = bias_max * exp(-decay * step / total_steps)
     早期β≈0.3，后期β→0.05，平滑过渡

  2. δ 稳定化 clip：限制单步δ的绝对值不超过 reward_scale 的若干倍
     防止 rollout 内个别异常 δ 污染整个 GAE 展开

  3. 新增 returns 目标：Critic 用 GAE returns 训练（而非 A+V），
     与 on-policy 数据一致性更强

  数学表示：
    β(t) = β_max * exp(-κ * t/T)
    V_safe(s') = V(s') - β(t) * |V(s') - V_ema(s')|
    δ_t = clip(r_t + γ V_safe(s') - V(s), -δ_clip, δ_clip)
    A_t = GAE(δ, λ)
"""
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from ..utils.logger import MetricLogger
from ..utils.networks import ActorNetwork, CriticNetwork
from ..utils.rollout_buffer import RolloutBuffer


class ConservativeBootstrapPPO:
    """
    PPO + 保守 Bootstrap GAE
    使用 EMA Critic 作为 target network，减少价值估计的乐观偏差。
    """

    NAME = "Conservative_Bootstrap_GAE"

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
        # 保守 bootstrap 参数 (v2)
        ema_tau: float = 0.005,      # EMA 更新速率（类似 DQN target network soft update）
        bias_coef_max: float = 0.3,  # 早期保守惩罚系数（较大）
        bias_coef_min: float = 0.05, # 后期保守惩罚系数（较小）
        bias_decay: float = 3.0,     # 衰减速率（exp(-decay * t/T)）
        delta_clip: float = 10.0,    # δ 截断幅度（防止异常δ污染GAE）
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
        self.ema_tau = ema_tau
        self.bias_coef_max = bias_coef_max
        self.bias_coef_min = bias_coef_min
        self.bias_decay = bias_decay
        self.delta_clip = delta_clip
        self.device = torch.device(device)
        self._total_timesteps = 1  # 训练时更新

        obs_dim = env.observation_space.shape[0]
        if isinstance(env.action_space, gym.spaces.Discrete):
            action_dim = env.action_space.n
            self.continuous = False
        else:
            action_dim = env.action_space.shape[0]
            self.continuous = True

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # 在线 Critic（正常训练）
        self.actor  = ActorNetwork(obs_dim, action_dim, hidden_dim, self.continuous).to(self.device)
        self.critic = CriticNetwork(obs_dim, hidden_dim).to(self.device)

        # EMA Critic（慢速跟随，作为保守 target）
        self.critic_ema = CriticNetwork(obs_dim, hidden_dim).to(self.device)
        self.critic_ema.load_state_dict(self.critic.state_dict())
        for p in self.critic_ema.parameters():
            p.requires_grad_(False)   # EMA 不参与梯度计算

        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = RolloutBuffer(n_steps, obs_dim, action_dim, self.device, self.continuous)
        # 额外存储 EMA value（用于 GAE 计算的 safe bootstrap）
        self.ema_values_buffer = np.zeros(n_steps, dtype=np.float32)

        self.logger = MetricLogger(self.NAME, save_dir)
        self.total_steps = 0

        # 记录 bias 统计
        self._bias_history = []

    def _update_ema(self):
        """Polyak soft-update EMA Critic"""
        tau = self.ema_tau
        for p_online, p_ema in zip(self.critic.parameters(), self.critic_ema.parameters()):
            p_ema.data.mul_(1 - tau).add_(tau * p_online.data)

    def collect_rollout(self) -> tuple:
        """收集 rollout，同时记录在线 Critic 和 EMA Critic 的 V 估计"""
        self.buffer.reset()
        obs, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0

        for step in range(self.n_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, log_prob = self.actor.get_action_and_logprob(obs_tensor)
                v_online = self.critic(obs_tensor).item()
                v_ema    = self.critic_ema(obs_tensor).item()

            action_np = action.squeeze(0).cpu().numpy()
            log_prob_np = log_prob.item()

            if self.continuous:
                next_obs, reward, terminated, truncated, _ = self.env.step(action_np)
            else:
                next_obs, reward, terminated, truncated, _ = self.env.step(int(action_np))

            episode_reward += reward
            episode_length += 1

            self.buffer.add(obs, action_np, reward, float(terminated), log_prob_np, v_online)
            self.ema_values_buffer[step] = v_ema
            done = terminated or truncated
            obs = next_obs
            self.total_steps += 1

            if done:
                self.logger.log_episode(episode_reward, episode_length)
                obs, _ = self.env.reset()
                episode_reward = 0.0
                episode_length = 0

        with torch.no_grad():
            last_obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            last_v_online = self.critic(last_obs_t).item()
            last_v_ema    = self.critic_ema(last_obs_t).item()

        return last_v_online, last_v_ema

    def _current_bias_coef(self) -> float:
        """
        自适应 bias_coef 调度：
        β(t) = β_min + (β_max - β_min) * exp(-decay * t / T)
        训练早期保守（β≈β_max），后期精确（β→β_min）
        """
        progress = min(self.total_steps / max(self._total_timesteps, 1), 1.0)
        return self.bias_coef_min + (self.bias_coef_max - self.bias_coef_min) * np.exp(
            -self.bias_decay * progress
        )

    def compute_gae(self, last_v_online: float, last_v_ema: float) -> dict:
        """
        保守 Bootstrap GAE v2
        β(t) = β_min + (β_max - β_min) * exp(-decay * t/T)  [自适应衰减]
        V_safe(s') = V_online(s') - β(t) * |V_online(s') - V_ema(s')|
        δ_t = clip(r_t + γ * V_safe(s') - V_online(s), -δ_clip, δ_clip)
        """
        T = self.buffer.pos
        buf = self.buffer
        beta = self._current_bias_coef()

        # safe next values（保守化的 bootstrap）
        safe_nv = np.empty(T, dtype=np.float32)
        for t in range(T):
            if buf.terminated[t] > 0.5:
                safe_nv[t] = 0.0
            elif t == T - 1:
                safe_nv[t] = last_v_online - beta * abs(last_v_online - last_v_ema)
            else:
                v_next_online = buf.values[t + 1]
                v_next_ema    = self.ema_values_buffer[t + 1]
                safe_nv[t] = v_next_online - beta * abs(v_next_online - v_next_ema)

        # GAE with conservative deltas + delta_clip
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            delta    = buf.rewards[t] + self.gamma * safe_nv[t] - buf.values[t]
            # ★ v2: 裁断异常δ，防止rollout内少数极端值污染整个GAE展开
            delta    = np.clip(delta, -self.delta_clip, self.delta_clip)
            not_done = 1.0 - buf.terminated[t]
            gae      = delta + self.gamma * self.lam * not_done * gae
            adv[t]   = gae

        # 记录 bias 统计
        bias_arr = np.abs(self.buffer.values[:T] - self.ema_values_buffer[:T])
        self._bias_history.append(float(bias_arr.mean()))

        buf.advantages = adv
        buf.returns    = adv + buf.values[:T]

        # 返回中间统计（用原始δ报告，供监控使用）
        nv_online = np.empty(T, dtype=np.float32)
        for t in range(T):
            if buf.terminated[t] > 0.5:
                nv_online[t] = 0.0
            elif t == T - 1:
                nv_online[t] = last_v_online
            else:
                nv_online[t] = buf.values[t + 1]
        raw_deltas = buf.rewards[:T] + self.gamma * nv_online - buf.values[:T]
        autocorr   = float(np.corrcoef(raw_deltas[:-1], raw_deltas[1:])[0, 1]) if T > 2 else 0.0
        return {
            "delta_mean"    : float(raw_deltas.mean()),
            "delta_std"     : float(raw_deltas.std()),
            "delta_autocorr": autocorr,
            "mean_ema_bias" : float(bias_arr.mean()),
            "current_beta"  : beta,
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

                batch_obs        = obs[batch_idx]
                batch_actions    = actions[batch_idx]
                batch_old_lp     = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns    = returns[batch_idx]
                batch_old_values = old_values[batch_idx]

                new_log_probs, entropy = self.actor.evaluate_actions(batch_obs, batch_actions)
                new_values = self.critic(batch_obs)

                ratio = torch.exp(new_log_probs - batch_old_lp)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
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

                # 每 batch 更新一次 EMA（相当于更慢的 target network 追踪）
                self._update_ema()

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

        # 记录 bias 信息
        if self._bias_history:
            metrics["mean_bias"] = float(np.mean(self._bias_history[-5:]))

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
        self._total_timesteps = total_timesteps  # 用于自适应 β 调度

        while self.total_steps < total_timesteps:
            last_v_online, last_v_ema = self.collect_rollout()
            gae_stats = self.compute_gae(last_v_online, last_v_ema)
            metrics = self.update()
            update_idx += 1
            metrics.update(gae_stats)   # 合并 GAE 统计

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
                elapsed = time.time() - train_start
                recent = self.logger.get_recent_reward(20)
                progress = self._progress_bar(self.total_steps, total_timesteps)
                fps = int(self.total_steps / (elapsed + 1e-8))
                eval_str = f"{last_eval_reward:7.1f}" if not np.isnan(last_eval_reward) else "    N/A"
                delta_str = (
                    f"δ:μ={metrics.get('delta_mean',0):+.3f}"
                    f"/σ={metrics.get('delta_std',0):.3f}"
                    f"/r1={metrics.get('delta_autocorr',0):+.2f}"
                )
                ema_str = (
                    f"EMA_bias={metrics.get('mean_ema_bias', 0):.4f}"
                    f" β={metrics.get('current_beta', 0):.3f}"
                )
                print(
                    f"  [{self.NAME:<30}] "
                    f"{self.total_steps:7d}/{total_timesteps} "
                    f"{progress} "
                    f"| Eval={eval_str} Recent={recent:6.1f} "
                    f"| VLoss={metrics['value_loss']:.3f} EV={metrics['explained_variance']:+.2f} "
                    f"| KL={metrics['approx_kl']:.4f} clip={metrics['clip_frac']:.2f}"
                    f"| {delta_str} | {ema_str} | {fps:5d}fps {elapsed:6.0f}s"
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

