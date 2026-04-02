"""
改进二：自适应 λ GAE（State-Dependent λ）
λ_t = f_φ(s_t) ∈ [0,1]
高不确定性 → λ → 0（更信任 TD，少展开）
低不确定性 → λ → 1（更信任 MC，多展开）

训练信号：使用价值函数预测误差作为不确定性代理
λ_t = σ(-β * |V(s_t) - G_t|) 的近似
端到端：将 λ 网络与策略/价值网络一起训练
"""
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from ..utils.logger import MetricLogger
from ..utils.networks import ActorNetwork, CriticNetwork, LambdaNetwork
from ..utils.rollout_buffer import RolloutBuffer


class AdaptiveLambdaPPO:
    """
    PPO + 自适应 λ GAE
    λ 网络根据状态动态输出 λ_t，使 GAE 展开长度随不确定性自适应。
    """

    NAME = "Adaptive_Lambda_GAE"

    def __init__(
        self,
        env: gym.Env,
        hidden_dim: int = 64,
        lambda_hidden_dim: int = 32,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        lr_lambda: float = 1e-3,
        gamma: float = 0.99,
        lam_init: float = 0.95,      # λ 网络的初始化偏置（使其初始输出接近 lam_init）
        eps_clip: float = 0.2,
        n_epochs: int = 10,
        batch_size: int = 64,
        n_steps: int = 2048,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        lambda_coef: float = 0.1,    # λ 网络的辅助损失系数
        max_grad_norm: float = 0.5,
        device: str = "cpu",
        save_dir: str = "results",
    ):
        self.env = env
        self.gamma = gamma
        self.lam_init = lam_init
        self.eps_clip = eps_clip
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.lambda_coef = lambda_coef
        self.max_grad_norm = max_grad_norm
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

        # 网络
        self.actor = ActorNetwork(obs_dim, action_dim, hidden_dim, self.continuous).to(self.device)
        self.critic = CriticNetwork(obs_dim, hidden_dim).to(self.device)
        self.lambda_net = LambdaNetwork(obs_dim, lambda_hidden_dim).to(self.device)

        # 优化器：Actor/Critic 分开，可独立控制学习率
        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),    lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),   lr=lr_critic)
        self.lambda_optimizer = torch.optim.Adam(self.lambda_net.parameters(), lr=lr_lambda)

        self.buffer = RolloutBuffer(n_steps, obs_dim, action_dim, self.device, self.continuous)

        self.logger = MetricLogger(self.NAME, save_dir)
        self.total_steps = 0

        # 存储 rollout 中每个状态的 λ 值
        self.lambda_buffer = np.zeros(n_steps, dtype=np.float32)

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
                lam_val = self.lambda_net(obs_tensor)

            action_np = action.squeeze(0).cpu().numpy()
            value_np = value.item()
            log_prob_np = log_prob.item()
            lam_np = lam_val.item()

            if self.continuous:
                next_obs, reward, terminated, truncated, _ = self.env.step(action_np)
            else:
                next_obs, reward, terminated, truncated, _ = self.env.step(int(action_np))

            episode_reward += reward
            episode_length += 1

            # ★ 存 terminated 而非 done，truncated 时 next state 仍有价值
            self.buffer.add(obs, action_np, reward, float(terminated), log_prob_np, value_np)
            self.lambda_buffer[step] = lam_np
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
        """使用状态相关的 λ_t 计算自适应 GAE，返回中间统计"""
        T = self.buffer.pos
        self.buffer.compute_adaptive_lambda_gae(
            last_value=last_value,
            gamma=self.gamma,
            lambda_values=self.lambda_buffer[:T],
        )
        # 计算 delta 统计
        nv = self.buffer._next_values(last_value)
        deltas = self.buffer.rewards[:T] + self.gamma * nv - self.buffer.values[:T]
        autocorr = float(np.corrcoef(deltas[:-1], deltas[1:])[0, 1]) if T > 2 else 0.0
        lambdas  = self.lambda_buffer[:T]
        return {
            "delta_mean"    : float(deltas.mean()),
            "delta_std"     : float(deltas.std()),
            "delta_autocorr": autocorr,
            "lambda_rollout_mean": float(lambdas.mean()),
            "lambda_rollout_std" : float(lambdas.std()),
        }

    def update_lambda_network(self, obs: torch.Tensor, mc_returns: torch.Tensor):
        """
        更新 λ 网络（重新设计的训练信号）。

        核心思想：
          GAE 中 λ 控制偏差-方差权衡：
            - λ=1: 完全展开 → 低偏差、高方差（类 MC）
            - λ=0: 单步 TD  → 高偏差、低方差

          对于 PPO，Critic 已经在学习，大多数情况 λ 应该偏高（接近0.9）。
          只有当 Critic 预测非常不准时，才需要减小 λ。

        改进设计：
          1. 将误差归一化到 [0,1]: norm_err = |V-G| / (|G|.mean() + 1e-8)
             相对误差比绝对误差更稳定
          2. target_lambda = lam_max - (lam_max - lam_min) * norm_err_clipped
             线性映射，范围在 [lam_min, lam_max] = [0.7, 0.99]
             - norm_err=0（Critic完全准确）→ λ → 0.99
             - norm_err=1（误差等于return均值）→ λ → 0.70
             这保证 λ 始终在合理范围内，不会崩到 0
          3. 分化正则：轻度鼓励 λ 多样性，但系数降低到 0.01
        """
        lam_max = 0.99
        lam_min = 0.70

        with torch.no_grad():
            current_v = self.critic(obs)
            errors    = (current_v - mc_returns).abs()
            # 用 returns 的绝对均值做相对归一化（与数值尺度无关）
            returns_scale = mc_returns.abs().mean() + 1e-8
            norm_err  = torch.clamp(errors / returns_scale, 0.0, 1.0)
            # 线性映射到 [lam_min, lam_max]
            target_lambda = lam_max - (lam_max - lam_min) * norm_err

        lambda_preds = self.lambda_net(obs)
        mse_loss     = nn.MSELoss()(lambda_preds, target_lambda)

        # 轻度分化正则（系数从 0.05 降到 0.01，避免过度影响主目标）
        diversity_reg = -0.01 * lambda_preds.var()
        lambda_loss   = mse_loss + diversity_reg

        self.lambda_optimizer.zero_grad()
        lambda_loss.backward()
        nn.utils.clip_grad_norm_(self.lambda_net.parameters(), self.max_grad_norm)
        self.lambda_optimizer.step()

        return mse_loss.item(), lambda_preds.mean().item()

    def update(self) -> dict:
        obs, actions, old_log_probs, advantages, returns, old_values = self.buffer.get_batch()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        T = self.buffer.pos
        indices = np.arange(T)
        metrics = {"value_loss": 0.0, "policy_loss": 0.0, "entropy_loss": 0.0,
                   "approx_kl": 0.0, "clip_frac": 0.0, "lambda_loss": 0.0}
        update_count = 0
        mean_lambda_list = []

        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, self.batch_size):
                end = start + self.batch_size
                if end > T:
                    break
                batch_idx = indices[start:end]

                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                batch_old_values = old_values[batch_idx]

                new_log_probs, entropy = self.actor.evaluate_actions(batch_obs, batch_actions)
                new_values = self.critic(batch_obs)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                v_loss_unclipped = (new_values - batch_returns) ** 2
                v_clipped = batch_old_values + torch.clamp(new_values - batch_old_values, -self.eps_clip, self.eps_clip)
                v_loss_clipped = (v_clipped - batch_returns) ** 2
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = -entropy.mean()

                # Actor 更新
                self.actor_optimizer.zero_grad()
                (policy_loss + self.ent_coef * entropy_loss).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Critic 更新
                self.critic_optimizer.zero_grad()
                (self.vf_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                # ★ λ 网络与 Critic 同步更新（每 batch 更新一次，避免欠训练）
                # 用最新 Critic 估计误差作为不确定性信号
                lam_loss_val, lam_mean_val = self.update_lambda_network(batch_obs, batch_returns)
                metrics["lambda_loss"] += lam_loss_val
                mean_lambda_list.append(lam_mean_val)

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    clip_frac = ((ratio - 1).abs() > self.eps_clip).float().mean().item()

                metrics["value_loss"] += value_loss.item()
                metrics["policy_loss"] += policy_loss.item()
                metrics["entropy_loss"] += entropy_loss.item()
                metrics["approx_kl"] += approx_kl
                metrics["clip_frac"] += clip_frac
                update_count += 1

        if update_count > 0:
            for k in ["value_loss", "policy_loss", "entropy_loss", "approx_kl", "clip_frac"]:
                metrics[k] /= update_count

        with torch.no_grad():
            y_pred = old_values.cpu().numpy()
            y_true = returns.cpu().numpy()
            var_y = np.var(y_true)
            ev = 1 - np.var(y_true - y_pred) / (var_y + 1e-8)
            metrics["explained_variance"] = float(ev)

        metrics["mean_lambda"] = float(np.mean(mean_lambda_list))
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
            gae_stats = self.compute_gae(last_value)
            metrics = self.update()
            update_idx += 1
            metrics.update(gae_stats)    # 合并 GAE 中间统计

            self.logger.log_update(
                value_loss=metrics["value_loss"],
                policy_loss=metrics["policy_loss"],
                entropy_loss=metrics["entropy_loss"],
                approx_kl=metrics["approx_kl"],
                clip_frac=metrics["clip_frac"],
                explained_variance=metrics["explained_variance"],
                total_steps=self.total_steps,
                mean_lambda=metrics.get("mean_lambda"),
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
                lam_mean = metrics.get('mean_lambda', 0)
                lam_rm   = metrics.get('lambda_rollout_mean', 0)
                lam_rs   = metrics.get('lambda_rollout_std', 0)
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
                    f"| KL={metrics['approx_kl']:.4f} clip={metrics['clip_frac']:.2f} "
                    f"| λ_update={lam_mean:.3f} λ_rollout={lam_rm:.3f}±{lam_rs:.3f}"
                    f"| {delta_str} | {fps:5d}fps {elapsed:6.0f}s"
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

