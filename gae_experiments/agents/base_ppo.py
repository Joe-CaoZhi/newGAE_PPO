"""
基线 PPO + 标准 GAE
"""
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from ..utils.logger import MetricLogger
from ..utils.networks import ActorNetwork, CriticNetwork
from ..utils.rollout_buffer import RolloutBuffer


class BasePPO:
    """
    标准 PPO + GAE（固定 λ）
    作为所有改进方案的基线。
    """

    NAME = "Standard_GAE"

    def __init__(
        self,
        env: gym.Env,
        # 网络结构
        hidden_dim: int = 64,
        # PPO 超参数
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        lam: float = 0.95,
        eps_clip: float = 0.2,
        n_epochs: int = 10,
        batch_size: int = 64,
        # Rollout
        n_steps: int = 2048,
        # 熵正则化
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        # 设备
        device: str = "cpu",
        # 日志
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
        self.device = torch.device(device)

        # 环境信息
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

        # 优化器：Actor 和 Critic 分开，可独立控制学习率
        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Rollout Buffer
        self.buffer = RolloutBuffer(n_steps, obs_dim, action_dim, self.device, self.continuous)

        # 日志
        self.logger = MetricLogger(self.NAME, save_dir)
        self.total_steps = 0

    def collect_rollout(self) -> float:
        """收集 n_steps 步的交互数据，返回最后状态的价值"""
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
            value_np = value.squeeze(0).item()
            log_prob_np = log_prob.squeeze(0).item()

            if self.continuous:
                next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            else:
                next_obs, reward, terminated, truncated, info = self.env.step(int(action_np))

            episode_reward += reward
            episode_length += 1

            # ★ 存 terminated（非 done），truncated 时下一状态仍有价值
            self.buffer.add(obs, action_np, reward, float(terminated), log_prob_np, value_np)
            done = terminated or truncated
            obs = next_obs
            self.total_steps += 1

            if done:
                self.logger.log_episode(episode_reward, episode_length)
                obs, _ = self.env.reset()
                episode_reward = 0.0
                episode_length = 0

        # 最后一个状态的价值（用于 GAE 边界条件）
        with torch.no_grad():
            last_obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            last_value = self.critic(last_obs_tensor).item()

        return last_value

    def compute_gae(self, last_value: float) -> dict:
        """计算标准 GAE 优势函数，返回中间统计信息"""
        self.buffer.compute_standard_gae(last_value, self.gamma, self.lam)
        T  = self.buffer.pos
        nv = self.buffer._next_values(last_value)
        deltas = self.buffer.rewards[:T] + self.gamma * nv - self.buffer.values[:T]
        # δ 一阶自相关（衡量 Critic 残差时序结构）
        autocorr = float(np.corrcoef(deltas[:-1], deltas[1:])[0, 1]) if T > 2 else 0.0
        return {
            "delta_mean": float(deltas.mean()),
            "delta_std" : float(deltas.std()),
            "delta_autocorr": autocorr,
            "adv_mean" : float(self.buffer.advantages.mean()),
            "adv_std"  : float(self.buffer.advantages.std()),
        }

    def update(self) -> dict:
        """PPO 更新步骤，返回训练指标"""
        obs, actions, old_log_probs, advantages, returns, old_values = self.buffer.get_batch()

        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        T = self.buffer.pos
        indices = np.arange(T)
        metrics = {
            "value_loss": 0.0,
            "policy_loss": 0.0,
            "entropy_loss": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
        }
        update_count = 0

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

                # 计算新的 log prob 和熵
                new_log_probs, entropy = self.actor.evaluate_actions(batch_obs, batch_actions)
                new_values = self.critic(batch_obs)

                # Policy loss（PPO clip）
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss（clip）
                v_loss_unclipped = (new_values - batch_returns) ** 2
                v_clipped = batch_old_values + torch.clamp(
                    new_values - batch_old_values, -self.eps_clip, self.eps_clip
                )
                v_loss_clipped = (v_clipped - batch_returns) ** 2
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

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

                # 记录指标
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    clip_frac = ((ratio - 1).abs() > self.eps_clip).float().mean().item()

                metrics["value_loss"] += value_loss.item()
                metrics["policy_loss"] += policy_loss.item()
                metrics["entropy_loss"] += entropy_loss.item()
                metrics["approx_kl"] += approx_kl
                metrics["clip_frac"] += clip_frac
                update_count += 1

        # 平均
        if update_count > 0:
            for k in metrics:
                metrics[k] /= update_count

        # 计算 explained variance
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
        pct = 100 * current / total
        return f"[{bar}] {pct:5.1f}%"

    def train(
        self,
        total_timesteps: int,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 5000,
        n_eval_episodes: int = 5,
        verbose: bool = True,
    ):
        """主训练循环"""
        import time
        train_start = time.time()
        last_eval_reward = float("nan")
        update_idx = 0

        if verbose:
            print(f"\n  {'─'*62}")
            print(f"  {'方法':<26} {'步数':>8}  {'进度':>32}")
            print(f"  {'─'*62}")

        while self.total_steps < total_timesteps:
            # 收集数据
            last_value = self.collect_rollout()

            # 计算 GAE
            gae_stats = self.compute_gae(last_value)

            # 更新网络
            metrics = self.update()
            update_idx += 1
            metrics.update(gae_stats)    # 合并 GAE 统计信息

            # 记录
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

            # 评估
            do_eval = eval_env is not None and self.total_steps % eval_freq < self.n_steps
            if do_eval:
                last_eval_reward = self.evaluate(eval_env, n_eval_episodes)
                self.logger.log_eval(last_eval_reward, self.total_steps)

            # 打印中间状态（丰富版：包含数学量分析）
            if verbose and (do_eval or update_idx % 5 == 1):
                elapsed = time.time() - train_start
                recent = self.logger.get_recent_reward(20)
                progress = self._progress_bar(self.total_steps, total_timesteps)
                fps = int(self.total_steps / (elapsed + 1e-8))
                eval_str = f"{last_eval_reward:7.1f}" if not np.isnan(last_eval_reward) else "    N/A"
                # 构建额外统计信息字符串
                extra_parts = []
                if "mean_lambda" in metrics and metrics["mean_lambda"] is not None:
                    extra_parts.append(f"λ={metrics['mean_lambda']:.3f}")
                if "mean_bias" in metrics:
                    extra_parts.append(f"bias={metrics['mean_bias']:.3f}")
                if "delta_mean" in metrics:
                    extra_parts.append(
                        f"δ:μ={metrics['delta_mean']:+.3f}"
                        f"/σ={metrics['delta_std']:.3f}"
                        f"/r1={metrics['delta_autocorr']:+.2f}"
                    )
                if "mean_confidence" in metrics:
                    extra_parts.append(f"c={metrics['mean_confidence']:.3f}")
                extra = " | " + " | ".join(extra_parts) if extra_parts else ""
                print(
                    f"  [{self.NAME:<24}] "
                    f"{self.total_steps:7d}/{total_timesteps} "
                    f"{progress} "
                    f"| Eval={eval_str} Recent={recent:6.1f} "
                    f"| VLoss={metrics['value_loss']:.3f} EV={metrics['explained_variance']:+.2f} "
                    f"| KL={metrics['approx_kl']:.4f} clip={metrics['clip_frac']:.2f}"
                    f"{extra} | {fps:5d}fps {elapsed:6.0f}s"
                )

        # 最终汇总
        if verbose:
            elapsed = time.time() - train_start
            final_r = np.mean(self.logger.eval_rewards[-5:]) if self.logger.eval_rewards else 0.0
            best_r  = max(self.logger.eval_rewards) if self.logger.eval_rewards else 0.0
            print(f"\n  ✓ [{self.NAME}] 完成 | 总耗时={elapsed:.1f}s "
                  f"| 最终评估={final_r:.1f} | 最高评估={best_r:.1f}")

        # 保存日志
        self.logger.save()
        return self.logger

    def evaluate(self, eval_env: gym.Env, n_episodes: int = 5) -> float:
        """评估当前策略"""
        total_reward = 0.0
        for _ in range(n_episodes):
            obs, _ = eval_env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    dist = self.actor(obs_tensor)
                    if self.continuous:
                        action = dist.mean
                    else:
                        action = dist.probs.argmax(dim=-1)
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

