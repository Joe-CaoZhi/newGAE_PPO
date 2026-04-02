"""
改进三：置信度加权 GAE（Confidence-Weighted GAE）
不同时刻的 TD 残差可信度不同，按可信度加权归一化，而不是只按时间衰减。

c_t = exp(-α * Var(δ_t)) = 1 / (1 + running_var(δ_t))
A_t = Σ (γλ)^l * c_{t+l} * δ_{t+l} / Σ (γλ)^l * c_{t+l}
"""
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from ..utils.logger import MetricLogger
from ..utils.networks import ActorNetwork, CriticNetwork
from ..utils.rollout_buffer import RolloutBuffer


class ConfidenceWeightedPPO:
    """
    PPO + 置信度加权 GAE
    基于 TD 残差历史方差来估计每个状态的置信度。
    方差大 → 置信度低；方差小 → 置信度高。
    """

    NAME = "Confidence_Weighted_GAE"

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
        # 置信度相关
        conf_alpha: float = 1.0,         # 置信度敏感度（控制 delta 影响程度）
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
        self.conf_alpha = conf_alpha
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

        self.actor = ActorNetwork(obs_dim, action_dim, hidden_dim, self.continuous).to(self.device)
        self.critic = CriticNetwork(obs_dim, hidden_dim).to(self.device)

        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = RolloutBuffer(n_steps, obs_dim, action_dim, self.device, self.continuous)

        # 滑动窗口方差（最近 K 个 rollout 的 δ 统计）
        # 用固定大小的滑动窗口，方差稳定在最近分布上
        self._delta_window_size = 2048 * 4   # 保留最近 4 个 rollout 的 delta
        self._delta_window     = np.zeros(self._delta_window_size, dtype=np.float32)
        self._delta_win_ptr    = 0
        self._delta_win_n      = 0          # 已填充数量
        # EMA 平滑置信度，防止单步置信度突变引发训练不稳定
        self._ema_confidence   = None       # shape (n_steps,)，懒初始化
        self._conf_ema_alpha   = 0.3        # EMA 权重（0=完全历史，1=完全当前）

        self.logger = MetricLogger(self.NAME, save_dir)
        self.total_steps = 0

    def _update_delta_window(self, deltas: np.ndarray):
        """滑动窗口更新：用环形缓冲区存最近 K 个 delta"""
        n = len(deltas)
        for i in range(n):
            self._delta_window[self._delta_win_ptr] = deltas[i]
            self._delta_win_ptr = (self._delta_win_ptr + 1) % self._delta_window_size
            self._delta_win_n   = min(self._delta_win_n + 1, self._delta_window_size)

    def compute_confidence(self, deltas: np.ndarray) -> np.ndarray:
        """
        置信度 c_t（稳健版：分位数归一化 + EMA 平滑）

        设计目标：
          - 高δ²（Critic预测差）→ 低置信度（不信任这个TD残差）
          - 低δ²（Critic预测准）→ 高置信度（信任这个TD残差）
          - 权重变化不能太剧烈，防止训练崩溃

        方案：
          1. 用滑动窗口的中位数绝对偏差（MAD）替代方差，更鲁棒
             MAD 对异常值不敏感，不会因少数大δ导致方差膨胀
          2. 原始 c_t = 1 / (1 + alpha * |delta| / (MAD + 1e-8))
             改用 |delta| 而非 delta²，线性衰减更稳定
          3. 权重归一化：除以均值，使平均权重=1（量纲保持）
          4. EMA 平滑：c_ema = (1-α) * c_prev + α * c_new
             平滑系数 α=0.3，减缓权重突变引起的训练不稳定
        """
        # 更新滑动窗口
        self._update_delta_window(deltas)

        # 用 MAD（中位数绝对偏差）估计规模，对异常值鲁棒
        valid = self._delta_window[:self._delta_win_n]
        if len(valid) > 1:
            median_abs = float(np.median(np.abs(valid)))
            window_scale = max(median_abs, 1e-8) * 1.4826   # 1.4826 使MAD与σ相当
        else:
            window_scale = 1.0

        # 原始置信度：基于 |delta| 的线性衰减，比 delta² 更稳定
        raw_c = 1.0 / (1.0 + self.conf_alpha * np.abs(deltas) / window_scale)

        # 归一化：除以均值，使均值=1（量纲保持与标准GAE一致）
        c_mean = raw_c.mean()
        if c_mean > 1e-8:
            raw_c = raw_c / c_mean
        raw_c = np.clip(raw_c, 0.2, 5.0)   # 限制范围，防止极端权重

        # EMA 平滑：减缓权重突变
        T = len(deltas)
        if self._ema_confidence is None or len(self._ema_confidence) != T:
            # 初始化或长度变化时重置
            self._ema_confidence = raw_c.copy()
        else:
            alpha = self._conf_ema_alpha
            self._ema_confidence = (1 - alpha) * self._ema_confidence + alpha * raw_c
            # 再次归一化确保均值=1
            ema_mean = self._ema_confidence.mean()
            if ema_mean > 1e-8:
                self._ema_confidence = self._ema_confidence / ema_mean
            self._ema_confidence = np.clip(self._ema_confidence, 0.2, 5.0)

        return self._ema_confidence.astype(np.float32)

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

            # ★ 存 terminated而非 done
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
        """计算置信度加权 GAE，返回中间统计"""
        T  = self.buffer.pos
        nv = self.buffer._next_values(last_value)

        deltas = (self.buffer.rewards[:T]
                  + self.gamma * nv
                  - self.buffer.values[:T])

        # 计算置信度（内部用滑动窗口方差 + Softmax 归一化）
        confidence = self.compute_confidence(deltas)

        self.buffer.compute_confidence_weighted_gae(
            last_value=last_value,
            gamma=self.gamma,
            lam=self.lam,
            confidence=confidence,
        )

        autocorr = float(np.corrcoef(deltas[:-1], deltas[1:])[0, 1]) if T > 2 else 0.0
        return {
            "delta_mean"     : float(deltas.mean()),
            "delta_std"      : float(deltas.std()),
            "delta_autocorr" : autocorr,
            "mean_confidence": float(confidence.mean()),
            "min_confidence" : float(confidence.min()),
            "max_confidence" : float(confidence.max()),
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

                metrics["value_loss"] += value_loss.item()
                metrics["policy_loss"] += policy_loss.item()
                metrics["entropy_loss"] += entropy_loss.item()
                metrics["approx_kl"] += approx_kl
                metrics["clip_frac"] += clip_frac
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
            gae_stats = self.compute_gae(last_value)
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
                conf_str = (
                    f"c: μ={metrics.get('mean_confidence',0):.3f}"
                    f" [̲{metrics.get('min_confidence',0):.2f}"
                    f",{metrics.get('max_confidence',0):.2f}]"
                )
                print(
                    f"  [{self.NAME:<24}] "
                    f"{self.total_steps:7d}/{total_timesteps} "
                    f"{progress} "
                    f"| Eval={eval_str} Recent={recent:6.1f} "
                    f"| VLoss={metrics['value_loss']:.3f} EV={metrics['explained_variance']:+.2f} "
                    f"| KL={metrics['approx_kl']:.4f} clip={metrics['clip_frac']:.2f}"
                    f"| {delta_str} | {conf_str} | {fps:5d}fps {elapsed:6.0f}s"
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

