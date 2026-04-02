"""
改进四：组合改进 GAE（简化版）
= 自适应 λ + 置信度归一化

核心原则：
  组合改进应当"1+1>2"，而非引入更多超参数。
  旧版 Combined 引入双 Critic，导致：
    1. λ 网络收到的训练信号与实际 δ 不一致（double-critic δ vs 单Critic target）
    2. 两个 Critic 竞争同一个 returns target，优化混乱
    3. 训练速度降低，计算量翻倍

新版设计：
  1. 单 Critic（与基线保持一致）
  2. 自适应 λ: λ_t = f_φ(s_t) ∈ [lam_min, lam_max]
     基于 Critic 预测误差的相对大小调节，λ 范围 [0.7, 0.99]
  3. 置信度归一化: 基于 MAD + EMA 平滑，稳定权重分配
  4. 组合优势:
     A_t = [Σ_l (γλ_t)^l c_{t+l} δ_{t+l}] / [Σ_l (γλ_t)^l c_{t+l}]
     returns = 标准 GAE returns（独立计算，避免归一化污染Critic目标）
"""
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from ..utils.logger import MetricLogger
from ..utils.networks import ActorNetwork, CriticNetwork, LambdaNetwork
from ..utils.rollout_buffer import RolloutBuffer


class CombinedPPO:
    """
    PPO + 组合改进 GAE（自适应λ + 置信度加权）
    单 Critic 设计，减少复杂度，提高训练一致性。
    """

    NAME = "Combined_GAE"

    def __init__(
        self,
        env: gym.Env,
        hidden_dim: int = 64,
        lambda_hidden_dim: int = 32,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        lr_lambda: float = 1e-3,
        gamma: float = 0.99,
        lam: float = 0.95,       # 置信度加权用的基础 λ（用于 returns 计算）
        eps_clip: float = 0.2,
        n_epochs: int = 10,
        batch_size: int = 64,
        n_steps: int = 2048,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        # 置信度相关
        conf_alpha: float = 1.0,
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

        # 单 Actor + 单 Critic（简化，避免双Critic的优化混乱）
        self.actor  = ActorNetwork(obs_dim, action_dim, hidden_dim, self.continuous).to(self.device)
        self.critic = CriticNetwork(obs_dim, hidden_dim).to(self.device)

        # 自适应 λ 网络
        self.lambda_net = LambdaNetwork(obs_dim, lambda_hidden_dim).to(self.device)

        # 优化器
        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),       lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),      lr=lr_critic)
        self.lambda_optimizer = torch.optim.Adam(self.lambda_net.parameters(),  lr=lr_lambda)

        # Rollout Buffer
        self.buffer = RolloutBuffer(n_steps, obs_dim, action_dim, self.device, self.continuous)
        self.lambda_buffer = np.zeros(n_steps, dtype=np.float32)

        # 滑动窗口方差（MAD 计算用）
        self._delta_window_size = 2048 * 4
        self._delta_window     = np.zeros(self._delta_window_size, dtype=np.float32)
        self._delta_win_ptr    = 0
        self._delta_win_n      = 0

        # EMA 平滑置信度
        self._ema_confidence  = None
        self._conf_ema_alpha  = 0.3

        self.logger = MetricLogger(self.NAME, save_dir)
        self.total_steps = 0

    def _update_delta_window(self, deltas: np.ndarray):
        """滑动窗口更新"""
        for x in deltas:
            self._delta_window[self._delta_win_ptr] = x
            self._delta_win_ptr = (self._delta_win_ptr + 1) % self._delta_window_size
            self._delta_win_n   = min(self._delta_win_n + 1, self._delta_window_size)

    def compute_confidence(self, deltas: np.ndarray) -> np.ndarray:
        """
        置信度（MAD 尺度估计 + EMA 平滑，与 Confidence_Weighted_GAE 保持一致）
        c_t = 1 / (1 + alpha * |delta_t| / MAD_scale)
        归一化后 EMA 平滑
        """
        self._update_delta_window(deltas)

        valid = self._delta_window[:self._delta_win_n]
        if len(valid) > 1:
            median_abs = float(np.median(np.abs(valid)))
            window_scale = max(median_abs, 1e-8) * 1.4826
        else:
            window_scale = 1.0

        raw_c = 1.0 / (1.0 + self.conf_alpha * np.abs(deltas) / window_scale)

        c_mean = raw_c.mean()
        if c_mean > 1e-8:
            raw_c = raw_c / c_mean
        raw_c = np.clip(raw_c, 0.2, 5.0)

        T = len(deltas)
        if self._ema_confidence is None or len(self._ema_confidence) != T:
            self._ema_confidence = raw_c.copy()
        else:
            alpha = self._conf_ema_alpha
            self._ema_confidence = (1 - alpha) * self._ema_confidence + alpha * raw_c
            ema_mean = self._ema_confidence.mean()
            if ema_mean > 1e-8:
                self._ema_confidence = self._ema_confidence / ema_mean
            self._ema_confidence = np.clip(self._ema_confidence, 0.2, 5.0)

        return self._ema_confidence.astype(np.float32)

    def collect_rollout(self):
        self.buffer.reset()
        obs, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0

        for step in range(self.n_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, log_prob = self.actor.get_action_and_logprob(obs_tensor)
                v = self.critic(obs_tensor)
                lam_val = self.lambda_net(obs_tensor)

            action_np = action.squeeze(0).cpu().numpy()
            v_np = v.item()
            log_prob_np = log_prob.item()
            lam_np = lam_val.item()

            if self.continuous:
                next_obs, reward, terminated, truncated, _ = self.env.step(action_np)
            else:
                next_obs, reward, terminated, truncated, _ = self.env.step(int(action_np))

            episode_reward += reward
            episode_length += 1

            self.buffer.add(obs, action_np, reward, float(terminated), log_prob_np, v_np)
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
        """
        组合 GAE（单Critic）：
        δ_t = r_t + γ V(s') - V(s)（单Critic，干净一致）
        A_t = [Σ_l (γλ_t)^l c_{t+l} δ_{t+l}] / [Σ_l (γλ_t)^l c_{t+l}]
        returns = 标准 GAE returns（避免置信度归一化污染Critic目标）
        """
        T  = self.buffer.pos
        nv = self.buffer._next_values(last_value)
        deltas = self.buffer.rewards[:T] + self.gamma * nv - self.buffer.values[:T]

        confidence = self.compute_confidence(deltas)
        lambdas    = self.lambda_buffer[:T]

        # 计算置信度加权 + 自适应λ 的组合优势
        adv = np.zeros(T, dtype=np.float32)
        H   = min(T, 64)

        for t in range(T):
            w_sum  = 0.0
            w_tot  = 0.0
            w_step = 1.0  # 初始累积权重

            for l in range(H):
                idx = t + l
                if idx >= T:
                    break
                if l > 0 and self.buffer.terminated[idx - 1] > 0.5:
                    break

                w      = w_step * confidence[idx]
                w_sum += w * deltas[idx]
                w_tot += w
                # 用该步的自适应 λ 更新下一步累积权重
                w_step *= self.gamma * float(lambdas[idx])

            adv[t] = w_sum / w_tot if w_tot > 1e-8 else deltas[t]

        self.buffer.advantages = adv
        # ★ returns 用标准 GAE（固定 λ=self.lam），给 Critic 提供稳定目标
        self.buffer.returns = self.buffer._compute_standard_returns(last_value, self.gamma, self.lam)

        autocorr = float(np.corrcoef(deltas[:-1], deltas[1:])[0, 1]) if T > 2 else 0.0
        return {
            "delta_mean"          : float(deltas.mean()),
            "delta_std"           : float(deltas.std()),
            "delta_autocorr"      : autocorr,
            "mean_confidence"     : float(confidence.mean()),
            "lambda_rollout_mean" : float(lambdas.mean()),
            "lambda_rollout_std"  : float(lambdas.std()),
        }

    def update_lambda_network(self, obs: torch.Tensor, mc_returns: torch.Tensor):
        """
        更新 λ 网络（与 AdaptiveLambdaPPO 保持一致的训练信号）
        target_lambda = lam_max - (lam_max - lam_min) * clamp(|V-G| / |G|.mean(), 0, 1)
        λ 范围限制在 [0.70, 0.99]，避免 λ 过低导致偏差过大
        """
        lam_max = 0.99
        lam_min = 0.70

        with torch.no_grad():
            current_v     = self.critic(obs)
            errors        = (current_v - mc_returns).abs()
            returns_scale = mc_returns.abs().mean() + 1e-8
            norm_err      = torch.clamp(errors / returns_scale, 0.0, 1.0)
            target_lambda = lam_max - (lam_max - lam_min) * norm_err

        lambda_preds = self.lambda_net(obs)
        mse_loss     = nn.MSELoss()(lambda_preds, target_lambda)
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

                batch_obs         = obs[batch_idx]
                batch_actions     = actions[batch_idx]
                batch_old_lp      = old_log_probs[batch_idx]
                batch_advantages  = advantages[batch_idx]
                batch_returns     = returns[batch_idx]
                batch_old_values  = old_values[batch_idx]

                # Policy loss
                new_log_probs, entropy = self.actor.evaluate_actions(batch_obs, batch_actions)
                ratio  = torch.exp(new_log_probs - batch_old_lp)
                surr1  = ratio * batch_advantages
                surr2  = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                policy_loss  = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()

                self.actor_optimizer.zero_grad()
                (policy_loss + self.ent_coef * entropy_loss).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Value loss（clipped）
                new_values = self.critic(batch_obs)
                v_loss_unclipped = (new_values - batch_returns) ** 2
                v_clipped = batch_old_values + torch.clamp(
                    new_values - batch_old_values, -self.eps_clip, self.eps_clip
                )
                v_loss_clipped = (v_clipped - batch_returns) ** 2
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                self.critic_optimizer.zero_grad()
                (self.vf_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                # λ 网络每 batch 更新（与 AdaptiveLambdaPPO 一致）
                lam_loss_v, lam_mean_v = self.update_lambda_network(batch_obs, batch_returns)
                metrics["lambda_loss"] += lam_loss_v
                mean_lambda_list.append(lam_mean_v)

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

        metrics["mean_lambda"] = float(np.mean(mean_lambda_list)) if mean_lambda_list else 0.9
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
                mean_lambda=metrics.get("mean_lambda"),
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
                delta_str = (
                    f"δ:μ={metrics.get('delta_mean',0):+.3f}"
                    f"/σ={metrics.get('delta_std',0):.3f}"
                    f"/r1={metrics.get('delta_autocorr',0):+.2f}"
                )
                lam_rm = metrics.get('lambda_rollout_mean', 0)
                lam_rs = metrics.get('lambda_rollout_std', 0)
                conf_m = metrics.get('mean_confidence', 0)
                print(
                    f"  [{self.NAME:<24}] "
                    f"{self.total_steps:7d}/{total_timesteps} "
                    f"{progress} "
                    f"| Eval={eval_str} Recent={recent:6.1f} "
                    f"| VLoss={metrics['value_loss']:.3f} EV={metrics['explained_variance']:+.2f} "
                    f"| KL={metrics['approx_kl']:.4f} clip={metrics['clip_frac']:.2f}"
                    f"| {delta_str} | c={conf_m:.3f} λ={lam_rm:.3f}±{lam_rs:.3f}"
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

