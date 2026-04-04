"""
革新方法一：Hindsight-Corrected GAE（HCGAE）v2

改进清单（v2）：
─────────────────────────────────────────────────────────────────────
① EMA 归一化修正
   旧：alpha = sigmoid(β * err / err_ema)
       问题：err_ema 是慢速 EMA，Critic 快速收敛时 err_ema 滞后偏大
            导致 err/err_scale << 1，修正被过早关闭
   新：sigmoid 以「当前批次均值 + 批次标准差」双统计量做中心化归一化
       alpha = sigmoid(β * (err - μ_batch) / (σ_batch + ε))
       物理含义：只有「高于当前平均误差水平」的步骤才触发强修正，
       sigmoid 中心在 err = μ_batch（而非 err = err_ema）

② Critic 目标自适应混合（EV 驱动）
   旧：returns = 0.5 * G + 0.5 * gae_returns（固定混合）
   新：c_mc = clip(1 - EV, 0, 1)（EV 低→多用 MC；EV 高→多用 GAE returns）
   物理含义：用 Critic 自身精度指标驱动训练目标质量，与 α_max 自适应机制对称

③ rollout 末端 bootstrap 不一致修正
   旧：V_corrected_next[T-1] = last_value（未修正，与 rollout 内其他步不一致）
   新：V_corrected_next[T-1] = (1 - α_last) * last_value + α_last * approx_G_last
       其中 approx_G_last = last_value + δ_trend（用 rollout 末端 δ 趋势外推）
       物理含义：末端状态的期望价值同样存在 Critic 偏差，用邻近误差的平均外推

④ 优势归一化统计量冻结
   旧：update() 内对整个 rollout 归一化，10 个 epoch 中各 minibatch 共享同一统计量
   新：在 compute_gae 阶段冻结 (adv_mean, adv_std)，update 内直接用冻结值
       防止不同 epoch 的 minibatch 接收到不同尺度的梯度信号
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
    PPO + Hindsight-Corrected GAE（HCGAE）v2
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
        hindsight_beta: float = 3.0,
        hindsight_alpha_max: float = 0.7,
        hindsight_alpha_min: float = 0.1,
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
        self.device = torch.device(device)
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

        # ── v2 改进：EMA 仅保留慢速追踪的 err_ema 用于历史参考（不再用于归一化）
        self._err_ema = 1.0
        self._err_ema_alpha = 0.05
        # EV 的 EMA，驱动 α_max 退火 + Critic 目标混合系数
        self._ev_ema = 0.0
        self._ev_ema_alpha = 0.1
        # ── v2 改进④：冻结归一化统计量
        self._adv_mean_frozen = 0.0
        self._adv_std_frozen  = 1.0

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
        """计算 MC returns（从轨迹末尾反向展开）"""
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
        Hindsight-Corrected GAE v2：

        ① 批内中心化归一化（替代慢速 EMA 归一化）
        ② EV 驱动的 Critic 目标混合系数
        ③ 末端 bootstrap 不一致修正
        ④ 冻结优势归一化统计量
        """
        T   = self.buffer.pos
        buf = self.buffer

        # Step 1: MC returns
        G = self._compute_mc_returns(last_value)

        # Step 2: 逐步误差
        V = buf.values[:T]
        err = np.abs(V - G)

        # ── 改进①：批内统计量归一化（彻底去除 EMA 滞后问题）
        # 用当前批次的均值和标准差做中心化，sigmoid 中心 = 当前平均误差水平
        err_batch_mean = float(err.mean())
        err_batch_std  = float(err.std()) + 1e-8
        # 同时更新历史 EMA（仅供监控，不再用于归一化）
        self._err_ema = (1 - self._err_ema_alpha) * self._err_ema + self._err_ema_alpha * err_batch_mean

        # Step 3: 自适应 α_max（余弦退火 × EV 门控）
        progress = min(self.total_steps / max(self._total_timesteps, 1), 1.0)
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        ev_factor = max(1.0 - max(self._ev_ema, 0.0), 0.2)
        dynamic_alpha_max = (
            self.hindsight_alpha_min
            + (self.hindsight_alpha_max - self.hindsight_alpha_min)
            * cosine_decay * ev_factor
        )

        # ── 改进①：用批内中心化做 sigmoid 的输入
        # z = β * (err - μ_batch) / σ_batch
        # 当 err > μ_batch: z > 0 → alpha > alpha_max/2（强修正）
        # 当 err < μ_batch: z < 0 → alpha < alpha_max/2（弱修正）
        z = self.hindsight_beta * (err - err_batch_mean) / err_batch_std
        alpha = dynamic_alpha_max * (1.0 / (1.0 + np.exp(-z)))

        # Step 4: 修正价值
        V_corrected = (1.0 - alpha) * V + alpha * G  # shape (T,)

        # ── 改进③：末端 bootstrap 修正
        # 用 rollout 末端若干步 δ 的趋势外推 last_value 的近似 MC 误差
        # approx_err_last = 末端10步误差均值（作为 last_value 偏差的保守估计）
        tail_n = min(10, T)
        approx_err_last = float(err[-tail_n:].mean())
        alpha_last = dynamic_alpha_max * (1.0 / (1.0 + np.exp(
            -self.hindsight_beta * (approx_err_last - err_batch_mean) / err_batch_std
        )))
        # 用尾部 MC 值外推近似 last_value 的 hindsight 修正
        approx_G_last = G[-1]  # 最后一步的 MC return 作为保守估计
        last_value_corrected = (1.0 - alpha_last) * last_value + alpha_last * approx_G_last

        # 构建 V_corrected_next（含末端修正）
        V_corrected_next = np.empty(T, dtype=np.float32)
        for t in range(T):
            if buf.terminated[t] > 0.5:
                V_corrected_next[t] = 0.0
            elif t == T - 1:
                # ── 改进③：使用修正后的 last_value
                V_corrected_next[t] = last_value_corrected
            else:
                V_corrected_next[t] = V_corrected[t + 1]

        # Step 5: 用修正后的 V 重新计算 δ，然后标准 GAE 展开
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            delta_corrected = (
                buf.rewards[t]
                + self.gamma * V_corrected_next[t]
                - V_corrected[t]
            )
            not_done = 1.0 - buf.terminated[t]
            gae    = delta_corrected + self.gamma * self.lam * not_done * gae
            adv[t] = gae

        buf.advantages = adv

        # ── 改进②：EV 驱动的 Critic 目标混合系数
        # 关键设计：Critic 目标使用【未校正原始 V】计算的标准 GAE returns
        # R_t^{Critic} = A_t^{std}(V_orig) + V_orig(s_t)
        # 这确保 Critic 训练目标不受优势估计校正路径的污染（两个更新通道完全分离）。
        # _compute_standard_returns 内部使用 buf.values（原始 V），不受 V_corrected 影响。
        ev_current = max(0.0, min(1.0, self._ev_ema))
        c_mc = float(np.clip(1.0 - ev_current, 0.1, 1.0))  # 至少保留 10% MC（防止完全丢弃无偏性）
        std_gae_returns = buf._compute_standard_returns(last_value, self.gamma, self.lam)
        buf.returns = c_mc * G + (1.0 - c_mc) * std_gae_returns

        # ── 改进④：冻结归一化统计量（在 compute_gae 阶段计算，update 阶段直接使用）
        self._adv_mean_frozen = float(adv.mean())
        self._adv_std_frozen  = float(adv.std()) + 1e-8

        # 统计
        raw_deltas = buf.rewards[:T] + self.gamma * buf._next_values(last_value) - V
        autocorr = float(np.corrcoef(raw_deltas[:-1], raw_deltas[1:])[0, 1]) if T > 2 else 0.0
        return {
            "delta_mean"         : float(raw_deltas.mean()),
            "delta_std"          : float(raw_deltas.std()),
            "delta_autocorr"     : autocorr,
            "mean_alpha"         : float(alpha.mean()),
            "dynamic_alpha_max"  : float(dynamic_alpha_max),
            "err_batch_mean"     : err_batch_mean,
            "err_batch_std"      : err_batch_std,
            "c_mc"               : c_mc,
            "alpha_last"         : float(alpha_last),
            # 诊断：使用 std_gae_returns（原始V路径）计算 MC vs Critic目标差异
            "mc_gae_diff"        : float(np.mean(np.abs(G - std_gae_returns))),
        }

    def update(self) -> dict:
        obs, actions, old_log_probs, advantages, returns, old_values = self.buffer.get_batch()

        # ── 改进④：使用冻结统计量归一化（而非当前 minibatch 的统计量）
        advantages = (advantages - self._adv_mean_frozen) / self._adv_std_frozen

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

        self._total_timesteps = total_timesteps
        while self.total_steps < total_timesteps:
            last_value = self.collect_rollout()
            gae_stats  = self.compute_gae(last_value)
            metrics    = self.update()
            update_idx += 1
            metrics.update(gae_stats)
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
                alpha_m   = metrics.get('mean_alpha', 0)
                alpha_max = metrics.get('dynamic_alpha_max', 0)
                c_mc      = metrics.get('c_mc', 0)
                print(
                    f"  [{self.NAME:<24}] "
                    f"{self.total_steps:7d}/{total_timesteps} "
                    f"{progress} "
                    f"| Eval={eval_str} Recent={recent:6.1f} "
                    f"| VLoss={metrics['value_loss']:.3f} EV={metrics['explained_variance']:+.2f} "
                    f"| KL={metrics['approx_kl']:.4f} clip={metrics['clip_frac']:.2f} "
                    f"| α={alpha_m:.3f}(αmax={alpha_max:.2f}) c_mc={c_mc:.2f}"
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

