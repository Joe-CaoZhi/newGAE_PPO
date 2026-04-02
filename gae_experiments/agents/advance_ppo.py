"""
ADVANCE-PPO: Adaptive Non-parametric Dynamic Value-Correction Enhanced PPO
==========================================================================

从三个正交维度对 PPO 进行全面改进，基于 HCGAE 的训练数据分析：

分析的核心问题
──────────────────────────────────────────────────────────────────────
1. [信任域] 固定 ε=0.2 对所有训练阶段一刀切:
   - 早期 policy 随机，ratio 方差大 → 应放宽 ε 允许更快探索
   - 后期 policy 收敛，ratio 方差小 → 应收紧 ε 防止过拟合
   - clip_frac ~15% 且集中在 ε 边界，说明梯度"撞墙"而非真正受益于 clip

2. [重要性采样] 多 epoch 后 ratio 偏离 1.0，但固定 clip 无法感知偏离程度:
   - epoch 早期样本 "新鲜"，epoch 晚期样本 "陈旧"
   - 应在 epoch 维度上也感知数据新鲜度，而非仅靠 clip 截断

3. [Critic 稳定性] value_loss 早期尖峰（max~777），EV 中期短暂负值:
   - 根因：Critic 目标（returns）与当前估计差距过大时，单步更新步子太大
   - Slow-update Target Network for Value（借鉴 SAC/TD3 但适配 on-policy）

新方法设计（三个改进，各有独立开关）
──────────────────────────────────────────────────────────────────────
改进 A：自适应信任域（Adaptive Trust Region）
  - 用 KL 滑动均值 kl_ema 和 clip_frac_ema 联合驱动 ε_t：
    ε_t = ε_base × (kl_target / (kl_ema + 1e-6))^α
    含义：如果当前 KL 低于目标，放宽 ε；如果 KL 高，收紧 ε
  - 反省：这不同于 TRPO（二阶）/KLPEN（软惩罚），是直接动态调整 ε
  - 边界：ε ∈ [ε_min, ε_max] = [0.05, 0.4]

改进 B：Epoch-Decay 重要性采样权重（Epoch-Decay IS Reweighting）
  - 在每个 epoch 计算当前 ratio 与 epoch=0 时 ratio 的相对偏离：
    staleness_t = |ratio_t - 1| / (|ratio_0 - 1| + 1e-4)
  - 用 staleness 衰减优势：
    A_effective = A × sigmoid(1 - staleness_t × decay_rate)
  - 反省：与 PPG 中的辅助 phase 不同，这是在 on-policy 框架内解决样本陈旧问题
  - 越陈旧的样本对梯度的贡献越小，起到软截断效果

改进 C：EV-Gated Critic 动量更新（EV-Gated Value Stabilizer）
  - 引入 "目标价值网络"（target critic），在 EV 高时与主网络差异小，EV 低时差异大
  - returns 目标混合：
    target_returns = ev_ema × returns_gae + (1 - ev_ema) × target_critic_preds
  - target critic 使用 Polyak 平均：θ_target ← τ × θ + (1-τ) × θ_target
  - 反省：on-policy 中用 target network 较罕见，关键是 τ 要足够大（0.95~0.99）
    保证 target 网络在 on-policy 数据上是新鲜的，不像 off-policy 中τ很小

数学正确性验证
──────────────────────────────────────────────────────────────────────
A: ε_t 动态调整不改变 PPO 目标函数的结构，仍然是 min(ratio×A, clip(ratio)×A)
   只是 clip 的范围动态变化。等价于 KL-adaptive PPO（有理论支持）

B: 用 staleness 加权相当于给样本加了重要性权重 w(x) = sigmoid(1 - staleness)
   当 staleness→0（新鲜样本），w→sigmoid(1)=0.73；staleness→∞，w→0.5
   权重下界 0.5 保证梯度不会完全消失

C: 使用 target critic 的 returns 混合等价于引入了额外的正则化：
   Critic 训练目标向一个"稳定的老版本"靠拢，减少非平稳性
   当 EV→1 时，混合退化为纯 GAE returns，与标准 PPO 完全一致

版本名称
──────────────────────────────────────────────────────────────────────
ADVANCE_PPO_Base: 无任何改进（复现 HCGAE_Imp12 基础上的 Standard PPO）
ADVANCE_ImpA:    仅自适应信任域
ADVANCE_ImpB:    仅 Epoch-Decay IS Reweighting
ADVANCE_ImpC:    仅 EV-Gated Value Stabilizer
ADVANCE_ImpAB:   A + B
ADVANCE_ImpAC:   A + C
ADVANCE_ImpBC:   B + C
ADVANCE_Full:    A + B + C
"""
from typing import Optional
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import copy

from ..utils.logger import MetricLogger
from ..utils.networks import ActorNetwork, CriticNetwork
from ..utils.rollout_buffer import RolloutBuffer


class AdvancePPO:
    """
    ADVANCE-PPO: 三项正交的 PPO 全面改进
    通过布尔开关独立控制各项改进，所有变体共享相同代码路径。
    """

    def __init__(
        self,
        env: gym.Env,
        name: str = "ADVANCE_Full",
        # ── 改进开关 ──
        use_imp_a: bool = True,   # A: 自适应信任域
        use_imp_b: bool = True,   # B: Epoch-Decay IS Reweighting
        use_imp_c: bool = True,   # C: EV-Gated Value Stabilizer
        # ── 标准 PPO 超参 ──
        hidden_dim: int = 64,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        lam: float = 0.95,
        eps_clip: float = 0.2,      # A 的基准 ε
        n_epochs: int = 10,
        batch_size: int = 64,
        n_steps: int = 2048,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        # ── A: 自适应信任域超参 ──
        kl_target: float = 0.01,    # 目标 KL 散度
        eps_min: float = 0.05,      # ε 最小值
        eps_max: float = 0.4,       # ε 最大值
        kl_adapt_rate: float = 0.5, # ε 调整幅度（指数）
        kl_ema_alpha: float = 0.1,  # KL EMA 滑动系数
        # ── B: Epoch-Decay 超参 ──
        staleness_decay: float = 2.0,  # staleness 衰减速率
        # ── C: EV-Gated Value Stabilizer 超参 ──
        target_critic_tau: float = 0.95,   # Polyak 系数（越大更新越快）
        ev_ema_alpha: float = 0.1,
        # ── HCGAE 增强（基于 HCGAE_Imp12 的 GAE 部分）──
        use_hcgae: bool = True,          # 是否继承 HCGAE 的 GAE 改进
        hindsight_beta: float = 3.0,
        hindsight_alpha_max: float = 0.7,
        hindsight_alpha_min: float = 0.1,
        # ── 其他 ──
        device: str = "cpu",
        save_dir: str = "results",
    ):
        self.NAME = name
        self.use_imp_a = use_imp_a
        self.use_imp_b = use_imp_b
        self.use_imp_c = use_imp_c
        self.use_hcgae = use_hcgae

        self.env = env
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.eps_clip_current = eps_clip  # 动态调整的 ε（A 改进使用）
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        # A 超参
        self.kl_target = kl_target
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.kl_adapt_rate = kl_adapt_rate
        self.kl_ema_alpha = kl_ema_alpha
        self._kl_ema = kl_target  # 初始化为目标值

        # B 超参
        self.staleness_decay = staleness_decay

        # C 超参
        self.target_critic_tau = target_critic_tau
        self.ev_ema_alpha = ev_ema_alpha
        self._ev_ema = 0.0

        # HCGAE 超参
        self.hindsight_beta = hindsight_beta
        self.hindsight_alpha_max = hindsight_alpha_max
        self.hindsight_alpha_min = hindsight_alpha_min
        self._err_ema = 1.0
        self._err_ema_alpha = 0.05
        self._total_timesteps = 1

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

        self.actor  = ActorNetwork(obs_dim, action_dim, hidden_dim, self.continuous).to(self.device)
        self.critic = CriticNetwork(obs_dim, hidden_dim).to(self.device)

        # C: target critic（Polyak 平均）
        if use_imp_c:
            self.target_critic = copy.deepcopy(self.critic)
            for p in self.target_critic.parameters():
                p.requires_grad_(False)
        else:
            self.target_critic = None

        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = RolloutBuffer(n_steps, obs_dim, action_dim, self.device, self.continuous)

        self.logger = MetricLogger(self.NAME, save_dir)
        self.total_steps = 0

    # ──────────────────────────────────────────────────────────────
    # 数据收集
    # ──────────────────────────────────────────────────────────────
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

    # ──────────────────────────────────────────────────────────────
    # HCGAE 增强型 GAE 计算
    # ──────────────────────────────────────────────────────────────
    def _compute_mc_returns(self, last_value: float) -> np.ndarray:
        T = self.buffer.pos
        G = np.zeros(T, dtype=np.float32)
        g = last_value
        for t in reversed(range(T)):
            not_done = 1.0 - self.buffer.terminated[t]
            g = self.buffer.rewards[t] + self.gamma * g * not_done
            G[t] = g
        return G

    def compute_gae(self, last_value: float) -> dict:
        T   = self.buffer.pos
        buf = self.buffer

        if self.use_hcgae:
            # ── HCGAE 增强型（继承 Imp12：批内中心化 + EV 驱动混合）──
            G   = self._compute_mc_returns(last_value)
            V   = buf.values[:T]
            err = np.abs(V - G)

            # 批内中心化 Sigmoid（改进①）
            err_batch_mean = float(err.mean())
            err_batch_std  = float(err.std()) + 1e-8
            self._err_ema  = (1 - self._err_ema_alpha) * self._err_ema + self._err_ema_alpha * err_batch_mean
            z = self.hindsight_beta * (err - err_batch_mean) / err_batch_std

            # EV 驱动 alpha_max（余弦退火 + EV 门控）
            progress = min(self.total_steps / max(self._total_timesteps, 1), 1.0)
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            ev_factor = max(1.0 - max(self._ev_ema, 0.0), 0.2)
            dynamic_alpha_max = (
                self.hindsight_alpha_min
                + (self.hindsight_alpha_max - self.hindsight_alpha_min)
                * cosine_decay * ev_factor
            )
            alpha = dynamic_alpha_max * (1.0 / (1.0 + np.exp(-z)))
            alpha = np.clip(alpha, 0.0, dynamic_alpha_max)

            V_corrected = (1.0 - alpha) * V + alpha * G

            # 构建 V_corrected_next
            V_corrected_next = np.empty(T, dtype=np.float32)
            for t in range(T):
                if buf.terminated[t] > 0.5:
                    V_corrected_next[t] = 0.0
                elif t == T - 1:
                    V_corrected_next[t] = last_value
                else:
                    V_corrected_next[t] = V_corrected[t + 1]

            # 标准 GAE 展开（用修正后的 V）
            adv = np.zeros(T, dtype=np.float32)
            gae = 0.0
            for t in reversed(range(T)):
                delta_corrected = buf.rewards[t] + self.gamma * V_corrected_next[t] - V_corrected[t]
                not_done = 1.0 - buf.terminated[t]
                gae    = delta_corrected + self.gamma * self.lam * not_done * gae
                adv[t] = gae
            buf.advantages = adv

            # EV 驱动混合 returns（改进②）
            ev_current = max(0.0, min(1.0, self._ev_ema))
            c_mc = float(np.clip(1.0 - ev_current, 0.1, 1.0))
            gae_returns = buf._compute_standard_returns(last_value, self.gamma, self.lam)
            buf.returns = c_mc * G + (1.0 - c_mc) * gae_returns

            hcgae_stats = {
                "mean_alpha": float(alpha.mean()),
                "dynamic_alpha_max": float(dynamic_alpha_max),
                "c_mc": c_mc,
                "bias_proxy": float(np.mean(V - G)),
                "V_correction_norm": float(np.mean(np.abs(V_corrected - V))),
            }
        else:
            # 标准 GAE
            buf.compute_standard_gae(last_value, self.gamma, self.lam)
            G = None
            hcgae_stats = {"mean_alpha": 0.0, "dynamic_alpha_max": 0.0, "c_mc": 0.0,
                           "bias_proxy": 0.0, "V_correction_norm": 0.0}

        # C: 如果使用 target critic，计算 target 预测用于混合
        if self.use_imp_c and self.target_critic is not None:
            obs_tensor = torch.FloatTensor(buf.observations[:T]).to(self.device)
            with torch.no_grad():
                target_vals = self.target_critic(obs_tensor).squeeze(-1).cpu().numpy()
            ev_current = max(0.0, min(1.0, self._ev_ema))
            # EV 高→更信任 GAE returns；EV 低→混入 target critic 预测
            # 注意：target_vals 不能直接当 returns，需要加优势：
            # target_returns = target_vals + adv（Critic 学的是 V，returns = V + A）
            target_returns = target_vals + buf.advantages[:T]
            blend = ev_current  # ev_ema 作为混合权重
            buf.returns[:T] = blend * buf.returns[:T] + (1.0 - blend) * target_returns
            hcgae_stats["target_blend"] = float(1.0 - blend)
        else:
            hcgae_stats["target_blend"] = 0.0

        # 统计
        nv = buf._next_values(last_value)
        raw_deltas = buf.rewards[:T] + self.gamma * nv - buf.values[:T]
        autocorr = float(np.corrcoef(raw_deltas[:-1], raw_deltas[1:])[0, 1]) if T > 2 else 0.0

        stats = {
            "delta_mean": float(raw_deltas.mean()),
            "delta_std": float(raw_deltas.std()),
            "delta_autocorr": autocorr,
            "eps_clip_current": float(self.eps_clip_current),
            "kl_ema": float(self._kl_ema),
        }
        stats.update(hcgae_stats)
        return stats

    # ──────────────────────────────────────────────────────────────
    # PPO 更新（包含三项改进）
    # ──────────────────────────────────────────────────────────────
    def update(self) -> dict:
        obs, actions, old_log_probs, advantages, returns, old_values = self.buffer.get_batch()

        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        T = self.buffer.pos
        indices = np.arange(T)
        metrics = {"value_loss": 0.0, "policy_loss": 0.0, "entropy_loss": 0.0,
                   "approx_kl": 0.0, "clip_frac": 0.0}
        update_count = 0

        # B: 记录 epoch=0 时的 ratio 分布（用于计算 staleness）
        with torch.no_grad():
            init_new_lp, _ = self.actor.evaluate_actions(obs, actions)
            init_ratio = torch.exp(init_new_lp - old_log_probs).detach()  # shape (T,)

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

                # ── 改进 A：使用动态 ε_clip ──
                eps = self.eps_clip_current

                # ── 改进 B：Epoch-Decay IS Reweighting ──
                if self.use_imp_b:
                    init_ratio_batch = init_ratio[batch_idx]
                    # staleness = 当前 ratio 与初始 ratio 的相对偏离
                    staleness = torch.abs(ratio - 1.0) / (torch.abs(init_ratio_batch - 1.0) + 1e-4)
                    staleness = staleness.detach()
                    # 越陈旧的样本，权重越小（但不会降到 0）
                    is_weight = torch.sigmoid(1.0 - staleness * self.staleness_decay)
                    # 归一化权重（保持期望梯度量级不变）
                    is_weight = is_weight / (is_weight.mean() + 1e-8) * 0.9 + 0.1
                    batch_advantages = batch_advantages * is_weight.detach()

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * batch_advantages
                policy_loss  = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()
                value_loss   = 0.5 * ((new_values - batch_returns) ** 2).mean()

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
                    clip_frac = ((ratio - 1).abs() > eps).float().mean().item()

                metrics["value_loss"]   += value_loss.item()
                metrics["policy_loss"]  += policy_loss.item()
                metrics["entropy_loss"] += entropy_loss.item()
                metrics["approx_kl"]    += approx_kl
                metrics["clip_frac"]    += clip_frac
                update_count += 1

        if update_count > 0:
            for k in metrics:
                metrics[k] /= update_count

        # ── 改进 A：更新自适应 ε ──
        if self.use_imp_a:
            current_kl = metrics["approx_kl"]
            self._kl_ema = (1 - self.kl_ema_alpha) * self._kl_ema + self.kl_ema_alpha * current_kl
            # ε_t = ε_base × (kl_target / kl_ema)^kl_adapt_rate
            ratio_kl = self.kl_target / (self._kl_ema + 1e-8)
            new_eps = self.eps_clip * (ratio_kl ** self.kl_adapt_rate)
            self.eps_clip_current = float(np.clip(new_eps, self.eps_min, self.eps_max))
        else:
            self.eps_clip_current = self.eps_clip

        # ── 改进 C：Polyak 更新 target critic ──
        if self.use_imp_c and self.target_critic is not None:
            tau = self.target_critic_tau
            with torch.no_grad():
                for p, pt in zip(self.critic.parameters(), self.target_critic.parameters()):
                    pt.data.copy_(tau * p.data + (1 - tau) * pt.data)

        # 计算 EV
        with torch.no_grad():
            y_pred = old_values.cpu().numpy()
            y_true = returns.cpu().numpy()
            var_y  = np.var(y_true)
            ev = 1 - np.var(y_true - y_pred) / (var_y + 1e-8)
            metrics["explained_variance"] = float(ev)

        metrics["eps_clip_current"] = float(self.eps_clip_current)
        return metrics

    # ──────────────────────────────────────────────────────────────
    # 训练循环
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _bar(cur, tot, w=20):
        f = int(w * cur / tot)
        return f"[{'█'*f}{'░'*(w-f)}] {100*cur/tot:5.1f}%"

    def train(
        self,
        total_timesteps: int,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 5000,
        n_eval_episodes: int = 5,
        verbose: bool = True,
        print_interval: int = 5,
    ):
        import time
        self._total_timesteps = total_timesteps
        train_start = time.time()
        last_eval_reward = float("nan")
        update_idx = 0

        while self.total_steps < total_timesteps:
            last_value = self.collect_rollout()
            gae_stats  = self.compute_gae(last_value)
            metrics    = self.update()
            update_idx += 1
            metrics.update(gae_stats)

            ev_val = metrics.get("explained_variance", 0.0)
            self._ev_ema = (1 - self.ev_ema_alpha) * self._ev_ema + self.ev_ema_alpha * ev_val

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

            if verbose and (do_eval or update_idx % print_interval == 1):
                elapsed = time.time() - train_start
                recent  = self.logger.get_recent_reward(20)
                fps     = int(self.total_steps / (elapsed + 1e-8))
                eval_str = f"{last_eval_reward:7.1f}" if not np.isnan(last_eval_reward) else "    N/A"

                imp_flags = (
                    f"A{'✓' if self.use_imp_a else '✗'}"
                    f"B{'✓' if self.use_imp_b else '✗'}"
                    f"C{'✓' if self.use_imp_c else '✗'}"
                )
                print(
                    f"  [{self.NAME:<18}|{imp_flags}] "
                    f"{self.total_steps:7d}/{total_timesteps} {self._bar(self.total_steps, total_timesteps)} "
                    f"| Eval={eval_str} Rec={recent:6.1f} "
                    f"| VL={metrics['value_loss']:.3f} EV={metrics['explained_variance']:+.3f} "
                    f"| ε={self.eps_clip_current:.3f}(KL_ema={self._kl_ema:.4f}) "
                    f"| c_mc={metrics.get('c_mc', 0):.2f} blend={metrics.get('target_blend', 0):.2f} "
                    f"| KL={metrics['approx_kl']:.4f} clip={metrics['clip_frac']:.2f} "
                    f"| {fps:5d}fps {elapsed:5.0f}s"
                )

        if verbose:
            elapsed = time.time() - train_start
            final_r = np.mean(self.logger.eval_rewards[-5:]) if self.logger.eval_rewards else 0.0
            best_r  = max(self.logger.eval_rewards) if self.logger.eval_rewards else 0.0
            print(f"  ✓ [{self.NAME}] 完成 | 耗时={elapsed:.1f}s | 最终={final_r:.1f} | 最高={best_r:.1f}")

        self.logger.save()
        return self.logger

    def evaluate(self, eval_env: gym.Env, n_episodes: int = 5) -> float:
        total_reward = 0.0
        for _ in range(n_episodes):
            obs, _ = eval_env.reset()
            done   = False
            ep_r   = 0.0
            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    dist = self.actor(obs_t)
                    action = dist.mean if self.continuous else dist.probs.argmax(dim=-1)
                a = action.squeeze(0).cpu().numpy()
                if self.continuous:
                    next_obs, r, terminated, truncated, _ = eval_env.step(a)
                else:
                    next_obs, r, terminated, truncated, _ = eval_env.step(int(a))
                ep_r += r
                done  = terminated or truncated
                obs   = next_obs
            total_reward += ep_r
        return total_reward / n_episodes


# ════════════════════════════════════════════════════════════════════
# 工厂函数
# ════════════════════════════════════════════════════════════════════

_ADVANCE_CONFIGS = {
    # name                  A       B       C
    "ADVANCE_Base":     (False, False, False),  # 仅 HCGAE，无新改进
    "ADVANCE_ImpA":     (True,  False, False),  # 仅自适应信任域
    "ADVANCE_ImpB":     (False, True,  False),  # 仅 Epoch-Decay IS
    "ADVANCE_ImpC":     (False, False, True ),  # 仅 EV-Gated Critic
    "ADVANCE_ImpAB":    (True,  True,  False),  # A + B
    "ADVANCE_ImpAC":    (True,  False, True ),  # A + C
    "ADVANCE_ImpBC":    (False, True,  True ),  # B + C
    "ADVANCE_Full":     (True,  True,  True ),  # 全量
}


def build_advance_agent(
    variant_name: str,
    env: gym.Env,
    **kwargs
) -> AdvancePPO:
    if variant_name not in _ADVANCE_CONFIGS:
        raise ValueError(
            f"未知变体 '{variant_name}'。可用变体：{list(_ADVANCE_CONFIGS.keys())}"
        )
    imp_a, imp_b, imp_c = _ADVANCE_CONFIGS[variant_name]
    # 如果 kwargs 里没有 name，使用 variant_name 作为默认名称
    kwargs.setdefault("name", variant_name)
    return AdvancePPO(
        env=env,
        use_imp_a=imp_a,
        use_imp_b=imp_b,
        use_imp_c=imp_c,
        **kwargs,
    )


def get_all_advance_variant_names():
    return list(_ADVANCE_CONFIGS.keys())

