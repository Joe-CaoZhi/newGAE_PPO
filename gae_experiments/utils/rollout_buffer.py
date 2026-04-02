"""
Rollout Buffer：存储交互数据，支持多种 GAE 计算方式

关键设计说明
-----------
1. buffer 存储 `terminated`（非 `done`），用于区分：
   - terminated=True  → episode 真正结束，next_value = 0
   - truncated=True   → 时间截断，next_value = V(last_obs)（有未来价值）
   因此 next_non_terminal = 1 - terminated

2. confidence_weighted_gae:
   - A_t 使用置信度归一化加权（量纲保持与标准 GAE 一致）
   - returns 使用独立的 n-step TD return（而非 A+V），避免 Critic 目标污染

3. combined_gae 的自适应λ展开：
   A_t = Σ_{l=0}^{H} [Π_{k=0}^{l-1} (γλ_{t+k})] * c_{t+l} * δ_{t+l}
        ─────────────────────────────────────────────────────────────
        Σ_{l=0}^{H} [Π_{k=0}^{l-1} (γλ_{t+k})] * c_{t+l}
   returns 同样使用标准 GAE returns（A_std + V），不做归一化污染
"""
import numpy as np
import torch


class RolloutBuffer:
    """
    存储一条 rollout 轨迹所需的所有数据，
    并提供多种 GAE 计算方法。
    """

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        continuous: bool = False,
    ):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.continuous = continuous
        self.pos = 0
        self.full = False
        self._init_buffers()

    def _init_buffers(self):
        self.observations  = np.zeros((self.buffer_size, self.obs_dim), dtype=np.float32)
        if self.continuous:
            self.actions   = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        else:
            self.actions   = np.zeros((self.buffer_size,), dtype=np.int64)
        self.rewards       = np.zeros(self.buffer_size, dtype=np.float32)
        # ★ 存 terminated（真正结束），而非 done=terminated|truncated
        # terminated=1 → next state 无价值；truncated=1 → next state 仍有价值
        self.terminated    = np.zeros(self.buffer_size, dtype=np.float32)
        self.log_probs     = np.zeros(self.buffer_size, dtype=np.float32)
        self.values        = np.zeros(self.buffer_size, dtype=np.float32)
        self.advantages    = np.zeros(self.buffer_size, dtype=np.float32)
        self.returns       = np.zeros(self.buffer_size, dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action,
        reward: float,
        terminated: float,   # ← 只传 terminated
        log_prob: float,
        value: float,
    ):
        self.observations[self.pos] = obs
        self.actions[self.pos]      = action
        self.rewards[self.pos]      = reward
        self.terminated[self.pos]   = terminated
        self.log_probs[self.pos]    = log_prob
        self.values[self.pos]       = value
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def reset(self):
        self.pos  = 0
        self.full = False

    # -----------------------------------------------------------------------
    # 内部工具
    # -----------------------------------------------------------------------
    def _next_values(self, last_value: float) -> np.ndarray:
        """
        返回 shape (T,) 的 next_value 数组：
          - terminated[t]=1 → next_value[t] = 0（真正结束，无未来价值）
          - terminated[t]=0, t<T-1 → next_value[t] = values[t+1]
          - terminated[t]=0, t=T-1 → next_value[t] = last_value（bootstrap）
        正确处理 truncation（time-limit）：truncated 时 done=True 但 terminated=False，
        故 next_value 不清零，而是用 bootstrap 值。
        """
        T = self.pos
        nv = np.empty(T, dtype=np.float32)
        for t in range(T):
            if self.terminated[t] > 0.5:
                nv[t] = 0.0
            elif t == T - 1:
                nv[t] = last_value
            else:
                nv[t] = self.values[t + 1]
        return nv

    def _compute_standard_returns(self, last_value: float, gamma: float, lam: float) -> np.ndarray:
        """
        计算标准 GAE returns（= 标准优势 + V），供置信度方法使用。
        这提供了一个无偏的 Critic 训练目标，不受置信度归一化影响。
        """
        T   = self.pos
        nv  = self._next_values(last_value)
        gae = 0.0
        ret = np.zeros(T, dtype=np.float32)
        for t in reversed(range(T)):
            delta    = self.rewards[t] + gamma * nv[t] - self.values[t]
            not_done = 1.0 - self.terminated[t]
            gae      = delta + gamma * lam * not_done * gae
            ret[t]   = gae + self.values[t]
        return ret

    # -----------------------------------------------------------------------
    # GAE 计算方法
    # -----------------------------------------------------------------------

    def compute_standard_gae(self, last_value: float, gamma: float, lam: float):
        """
        标准 GAE（固定 λ）
        δ_t = r_t + γ V(s_{t+1}) - V(s_t)，terminated 时 V(s_{t+1})=0
        A_t = Σ_{l≥0} (γλ)^l δ_{t+l}，episode 边界自动截断
        returns = A_t + V(s_t)
        """
        T   = self.pos
        nv  = self._next_values(last_value)
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(T)):
            delta    = self.rewards[t] + gamma * nv[t] - self.values[t]
            not_done = 1.0 - self.terminated[t]
            gae      = delta + gamma * lam * not_done * gae
            adv[t]   = gae

        self.advantages = adv
        self.returns    = adv + self.values[:T]

    def compute_adaptive_lambda_gae(
        self,
        last_value: float,
        gamma: float,
        lambda_values: np.ndarray,   # shape (T,)，每步的 λ_t ∈ [0,1]
    ):
        """
        自适应 λ GAE
        A_t = δ_t + γλ_{t}(1-term_t)(δ_{t+1} + γλ_{t+1}(1-term_{t+1})(...))
        λ_t 由 LambdaNetwork 根据 s_t 输出，不确定性高→λ小→更信任 TD
        returns = A_t + V(s_t)（Critic 直接拟合多步 TD target）
        """
        T   = self.pos
        nv  = self._next_values(last_value)
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(T)):
            delta    = self.rewards[t] + gamma * nv[t] - self.values[t]
            not_done = 1.0 - self.terminated[t]
            lam_t    = float(lambda_values[t])
            gae      = delta + gamma * lam_t * not_done * gae
            adv[t]   = gae

        self.advantages = adv
        self.returns    = adv + self.values[:T]

    def compute_double_critic_gae(
        self,
        last_value: float,
        last_value2: float,
        gamma: float,
        lam: float,
        values2: np.ndarray,   # shape (T,)，第二个 Critic 的估计
    ):
        """
        双 Critic 保守 GAE（借鉴 TD3）
        δ_t = r_t + γ min(V1(s'), V2(s')) - (V1(s) + V2(s))/2
          当前状态用均值（无偏估计），下一状态用 min（保守/防过估计）

        returns: 为了让 Critic1 和 Critic2 各自有正确的训练目标，
          这里 returns = A_t + mean(V1,V2)，作为两个 Critic 共同的目标。
          实践中这让两个 Critic 向同一 conservative target 靠拢。
        """
        T           = self.pos
        mean_values = (self.values[:T] + values2[:T]) / 2.0
        adv         = np.zeros(T, dtype=np.float32)
        gae         = 0.0

        for t in reversed(range(T)):
            not_done = 1.0 - self.terminated[t]
            if self.terminated[t] > 0.5:
                next_min_v = 0.0
            elif t == T - 1:
                next_min_v = min(last_value, last_value2)
            else:
                next_min_v = min(self.values[t + 1], values2[t + 1])

            delta  = self.rewards[t] + gamma * next_min_v - mean_values[t]
            gae    = delta + gamma * lam * not_done * gae
            adv[t] = gae

        self.advantages = adv
        self.returns    = adv + mean_values   # conservative target

    def compute_confidence_weighted_gae(
        self,
        last_value: float,
        gamma: float,
        lam: float,
        confidence: np.ndarray,   # shape (T,)，c_t ∈ (0,1]
    ):
        """
        置信度加权 GAE（归一化优势 + 标准 returns）

        优势：A_t = [Σ_l (γλ)^l c_{t+l} δ_{t+l}] / [Σ_l (γλ)^l c_{t+l}]
          - 置信度归一化确保量纲与标准 GAE 一致
          - c_t 低（δ方差大）时，该步 δ 贡献减少

        ★ returns 使用独立的标准 GAE returns（而非 A_normalized + V）
          原因：A_normalized 已经不是真实的多步回报，
          用它加 V 训练 Critic 会引入偏差。
          标准 returns = A_std + V 提供了无偏的 Critic 目标。
        """
        T   = self.pos
        nv  = self._next_values(last_value)
        deltas = self.rewards[:T] + gamma * nv - self.values[:T]

        adv = np.zeros(T, dtype=np.float32)
        H   = min(T, 64)

        for t in range(T):
            w_sum  = 0.0
            w_tot  = 0.0
            w_step = 1.0

            for l in range(H):
                idx = t + l
                if idx >= T:
                    break
                if l > 0 and self.terminated[idx - 1] > 0.5:
                    break

                w       = w_step * confidence[idx]
                w_sum  += w * deltas[idx]
                w_tot  += w
                w_step *= gamma * lam

            adv[t] = w_sum / w_tot if w_tot > 1e-8 else deltas[t]

        self.advantages = adv
        # ★ 用标准 GAE returns 训练 Critic，而不是归一化优势+V
        self.returns = self._compute_standard_returns(last_value, gamma, lam)

    def compute_combined_gae(
        self,
        last_value: float,
        last_value2: float,
        gamma: float,
        lambda_values: np.ndarray,   # shape (T,)，自适应 λ_t
        values2: np.ndarray,         # shape (T,)，第二个 Critic
        confidence: np.ndarray,      # shape (T,)，置信度 c_t
    ):
        """
        组合改进：双 Critic δ + 自适应 λ 展开 + 置信度归一化

        δ_t = r_t + γ min(V1(s'), V2(s')) - (V1(s)+V2(s))/2  [double-critic，保守]

        A_t = [Σ_l w_l c_{t+l} δ_{t+l}] / [Σ_l w_l c_{t+l}]
        w_l = Π_{k=0}^{l-1} (γ λ_{t+k})   每步用该步的自适应 λ

        ★ returns 使用 double-critic 标准 GAE returns（固定λ=mean_λ），
          给 Critic 提供稳定目标，不被置信度归一化污染。
        """
        T           = self.pos
        mean_values = (self.values[:T] + values2[:T]) / 2.0

        # ① 计算 double-critic δ
        deltas = np.zeros(T, dtype=np.float32)
        for t in range(T):
            if self.terminated[t] > 0.5:
                next_min_v = 0.0
            elif t == T - 1:
                next_min_v = min(last_value, last_value2)
            else:
                next_min_v = min(self.values[t + 1], values2[t + 1])
            deltas[t] = self.rewards[t] + gamma * next_min_v - mean_values[t]

        # ② 自适应 λ + 置信度归一化展开 → 优势
        adv = np.zeros(T, dtype=np.float32)
        H   = min(T, 64)

        for t in range(T):
            w_sum  = 0.0
            w_tot  = 0.0
            w_step = 1.0

            for l in range(H):
                idx = t + l
                if idx >= T:
                    break
                if l > 0 and self.terminated[idx - 1] > 0.5:
                    break

                w       = w_step * confidence[idx]
                w_sum  += w * deltas[idx]
                w_tot  += w
                # 下一步的累积权重 = 当前步 γλ_{t+l}
                w_step *= gamma * float(lambda_values[idx])

            adv[t] = w_sum / w_tot if w_tot > 1e-8 else deltas[t]

        self.advantages = adv
        # ★ returns = conservative A_double_std + mean_values
        # 先用平均λ计算一个标准的 double-critic GAE returns 作为 Critic 目标
        lam_mean = float(np.mean(lambda_values))
        std_gae  = np.zeros(T, dtype=np.float32)
        gae_r    = 0.0
        for t in reversed(range(T)):
            not_done = 1.0 - self.terminated[t]
            gae_r    = deltas[t] + gamma * lam_mean * not_done * gae_r
            std_gae[t] = gae_r
        self.returns = std_gae + mean_values

    def get_batch(self):
        """获取 tensor 格式的数据，返回顺序与 update() 期望一致"""
        T = self.pos
        obs      = torch.FloatTensor(self.observations[:T]).to(self.device)
        if self.continuous:
            actions = torch.FloatTensor(self.actions[:T]).to(self.device)
        else:
            actions = torch.LongTensor(self.actions[:T]).to(self.device)
        log_probs  = torch.FloatTensor(self.log_probs[:T]).to(self.device)
        advantages = torch.FloatTensor(self.advantages[:T]).to(self.device)
        returns    = torch.FloatTensor(self.returns[:T]).to(self.device)
        values     = torch.FloatTensor(self.values[:T]).to(self.device)
        return obs, actions, log_probs, advantages, returns, values

