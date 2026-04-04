"""
Optimal PPO Implementation (Andrychowicz et al., 2021 Best Practices)
======================================================================

This implements the "optimal PPO" baseline based on:
  - Andrychowicz et al. (2021): "What Matters for On-Policy Deep Actor-Critic Methods?
    A Large-Scale Study" (arXiv:2006.05990)
  - Engstrom et al. (2020): "Implementation Matters in Deep RL"
  - CleanRL's PPO implementation (Costa et al., 2022)

Key improvements over our previous "Standard_PPO":
  1. Observation normalization (running mean/var, crucial!)
  2. Per-minibatch advantage normalization (instead of per-rollout)
  3. Orthogonal initialization with std=sqrt(2) for hidden, 0.01 for policy head, 1.0 for value head
  4. Shared actor-critic backbone option (separate by default)
  5. Value loss clipping (optional, from Engstrom 2020)
  6. Entropy coefficient (positive, for exploration)
  7. LR annealing to 0
  8. Gradient clipping (0.5)

Usage:
  - OptimalPPO: The full "best practice" PPO (with obs normalization etc.)
  - OptimalHCGAE: HCGAE built on top of OptimalPPO
"""
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from ..utils.logger import MetricLogger
from ..utils.networks import ActorNetwork, CriticNetwork
from ..utils.rollout_buffer import RolloutBuffer


class RunningMeanStd:
    """
    Running mean and standard deviation for observation normalization.
    Uses Welford's online algorithm for numerical stability.
    """
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4  # small init to avoid divide-by-zero

    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = m2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        std = np.sqrt(self.var + 1e-8)
        x_norm = (x - self.mean) / std
        return np.clip(x_norm, -clip, clip)


class OptimalPPO:
    """
    Optimal PPO with all best practices from Andrychowicz et al. (2021).

    This is the 'gold standard' PPO baseline against which HCGAE should be
    compared. Every trick here is standard and well-justified.
    """

    NAME = "Optimal_PPO"

    def __init__(
        self,
        env: gym.Env,
        name: str = "Optimal_PPO",
        # ── Network ──
        hidden_dim: int = 64,
        # ── Optimizer ──
        lr: float = 3e-4,           # shared LR for actor & critic
        # ── PPO core ──
        gamma: float = 0.99,
        lam: float = 0.95,
        eps_clip: float = 0.2,
        n_epochs: int = 10,
        batch_size: int = 64,
        n_steps: int = 2048,
        # ── Losses ──
        ent_coef: float = 0.0,      # entropy coefficient (0.0 for MuJoCo)
        vf_coef: float = 0.5,       # value loss coefficient
        max_grad_norm: float = 0.5,
        # ── Key tricks ──
        use_obs_norm: bool = True,   # ★ Obs normalization (crucial)
        use_adv_norm: bool = True,   # ★ Per-minibatch advantage normalization
        use_lr_anneal: bool = True,  # ★ LR annealing to 0
        use_vclip: bool = False,     # Value clipping (controversial, default off)
        vclip_eps: float = 0.2,      # value clip epsilon (same as eps_clip)
        # ── Other ──
        device: str = "cpu",
        save_dir: str = "results",
    ):
        self.NAME = name
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
        self.use_obs_norm = use_obs_norm
        self.use_adv_norm = use_adv_norm
        self.use_lr_anneal = use_lr_anneal
        self.use_vclip = use_vclip
        self.vclip_eps = vclip_eps
        self.device = torch.device(device)
        self.lr_init = lr
        self._total_timesteps = 1
        self.total_steps = 0

        obs_dim = env.observation_space.shape[0]
        if isinstance(env.action_space, gym.spaces.Discrete):
            action_dim = env.action_space.n
            self.continuous = False
        else:
            action_dim = env.action_space.shape[0]
            self.continuous = True

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Networks (orthogonal init already in ActorNetwork / CriticNetwork)
        self.actor = ActorNetwork(obs_dim, action_dim, hidden_dim, self.continuous).to(self.device)
        self.critic = CriticNetwork(obs_dim, hidden_dim).to(self.device)

        # Single optimizer with combined parameters (CleanRL style)
        # Separate LRs for actor and critic can be set via param_groups
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr, eps=1e-5
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(n_steps, obs_dim, action_dim, self.device, self.continuous)

        # Running stats for observation normalization
        if use_obs_norm:
            self.obs_rms = RunningMeanStd(shape=(obs_dim,))
        else:
            self.obs_rms = None

        self.logger = MetricLogger(self.NAME, save_dir)

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation using running stats."""
        if self.obs_rms is None:
            return obs
        return self.obs_rms.normalize(obs).astype(np.float32)

    def update_obs_rms(self, obs: np.ndarray):
        """Update running stats for observation normalization."""
        if self.obs_rms is not None:
            self.obs_rms.update(obs.reshape(1, -1) if obs.ndim == 1 else obs)

    def compute_gae(self, last_value: float):
        self.buffer.compute_standard_gae(last_value, self.gamma, self.lam)

    def update(self) -> dict:
        obs, actions, old_log_probs, advantages, returns, old_values = self.buffer.get_batch()

        # LR annealing
        if self.use_lr_anneal:
            frac = 1.0 - (self.total_steps / max(self._total_timesteps, 1))
            frac = max(frac, 0.0)
            new_lr = frac * self.lr_init
            for pg in self.optimizer.param_groups:
                pg['lr'] = max(new_lr, 1e-8)

        T = self.buffer.pos
        indices = np.arange(T)
        metrics = {
            "value_loss": 0.0,
            "policy_loss": 0.0,
            "entropy_loss": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
            "explained_variance": 0.0,
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
                batch_old_lp = old_log_probs[batch_idx]
                batch_adv = advantages[batch_idx]
                batch_ret = returns[batch_idx]
                batch_old_val = old_values[batch_idx]

                # Per-minibatch advantage normalization (★ key trick)
                if self.use_adv_norm:
                    batch_adv = (batch_adv - batch_adv.mean()) / (batch_adv.std() + 1e-8)

                new_log_probs, entropy = self.actor.evaluate_actions(batch_obs, batch_actions)
                new_values = self.critic(batch_obs).squeeze(-1)

                ratio = torch.exp(new_log_probs - batch_old_lp)

                # Policy loss (PPO clip)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                clip_frac_val = ((ratio - 1).abs() > self.eps_clip).float().mean().item()

                # Value loss
                if self.use_vclip:
                    v_clipped = batch_old_val + torch.clamp(
                        new_values - batch_old_val, -self.vclip_eps, self.vclip_eps
                    )
                    value_loss = 0.5 * torch.max(
                        (new_values - batch_ret) ** 2,
                        (v_clipped - batch_ret) ** 2,
                    ).mean()
                else:
                    value_loss = 0.5 * ((new_values - batch_ret) ** 2).mean()

                entropy_loss = -entropy.mean()

                # Combined loss (CleanRL style: single optimizer)
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()

                metrics["value_loss"] += value_loss.item()
                metrics["policy_loss"] += policy_loss.item()
                metrics["entropy_loss"] += entropy_loss.item()
                metrics["approx_kl"] += approx_kl
                metrics["clip_frac"] += clip_frac_val
                update_count += 1

        if update_count > 0:
            for k in ["value_loss", "policy_loss", "entropy_loss", "approx_kl", "clip_frac"]:
                metrics[k] /= update_count

        # Explained variance
        with torch.no_grad():
            y_pred = old_values.cpu().numpy()
            y_true = returns.cpu().numpy()
            var_y = np.var(y_true)
            ev = 1 - np.var(y_true - y_pred) / (var_y + 1e-8)
            metrics["explained_variance"] = float(ev)

        return metrics


class OptimalHCGAE(OptimalPPO):
    """
    HCGAE built on top of OptimalPPO.

    Shares ALL implementation tricks from OptimalPPO (obs norm, adv norm,
    LR annealing, etc.) and adds the HCGAE GAE correction on top.

    This is the fair comparison: OptimalPPO vs OptimalHCGAE, where the
    ONLY difference is the GAE computation method.
    """

    NAME = "Optimal_HCGAE"

    def __init__(
        self,
        env: gym.Env,
        name: str = "Optimal_HCGAE",
        # HCGAE-specific parameters
        hindsight_beta: float = 3.0,
        hindsight_alpha_max: float = 0.7,
        hindsight_alpha_min: float = 0.1,
        use_scr_adapt: bool = False,  # SCR-adaptive correction strength
        scr_threshold: float = 1.0,
        scr_min_scale: float = 0.1,
        # All OptimalPPO parameters
        **kwargs,
    ):
        super().__init__(env=env, name=name, **kwargs)
        self.hindsight_beta = hindsight_beta
        self.hindsight_alpha_max = hindsight_alpha_max
        self.hindsight_alpha_min = hindsight_alpha_min
        self.use_scr_adapt = use_scr_adapt
        self.scr_threshold = scr_threshold
        self.scr_min_scale = scr_min_scale

        # EV tracking for adaptive alpha_max
        self._ev_ema = 0.0
        self._ev_ema_alpha = 0.05
        self._total_rollouts = 0

        # SCR tracking
        self._scr_ema = 1.0
        self._scr_ema_alpha = 0.1

    def compute_hindsight_gae(self, last_value: float):
        """
        HCGAE: Hindsight-Corrected GAE computation.
        This is the key difference from standard PPO.
        """
        T = self.buffer.pos
        obs_arr = self.buffer.observations[:T]
        rewards = self.buffer.rewards[:T]
        terminated = self.buffer.terminated[:T]
        values = self.buffer.values[:T]

        # Step 1: Compute Monte Carlo returns (rollout returns)
        returns_mc = np.zeros(T, dtype=np.float32)
        running_return = last_value
        for t in reversed(range(T)):
            if terminated[t]:
                running_return = 0.0
            running_return = rewards[t] + self.gamma * running_return
            returns_mc[t] = running_return

        # Step 2: Compute per-step error |V(s_t) - G_t|
        errors = np.abs(values - returns_mc)
        mu_e = errors.mean()
        sigma_e = errors.std() + 1e-8

        # Step 3: EV-gated adaptive alpha_max (with cosine annealing)
        self._total_rollouts += 1
        K = max(1, self._total_timesteps // self.n_steps)
        k = self._total_rollouts
        cosine_factor = 0.5 * (1 + np.cos(np.pi * k / K))
        ev_gate = max(1.0 - self._ev_ema, 0.2)
        alpha_max_k = (
            self.hindsight_alpha_min
            + (self.hindsight_alpha_max - self.hindsight_alpha_min)
            * cosine_factor * ev_gate
        )

        # SCR-based scaling (optional)
        if self.use_scr_adapt:
            # Estimate SCR: |bias| / std[G]
            bias_est = np.abs(values - returns_mc).mean()
            var_G = np.var(returns_mc) + 1e-8
            scr_raw = bias_est / (var_G ** 0.5)
            self._scr_ema = (1 - self._scr_ema_alpha) * self._scr_ema + self._scr_ema_alpha * scr_raw
            scr_scale = np.clip(self._scr_ema / self.scr_threshold, self.scr_min_scale, 1.0)
            alpha_max_k *= scr_scale

        # Step 4: Compute per-step alpha_t (batch-normalized sigmoid)
        z = self.hindsight_beta * (errors - mu_e) / sigma_e
        sigmoid_z = 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
        alpha = alpha_max_k * sigmoid_z

        # Step 5: Compute corrected V^c(s_t) = (1 - alpha) * V + alpha * G
        v_corrected = (1 - alpha) * values + alpha * returns_mc

        # Step 6: Corrected bootstrap value at boundary
        v_corrected_next_last = last_value  # boundary bootstrap (use raw V for boundary)

        # Step 7: Compute corrected GAE
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - terminated[t]
                next_v_c = v_corrected_next_last
            else:
                next_non_terminal = 1.0 - terminated[t]
                next_v_c = v_corrected[t + 1]
            delta = rewards[t] + self.gamma * next_v_c * next_non_terminal - v_corrected[t]
            last_gae = delta + self.gamma * self.lam * next_non_terminal * last_gae
            advantages[t] = last_gae

        # Step 8: Compute critic targets
        # Use EV-driven MC mixing for critic target
        # c_mc = clip(1 - EV, 0.1, 1): EV low → more MC; EV high → more GAE returns
        # Lower bound 0.1 ensures we always retain at least 10% MC (matches paper §2.2 and hindsight_ppo.py)
        c_mc = float(np.clip(1.0 - self._ev_ema, 0.1, 1.0))
        gae_returns = advantages + values  # standard GAE returns
        critic_returns = c_mc * returns_mc + (1 - c_mc) * gae_returns

        # Store in buffer
        self.buffer.advantages[:T] = advantages
        self.buffer.returns[:T] = critic_returns

        # Update EV EMA (based on current rollout)
        var_y = np.var(returns_mc) + 1e-8
        ev_now = float(1.0 - np.var(returns_mc - values) / var_y)
        self._ev_ema = (1 - self._ev_ema_alpha) * self._ev_ema + self._ev_ema_alpha * ev_now

    def compute_gae(self, last_value: float):
        """Override to use HCGAE instead of standard GAE."""
        self.compute_hindsight_gae(last_value)


class OptimalHCGAE_v2(OptimalHCGAE):
    """
    HCGAE v2 built on OptimalPPO — all fixes applied.

    Improvements over OptimalHCGAE (v1):
    ① Boundary bootstrap correction (from hindsight_ppo.py):
       last_value_corrected = (1-α_last)*V(sT) + α_last*G_{T-1}
       Eliminates the inconsistency at the rollout boundary step.

    ② EV Growth-Rate Gate (new, for HalfCheetah fix):
       Suppress HCGAE correction when EV is RISING FAST (Critic converging quickly).
       ev_rate = (ev_now - ev_prev) / rollout_interval
       If ev_rate > ev_rate_threshold: scale = max(1 - ev_rate / ev_rate_max, min_scale)
       Physical: if EV jumps >0.05/rollout, Critic is already learning rapidly;
       HCGAE's MC correction adds noise without benefit.

    ③ c_mc lower bound = 0.1 (not 0.0, matches paper §2.2).

    All three improvements are individually switchable for ablation.
    """

    NAME = "Optimal_HCGAE_v2"

    def __init__(
        self,
        env: gym.Env,
        name: str = "Optimal_HCGAE_v2",
        # v2-specific: boundary bootstrap correction
        use_boundary_correction: bool = True,
        # v2-specific: EV growth rate gate
        use_ev_rate_gate: bool = True,
        ev_rate_threshold: float = 0.05,   # suppress if EV grows > 5%/rollout
        ev_rate_max: float = 0.15,         # full suppression at 15%/rollout
        ev_gate_min_scale: float = 0.1,    # minimum scale (never fully zero)
        # All OptimalHCGAE parameters
        **kwargs,
    ):
        super().__init__(env=env, name=name, **kwargs)
        self.use_boundary_correction = use_boundary_correction
        self.use_ev_rate_gate = use_ev_rate_gate
        self.ev_rate_threshold = ev_rate_threshold
        self.ev_rate_max = ev_rate_max
        self.ev_gate_min_scale = ev_gate_min_scale

        # EV growth rate tracking
        self._ev_prev = 0.0
        self._ev_rate_ema = 0.0
        self._ev_rate_ema_alpha = 0.2

    def compute_hindsight_gae(self, last_value: float):
        """HCGAE v2: all fixes applied."""
        T = self.buffer.pos
        rewards = self.buffer.rewards[:T]
        terminated = self.buffer.terminated[:T]
        values = self.buffer.values[:T]

        # Step 1: MC returns
        returns_mc = np.zeros(T, dtype=np.float32)
        running_return = last_value
        for t in reversed(range(T)):
            if terminated[t]:
                running_return = 0.0
            running_return = rewards[t] + self.gamma * running_return
            returns_mc[t] = running_return

        # Step 2: Per-step error (batch-centred sigmoid)
        errors = np.abs(values - returns_mc)
        mu_e = errors.mean()
        sigma_e = errors.std() + 1e-8

        # Step 3: EV-gated adaptive alpha_max (cosine annealing)
        self._total_rollouts += 1
        K = max(1, self._total_timesteps // self.n_steps)
        k = self._total_rollouts
        cosine_factor = 0.5 * (1 + np.cos(np.pi * k / K))
        ev_gate = max(1.0 - self._ev_ema, 0.2)
        alpha_max_k = (
            self.hindsight_alpha_min
            + (self.hindsight_alpha_max - self.hindsight_alpha_min)
            * cosine_factor * ev_gate
        )

        # ── v2 FIX ②: EV growth-rate gate
        # If the Critic is converging rapidly, suppress MC correction.
        # Physical: fast EV growth → Critic already learning → MC adds noise.
        if self.use_ev_rate_gate:
            var_y_tmp = np.var(returns_mc) + 1e-8
            ev_now_tmp = float(1.0 - np.var(returns_mc - values) / var_y_tmp)
            ev_rate_raw = ev_now_tmp - self._ev_prev  # per-rollout EV increase
            self._ev_rate_ema = ((1 - self._ev_rate_ema_alpha) * self._ev_rate_ema
                                 + self._ev_rate_ema_alpha * max(ev_rate_raw, 0.0))
            # Scale: 1.0 if ev_rate <= threshold; decreasing to min_scale if rate >= max
            if self._ev_rate_ema > self.ev_rate_threshold:
                excess = min(self._ev_rate_ema - self.ev_rate_threshold,
                             self.ev_rate_max - self.ev_rate_threshold)
                suppression_frac = excess / max(self.ev_rate_max - self.ev_rate_threshold, 1e-8)
                ev_rate_scale = max(1.0 - suppression_frac * (1.0 - self.ev_gate_min_scale),
                                    self.ev_gate_min_scale)
            else:
                ev_rate_scale = 1.0
            alpha_max_k *= ev_rate_scale
        else:
            ev_now_tmp = None
            ev_rate_scale = 1.0

        # SCR-based scaling (optional, from v1)
        if self.use_scr_adapt:
            bias_est = np.abs(values - returns_mc).mean()
            var_G = np.var(returns_mc) + 1e-8
            scr_raw = bias_est / (var_G ** 0.5)
            self._scr_ema = (1 - self._scr_ema_alpha) * self._scr_ema + self._scr_ema_alpha * scr_raw
            scr_scale = np.clip(self._scr_ema / self.scr_threshold, self.scr_min_scale, 1.0)
            alpha_max_k *= scr_scale

        # Step 4: Per-step alpha (batch-centred sigmoid)
        z = self.hindsight_beta * (errors - mu_e) / sigma_e
        sigmoid_z = 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
        alpha = alpha_max_k * sigmoid_z

        # Step 5: Corrected V^c
        v_corrected = (1 - alpha) * values + alpha * returns_mc

        # ── v2 FIX ①: Boundary bootstrap correction
        # Instead of using raw last_value, apply error-gated blend at boundary.
        if self.use_boundary_correction:
            tail_n = min(10, T)
            approx_err_last = float(errors[-tail_n:].mean())
            alpha_last = alpha_max_k * (1.0 / (1.0 + np.exp(
                -self.hindsight_beta * (approx_err_last - mu_e) / sigma_e
            )))
            approx_G_last = returns_mc[-1]  # last MC return as conservative estimate
            last_value_corrected = (1.0 - alpha_last) * last_value + alpha_last * approx_G_last
        else:
            last_value_corrected = last_value

        # Step 6: Corrected GAE (using v2 boundary)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - terminated[t]
                next_v_c = last_value_corrected if next_non_terminal > 0 else 0.0
            else:
                next_non_terminal = 1.0 - terminated[t]
                next_v_c = v_corrected[t + 1] if next_non_terminal > 0 else 0.0
            delta = rewards[t] + self.gamma * next_v_c - v_corrected[t]
            last_gae = delta + self.gamma * self.lam * next_non_terminal * last_gae
            advantages[t] = last_gae

        # Step 7: Critic targets (EV-driven, c_mc lower bound = 0.1)
        c_mc = float(np.clip(1.0 - self._ev_ema, 0.1, 1.0))
        gae_returns = advantages + values
        critic_returns = c_mc * returns_mc + (1 - c_mc) * gae_returns

        self.buffer.advantages[:T] = advantages
        self.buffer.returns[:T] = critic_returns

        # Update EV EMA
        var_y = np.var(returns_mc) + 1e-8
        ev_now = float(1.0 - np.var(returns_mc - values) / var_y)
        self._ev_prev = self._ev_ema  # track previous EV for rate gate
        self._ev_ema = (1 - self._ev_ema_alpha) * self._ev_ema + self._ev_ema_alpha * ev_now

    def compute_gae(self, last_value: float):
        self.compute_hindsight_gae(last_value)


def build_optimal_agent(
    algo_name: str,
    env: gym.Env,
    name: str = None,
    **kwargs
):
    """Factory function for building optimal PPO variants."""
    name = name or algo_name

    # Common kwargs for OptimalPPO
    opt_defaults = dict(
        hidden_dim=64,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        eps_clip=0.2,
        n_epochs=10,
        batch_size=64,
        n_steps=2048,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_obs_norm=True,
        use_adv_norm=True,
        use_lr_anneal=True,
        use_vclip=False,
        device="cpu",
    )
    # Override with user kwargs
    for k, v in kwargs.items():
        opt_defaults[k] = v

    if algo_name == "Optimal_PPO":
        return OptimalPPO(env=env, name=name, **opt_defaults)

    elif algo_name == "Optimal_PPO_VClip":
        cfg = dict(opt_defaults)
        cfg['use_vclip'] = True
        return OptimalPPO(env=env, name=name, **cfg)

    elif algo_name == "Optimal_PPO_NoObsNorm":
        # Ablation: no obs normalization
        cfg = dict(opt_defaults)
        cfg['use_obs_norm'] = False
        return OptimalPPO(env=env, name=name, **cfg)

    elif algo_name == "Optimal_HCGAE":
        hcgae_kwargs = {k: v for k, v in opt_defaults.items()}
        return OptimalHCGAE(
            env=env, name=name,
            hindsight_beta=3.0,
            hindsight_alpha_max=0.7,
            hindsight_alpha_min=0.1,
            use_scr_adapt=False,
            **hcgae_kwargs
        )

    elif algo_name == "Optimal_HCGAE_SCR":
        hcgae_kwargs = {k: v for k, v in opt_defaults.items()}
        return OptimalHCGAE(
            env=env, name=name,
            hindsight_beta=3.0,
            hindsight_alpha_max=0.7,
            hindsight_alpha_min=0.1,
            use_scr_adapt=True,
            scr_threshold=1.0,
            scr_min_scale=0.1,
            **hcgae_kwargs
        )

    elif algo_name == "Optimal_HCGAE_v2":
        # v2: boundary bootstrap correction + EV growth-rate gate (all fixes)
        hcgae_kwargs = {k: v for k, v in opt_defaults.items()}
        return OptimalHCGAE_v2(
            env=env, name=name,
            hindsight_beta=3.0,
            hindsight_alpha_max=0.7,
            hindsight_alpha_min=0.1,
            use_scr_adapt=False,
            use_boundary_correction=True,
            use_ev_rate_gate=True,
            ev_rate_threshold=0.05,
            ev_rate_max=0.15,
            ev_gate_min_scale=0.1,
            **hcgae_kwargs
        )

    elif algo_name == "Optimal_HCGAE_v2_NoBdry":
        # Ablation: v2 without boundary correction (EV gate only)
        hcgae_kwargs = {k: v for k, v in opt_defaults.items()}
        return OptimalHCGAE_v2(
            env=env, name=name,
            hindsight_beta=3.0,
            hindsight_alpha_max=0.7,
            hindsight_alpha_min=0.1,
            use_scr_adapt=False,
            use_boundary_correction=False,
            use_ev_rate_gate=True,
            ev_rate_threshold=0.05,
            ev_rate_max=0.15,
            ev_gate_min_scale=0.1,
            **hcgae_kwargs
        )

    elif algo_name == "Optimal_HCGAE_v2_NoGate":
        # Ablation: v2 without EV rate gate (boundary correction only)
        hcgae_kwargs = {k: v for k, v in opt_defaults.items()}
        return OptimalHCGAE_v2(
            env=env, name=name,
            hindsight_beta=3.0,
            hindsight_alpha_max=0.7,
            hindsight_alpha_min=0.1,
            use_scr_adapt=False,
            use_boundary_correction=True,
            use_ev_rate_gate=False,
            **hcgae_kwargs
        )

    else:
        raise ValueError(f"Unknown optimal agent: {algo_name}")

