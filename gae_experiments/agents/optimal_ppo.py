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
        hidden_dim: int = 256,
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
        # Use EV-driven MC mixing for critic target.
        # CRITICAL: Critic target must use standard GAE returns computed from the
        # ORIGINAL (uncorrected) V, not advantages (computed from V_corrected) + values.
        # Using advantages + values would mix V_corrected-based deltas with V(s_t),
        # which is mathematically inconsistent.
        # c_mc = clip(1 - EV, 0.1, 1): EV low → more MC; EV high → more GAE returns
        c_mc = float(np.clip(1.0 - self._ev_ema, 0.1, 1.0))
        std_gae_returns = self.buffer._compute_standard_returns(last_value, self.gamma, self.lam)
        critic_returns = c_mc * returns_mc + (1 - c_mc) * std_gae_returns

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
        # CRITICAL: Use standard GAE returns from ORIGINAL (uncorrected) V,
        # not advantages (V_corrected-based) + values (original V).
        # This ensures the two update paths (advantage estimation vs critic training)
        # remain fully decoupled, as required by the paper §2.4.
        c_mc = float(np.clip(1.0 - self._ev_ema, 0.1, 1.0))
        std_gae_returns = self.buffer._compute_standard_returns(last_value, self.gamma, self.lam)
        critic_returns = c_mc * returns_mc + (1 - c_mc) * std_gae_returns

        self.buffer.advantages[:T] = advantages
        self.buffer.returns[:T] = critic_returns

        # Update EV EMA
        var_y = np.var(returns_mc) + 1e-8
        ev_now = float(1.0 - np.var(returns_mc - values) / var_y)
        self._ev_prev = self._ev_ema  # track previous EV for rate gate
        self._ev_ema = (1 - self._ev_ema_alpha) * self._ev_ema + self._ev_ema_alpha * ev_now

    def compute_gae(self, last_value: float):
        self.compute_hindsight_gae(last_value)


class OptimalHCGAE_v2_AutoSCR(OptimalHCGAE_v2):
    """
    HCGAE v2 with Automatic SCR-Based Mode Detection.

    This implements the "Future Work 1" direction from the paper:
    An automatic mode detector that estimates SCR = |B_t| / Var[G_t]^{1/2}
    online and uses it to jointly gate correction strength alongside the
    EV growth-rate gate from v2.

    Key improvements over v2:
    ─────────────────────────
    ① Joint gating: correction scale = min(ev_rate_scale, scr_scale)
       Both gates must agree before allowing strong correction.
       - EV rate gate: Critic already converging fast → suppress
       - SCR gate:     MC noise > Critic bias → suppress

    ② Improved SCR estimator:
       SCR_raw = |bias_est| / (std[G_t] + eps)
       where bias_est uses MEDIAN absolute error (robust to outliers)
       and a 0.05-quantile of |V-G| as the low-bias floor.

    ③ Adaptive SCR threshold with EV adjustment:
       As EV rises, bias typically falls, so SCR naturally increases.
       We correct for this: scr_adjusted = SCR_raw * sqrt(1 - EV_ema)
       so that the threshold stays meaningful across training.

    ④ Smooth transition: scr_scale = sigmoid((scr_ema - scr_threshold) * scr_sharpness)
       rather than hard clip, giving a softer, more stable gate.

    Parameters
    ----------
    scr_threshold : float
        If SCR_adjusted < threshold, gate is < 0.5 → suppress correction.
        Default: 1.0 (same as paper's SCR framework §5.1).
    scr_min_scale : float
        Minimum correction scale from SCR gate (never fully suppress).
        Default: 0.15 (slightly higher than v2's scr_min=0.1 to avoid over-suppression).
    scr_sharpness : float
        Sigmoid sharpness for SCR gate. Higher → sharper transition.
        Default: 3.0.
    use_auto_scr : bool
        Toggle automatic SCR gate on/off (for ablation).
        Default: True.
    """

    NAME = "Optimal_HCGAE_v2_AutoSCR"

    def __init__(
        self,
        env: gym.Env,
        name: str = "Optimal_HCGAE_v2_AutoSCR",
        # AutoSCR-specific parameters
        scr_threshold: float = 1.0,
        scr_min_scale: float = 0.15,
        scr_sharpness: float = 3.0,
        use_auto_scr: bool = True,
        # All OptimalHCGAE_v2 parameters
        **kwargs,
    ):
        super().__init__(env=env, name=name, **kwargs)
        self.scr_threshold = scr_threshold
        self.scr_min_scale = scr_min_scale
        self.scr_sharpness = scr_sharpness
        self.use_auto_scr = use_auto_scr

        # SCR tracking (EMA)
        self._scr_ema = 1.0
        self._scr_ema_alpha = 0.1
        # History for diagnostics
        self._scr_history = []
        self._scr_scale_history = []

    def _estimate_scr(self, values: np.ndarray, returns_mc: np.ndarray) -> float:
        """
        Robust online SCR estimator.

        SCR = |bias| / std[G]
        - bias = robust Critic bias (median absolute error, with 5%-quantile floor)
        - std[G] = std of MC returns (measure of MC noise level)

        EV-adjusted: scr_adj = scr_raw * sqrt(max(1 - EV_ema, 0.05))
        This normalizes for the fact that bias naturally falls as EV rises.
        """
        errors = np.abs(values - returns_mc)
        # Robust bias: use median + 5th percentile blend to avoid outlier domination
        median_err = float(np.median(errors))
        p05_err = float(np.percentile(errors, 5))
        bias_est = 0.7 * median_err + 0.3 * p05_err  # blend for robustness

        std_G = float(np.std(returns_mc)) + 1e-8
        scr_raw = bias_est / std_G

        # EV adjustment: scale by sqrt(1 - EV_ema) so threshold is EV-invariant
        ev_factor = float(np.sqrt(max(1.0 - self._ev_ema, 0.05)))
        scr_adjusted = scr_raw * ev_factor

        return scr_adjusted

    def _scr_gate_scale(self, scr_adjusted: float) -> float:
        """
        Compute correction scale from SCR gate.

        Uses sigmoid for smooth transition:
            scr_scale = scr_min + (1 - scr_min) * sigmoid(sharpness * (scr - threshold))

        When SCR >> threshold: scr_scale ≈ 1.0 (correction is beneficial)
        When SCR << threshold: scr_scale ≈ scr_min (suppress, MC noise dominates)
        """
        z = self.scr_sharpness * (scr_adjusted - self.scr_threshold)
        sigmoid_z = 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
        scr_scale = self.scr_min_scale + (1.0 - self.scr_min_scale) * sigmoid_z
        return float(scr_scale)

    def compute_hindsight_gae(self, last_value: float):
        """HCGAE v2 + AutoSCR gate: joint EV-rate + SCR gating."""
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

        # Step 3: EV-gated adaptive alpha_max (cosine annealing) — from v2
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

        # ── Gate A: EV growth-rate gate (from v2)
        if self.use_ev_rate_gate:
            var_y_tmp = np.var(returns_mc) + 1e-8
            ev_now_tmp = float(1.0 - np.var(returns_mc - values) / var_y_tmp)
            ev_rate_raw = ev_now_tmp - self._ev_prev
            self._ev_rate_ema = ((1 - self._ev_rate_ema_alpha) * self._ev_rate_ema
                                 + self._ev_rate_ema_alpha * max(ev_rate_raw, 0.0))
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

        # ── Gate B: Automatic SCR gate (new in AutoSCR)
        if self.use_auto_scr:
            scr_adjusted = self._estimate_scr(values, returns_mc)
            # Update SCR EMA
            self._scr_ema = ((1 - self._scr_ema_alpha) * self._scr_ema
                             + self._scr_ema_alpha * scr_adjusted)
            scr_scale = self._scr_gate_scale(self._scr_ema)

            # Joint gating: use MINIMUM of both gates
            # Both must agree that correction is beneficial
            alpha_max_k *= scr_scale
            # Track diagnostics
            self._scr_history.append(float(self._scr_ema))
            self._scr_scale_history.append(scr_scale)
        else:
            scr_scale = 1.0

        # Step 4: Per-step alpha (batch-centred sigmoid)
        z = self.hindsight_beta * (errors - mu_e) / sigma_e
        sigmoid_z = 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
        alpha = alpha_max_k * sigmoid_z

        # Step 5: Corrected V^c
        v_corrected = (1 - alpha) * values + alpha * returns_mc

        # ── v2 FIX ①: Boundary bootstrap correction
        if self.use_boundary_correction:
            tail_n = min(10, T)
            approx_err_last = float(errors[-tail_n:].mean())
            alpha_last = alpha_max_k * (1.0 / (1.0 + np.exp(
                -self.hindsight_beta * (approx_err_last - mu_e) / sigma_e
            )))
            approx_G_last = returns_mc[-1]
            last_value_corrected = (1.0 - alpha_last) * last_value + alpha_last * approx_G_last
        else:
            last_value_corrected = last_value

        # Step 6: Corrected GAE
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
        # Use standard GAE returns from ORIGINAL (uncorrected) V, to keep
        # advantage estimation and critic training paths fully decoupled.
        c_mc = float(np.clip(1.0 - self._ev_ema, 0.1, 1.0))
        std_gae_returns = self.buffer._compute_standard_returns(last_value, self.gamma, self.lam)
        critic_returns = c_mc * returns_mc + (1 - c_mc) * std_gae_returns

        self.buffer.advantages[:T] = advantages
        self.buffer.returns[:T] = critic_returns

        # Update EV EMA
        var_y = np.var(returns_mc) + 1e-8
        ev_now = float(1.0 - np.var(returns_mc - values) / var_y)
        self._ev_prev = self._ev_ema
        self._ev_ema = (1 - self._ev_ema_alpha) * self._ev_ema + self._ev_ema_alpha * ev_now

    def compute_gae(self, last_value: float):
        self.compute_hindsight_gae(last_value)


class OptimalHCGAE_v3(OptimalHCGAE_v2):
    """
    HCGAE v3: High-Variance MC Returns Adaptation (Ant-v4 Fix).

    Root cause of HCGAE failure on Ant-v4 (from experimental analysis):
    ─────────────────────────────────────────────────────────────────────
    Ant-v4 has FUNDAMENTALLY different reward structure vs Hopper/Walker:
      • Episode reward CV = 16.47  (Hopper: 0.93) — 17× more variable
      • Negative episode rate = 60%  (Hopper: 0%)
      • SNR (|mean|/std) = 0.06  (Hopper: 1.08)

    During early training, G_t (MC return) is dominated by large negative
    values from falling episodes. HCGAE uses G_t to "correct" V(s), but
    when G_t is highly noisy and negative:
      1. V^c = (1-α)V + α*G pulls V toward large negatives (pessimism bias)
      2. Pessimistic V^c → underestimated advantages → wrong gradient direction
      3. Boundary correction amplifies this: last_value_corrected is dragged
         toward negative G (returns_mc[-1] is often negative early on)

    Mathematical Derivation
    ─────────────────────────────────────────────────────────────────────
    Standard HCGAE assumes:
        MSE_correction[V^c] < MSE_correction[V]
    i.e., bias reduction > variance increase.

    This holds when: α² · Var[G] < (1-α)² · Bias[V]²
    i.e., SCR = |Bias[V]| / Std[G] > α / (1-α)

    For Ant-v4 early training:
        Bias[V] ≈ |V(0) - E[G]| ≈ |0 - 25| = 25
        Std[G] ≈ 393
        SCR_actual ≈ 25/393 = 0.064

    For any α > SCR/(1+SCR) = 0.064/1.064 ≈ 0.06, correction HURTS!
    But HCGAE uses α_max = 0.7 → massive over-correction.

    v3 Fixes (Three complementary mechanisms)
    ─────────────────────────────────────────────────────────────────────
    FIX ①: Positive-Only G Clamping for V correction
        Replace: v_corrected = (1-α)*V + α*G
        With:    v_corrected = (1-α)*V + α*max(G, V - margin)
        where margin = k_margin * Std[G_positive] (computed from positive-G timesteps)
        Physical: If G_t is much worse than V(s_t) (by more than k_margin * Std[G+]),
                  it likely reflects unlucky noise, not true V bias.
                  Use "loss-clamp": don't let MC pull V below V - margin.

        Key insight: This is NOT simply clamping G. It's setting a dynamic floor
        per-timestep based on the positive-episode return distribution.
        G_t ≥ V(s_t) - margin → correction is meaningful
        G_t < V(s_t) - margin → likely noise, clamp to avoid pessimism injection

    FIX ②: Variance-Weighted Alpha Scaling (VW Gate)
        Replace: alpha_max_k (fixed from EV/cosine)
        With:    alpha_max_k *= vw_scale
        where:   vw_scale = clip(snr_ema / snr_target, snr_min_scale, 1.0)
                 snr_ema = EMA(|mean_delta| / std_delta)
                 snr_target = target SNR for full-strength correction

        Physical: When MC returns are very noisy (low SNR), automatically
                  reduce correction strength. This generalizes the EV gate
                  to handle high-variance environments beyond just HalfCheetah.

        Math: The optimal α* = max(0, 1 - Var[G] / (2·Bias[V]²))
              In practice, we use SNR as a proxy: low SNR → low α*

    FIX ③: Boundary Correction with Positive-Return Prior
        Replace: approx_G_last = returns_mc[-1]  (can be very negative!)
        With:    approx_G_last = (pos_weight * mean_G_positive + neg_weight * returns_mc[-1])
        where pos_weight = fraction of positive G in rollout
        Physical: For boundary correction, blend in the positive-return
                  distribution as a prior, weighted by how often G>0.
                  This prevents boundary from being dragged down by single
                  large negative G[-1].

    Parameters
    ─────────────────────────────────────────────────────────────────────
    use_g_clamp: bool
        Enable G clamping for V correction (FIX ①). Default True.
    g_clamp_margin_k: float
        Number of G+ std-devs below V to allow. Default 1.5.
        Higher → more permissive (less clamping). Lower → stricter.
    use_vw_gate: bool
        Enable variance-weighted alpha scaling (FIX ②). Default True.
    snr_target: float
        Target SNR for full-strength correction. Default 0.5.
        SNR < snr_target → partial suppression. SNR >> snr_target → full correction.
    snr_min_scale: float
        Minimum correction scale from VW gate. Default 0.1.
    snr_ema_alpha: float
        EMA smoothing for SNR estimate. Default 0.05.
    use_boundary_prior: bool
        Enable positive-return prior for boundary correction (FIX ③). Default True.
    """

    NAME = "Optimal_HCGAE_v3"

    def __init__(
        self,
        env: gym.Env,
        name: str = "Optimal_HCGAE_v3",
        # FIX ①: G clamping
        use_g_clamp: bool = True,
        g_clamp_margin_k: float = 1.5,   # allow G down to V - k*Std[G+]
        # FIX ②: Variance-weighted gate
        use_vw_gate: bool = True,
        snr_target: float = 0.5,          # full correction when SNR >= snr_target
        snr_min_scale: float = 0.1,       # never fully suppress
        snr_ema_alpha: float = 0.05,      # EMA smoothing for SNR
        # FIX ③: Boundary positive-return prior
        use_boundary_prior: bool = True,
        # All OptimalHCGAE_v2 parameters (keep EV rate gate & boundary correction)
        **kwargs,
    ):
        super().__init__(env=env, name=name, **kwargs)
        self.use_g_clamp = use_g_clamp
        self.g_clamp_margin_k = g_clamp_margin_k
        self.use_vw_gate = use_vw_gate
        self.snr_target = snr_target
        self.snr_min_scale = snr_min_scale
        self.snr_ema_alpha = snr_ema_alpha
        self.use_boundary_prior = use_boundary_prior

        # Online SNR tracking
        self._snr_ema = snr_target  # initialize at target (neutral)
        # Online tracking of positive-G statistics
        self._g_pos_std_ema = 1.0
        self._g_pos_mean_ema = 0.0
        self._g_pos_frac_ema = 0.5

    def compute_hindsight_gae(self, last_value: float):
        """HCGAE v3: HV-MC adaptation for Ant-v4 style environments."""
        T = self.buffer.pos
        rewards = self.buffer.rewards[:T]
        terminated = self.buffer.terminated[:T]
        values = self.buffer.values[:T]

        # Step 1: MC returns (standard)
        returns_mc = np.zeros(T, dtype=np.float32)
        running_return = last_value
        for t in reversed(range(T)):
            if terminated[t]:
                running_return = 0.0
            running_return = rewards[t] + self.gamma * running_return
            returns_mc[t] = running_return

        # Step 2: Per-step error (batch-centred sigmoid) — same as v2
        errors = np.abs(values - returns_mc)
        mu_e = errors.mean()
        sigma_e = errors.std() + 1e-8

        # ──────────────────────────────────────────────────────────
        # FIX ②: Online SNR estimation & variance-weighted gate
        # SNR = |mean(δ)| / std(δ)  where δ = G_t - V(s_t)
        # ──────────────────────────────────────────────────────────
        deltas = returns_mc - values
        delta_mean = float(np.mean(deltas))
        delta_std = float(np.std(deltas)) + 1e-8
        snr_now = abs(delta_mean) / delta_std  # current rollout SNR

        # EMA of SNR (slow-moving estimate of environment SNR)
        self._snr_ema = ((1 - self.snr_ema_alpha) * self._snr_ema
                         + self.snr_ema_alpha * snr_now)

        if self.use_vw_gate:
            # vw_scale: 0 when SNR → 0, 1 when SNR ≥ snr_target
            # Use sigmoid-like: clip(SNR_ema / snr_target, snr_min, 1.0)
            vw_scale = float(np.clip(self._snr_ema / self.snr_target,
                                     self.snr_min_scale, 1.0))
        else:
            vw_scale = 1.0

        # ──────────────────────────────────────────────────────────
        # Online positive-G statistics (needed for FIX ① and FIX ③)
        # ──────────────────────────────────────────────────────────
        pos_mask = returns_mc > 0
        pos_frac = float(pos_mask.mean())
        if pos_mask.sum() > 1:
            g_pos_std = float(np.std(returns_mc[pos_mask]))
            g_pos_mean = float(np.mean(returns_mc[pos_mask]))
        else:
            g_pos_std = float(np.std(np.abs(returns_mc))) + 1e-8
            g_pos_mean = float(np.mean(returns_mc))

        # Smooth positive-G statistics
        ema_a = 0.1
        self._g_pos_std_ema = (1 - ema_a) * self._g_pos_std_ema + ema_a * g_pos_std
        self._g_pos_mean_ema = (1 - ema_a) * self._g_pos_mean_ema + ema_a * g_pos_mean
        self._g_pos_frac_ema = (1 - ema_a) * self._g_pos_frac_ema + ema_a * pos_frac

        # Step 3: EV-gated adaptive alpha_max (cosine annealing) — same as v2
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

        # EV growth-rate gate (from v2) — unchanged
        if self.use_ev_rate_gate:
            var_y_tmp = np.var(returns_mc) + 1e-8
            ev_now_tmp = float(1.0 - np.var(returns_mc - values) / var_y_tmp)
            ev_rate_raw = ev_now_tmp - self._ev_prev
            self._ev_rate_ema = ((1 - self._ev_rate_ema_alpha) * self._ev_rate_ema
                                 + self._ev_rate_ema_alpha * max(ev_rate_raw, 0.0))
            if self._ev_rate_ema > self.ev_rate_threshold:
                excess = min(self._ev_rate_ema - self.ev_rate_threshold,
                             self.ev_rate_max - self.ev_rate_threshold)
                suppression_frac = excess / max(self.ev_rate_max - self.ev_rate_threshold, 1e-8)
                ev_rate_scale = max(1.0 - suppression_frac * (1.0 - self.ev_gate_min_scale),
                                    self.ev_gate_min_scale)
            else:
                ev_rate_scale = 1.0
            alpha_max_k *= ev_rate_scale

        # ── Apply variance-weighted gate (FIX ②)
        alpha_max_k *= vw_scale

        # Step 4: Per-step alpha (batch-centred sigmoid) — same as v2
        z = self.hindsight_beta * (errors - mu_e) / sigma_e
        sigmoid_z = 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
        alpha = alpha_max_k * sigmoid_z

        # ──────────────────────────────────────────────────────────
        # FIX ①: G Clamping — prevent pessimism injection
        # Per-timestep: returns_mc_clamped[t] = max(returns_mc[t], V[t] - margin)
        # margin = g_clamp_margin_k * Std[G+]  (dynamic, based on positive returns)
        # ──────────────────────────────────────────────────────────
        if self.use_g_clamp:
            margin = self.g_clamp_margin_k * (self._g_pos_std_ema + 1e-8)
            # Clamp G_t to not go more than `margin` below V(s_t)
            returns_mc_clamped = np.maximum(returns_mc, values - margin)
        else:
            returns_mc_clamped = returns_mc

        # Step 5: Corrected V^c using clamped G
        v_corrected = (1 - alpha) * values + alpha * returns_mc_clamped

        # ──────────────────────────────────────────────────────────
        # FIX ③: Boundary correction with positive-return prior
        # ──────────────────────────────────────────────────────────
        if self.use_boundary_correction:
            tail_n = min(10, T)
            approx_err_last = float(errors[-tail_n:].mean())
            alpha_last = alpha_max_k * (1.0 / (1.0 + np.exp(
                -self.hindsight_beta * (approx_err_last - mu_e) / sigma_e
            )))
            # FIX ③: Blend G_last with positive-return prior
            raw_G_last = returns_mc[-1]
            if self.use_boundary_prior and self._g_pos_frac_ema > 0.1:
                # Weighted blend: more positive-G → more trust in raw G
                # If pos_frac low (bad environment): weight toward positive-G mean
                pos_w = self._g_pos_frac_ema  # fraction of positive G in history
                approx_G_last = (pos_w * raw_G_last
                                 + (1 - pos_w) * self._g_pos_mean_ema)
            else:
                approx_G_last = raw_G_last
            last_value_corrected = (1.0 - alpha_last) * last_value + alpha_last * approx_G_last
        else:
            last_value_corrected = last_value

        # Step 6: Corrected GAE (using v3 corrected V and boundary) — same structure as v2
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
        # Use clamped MC returns for MC component (avoids large negative targets in high-var envs).
        # Use standard GAE returns from ORIGINAL (uncorrected) V for GAE component,
        # to keep advantage estimation and critic training paths fully decoupled.
        c_mc = float(np.clip(1.0 - self._ev_ema, 0.1, 1.0))
        std_gae_returns = self.buffer._compute_standard_returns(last_value, self.gamma, self.lam)
        critic_returns = c_mc * returns_mc_clamped + (1 - c_mc) * std_gae_returns

        self.buffer.advantages[:T] = advantages
        self.buffer.returns[:T] = critic_returns

        # Update EV EMA (use original returns_mc for EV, not clamped version)
        var_y = np.var(returns_mc) + 1e-8
        ev_now = float(1.0 - np.var(returns_mc - values) / var_y)
        self._ev_prev = self._ev_ema
        self._ev_ema = (1 - self._ev_ema_alpha) * self._ev_ema + self._ev_ema_alpha * ev_now

    def compute_gae(self, last_value: float):
        self.compute_hindsight_gae(last_value)


class OptimalHCGAE_v4(OptimalHCGAE_v2):
    """
    HCGAE v4: SCR-Adaptive Optimal Correction Strength.

    Single targeted fix on top of v2: cap α_max at the MSE-optimal upper bound
    derived from the online Signal-Correction Ratio (SCR) estimate.

    Root cause analysis
    ───────────────────
    In v2, the EV growth-rate gate can suppress α_max down to s_min=0.1.
    But on large networks (256×256 MLP) where the Critic converges quickly,
    SCR = |Bias[V]| / Std[G] drops below 1.0 within ~50K steps.
    At that point even the residual α ≈ 0.05–0.10 adds MC noise that outweighs
    the bias reduction, causing the observed −30% on HalfCheetah.

    Mathematical Derivation of Optimal α*
    ──────────────────────────────────────
    Minimise MSE(α) = (1−α)²·B² + α²·σ_G²  w.r.t. α ∈ [0,1]:

        dMSE/dα = 0  →  α*(SCR) = SCR² / (1 + SCR²)

    where  SCR = |B| / σ_G = |mean(G−V)| / std(G).

    α*(SCR) is monotone increasing in SCR:
        SCR = 0.5 → α* = 0.20   (moderate)
        SCR = 1.0 → α* = 0.50   (balanced)
        SCR = 2.0 → α* = 0.80   (strong)

    v2's effective α_max ≈ 0.40–0.58 grossly exceeds α*(SCR) when SCR < 0.7.

    The Fix (one line of core logic)
    ─────────────────────────────────
    After the v2 EV-rate gate has computed α_max_k, apply one extra cap:

        α_max_v4(k) = min(α_max_v2(k),  SCR_ema²/(1+SCR_ema²) + scr_relax)

    where scr_relax=0.05 adds a small slack for estimation noise.
    Everything else (EV gate, boundary correction, critic mixing) is unchanged.

    Parameters
    ──────────────────────────────────────────────────────────────────────────
    scr_ema_alpha : float
        EMA smoothing rate for the online SCR estimate. Default 0.1.
    scr_relax : float
        Additive slack above the theoretical α* bound (estimation noise buffer).
        Default 0.05.  Set to 0 for the strict theoretical bound.
    """

    NAME = "Optimal_HCGAE_v4"

    def __init__(
        self,
        env: gym.Env,
        name: str = "Optimal_HCGAE_v4",
        scr_ema_alpha: float = 0.1,
        scr_relax: float = 0.05,
        # All OptimalHCGAE_v2 parameters (EV rate gate + boundary correction)
        **kwargs,
    ):
        super().__init__(env=env, name=name, **kwargs)
        self.scr_ema_alpha = scr_ema_alpha
        self.scr_relax = scr_relax

        # Online SCR tracking (EMA); initialise at 1.0 → α* = 0.50 (neutral)
        self._scr_ema = 1.0
        self._scr_history = []          # diagnostics

    def _scr_alpha_cap(self, values: np.ndarray, returns_mc: np.ndarray) -> float:
        """
        Compute the MSE-optimal α* cap from current rollout data.

        SCR_hat = |mean(G − V)| / (std(G) + ε)
        α*_cap  = SCR_ema² / (1 + SCR_ema²) + scr_relax
        """
        delta = returns_mc - values
        scr_hat = float(np.abs(np.mean(delta))) / (float(np.std(returns_mc)) + 1e-8)
        # EMA update
        self._scr_ema = (1 - self.scr_ema_alpha) * self._scr_ema + self.scr_ema_alpha * scr_hat
        self._scr_history.append(self._scr_ema)
        alpha_cap = (self._scr_ema ** 2) / (1.0 + self._scr_ema ** 2) + self.scr_relax
        return float(np.clip(alpha_cap, 0.0, 1.0))

    def compute_hindsight_gae(self, last_value: float):
        """HCGAE v4 = HCGAE v2 + one SCR-based α_max cap."""
        T = self.buffer.pos
        rewards = self.buffer.rewards[:T]
        terminated = self.buffer.terminated[:T]
        values = self.buffer.values[:T]

        # Step 1: MC returns (identical to v2)
        returns_mc = np.zeros(T, dtype=np.float32)
        running_return = last_value
        for t in reversed(range(T)):
            if terminated[t]:
                running_return = 0.0
            running_return = rewards[t] + self.gamma * running_return
            returns_mc[t] = running_return

        # Step 2: Per-step error (identical to v2)
        errors = np.abs(values - returns_mc)
        mu_e = errors.mean()
        sigma_e = errors.std() + 1e-8

        # Step 3a: EV-gated cosine α_max (identical to v2)
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

        # Step 3b: EV growth-rate gate (identical to v2)
        if self.use_ev_rate_gate:
            var_y_tmp = np.var(returns_mc) + 1e-8
            ev_now_tmp = float(1.0 - np.var(returns_mc - values) / var_y_tmp)
            ev_rate_raw = ev_now_tmp - self._ev_prev
            self._ev_rate_ema = ((1 - self._ev_rate_ema_alpha) * self._ev_rate_ema
                                 + self._ev_rate_ema_alpha * max(ev_rate_raw, 0.0))
            if self._ev_rate_ema > self.ev_rate_threshold:
                excess = min(self._ev_rate_ema - self.ev_rate_threshold,
                             self.ev_rate_max - self.ev_rate_threshold)
                suppression_frac = excess / max(self.ev_rate_max - self.ev_rate_threshold, 1e-8)
                ev_rate_scale = max(1.0 - suppression_frac * (1.0 - self.ev_gate_min_scale),
                                    self.ev_gate_min_scale)
            else:
                ev_rate_scale = 1.0
            alpha_max_k *= ev_rate_scale

        # ── v4 ADDITION: one SCR-based α_max cap ──────────────────────────────
        alpha_max_k = min(alpha_max_k, self._scr_alpha_cap(values, returns_mc))
        # ──────────────────────────────────────────────────────────────────────

        # Step 4: Per-step alpha (batch-centred sigmoid)
        z = self.hindsight_beta * (errors - mu_e) / sigma_e
        sigmoid_z = 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
        alpha = alpha_max_k * sigmoid_z

        # Step 5: Corrected V^c
        v_corrected = (1 - alpha) * values + alpha * returns_mc

        # Step 6: Boundary bootstrap correction (from v2)
        if self.use_boundary_correction:
            tail_n = min(10, T)
            approx_err_last = float(errors[-tail_n:].mean())
            alpha_last = alpha_max_k * (1.0 / (1.0 + np.exp(
                -self.hindsight_beta * (approx_err_last - mu_e) / sigma_e
            )))
            approx_G_last = returns_mc[-1]
            last_value_corrected = (1.0 - alpha_last) * last_value + alpha_last * approx_G_last
        else:
            last_value_corrected = last_value

        # Step 7: Corrected GAE
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - terminated[t]
                next_v_c = last_value_corrected if next_non_terminal > 0 else 0.0
            else:
                next_non_terminal = 1.0 - terminated[t]
                next_v_c = v_corrected[t + 1] if next_non_terminal > 0 else 0.0
            delta_td = rewards[t] + self.gamma * next_v_c - v_corrected[t]
            last_gae = delta_td + self.gamma * self.lam * next_non_terminal * last_gae
            advantages[t] = last_gae

        # Step 8: Critic targets (EV-driven mixing, c_mc lower bound = 0.1)
        # Use standard GAE returns from ORIGINAL (uncorrected) V to decouple
        # advantage estimation from critic training.
        c_mc = float(np.clip(1.0 - self._ev_ema, 0.1, 1.0))
        std_gae_returns = self.buffer._compute_standard_returns(last_value, self.gamma, self.lam)
        critic_returns = c_mc * returns_mc + (1 - c_mc) * std_gae_returns

        self.buffer.advantages[:T] = advantages
        self.buffer.returns[:T] = critic_returns

        # Update EV EMA
        var_y = np.var(returns_mc) + 1e-8
        ev_now = float(1.0 - np.var(returns_mc - values) / var_y)
        self._ev_prev = self._ev_ema
        self._ev_ema = (1 - self._ev_ema_alpha) * self._ev_ema + self._ev_ema_alpha * ev_now

    def compute_gae(self, last_value: float):
        self.compute_hindsight_gae(last_value)


class OptimalHCGAE_v4_FixSCR(OptimalHCGAE_v4):
    """
    HCGAE v4_FixSCR: 修正 SCR 分母的精确版本。

    ═══════════════════════════════════════════════════════════════════════════
    v4 的 SCR 分母问题
    ═══════════════════════════════════════════════════════════════════════════

    v4 原始公式（line 1202）：
        SCR = |mean(G - V)| / std(G)
        α_cap = SCR² / (1 + SCR²) + scr_relax

    这里 std(G) 包含两部分方差：
        Var(G) = Var(V*(s))      ← V*(s) 在不同时刻 t 的结构方差（时序差异）
               + E[Var(G|s)]    ← 固定 s_t 时 G_t 的条件方差（真正的 MC 噪声）

    MSE 最优的 α* 需要的是 **E[Var(G|s)]**（纯 MC 采样噪声），
    而不是 Var(G)（包含了与 alpha 无关的时序结构方差）。

    ── 全方差定律推导 ──
        Var(G) = Var(V*) + E[Var(G|s)]
        当 Critic 好时（V ≈ V*）：Var(V) ≈ Var(V*)
        ∴ E[Var(G|s)] ≈ max(Var(G) - Var(V), ε)

    ── 修正后的 SCR ──
        SCR_corrected = |mean(G-V)| / sqrt(max(Var(G) - Var(V), floor))
        α_cap = SCR_corrected² / (1 + SCR_corrected²) + scr_relax

    ── 影响量化（示例）──
        若 std(G)=60, std(V)=48, MAE(G-V)=12:
          v4:       SCR = 12/60 = 0.2  → α_cap = 0.04 + 0.05 = 0.09
          v4_Fix:   denom = sqrt(60²-48²) = sqrt(1296) = 36
                    SCR = 12/36 = 0.33 → α_cap = 0.10 + 0.05 = 0.15
          提升：α_cap 从 0.09 → 0.15（约 1.67x）

    ── Critic 质量对修正效果的影响 ──
        Critic 差（V ≈ const, Var(V)≈0）：分母 ≈ Var(G)，等同 v4（无变化）
        Critic 好（V ≈ V*, Var(V)≈Var(V*)）：分母 << Var(G)，α_cap 更大

    ── 在 v4 架构中的角色 ──
        SCR cap 是 v4 三层门控的最终约束层，
        修正分母使 cap 在"Critic 质量中等-良好"时不再过度保守，
        允许 per-step sigmoid 在准确 Critic 误差大的步骤上施加更多修正。

    Parameters
    ───────────────────────────────────────────────────────────────────────────
    var_floor_frac : float
        Var(G)-Var(V) 的最小值 = var_floor_frac × Var(G)。
        防止数值噪声导致分母为负。默认 0.05（即 5% 的 Var(G)）。
    所有其他参数继承自 OptimalHCGAE_v4。
    """

    NAME = "Optimal_HCGAE_v4_FixSCR"

    def __init__(
        self,
        env: gym.Env,
        name: str = "Optimal_HCGAE_v4_FixSCR",
        var_floor_frac: float = 0.05,
        **kwargs,
    ):
        super().__init__(env=env, name=name, **kwargs)
        self.var_floor_frac = var_floor_frac

    def _scr_alpha_cap(self, values: np.ndarray, returns_mc: np.ndarray) -> float:
        """
        修正版 SCR cap：使用 sqrt(Var(G) - Var(V)) 作为分母。

        原始 v4：  SCR = |mean(G-V)| / std(G)
        修正版：   SCR = |mean(G-V)| / sqrt(max(Var(G) - Var(V), floor))

        数学依据：全方差定律
            Var(G) = Var(V*(s)) + E[Var(G|s)]
            E[Var(G|s)] ≈ Var(G) - Var(V)  （当 V ≈ V* 时）
            E[Var(G|s)] 才是真正影响 α* 的 MC 噪声方差
        """
        delta = returns_mc - values
        bias = float(np.abs(np.mean(delta)))

        # 修正分母：Var(G) - Var(V)，下界为 floor × Var(G)
        var_G = float(np.var(returns_mc)) + 1e-8
        var_V = float(np.var(values))
        floor = self.var_floor_frac * var_G
        denom_sq = max(var_G - var_V, floor)
        denom = float(np.sqrt(denom_sq)) + 1e-8

        scr_hat = bias / denom

        # EMA 更新（与 v4 完全相同）
        self._scr_ema = (1 - self.scr_ema_alpha) * self._scr_ema + self.scr_ema_alpha * scr_hat
        self._scr_history.append(self._scr_ema)

        alpha_cap = (self._scr_ema ** 2) / (1.0 + self._scr_ema ** 2) + self.scr_relax
        return float(np.clip(alpha_cap, 0.0, 1.0))


class OptimalHCGAE_v5(OptimalHCGAE_v3):
    """
    HCGAE v5: Universal Adaptive Correction (v3 + v4 combined).

    Combines ALL fixes from v3 (G-Clamping, VW-Gate, Boundary Prior) with
    v4's SCR-adaptive α_max cap, creating a unified agent that handles both
    episodic (Hopper/Walker2d) and dense/high-variance (HalfCheetah, Ant-v4)
    environments under a single configuration.

    Architecture
    ────────────
    Gate hierarchy (applied in order, α_max_k reduced by each):
      1. EV level gate  (from v1):  ev_gate = max(1 - EV_ema, 0.2)
      2. Cosine decay   (from v1):  cosine_factor(k)
      3. EV rate gate   (from v2):  suppress when ΔEV > τ_rate
      4. VW (SNR) gate  (from v3):  suppress when SNR << snr_target
      5. SCR-α* cap     (from v4):  cap at α*(SCR_ema) = SCR²/(1+SCR²) + ε

    Plus:
      - G Clamping      (from v3):  prevents pessimism injection
      - Boundary Prior  (from v3):  positive-return-weighted boundary bootstrap

    Design intent
    ────────────────────────────────────────────────────────────────────────
    • On Hopper/Walker2d (episodic, high Critic bias, moderate SNR):
      Gates 1-3 are active; gate 4 gives ~full scale; gate 5 gives α* ≈ 0.5.
      G-clamping rarely fires (G > 0 most of the time).
      → Behaviour ≈ v2 (mild reduction from SCR cap on low-SNR episodes)

    • On HalfCheetah (dense, fast-converging Critic):
      Gate 3 (EV rate) suppresses strongly in phase 1.
      Gate 5 (SCR cap) further caps α* once bias falls.
      → Behaviour ≈ v4 (safe correction)

    • On Ant-v4 (dense, extreme variance, SNR ≈ 0.06):
      Gate 4 (VW/SNR) dominates: strong suppression via vw_scale ≪ 1.
      Gate 5 (SCR cap): α* small due to low SCR.
      G-clamping prevents negative-G pessimism injection.
      Boundary prior guards against large-negative last return.
      → Behaviour ≈ v3 with extra α* cap (safest config for Ant)

    Parameters
    ────────────────────────────────────────────────────────────────────────
    Inherits all from OptimalHCGAE_v3, plus:
    scr_ema_alpha : float
        EMA smoothing for SCR cap estimate. Default 0.1.
    scr_relax : float
        Additive slack above theoretical α* bound. Default 0.05.
    use_scr_cap : bool
        Toggle the SCR-α* cap (for ablation). Default True.
    """

    NAME = "Optimal_HCGAE_v5"

    def __init__(
        self,
        env: gym.Env,
        name: str = "Optimal_HCGAE_v5",
        # v4-style SCR α* cap
        scr_ema_alpha: float = 0.1,
        scr_relax: float = 0.05,
        use_scr_cap: bool = True,
        # All v3 parameters
        **kwargs,
    ):
        super().__init__(env=env, name=name, **kwargs)
        self.scr_ema_alpha = scr_ema_alpha
        self.scr_relax = scr_relax
        self.use_scr_cap = use_scr_cap
        # Online SCR tracking (initialise at 1.0 → α* = 0.50, neutral)
        self._scr_v5_ema = 1.0

    def _scr_alpha_cap_v5(self, values: np.ndarray, returns_mc: np.ndarray) -> float:
        """
        MSE-optimal α* cap (same formula as v4):
            SCR_hat = |mean(G − V)| / (std(G) + ε)
            α*_cap  = SCR_ema² / (1 + SCR_ema²) + scr_relax
        """
        delta = returns_mc - values
        scr_hat = float(np.abs(np.mean(delta))) / (float(np.std(returns_mc)) + 1e-8)
        self._scr_v5_ema = (1 - self.scr_ema_alpha) * self._scr_v5_ema + self.scr_ema_alpha * scr_hat
        alpha_cap = (self._scr_v5_ema ** 2) / (1.0 + self._scr_v5_ema ** 2) + self.scr_relax
        return float(np.clip(alpha_cap, 0.0, 1.0))

    def compute_hindsight_gae(self, last_value: float):
        """HCGAE v5 = v3 (G-clamp + VW-gate + BdryPrior) + v4 (SCR α* cap)."""
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

        # Step 2: Per-step error
        errors = np.abs(values - returns_mc)
        mu_e = errors.mean()
        sigma_e = errors.std() + 1e-8

        # ── FIX ②: Online SNR / VW gate (from v3) ──────────────────────────
        deltas = returns_mc - values
        delta_mean = float(np.mean(deltas))
        delta_std = float(np.std(deltas)) + 1e-8
        snr_now = abs(delta_mean) / delta_std
        self._snr_ema = ((1 - self.snr_ema_alpha) * self._snr_ema
                         + self.snr_ema_alpha * snr_now)
        vw_scale = float(np.clip(self._snr_ema / self.snr_target,
                                 self.snr_min_scale, 1.0)) if self.use_vw_gate else 1.0

        # ── Online positive-G statistics (needed for FIX ① and FIX ③) ──────
        pos_mask = returns_mc > 0
        pos_frac = float(pos_mask.mean())
        if pos_mask.sum() > 1:
            g_pos_std = float(np.std(returns_mc[pos_mask]))
            g_pos_mean = float(np.mean(returns_mc[pos_mask]))
        else:
            g_pos_std = float(np.std(np.abs(returns_mc))) + 1e-8
            g_pos_mean = float(np.mean(returns_mc))
        ema_a = 0.1
        self._g_pos_std_ema = (1 - ema_a) * self._g_pos_std_ema + ema_a * g_pos_std
        self._g_pos_mean_ema = (1 - ema_a) * self._g_pos_mean_ema + ema_a * g_pos_mean
        self._g_pos_frac_ema = (1 - ema_a) * self._g_pos_frac_ema + ema_a * pos_frac

        # Step 3: EV-gated cosine α_max
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

        # ── EV growth-rate gate (from v2) ────────────────────────────────────
        if self.use_ev_rate_gate:
            var_y_tmp = np.var(returns_mc) + 1e-8
            ev_now_tmp = float(1.0 - np.var(returns_mc - values) / var_y_tmp)
            ev_rate_raw = ev_now_tmp - self._ev_prev
            self._ev_rate_ema = ((1 - self._ev_rate_ema_alpha) * self._ev_rate_ema
                                 + self._ev_rate_ema_alpha * max(ev_rate_raw, 0.0))
            if self._ev_rate_ema > self.ev_rate_threshold:
                excess = min(self._ev_rate_ema - self.ev_rate_threshold,
                             self.ev_rate_max - self.ev_rate_threshold)
                suppression_frac = excess / max(self.ev_rate_max - self.ev_rate_threshold, 1e-8)
                ev_rate_scale = max(1.0 - suppression_frac * (1.0 - self.ev_gate_min_scale),
                                    self.ev_gate_min_scale)
            else:
                ev_rate_scale = 1.0
            alpha_max_k *= ev_rate_scale

        # ── Apply VW gate (FIX ② from v3) ───────────────────────────────────
        alpha_max_k *= vw_scale

        # ── Apply SCR α* cap (FIX from v4) ──────────────────────────────────
        if self.use_scr_cap:
            alpha_max_k = min(alpha_max_k, self._scr_alpha_cap_v5(values, returns_mc))

        # Step 4: Per-step alpha (batch-centred sigmoid)
        z = self.hindsight_beta * (errors - mu_e) / sigma_e
        sigmoid_z = 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
        alpha = alpha_max_k * sigmoid_z

        # ── FIX ①: G Clamping (from v3) ──────────────────────────────────────
        if self.use_g_clamp:
            margin = self.g_clamp_margin_k * (self._g_pos_std_ema + 1e-8)
            returns_mc_clamped = np.maximum(returns_mc, values - margin)
        else:
            returns_mc_clamped = returns_mc

        # Step 5: Corrected V^c (with clamped G)
        v_corrected = (1 - alpha) * values + alpha * returns_mc_clamped

        # ── FIX ③: Boundary correction with positive-return prior (from v3) ──
        if self.use_boundary_correction:
            tail_n = min(10, T)
            approx_err_last = float(errors[-tail_n:].mean())
            alpha_last = alpha_max_k * (1.0 / (1.0 + np.exp(
                -self.hindsight_beta * (approx_err_last - mu_e) / sigma_e
            )))
            raw_G_last = returns_mc[-1]
            if self.use_boundary_prior and self._g_pos_frac_ema > 0.1:
                pos_w = self._g_pos_frac_ema
                approx_G_last = (pos_w * raw_G_last
                                 + (1 - pos_w) * self._g_pos_mean_ema)
            else:
                approx_G_last = raw_G_last
            last_value_corrected = (1.0 - alpha_last) * last_value + alpha_last * approx_G_last
        else:
            last_value_corrected = last_value

        # Step 6: Corrected GAE
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

        # Step 7: Critic targets (EV-driven, clamped MC, c_mc lower bound = 0.1)
        # Use clamped MC returns for MC component; standard GAE returns (original V)
        # for GAE component to keep advantage and critic update paths decoupled.
        c_mc = float(np.clip(1.0 - self._ev_ema, 0.1, 1.0))
        std_gae_returns = self.buffer._compute_standard_returns(last_value, self.gamma, self.lam)
        critic_returns = c_mc * returns_mc_clamped + (1 - c_mc) * std_gae_returns

        self.buffer.advantages[:T] = advantages
        self.buffer.returns[:T] = critic_returns

        # Update EV EMA (use original returns_mc for EV)
        var_y = np.var(returns_mc) + 1e-8
        ev_now = float(1.0 - np.var(returns_mc - values) / var_y)
        self._ev_prev = self._ev_ema
        self._ev_ema = (1 - self._ev_ema_alpha) * self._ev_ema + self._ev_ema_alpha * ev_now

    def compute_gae(self, last_value: float):
        self.compute_hindsight_gae(last_value)


class OptimalHCGAE_v6(OptimalHCGAE_v4):
    """
    HCGAE v6: Variance-Normalized Boundary + Relative EV Gate.

    Targeted fix for "dense-reward, no-catastrophic-failure" environments
    such as HalfCheetah-v4, Swimmer-v4, etc., where:

      (a) G_0 ≈ 0 but V(s_0) quickly rises to 2000+
          → The absolute boundary prior δ_boundary = G_0 − V(s_0) is always
            very negative, flooring α to ~0 and blocking all correction.

      (b) EV grows slowly (EV_0 ≈ 0.05 → final ≈ 0.15) due to high reward
          variance (CV ≈ 2.8), so the raw EV growth-rate gate rarely fires
          or fires only after the critic has already committed to a sub-optimal
          local minimum.

    Root-cause (from ablation data on HC):
      • v2_NoBdry  vs v2: +15.7pp  →  boundary prior is primary source of harm
      • v2_NoGate  vs v2: +8.6pp   →  EV rate gate is secondary source of harm
      • v4 (SCR cap alone): +5.8%  →  SCR cap helps but insufficient alone

    Fixes on top of v4 (= v2 + SCR cap):
    ─────────────────────────────────────────────────────────────────────────
    FIX A: Variance-Normalised Boundary Prior
        v2 boundary check uses raw δ_boundary = G_last − V_last
        v6 normalises by online critic std σ_V:

            δ_boundary_norm = (G_last − V_last) / (σ_V + ε)

        boundary_ok = δ_boundary_norm > boundary_norm_threshold  (default −3.0)

        Physical meaning: only suppress boundary correction when G deviates
        more than 3 sigma from V, relative to the critic's own uncertainty.
        In HC, σ_V is large early (critic uncertain) → threshold is loose →
        boundary correction is NOT suppressed. As critic converges, σ_V falls
        and the boundary check becomes more stringent. This is the correct
        behaviour: trust boundary correction early (when critic is most biased)
        and become cautious later.

    FIX B: Relative EV Saturation Gate (Critic-Headroom-Normalised)
        Core HCGAE principle: when Critic is POOR (EV low) → apply MORE MC
        correction to reduce bias; when Critic is GOOD (EV high) → apply LESS
        correction because bias is already small and MC noise dominates.

        v2's EV-level gate already embeds this: ev_gate = max(1−EV_ema, 0.2).
        Fix B adds an additional SATURATION gate that suppresses correction
        once the Critic has captured a large fraction β of its EV headroom,
        i.e. when EV has improved substantially from its initial level.

            ev_threshold = EV_initial + β * (1 − EV_initial)

        If EV_current > ev_threshold → Critic has made large gains, bias is
        falling → begin to ramp down correction strength additively.
        If EV_current ≤ ev_threshold → Critic still improving or stuck →
        do NOT suppress (the existing ev_gate = max(1−EV, 0.2) already
        handles the level-based control).

        Environment behaviour (β=0.7 default):
          - Hopper  (EV_0≈0.3, final≈0.7): threshold = 0.3+0.7*0.7 = 0.79
            → fires only near convergence when correction is least needed ✓
          - Walker2d (EV_0≈0.3, final≈0.7): same as Hopper ✓
          - HC      (EV_0≈0.05, final≈0.15): threshold = 0.05+0.7*0.95 = 0.715
            → EV never reaches threshold → gate never fires → no change ✓
          - Ant-v4  (EV_0≈0.05, final≈0.2): threshold ≈ 0.715 → gate silent ✓

        This is universal: gate is silent on low-EV environments (HC, Ant)
        and only activates near the end of training on high-EV environments
        (Hopper, Walker) where it provides a gentle final suppression.

    Parameters (v6-specific, on top of v4):
    ─────────────────────────────────────────────────────────────────────────
    use_norm_boundary : bool
        Enable FIX A (variance-normalised boundary). Default True.
    boundary_norm_threshold : float
        Normalised deviation below which boundary correction is applied.
        Default −3.0 (suppress only when G < V − 3σ_V).
    use_relative_ev_gate : bool
        Enable FIX B (relative EV saturation gate). Default True.
    ev_headroom_frac : float
        Fraction of EV headroom after which saturation suppression begins.
        Default 0.7. Set higher (→1) to delay gate; lower (→0) to trigger
        earlier. Environments where EV never reaches the threshold are
        unaffected regardless of this value.
    v_std_ema_alpha : float
        EMA smoothing rate for critic value standard deviation. Default 0.05.
    """

    NAME = "Optimal_HCGAE_v6"

    def __init__(
        self,
        env: gym.Env,
        name: str = "Optimal_HCGAE_v6",
        # FIX A: variance-normalised boundary
        use_norm_boundary: bool = True,
        boundary_norm_threshold: float = -3.0,
        v_std_ema_alpha: float = 0.05,
        # FIX B: relative EV saturation gate (suppress only when EV >> EV_initial)
        use_relative_ev_gate: bool = True,
        ev_headroom_frac: float = 0.7,
        # All v4 parameters (v2 + SCR cap)
        **kwargs,
    ):
        super().__init__(env=env, name=name, **kwargs)
        self.use_norm_boundary = use_norm_boundary
        self.boundary_norm_threshold = boundary_norm_threshold
        self.v_std_ema_alpha = v_std_ema_alpha
        self.use_relative_ev_gate = use_relative_ev_gate
        self.ev_headroom_frac = ev_headroom_frac

        # Online critic std tracking (for FIX A)
        self._v_std_ema = 1.0        # initialise at 1 (neutral)
        self._ev_initial = None      # set on first rollout (for FIX B)

    def compute_hindsight_gae(self, last_value: float):
        """HCGAE v6 = v4 (v2 + SCR-cap) + Norm-Boundary (FIX A) + Relative-EV-Gate (FIX B)."""
        T = self.buffer.pos
        rewards = self.buffer.rewards[:T]
        terminated = self.buffer.terminated[:T]
        values = self.buffer.values[:T]

        # Step 1: MC returns (identical to v2/v4)
        returns_mc = np.zeros(T, dtype=np.float32)
        running_return = last_value
        for t in reversed(range(T)):
            if terminated[t]:
                running_return = 0.0
            running_return = rewards[t] + self.gamma * running_return
            returns_mc[t] = running_return

        # Step 2: Per-step error
        errors = np.abs(values - returns_mc)
        mu_e = errors.mean()
        sigma_e = errors.std() + 1e-8

        # ── FIX A: Track critic std σ_V ────────────────────────────────────
        v_std_now = float(np.std(values)) + 1e-8
        self._v_std_ema = ((1 - self.v_std_ema_alpha) * self._v_std_ema
                           + self.v_std_ema_alpha * v_std_now)

        # ── FIX B: Initialise EV baseline on first rollout ──────────────────
        var_y_base = np.var(returns_mc) + 1e-8
        ev_now_base = float(1.0 - np.var(returns_mc - values) / var_y_base)
        if self._ev_initial is None:
            self._ev_initial = max(ev_now_base, 0.0)

        # Step 3a: EV-gated cosine α_max (same as v2)
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

        # Step 3b: EV growth-rate gate (same as v2)
        if self.use_ev_rate_gate:
            ev_rate_raw = ev_now_base - self._ev_prev
            self._ev_rate_ema = ((1 - self._ev_rate_ema_alpha) * self._ev_rate_ema
                                 + self._ev_rate_ema_alpha * max(ev_rate_raw, 0.0))
            if self._ev_rate_ema > self.ev_rate_threshold:
                excess = min(self._ev_rate_ema - self.ev_rate_threshold,
                             self.ev_rate_max - self.ev_rate_threshold)
                suppression_frac = excess / max(self.ev_rate_max - self.ev_rate_threshold, 1e-8)
                ev_rate_scale = max(1.0 - suppression_frac * (1.0 - self.ev_gate_min_scale),
                                    self.ev_gate_min_scale)
            else:
                ev_rate_scale = 1.0
            alpha_max_k *= ev_rate_scale

        # ── FIX B: Relative EV saturation gate ─────────────────────────────
        # HCGAE principle: low EV → Critic is poor → keep correction strong.
        #                  high EV → Critic has improved greatly → reduce correction.
        # This gate adds suppression ONLY when EV has risen substantially above
        # its initial level (i.e. Critic has captured ev_headroom_frac of its
        # remaining headroom). On low-EV environments (HC, Ant) where EV never
        # reaches the threshold, this gate is completely silent → no change.
        if self.use_relative_ev_gate and self._ev_initial is not None:
            ev_threshold = self._ev_initial + self.ev_headroom_frac * (1.0 - self._ev_initial)
            # Only suppress once EV exceeds the saturation threshold
            if ev_now_base > ev_threshold:
                # Ramp down smoothly: the further past threshold, the more suppression
                # excess ∈ [0, 1-ev_threshold]; suppression_frac ∈ [0, 1]
                excess = ev_now_base - ev_threshold
                max_excess = max(1.0 - ev_threshold, 1e-8)
                suppression_frac = min(excess / max_excess, 1.0)
                # Scale: 1.0 at threshold → 0.2 at EV=1.0  (never fully suppress)
                sat_scale = max(1.0 - suppression_frac * 0.8, 0.2)
                alpha_max_k *= sat_scale

        # ── v4: SCR-based α* cap ─────────────────────────────────────────────
        alpha_max_k = min(alpha_max_k, self._scr_alpha_cap(values, returns_mc))

        # Step 4: Per-step alpha (batch-centred sigmoid)
        z = self.hindsight_beta * (errors - mu_e) / sigma_e
        sigmoid_z = 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
        alpha = alpha_max_k * sigmoid_z

        # Step 5: Corrected V^c
        v_corrected = (1 - alpha) * values + alpha * returns_mc

        # Step 6: Boundary bootstrap correction (FIX A applied here)
        if self.use_boundary_correction:
            tail_n = min(10, T)
            approx_err_last = float(errors[-tail_n:].mean())
            alpha_last = alpha_max_k * (1.0 / (1.0 + np.exp(
                -self.hindsight_beta * (approx_err_last - mu_e) / sigma_e
            )))
            raw_G_last = float(returns_mc[-1])

            if self.use_norm_boundary:
                # FIX A: variance-normalised boundary check
                # Only suppress boundary correction if G deviates > threshold sigma from V
                delta_norm = (raw_G_last - last_value) / self._v_std_ema
                if delta_norm < self.boundary_norm_threshold:
                    # G is unreliably far below V → don't let it drag V down
                    # Instead use a conservative mid-point estimate
                    approx_G_last = last_value + self.boundary_norm_threshold * self._v_std_ema
                else:
                    approx_G_last = raw_G_last
            else:
                approx_G_last = raw_G_last

            last_value_corrected = (1.0 - alpha_last) * last_value + alpha_last * approx_G_last
        else:
            last_value_corrected = last_value

        # Step 7: Corrected GAE
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - terminated[t]
                next_v_c = last_value_corrected if next_non_terminal > 0 else 0.0
            else:
                next_non_terminal = 1.0 - terminated[t]
                next_v_c = v_corrected[t + 1] if next_non_terminal > 0 else 0.0
            delta_td = rewards[t] + self.gamma * next_v_c - v_corrected[t]
            last_gae = delta_td + self.gamma * self.lam * next_non_terminal * last_gae
            advantages[t] = last_gae

        # Step 8: Critic targets (EV-driven mixing)
        # Use standard GAE returns from ORIGINAL (uncorrected) V to decouple
        # advantage estimation from critic training.
        c_mc = float(np.clip(1.0 - self._ev_ema, 0.1, 1.0))
        std_gae_returns = self.buffer._compute_standard_returns(last_value, self.gamma, self.lam)
        critic_returns = c_mc * returns_mc + (1 - c_mc) * std_gae_returns

        self.buffer.advantages[:T] = advantages
        self.buffer.returns[:T] = critic_returns

        # Update EV EMA
        var_y = np.var(returns_mc) + 1e-8
        ev_now = float(1.0 - np.var(returns_mc - values) / var_y)
        self._ev_prev = self._ev_ema
        self._ev_ema = (1 - self._ev_ema_alpha) * self._ev_ema + self._ev_ema_alpha * ev_now

    def compute_gae(self, last_value: float):
        self.compute_hindsight_gae(last_value)


class OptimalHCGAE_HeuristicV2(OptimalHCGAE_v4):
    """
    HCGAE HeuristicV2: 基于 Heuristic_HCGAE 的系统性改进。

    ═══════════════════════════════════════════════════════════════════════════
    背景：Heuristic_HCGAE (v2) 的问题分析
    ═══════════════════════════════════════════════════════════════════════════

    Heuristic_HCGAE 在 FinalExperiment 同批次对比中的表现:
      HC: +9.8% vs Optimal_PPO  (高模式逃脱概率提升：4/12 vs 3/12 seeds > 3000)
      Hopper: -0.2%  (统计不显著)
      Walker2d: -9.3%
      Ant: -14.4%

    根因分析：
    1. per-step sigmoid 使用批内 z-score 归一化（问题）
       z_t = β·(|V_t - G_t| - μ_e) / σ_e
       → 每批次固定 50% 步骤得到正 z 值（由于 sigmoid 对称性）
       → E[α_t] ≈ α_max/2，与 MC 噪声水平无关
       → HC 方差大时 σ_e 也大，σ_e 归一化后 α 实际上被"均匀化"了

    2. 无 SCR 上界约束（v4 已修复，但 Heuristic 没有）
       → 当 SCR 低（Var(G)大、Bias小）时 α_max 仍高
       → Ant/Walker 上过度修正 → Advantage 方差增加

    3. HC 优势的保留机制（需要继承）：
       α_max=0.7 的积极初始值在 HC 上有助于早期探索逃脱低模式

    ═══════════════════════════════════════════════════════════════════════════
    HeuristicV2 改进
    ═══════════════════════════════════════════════════════════════════════════

    改进1: 修正 per-step sigmoid 归一化（核心改进）
    ─────────────────────────────────────────────────
    原始: z_t = β·(|V_t-G_t| - μ_e) / σ_e   (批均值中心化)
    改进: z_t = β·(|V_t-G_t| / (σ_G_ema + ε) - noise_threshold)

    其中 σ_G_ema = EMA(std(G))，noise_threshold 为分位数（默认 0.5 对应中位数）

    物理含义:
    - 以"MC 噪声水平 σ_G"为参考，判断每步 Critic 误差是否显著
    - 当 Critic 普遍准确时（所有 |V-G| << σ_G），全批次 α → 0
    - 当 Critic 系统性偏差（所有 |V-G| >> σ_G）时，全批次 α → α_max
    - 相比 z-score，这个方式不再人为保证 50% 步骤得到修正

    改进2: 继承 v4 的 SCR_cap（理论最优上界）
    ─────────────────────────────────────────────
    同 v4：α_max_k = min(α_max_v2, SCR²/(1+SCR²) + relax)
    防止低 SCR 环境（Ant、Walker2d）上的过度修正

    改进3: 保留 Heuristic 的高初始 α_max（HC 优势保留）
    ─────────────────────────────────────────────────────
    继续使用 α_max=0.7, α_min=0.1（与 Heuristic 相同）
    SCR_cap 在 SCR 高时（HC 早期）允许足够的修正

    ═══════════════════════════════════════════════════════════════════════════
    参数
    ═══════════════════════════════════════════════════════════════════════════
    noise_threshold : float
        per-step sigmoid 的阈值（以 σ_G 为单位）。
        默认 0.5：误差 > 0.5·σ_G 才开始较强修正。
        较大值（如 1.0）→ 更保守（仅误差很大时才修正）。
        较小值（如 0.2）→ 更激进（接近原始批归一化行为）。
    g_std_ema_alpha : float
        MC 回报标准差的 EMA 系数。默认 0.1。
    use_noise_normalized_sigmoid : bool
        是否启用改进的 noise-normalized sigmoid。
        False → 退化为原始批归一化（兼容模式）。默认 True。
    所有其他参数继承自 OptimalHCGAE_v4（v2 + SCR_cap）。
    """

    NAME = "Optimal_HCGAE_HeuristicV2"

    def __init__(
        self,
        env,
        name: str = "Optimal_HCGAE_HeuristicV2",
        # 新增参数
        noise_threshold: float = 0.5,
        g_std_ema_alpha: float = 0.1,
        use_noise_normalized_sigmoid: bool = True,
        # 继承自 v4（v2 + SCR_cap）
        **kwargs,
    ):
        super().__init__(env=env, name=name, **kwargs)
        self.noise_threshold = noise_threshold
        self.g_std_ema_alpha = g_std_ema_alpha
        self.use_noise_normalized_sigmoid = use_noise_normalized_sigmoid

        # 在线跟踪 MC 回报标准差（用于 noise-normalized sigmoid）
        self._g_std_ema = 1.0          # 初始化为 1（中性）
        self._g_std_history = []       # 用于诊断

    def compute_hindsight_gae(self, last_value: float):
        """
        HCGAE HeuristicV2 = v4 (v2 + SCR_cap) + Noise-Normalized Per-Step Sigmoid
        """
        T = self.buffer.pos
        rewards = self.buffer.rewards[:T]
        terminated = self.buffer.terminated[:T]
        values = self.buffer.values[:T]

        # Step 1: MC returns（与 v4 完全相同）
        returns_mc = np.zeros(T, dtype=np.float32)
        running_return = last_value
        for t in reversed(range(T)):
            if terminated[t]:
                running_return = 0.0
            running_return = rewards[t] + self.gamma * running_return
            returns_mc[t] = running_return

        # Step 2: 更新 MC 回报标准差的 EMA
        g_std_now = float(np.std(returns_mc)) + 1e-8
        self._g_std_ema = (
            (1 - self.g_std_ema_alpha) * self._g_std_ema
            + self.g_std_ema_alpha * g_std_now
        )
        self._g_std_history.append(self._g_std_ema)

        # Step 3: per-step error
        errors = np.abs(values - returns_mc)
        mu_e = errors.mean()
        sigma_e = errors.std() + 1e-8

        # Step 4a: EV-gated cosine α_max（与 v4 完全相同）
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

        # Step 4b: EV 增长速率门控（与 v4 完全相同）
        if self.use_ev_rate_gate:
            var_y_tmp = np.var(returns_mc) + 1e-8
            ev_now_tmp = float(1.0 - np.var(returns_mc - values) / var_y_tmp)
            ev_rate_raw = ev_now_tmp - self._ev_prev
            self._ev_rate_ema = ((1 - self._ev_rate_ema_alpha) * self._ev_rate_ema
                                 + self._ev_rate_ema_alpha * max(ev_rate_raw, 0.0))
            if self._ev_rate_ema > self.ev_rate_threshold:
                excess = min(self._ev_rate_ema - self.ev_rate_threshold,
                             self.ev_rate_max - self.ev_rate_threshold)
                suppression_frac = excess / max(self.ev_rate_max - self.ev_rate_threshold, 1e-8)
                ev_rate_scale = max(1.0 - suppression_frac * (1.0 - self.ev_gate_min_scale),
                                    self.ev_gate_min_scale)
            else:
                ev_rate_scale = 1.0
            alpha_max_k *= ev_rate_scale

        # Step 4c: SCR-based α_max cap（继承自 v4）
        alpha_max_k = min(alpha_max_k, self._scr_alpha_cap(values, returns_mc))

        # ── HeuristicV2 核心改进: Noise-Normalized Per-Step Sigmoid ──────────
        if self.use_noise_normalized_sigmoid:
            # 用 σ_G_ema 归一化误差（不再用批内 z-score）
            # z_t = β·(|V_t-G_t|/σ_G_ema - noise_threshold)
            # 物理含义: 阈值为 noise_threshold × σ_G_ema（约为 MC 噪声的一半）
            z = self.hindsight_beta * (errors / self._g_std_ema - self.noise_threshold)
        else:
            # 退化为原始批归一化（兼容模式）
            z = self.hindsight_beta * (errors - mu_e) / sigma_e

        sigmoid_z = 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
        alpha = alpha_max_k * sigmoid_z
        # ─────────────────────────────────────────────────────────────────────

        # Step 5: 修正后的 V^c
        v_corrected = (1 - alpha) * values + alpha * returns_mc

        # Step 6: 边界修正（与 v4 完全相同）
        if self.use_boundary_correction:
            tail_n = min(10, T)
            approx_err_last = float(errors[-tail_n:].mean())
            if self.use_noise_normalized_sigmoid:
                z_last = self.hindsight_beta * (
                    approx_err_last / self._g_std_ema - self.noise_threshold
                )
            else:
                z_last = self.hindsight_beta * (approx_err_last - mu_e) / sigma_e
            alpha_last = alpha_max_k * (1.0 / (1.0 + np.exp(-np.clip(z_last, -20, 20))))
            approx_G_last = returns_mc[-1]
            last_value_corrected = (1.0 - alpha_last) * last_value + alpha_last * approx_G_last
        else:
            last_value_corrected = last_value

        # Step 7: 修正 GAE（与 v4 完全相同）
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - terminated[t]
                next_v_c = last_value_corrected if next_non_terminal > 0 else 0.0
            else:
                next_non_terminal = 1.0 - terminated[t]
                next_v_c = v_corrected[t + 1] if next_non_terminal > 0 else 0.0
            delta_td = rewards[t] + self.gamma * next_v_c - v_corrected[t]
            last_gae = delta_td + self.gamma * self.lam * next_non_terminal * last_gae
            advantages[t] = last_gae

        # Step 8: Critic 训练目标（与 v4 完全相同）
        c_mc = float(np.clip(1.0 - self._ev_ema, 0.1, 1.0))
        std_gae_returns = self.buffer._compute_standard_returns(last_value, self.gamma, self.lam)
        critic_returns = c_mc * returns_mc + (1 - c_mc) * std_gae_returns

        self.buffer.advantages[:T] = advantages
        self.buffer.returns[:T] = critic_returns

        # 更新 EV EMA
        var_y = np.var(returns_mc) + 1e-8
        ev_now = float(1.0 - np.var(returns_mc - values) / var_y)
        self._ev_prev = self._ev_ema
        self._ev_ema = (1 - self._ev_ema_alpha) * self._ev_ema + self._ev_ema_alpha * ev_now

    def compute_gae(self, last_value: float):
        self.compute_hindsight_gae(last_value)


class OptimalHCGAE_Optimal(OptimalHCGAE_v4_FixSCR):
    """
    HCGAE_Optimal: 基于三学科统一推导的最完整理论实现。

    ═══════════════════════════════════════════════════════════════════════════
    理论基础（James-Stein / 卡尔曼滤波 / 贝叶斯后验的等价推导）
    ═══════════════════════════════════════════════════════════════════════════

    核心问题: V_t = V*(s) + B + ε_V, G_t = V*(s) + ε_G
    最优混合: V^c_t = (1-α)V_t + αG_t
    MSE 最优: α* = (B² + σ²_V) / (B² + σ²_V + σ²_G)

    其中 σ²_G 应为 **条件 MC 噪声** E[Var(G|s)]，而非 Var(G)。
    由全方差定律: E[Var(G|s)] ≈ Var(G) - Var(V)  （当 V ≈ V* 时成立）

    ═══════════════════════════════════════════════════════════════════════════
    本版本 = v4_FixSCR (分母修正) + HeuristicV2 (MC 归一化 per-step sigmoid)
    ═══════════════════════════════════════════════════════════════════════════

    改进1 (继承 v4_FixSCR): 全局 α_cap 分母修正
    ─────────────────────────────────────────────
    原始 v4:   SCR = |mean(G-V)| / std(G)          [分母高估 σ_G]
    Optimal:   SCR = |mean(G-V)| / sqrt(Var(G)-Var(V))  [FixSCR 修正]

    理论依据: Var(G) = Var(V*) + E[Var(G|s)]
              当 Critic 好时 Var(V) ≈ Var(V*) → E[Var(G|s)] ≈ Var(G) - Var(V)
    实际影响: α_cap 约提升 1.5-2×（Critic 越好提升越大）

    改进2 (继承 HeuristicV2): MC 归一化 per-step sigmoid
    ─────────────────────────────────────────────────────
    原始:    z_t = β·(|δ_t| - μ_e) / σ_e           [批内 z-score，强制 50% 修正]
    Optimal: z_t = β·(|δ_t| / σ_G_ema - θ)         [MC 噪声归一化]

    理论依据: per-step α_t* ∝ SNR_t = |δ_t| / σ_G
              当所有 |δ_t| << σ_G (Critic 很好) → 全批 α → 0
              当所有 |δ_t| >> σ_G (Critic 系统偏差) → 全批 α → α_max

    两项改进的协同效果:
    ─────────────────────────────────────────────────────────────────────────
    环境         v4 问题                    Optimal 修复
    HalfCheetah  α_cap 过低（B 小）         FixSCR 放宽 α_cap；per-step 不再
                                           强制低误差步骤做无效修正
    Hopper       α_cap 较准，per-step 冗余  per-step 更智能分配
    Walker2d     α_cap 过低               FixSCR 放宽 2×；改善 Walker 表现
    Ant          α_cap 需精准控制           FixSCR 依然保守（高 σ_G）

    参数
    ─────────────────────────────────────────────────────────────────────────
    noise_threshold : float
        per-step sigmoid 阈值（以 σ_G 为单位）。默认 0.5。
        值越大 → 仅大误差步骤才获得修正（更保守）。
        值越小 → 更多步骤获得修正（更激进）。
    g_std_ema_alpha : float
        MC 回报标准差 EMA 平滑系数。默认 0.1。
    var_floor_frac : float (继承自 v4_FixSCR)
        Var(G)-Var(V) 下界 = var_floor_frac × Var(G)。默认 0.05。
    所有其他参数继承自 OptimalHCGAE_v4 / v2。
    """

    NAME = "Optimal_HCGAE_Optimal"

    def __init__(
        self,
        env,
        name: str = "Optimal_HCGAE_Optimal",
        # MC 归一化 per-step sigmoid 参数
        noise_threshold: float = 0.5,
        g_std_ema_alpha: float = 0.1,
        # 所有其他参数继承自 v4_FixSCR (包含 var_floor_frac, scr_ema_alpha, scr_relax)
        **kwargs,
    ):
        super().__init__(env=env, name=name, **kwargs)
        self.noise_threshold = noise_threshold
        self.g_std_ema_alpha = g_std_ema_alpha

        # 在线跟踪 MC 回报标准差
        self._g_std_ema = 1.0

    def compute_hindsight_gae(self, last_value: float):
        """
        HCGAE_Optimal = v4_FixSCR (FixSCR 分母) + MC 归一化 per-step sigmoid

        三学科统一的理论最优实现:
          全局 α_cap = FixSCR 修正的 James-Stein/Kalman 最优上界
          per-step = MC 噪声感知的局部最优分配
        """
        T = self.buffer.pos
        rewards = self.buffer.rewards[:T]
        terminated = self.buffer.terminated[:T]
        values = self.buffer.values[:T]

        # Step 1: MC 回报（标准）
        returns_mc = np.zeros(T, dtype=np.float32)
        running_return = last_value
        for t in reversed(range(T)):
            if terminated[t]:
                running_return = 0.0
            running_return = rewards[t] + self.gamma * running_return
            returns_mc[t] = running_return

        # Step 2: 更新 MC 回报标准差 EMA（用于 per-step 归一化）
        g_std_now = float(np.std(returns_mc)) + 1e-8
        self._g_std_ema = (
            (1 - self.g_std_ema_alpha) * self._g_std_ema
            + self.g_std_ema_alpha * g_std_now
        )

        # Step 3: Per-step 误差
        errors = np.abs(values - returns_mc)

        # Step 4a: EV-gated cosine α_max（继承 v4/v2）
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

        # Step 4b: EV 增长速率门控（继承 v4/v2）
        if self.use_ev_rate_gate:
            var_y_tmp = np.var(returns_mc) + 1e-8
            ev_now_tmp = float(1.0 - np.var(returns_mc - values) / var_y_tmp)
            ev_rate_raw = ev_now_tmp - self._ev_prev
            self._ev_rate_ema = ((1 - self._ev_rate_ema_alpha) * self._ev_rate_ema
                                 + self._ev_rate_ema_alpha * max(ev_rate_raw, 0.0))
            if self._ev_rate_ema > self.ev_rate_threshold:
                excess = min(self._ev_rate_ema - self.ev_rate_threshold,
                             self.ev_rate_max - self.ev_rate_threshold)
                suppression_frac = excess / max(self.ev_rate_max - self.ev_rate_threshold, 1e-8)
                ev_rate_scale = max(1.0 - suppression_frac * (1.0 - self.ev_gate_min_scale),
                                    self.ev_gate_min_scale)
            else:
                ev_rate_scale = 1.0
            alpha_max_k *= ev_rate_scale

        # Step 4c: FixSCR-based α_max cap（理论最优上界，FixSCR 修正分母）
        # 使用 v4_FixSCR 的 _scr_alpha_cap: SCR = |B| / sqrt(Var(G)-Var(V))
        alpha_max_k = min(alpha_max_k, self._scr_alpha_cap(values, returns_mc))

        # Step 5: MC 噪声归一化 per-step sigmoid（HeuristicV2 核心改进）
        # z_t = β·(|δ_t| / σ_G_ema - noise_threshold)
        # 理论依据: per-step α_t* ∝ SNR_t = |δ_t| / σ_G (卡尔曼局部最优)
        z = self.hindsight_beta * (errors / self._g_std_ema - self.noise_threshold)
        sigmoid_z = 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
        alpha = alpha_max_k * sigmoid_z

        # Step 6: 修正后的 V^c = (1-α)V + αG
        v_corrected = (1 - alpha) * values + alpha * returns_mc

        # Step 7: 边界修正（继承 v4/v2，用 noise-normalized sigmoid）
        if self.use_boundary_correction:
            tail_n = min(10, T)
            approx_err_last = float(errors[-tail_n:].mean())
            z_last = self.hindsight_beta * (
                approx_err_last / self._g_std_ema - self.noise_threshold
            )
            alpha_last = alpha_max_k * (1.0 / (1.0 + np.exp(-np.clip(z_last, -20, 20))))
            approx_G_last = returns_mc[-1]
            last_value_corrected = (1.0 - alpha_last) * last_value + alpha_last * approx_G_last
        else:
            last_value_corrected = last_value

        # Step 8: 修正 GAE（标准 δ_TD 递推）
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - terminated[t]
                next_v_c = last_value_corrected if next_non_terminal > 0 else 0.0
            else:
                next_non_terminal = 1.0 - terminated[t]
                next_v_c = v_corrected[t + 1] if next_non_terminal > 0 else 0.0
            delta_td = rewards[t] + self.gamma * next_v_c - v_corrected[t]
            last_gae = delta_td + self.gamma * self.lam * next_non_terminal * last_gae
            advantages[t] = last_gae

        # Step 9: Critic 训练目标（EV 驱动的 MC/GAE 混合）
        c_mc = float(np.clip(1.0 - self._ev_ema, 0.1, 1.0))
        std_gae_returns = self.buffer._compute_standard_returns(last_value, self.gamma, self.lam)
        critic_returns = c_mc * returns_mc + (1 - c_mc) * std_gae_returns

        self.buffer.advantages[:T] = advantages
        self.buffer.returns[:T] = critic_returns

        # 更新 EV EMA
        var_y = np.var(returns_mc) + 1e-8
        ev_now = float(1.0 - np.var(returns_mc - values) / var_y)
        self._ev_prev = self._ev_ema
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
        hidden_dim=256,
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

    elif algo_name == "Optimal_HCGAE_v2_AutoSCR":
        # Future Work 1: v2 + automatic SCR-based mode detection
        hcgae_kwargs = {k: v for k, v in opt_defaults.items()}
        return OptimalHCGAE_v2_AutoSCR(
            env=env, name=name,
            hindsight_beta=3.0,
            hindsight_alpha_max=0.7,
            hindsight_alpha_min=0.1,
            use_scr_adapt=False,       # disable old SCR (use new AutoSCR instead)
            use_boundary_correction=True,
            use_ev_rate_gate=True,
            ev_rate_threshold=0.05,
            ev_rate_max=0.15,
            ev_gate_min_scale=0.1,
            scr_threshold=1.0,
            scr_min_scale=0.15,
            scr_sharpness=3.0,
            use_auto_scr=True,
            **hcgae_kwargs
        )

    elif algo_name == "Optimal_HCGAE_v2_AutoSCR_NoGate":
        # Ablation: AutoSCR without EV rate gate (SCR-only gating)
        hcgae_kwargs = {k: v for k, v in opt_defaults.items()}
        return OptimalHCGAE_v2_AutoSCR(
            env=env, name=name,
            hindsight_beta=3.0,
            hindsight_alpha_max=0.7,
            hindsight_alpha_min=0.1,
            use_scr_adapt=False,
            use_boundary_correction=True,
            use_ev_rate_gate=False,    # disable EV gate → SCR-only
            scr_threshold=1.0,
            scr_min_scale=0.15,
            scr_sharpness=3.0,
            use_auto_scr=True,
            **hcgae_kwargs
        )

    elif algo_name == "Optimal_HCGAE_v3":
        # v3: High-Variance MC adaptation for Ant-v4 style environments
        # Three complementary fixes: G-Clamping, VW-Gate, Boundary Prior
        hcgae_kwargs = {k: v for k, v in opt_defaults.items()}
        return OptimalHCGAE_v3(
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
            # v3-specific
            use_g_clamp=True,
            g_clamp_margin_k=1.5,
            use_vw_gate=True,
            snr_target=0.5,
            snr_min_scale=0.1,
            snr_ema_alpha=0.05,
            use_boundary_prior=True,
            **hcgae_kwargs
        )

    elif algo_name == "Optimal_HCGAE_v3_NoClamp":
        # Ablation: v3 without G-Clamping (VW-Gate + Boundary Prior only)
        hcgae_kwargs = {k: v for k, v in opt_defaults.items()}
        return OptimalHCGAE_v3(
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
            use_g_clamp=False,
            use_vw_gate=True,
            snr_target=0.5,
            snr_min_scale=0.1,
            snr_ema_alpha=0.05,
            use_boundary_prior=True,
            **hcgae_kwargs
        )

    elif algo_name == "Optimal_HCGAE_v3_NoVWGate":
        # Ablation: v3 without VW-Gate (G-Clamping + Boundary Prior only)
        hcgae_kwargs = {k: v for k, v in opt_defaults.items()}
        return OptimalHCGAE_v3(
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
            use_g_clamp=True,
            g_clamp_margin_k=1.5,
            use_vw_gate=False,
            use_boundary_prior=True,
            **hcgae_kwargs
        )

    elif algo_name == "Optimal_HCGAE_v3_NoBdryPrior":
        # Ablation: v3 without Boundary Prior (G-Clamping + VW-Gate only)
        hcgae_kwargs = {k: v for k, v in opt_defaults.items()}
        return OptimalHCGAE_v3(
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
            use_g_clamp=True,
            g_clamp_margin_k=1.5,
            use_vw_gate=True,
            snr_target=0.5,
            snr_min_scale=0.1,
            snr_ema_alpha=0.05,
            use_boundary_prior=False,
            **hcgae_kwargs
        )

    elif algo_name == "Standard_HCGAE_v2":
        # HCGAE v2 built on Standard PPO (no obs norm, no adv norm, no LR anneal).
        # This isolates whether HCGAE improvement is independent of Optimal PPO tricks
        # or whether it only emerges when coupled with those implementation improvements.
        std_cfg = dict(opt_defaults)
        std_cfg['use_obs_norm'] = False   # No observation normalization
        std_cfg['use_adv_norm'] = False   # No per-minibatch advantage normalization
        std_cfg['use_lr_anneal'] = False  # No LR annealing
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
            **std_cfg
        )

    elif algo_name == "Standard_PPO_Optimal":
        # Standard PPO using OptimalPPO framework but with all best-practice
        # tricks disabled — equivalent to a vanilla PPO baseline for fair comparison
        # against Standard_HCGAE_v2.
        std_cfg = dict(opt_defaults)
        std_cfg['use_obs_norm'] = False
        std_cfg['use_adv_norm'] = False
        std_cfg['use_lr_anneal'] = False
        return OptimalPPO(env=env, name=name, **std_cfg)

    elif algo_name == "Optimal_HCGAE_v4":
        # v4: v2 + SCR-adaptive α_max cap (single targeted fix)
        # α_max_v4 = min(α_max_v2,  SCR_ema²/(1+SCR_ema²) + 0.05)
        hcgae_kwargs = {k: v for k, v in opt_defaults.items()}
        return OptimalHCGAE_v4(
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
            scr_ema_alpha=0.1,
            scr_relax=0.05,
            **hcgae_kwargs
        )

    elif algo_name == "Optimal_HCGAE_v4_NoSCRCap":
        # Ablation: v4 with SCR cap disabled → identical to v2
        # (verifies that the SCR cap is the sole driver of any improvement)
        hcgae_kwargs = {k: v for k, v in opt_defaults.items()}
        return OptimalHCGAE_v4(
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
            scr_ema_alpha=0.1,
            scr_relax=1.0,  # relax=1.0 makes cap ≥ 1.0, always inactive
            **hcgae_kwargs
        )

    elif algo_name == "Optimal_HCGAE_v4_FixSCR":
        # v4_FixSCR: 修正 v4 的 SCR 分母问题
        # v4 原始: SCR = |mean(G-V)| / std(G)   ← 分母 std(G) 包含 V*(s) 结构方差
        # 修正后:  SCR = |mean(G-V)| / sqrt(max(Var(G)-Var(V), floor))
        #              ← 分母改为纯 MC 条件方差 E[Var(G|s)]
        # 全方差定律: Var(G) = Var(V*) + E[Var(G|s)]
        # 当 Critic 质量中等-好时，修正后 SCR 更大，alpha_cap 更宽松
        hcgae_kwargs = {k: v for k, v in opt_defaults.items()}
        return OptimalHCGAE_v4_FixSCR(
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
            scr_ema_alpha=0.1,
            scr_relax=0.05,
            var_floor_frac=0.05,
            **hcgae_kwargs
        )

    elif algo_name == "Optimal_HCGAE_v5":
        # v5: Universal agent = v3 (G-Clamping + VW-Gate + Boundary Prior)
        #                      + v4 (SCR-adaptive α_max cap)
        # Primary agent for the large-scale 1M-step / 12-seed experiment.
        # Single config works across episodic (Hopper/Walker2d), dense (HalfCheetah),
        # and high-variance dense (Ant-v4) environments.
        hcgae_kwargs = {k: v for k, v in opt_defaults.items()}
        return OptimalHCGAE_v5(
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
            # v3: G-Clamping, VW-Gate, Boundary Prior
            use_g_clamp=True,
            g_clamp_margin_k=1.5,
            use_vw_gate=True,
            snr_target=0.5,
            snr_min_scale=0.1,
            snr_ema_alpha=0.05,
            use_boundary_prior=True,
            # v4: SCR α* cap
            scr_ema_alpha=0.1,
            scr_relax=0.05,
            use_scr_cap=True,
            **hcgae_kwargs
        )

    elif algo_name == "Optimal_HCGAE_v6":
        # v6: v4 (v2 + SCR-cap) + FIX A (variance-normalised boundary)
        #                        + FIX B (relative EV saturation gate)
        # Universal fix: works across all environments.
        #   FIX A: boundary suppression is now scaled by critic std σ_V,
        #          so early-training HC boundary is no longer locked at 0.
        #   FIX B: suppresses correction ONLY when EV >> EV_initial (Critic
        #          already good); gate is silent on low-EV envs (HC, Ant).
        hcgae_kwargs = {k: v for k, v in opt_defaults.items()}
        return OptimalHCGAE_v6(
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
            # v4: SCR α* cap
            scr_ema_alpha=0.1,
            scr_relax=0.05,
            # v6-specific: normalised boundary + relative EV gate
            use_norm_boundary=True,
            boundary_norm_threshold=-3.0,
            v_std_ema_alpha=0.05,
            use_relative_ev_gate=True,
            ev_headroom_frac=0.7,   # gate silent unless EV rises 70% of headroom
            **hcgae_kwargs
        )

    elif algo_name == "Optimal_HCGAE_HeuristicV2":
        # HeuristicV2: Heuristic_HCGAE 的系统性改进
        # = v4 (v2 + SCR_cap) + Noise-Normalized Per-Step Sigmoid
        #
        # 核心改进：修正 per-step sigmoid 的批内 z-score 归一化缺陷
        # 原始 (Heuristic/v2): z_t = β·(|V-G| - μ_e) / σ_e  (强制 50% 步骤修正)
        # 改进 (HeuristicV2):  z_t = β·(|V-G|/σ_G_ema - threshold)  (信噪比感知)
        #
        # 保留: α_max=0.7 的高初始值（HC 高模式逃脱优势）
        # 新增: SCR_cap（v4 继承，防止 Ant/Walker 过度修正）
        # 新增: noise_threshold 归一化（使修正量与 MC 噪声水平挂钩）
        hcgae_kwargs = {k: v for k, v in opt_defaults.items()}
        return OptimalHCGAE_HeuristicV2(
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
            # v4: SCR α* cap
            scr_ema_alpha=0.1,
            scr_relax=0.05,
            # HeuristicV2: noise-normalized sigmoid
            noise_threshold=0.5,
            g_std_ema_alpha=0.1,
            use_noise_normalized_sigmoid=True,
            **hcgae_kwargs
        )

    elif algo_name == "Optimal_HCGAE_Optimal":
        # HCGAE_Optimal: 三学科统一推导的理论最完整实现
        # = v4_FixSCR (FixSCR 分母修正) + HeuristicV2 (MC 归一化 per-step sigmoid)
        #
        # 理论依据（三者等价）:
        #   James-Stein 估计器:  α* = (B²+σ²_V) / (B²+σ²_V+σ²_G)
        #   卡尔曼滤波最优增益:  K = (B²+σ²_V) / (B²+σ²_V+σ²_G)
        #   贝叶斯后验期望:      α = 1 - K (命名约定不同)
        #
        # 改进1 (vs v4): SCR 分母从 std(G) → sqrt(Var(G)-Var(V))
        #   依据: E[Var(G|s)] ≈ Var(G) - Var(V)  (全方差定律 + Critic 质量假设)
        #   效果: α_cap 提升 1.5-2× (Critic 越好提升越大)
        #
        # 改进2 (vs v4): per-step sigmoid 从批内 z-score → MC 噪声归一化
        #   依据: per-step α_t* ∝ SNR_t = |δ_t| / σ_G (局部卡尔曼最优)
        #   效果: Critic 好时整批 α→0，避免无效修正
        hcgae_kwargs = {k: v for k, v in opt_defaults.items()}
        return OptimalHCGAE_Optimal(
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
            # FixSCR: 修正分母
            scr_ema_alpha=0.1,
            scr_relax=0.05,
            var_floor_frac=0.05,
            # Noise-Normalized Per-Step Sigmoid
            noise_threshold=0.5,
            g_std_ema_alpha=0.1,
            **hcgae_kwargs
        )

    elif algo_name in ("Standard_GRPO", "Optimal_GRPO"):
        # Standard/Optimal GRPO: Group Relative Policy Optimization (基线)
        # Shao et al. (2024): DeepSeekMath. 适配连续控制环境。
        # 组优势 = (r_i - mean) / std  [批内标准化]，附带 Critic baseline + obs_norm 等
        # Optimal_GRPO 是 Standard_GRPO 的改名别名（保持向后兼容）
        return StandardGRPO(env=env, name=name, **opt_defaults)

    elif algo_name in ("HCGAE_GRPO", "HCGAE_Optimal_GRPO"):
        # HCGAE_GRPO / HCGAE_Optimal_GRPO: GRPO + HCGAE 方差分解 + SNR 感知加权
        # 理论依据 (James-Stein/Kalman):
        #   GRPO 标准差 std(r) = sqrt(Var(V*) + E[Var(G|s)])
        #   纯MC噪声  σ_G = sqrt(E[Var(G|s)]) = sqrt(max(Var(r)-Var(V),floor))
        #   加权系数  w_i ∝ SNR_i = |r_i - mean(r)| / σ_G   [局部卡尔曼]
        # HCGAE_Optimal_GRPO 是 HCGAE_GRPO 的改名别名（保持向后兼容）
        return HCGAE_GRPO(env=env, name=name, **opt_defaults)

    elif algo_name == "HCGAE_GRPO_NoCritic":
        # 消融：去掉 Critic，纯组内 SNR 加权 (无价值函数基线)
        return HCGAE_GRPO(env=env, name=name, use_critic=False, **opt_defaults)

    elif algo_name == "Standard_GRPO_NoTrick":
        # Standard GRPO (No Trick): 最纯粹的 GRPO 基线，去掉所有 PPO-style tricks
        # 对比 Standard_GRPO (Optimal_GRPO)，本算法去掉:
        #   - obs normalization (use_obs_norm=False)
        #   - advantage normalization (use_adv_norm=False)
        #   - lr annealing (use_lr_anneal=False)
        #   - Critic baseline (use_critic_baseline=False): 纯组内 MC 标准化
        # 完全对应原始 GRPO 论文设定，无任何额外稳定化 trick
        no_trick_cfg = dict(opt_defaults)
        no_trick_cfg['use_obs_norm']  = False
        no_trick_cfg['use_adv_norm']  = False
        no_trick_cfg['use_lr_anneal'] = False
        return StandardGRPO(
            env=env, name=name,
            grpo_group_norm=True,
            use_critic_baseline=False,   # 纯组内标准化，无 Critic
            **no_trick_cfg
        )

    elif algo_name == "HCGAE_Standard_GRPO":
        # HCGAE_Standard_GRPO: 在 Standard_GRPO_NoTrick 基础上叠加 HCGAE 方差修正
        # 对比关系:
        #   Standard_GRPO_NoTrick: 纯 GRPO，无任何 trick
        #   HCGAE_Standard_GRPO:   纯 GRPO + HCGAE FixSCR + SNR 加权 (无其他 trick)
        # 验证 HCGAE 修正在"公平"基线上的增益（排除 obs_norm 等 PPO tricks 影响）
        no_trick_cfg = dict(opt_defaults)
        no_trick_cfg['use_obs_norm']  = False
        no_trick_cfg['use_adv_norm']  = False
        no_trick_cfg['use_lr_anneal'] = False
        return HCGAE_GRPO(
            env=env, name=name,
            use_critic=True,             # 保留 Critic 供 FixSCR 分母修正使用
            use_snr_weight=True,         # SNR 感知加权
            use_gae_blend=True,          # EV-driven GAE/GRPO 混合
            **no_trick_cfg
        )

    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


# =============================================================================
#  GRPO for Continuous Control (MuJoCo)
#  理论框架: 跨学科统一 (James-Stein / Kalman / Bayes)
# =============================================================================

class StandardGRPO(OptimalPPO):
    """
    Standard GRPO (Group Relative Policy Optimization) 适配连续控制环境。

    ═══════════════════════════════════════════════════════════════════════════
    原始 GRPO (Shao et al. 2024, DeepSeekMath) 的核心：
      对问题 q 采样 G 个响应 {o_i}，用组内相对优势替代 GAE：
        A_i = (r_i - mean_j(r_j)) / std_j(r_j)
      不需要 Critic，完全依赖 MC 回报的组内标准化。

    ═══════════════════════════════════════════════════════════════════════════
    连续控制适配：
      "组" = 一个 rollout 窗口内的 n_steps 步
      "响应" = 每步的折扣累积回报 G_t (MC returns)
      优势 = (G_t - mean(G)) / std(G)  [与 GRPO 原始公式完全对应]

      对比 PPO-clip 的差异:
        PPO: A_t = GAE_t (TD-λ 混合估计, 需要 Critic)
        GRPO: A_t = (G_t - μ_G) / σ_G  (纯组内 MC, 不需要 Critic)

    ═══════════════════════════════════════════════════════════════════════════
    参数
    ─────────────────────────────────────────────────────────────────────────
    grpo_group_norm : bool
        是否使用 GRPO 式组内标准化。True 为 GRPO，False 退化为 MC-returns PPO。
    use_critic_baseline : bool
        True = 用 Critic V(s) 做基线，优势 = G_t - V(s_t) [减方差]
        False = 纯 GRPO 组内标准化 [原始 GRPO，无 Critic 基线]
    """

    NAME = "Standard_GRPO"

    def __init__(
        self,
        env: gym.Env,
        name: str = "Standard_GRPO",
        grpo_group_norm: bool = True,
        use_critic_baseline: bool = True,   # True: GRPO+Critic baseline
        **kwargs,
    ):
        super().__init__(env=env, name=name, **kwargs)
        self.grpo_group_norm = grpo_group_norm
        self.use_critic_baseline = use_critic_baseline

    def compute_gae(self, last_value: float):
        """
        GRPO 优势计算：组内 MC 回报标准化。

        A_t = (G_t - μ_G) / σ_G    [纯 GRPO]
        or
        A_t = (G_t - V(s_t))        [GRPO with Critic baseline, 后续 minibatch 归一化]

        Critic 训练目标: MC 回报 G_t (不使用 λ-return)
        """
        T = self.buffer.pos
        rewards = self.buffer.rewards[:T]
        terminated = self.buffer.terminated[:T]
        values = self.buffer.values[:T]

        # Step 1: MC returns (折扣累积回报)
        returns_mc = np.zeros(T, dtype=np.float32)
        running_return = last_value
        for t in reversed(range(T)):
            if terminated[t]:
                running_return = 0.0
            running_return = rewards[t] + self.gamma * running_return
            returns_mc[t] = running_return

        if self.use_critic_baseline:
            # GRPO + Critic baseline: A_t = G_t - V(s_t)
            # 后续 minibatch 归一化会进一步标准化
            advantages = returns_mc - values
        else:
            # 纯 GRPO: A_t = (G_t - μ) / σ
            mu_g = np.mean(returns_mc)
            sigma_g = np.std(returns_mc) + 1e-8
            advantages = (returns_mc - mu_g) / sigma_g

        self.buffer.advantages[:T] = advantages
        self.buffer.returns[:T] = returns_mc   # Critic 学 MC 回报


class HCGAE_GRPO(StandardGRPO):
    """
    HCGAE_GRPO: GRPO + Hindsight-Corrected 方差分解 + SNR-Aware 加权

    ═══════════════════════════════════════════════════════════════════════════
    理论框架（跨学科统一视角验证）
    ═══════════════════════════════════════════════════════════════════════════

    **问题诊断：标准 GRPO 的方差高估**

    GRPO 组内标准化: A_i = (r_i - μ) / σ_r
    其中 σ_r = std(G_t) = sqrt(Var(G_t))

    全方差定律分解:
        Var(G_t) = Var(V*(s_t)) + E[Var(G_t | s_t)]
                   ────────────   ──────────────────
                   状态价值结构    纯 MC 随机噪声

    问题: σ_r 高估了"真正的噪声"，导致优势被过度缩放：
        若 Var(V*(s)) 大（如 HalfCheetah），则 σ_r >> σ_G
        → 真实信号被稀释，梯度过小，学习缓慢

    **HCGAE_GRPO 修正**

    ① 分母修正 (FixSCR): σ_G = sqrt(max(Var(G) - Var(V), floor))
        依据: E[Var(G|s)] ≈ Var(G) - Var(V)  (全方差定律 + Critic 质量假设)
        效果: 优势量级提升 1.5-2× (Critic 越好提升越大)

    ② SNR 感知加权 (per-step Kalman):
        w_t = σ(β · (|G_t - V(s_t)| / σ_G - θ))
        依据: per-step α_t* ∝ SNR_t = |δ_t| / σ_G (卡尔曼局部最优)
        效果: 误差大步骤权重高，Critic 好时整批权重低

    ③ 修正优势 (混合):
        A_t^HCGAE = w_t · (G_t - V(s_t)) / σ_G_corrected
                  + (1-w_t) · GAE_t / std(GAE)   [如果启用 GAE blend]

    ═══════════════════════════════════════════════════════════════════════════
    与 HCGAE_Optimal (PPO 版) 的对比
    ─────────────────────────────────────────────────────────────────────────
    PPO 版:  修正 V^c = (1-α)V + αG, 再用 V^c 计算 TD-δ → 影响 Actor
    GRPO 版: 直接修正 MC 优势的标准化分母 + 加权 → 更纯粹，无需 TD 传播

    ═══════════════════════════════════════════════════════════════════════════
    参数
    ─────────────────────────────────────────────────────────────────────────
    use_critic : bool
        是否使用 Critic 网络 (True: 提供 V(s) 基线 + FixSCR)
    noise_threshold : float
        SNR 阈值: |δ_t| / σ_G > threshold → 开始加权修正 (默认 0.5)
    snr_beta : float
        SNR sigmoid 的 β 系数 (默认 3.0, 与 HCGAE v4 一致)
    g_std_ema_alpha : float
        MC 回报标准差 σ_G 的 EMA 平滑系数 (默认 0.1)
    var_floor_frac : float
        分母下界 = var_floor_frac × Var(G) (防止分母趋零, 默认 0.05)
    use_snr_weight : bool
        是否启用 SNR 感知加权 (True: HCGAE_GRPO, False: 仅 FixSCR)
    use_gae_blend : bool
        是否与标准 GAE 优势混合 (EV-driven blend, 默认 True)
    ev_ema_alpha : float
        EV EMA 平滑系数 (默认 0.1)
    """

    NAME = "HCGAE_GRPO"

    def __init__(
        self,
        env: gym.Env,
        name: str = "HCGAE_GRPO",
        # ── HCGAE 核心参数 ──
        use_critic: bool = True,
        noise_threshold: float = 0.5,      # SNR 阈值 θ
        snr_beta: float = 3.0,             # sigmoid 斜率
        g_std_ema_alpha: float = 0.1,      # σ_G EMA 系数
        var_floor_frac: float = 0.05,      # 分母下界
        use_snr_weight: bool = True,       # SNR 感知加权
        use_gae_blend: bool = True,        # EV-driven GAE/GRPO 混合
        ev_ema_alpha: float = 0.1,         # EV EMA 系数
        # ── 继承所有 StandardGRPO 参数 ──
        **kwargs,
    ):
        # GRPO: use_critic_baseline=True 保留 Critic (V(s) 做基线)
        super().__init__(env=env, name=name, use_critic_baseline=True, **kwargs)
        self.use_critic = use_critic
        self.noise_threshold = noise_threshold
        self.snr_beta = snr_beta
        self.g_std_ema_alpha = g_std_ema_alpha
        self.var_floor_frac = var_floor_frac
        self.use_snr_weight = use_snr_weight
        self.use_gae_blend = use_gae_blend
        self.ev_ema_alpha = ev_ema_alpha

        # 在线状态
        self._g_std_ema = 1.0        # σ_G (MC 回报标准差) EMA
        self._ev_ema = 0.0           # EV EMA

    def compute_gae(self, last_value: float):
        """
        HCGAE_GRPO 优势计算：FixSCR + SNR-Aware 加权。

        数学:
          σ_G_corrected = sqrt(max(Var(G) - Var(V), floor))  [FixSCR]
          w_t = σ(β·(|G_t - V_t| / σ_G - θ))               [SNR 加权]
          A_t = w_t·(G_t - V_t)/σ_G + (1-w_t)·GAE_t/std(GAE)  [混合]
        """
        T = self.buffer.pos
        rewards = self.buffer.rewards[:T]
        terminated = self.buffer.terminated[:T]
        values = self.buffer.values[:T]

        # ── Step 1: MC returns ──────────────────────────────────────────────
        returns_mc = np.zeros(T, dtype=np.float32)
        running_return = last_value
        for t in reversed(range(T)):
            if terminated[t]:
                running_return = 0.0
            running_return = rewards[t] + self.gamma * running_return
            returns_mc[t] = running_return

        # ── Step 2: 标准 GAE (用于混合基准) ────────────────────────────────
        nv = np.zeros(T, dtype=np.float32)
        for t in range(T):
            if terminated[t] > 0.5:
                nv[t] = 0.0
            elif t == T - 1:
                nv[t] = last_value
            else:
                nv[t] = values[t + 1]
        std_gae = np.zeros(T, dtype=np.float32)
        gae_acc = 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * nv[t] - values[t]
            not_done = 1.0 - terminated[t]
            gae_acc = delta + self.gamma * self.lam * not_done * gae_acc
            std_gae[t] = gae_acc

        # ── Step 3: FixSCR 分母修正 ─────────────────────────────────────────
        var_g = np.var(returns_mc) + 1e-8
        var_v = np.var(values) + 1e-8

        # EV: 当 Critic 好时，Var(V) ≈ Var(V*)，E[Var(G|s)] ≈ Var(G)-Var(V)
        var_g_corrected = max(var_g - var_v, self.var_floor_frac * var_g)
        sigma_g_corrected = float(np.sqrt(var_g_corrected)) + 1e-8

        # 更新 σ_G EMA (用于在线诊断和 per-step SNR)
        self._g_std_ema = (
            (1 - self.g_std_ema_alpha) * self._g_std_ema
            + self.g_std_ema_alpha * sigma_g_corrected
        )

        # ── Step 4: per-step 误差 (|G_t - V_t|) ────────────────────────────
        errors = np.abs(returns_mc - values)  # = |δ_hindsight_t|

        # ── Step 5: SNR 感知权重 w_t ────────────────────────────────────────
        if self.use_snr_weight:
            # w_t = σ(β·(SNR_t - θ))   SNR_t = |δ_t| / σ_G
            snr_t = errors / self._g_std_ema
            z = self.snr_beta * (snr_t - self.noise_threshold)
            w = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
        else:
            w = np.ones(T, dtype=np.float32)

        # ── Step 6: EV 驱动混合系数 ─────────────────────────────────────────
        var_y = np.var(returns_mc) + 1e-8
        ev_now = float(1.0 - np.var(returns_mc - values) / var_y)
        self._ev_ema = (1 - self.ev_ema_alpha) * self._ev_ema + self.ev_ema_alpha * ev_now

        # ev_blend: EV 高 (Critic 好) → 更依赖 GRPO-FixSCR 修正；
        #            EV 低 (Critic 差) → 退回标准 GAE (稳健)
        ev_blend = float(np.clip(self._ev_ema, 0.0, 1.0))

        # ── Step 7: 最终优势 ─────────────────────────────────────────────────
        # GRPO 部分: w_t·(G_t - V_t) / σ_G_corrected
        grpo_adv = w * (returns_mc - values) / sigma_g_corrected

        if self.use_gae_blend:
            # GAE 部分 (归一化, 量纲对齐)
            std_gae_std = np.std(std_gae) + 1e-8
            gae_adv_normed = std_gae / std_gae_std

            # EV-driven 混合: ev_blend=1→全 GRPO, ev_blend=0→全 GAE
            advantages = ev_blend * grpo_adv + (1.0 - ev_blend) * gae_adv_normed
        else:
            advantages = grpo_adv

        # ── Step 8: Critic 目标 (EV-adaptive MC/GAE 混合) ───────────────────
        c_mc = float(np.clip(1.0 - self._ev_ema, 0.1, 1.0))
        std_gae_returns = std_gae + values  # GAE-returns
        critic_returns = c_mc * returns_mc + (1 - c_mc) * std_gae_returns

        self.buffer.advantages[:T] = advantages
        self.buffer.returns[:T] = critic_returns
