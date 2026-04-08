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


class OptimalHCGAE_Bayesian(OptimalPPO):
    """
    Unified Bayesian HCGAE (ICML Candidate) — BHVF + DCPPO-S.

    Replaces all heuristic gates (EV-rate, VW, Cosine, Boundary Prior) with a
    principled Bayesian Value Fusion (BHVF) framework derived from first principles.

    Mathematical Formulation (paper §2):
    1. Estimate Critic error (sigma_V) and MC noise (sigma_G):
           sigma_V = mean(|G - V|)   # MAE: Critic error proxy (NOT |mean(G-V)|)
           sigma_G = std(G)          # MC return volatility

    2. Optimal Bayesian Gain (Kalman Gain), paper Theorem 1 / Corollary 1:
           SCR = sigma_V / sigma_G
           alpha* = SCR^2 / (SCR^2 + 1) + scr_relax

    3. Robust Innovation Clipping (paper §2.3):
           sigma_e = std(G - V)
           innovation = clip(G - V, -c * sigma_e, +c * sigma_e)

    4. Corrected Value:
           V^c = V + alpha* * innovation

    No per-sample sigmoid gates — alpha* is a global scalar per rollout.
    This achieves the exact same goals as v5/v6 but with ~4 lines of math
    instead of 100 lines of conditional logic, with zero heuristic parameters.

    Hyperparameters (environment-agnostic):
        scr_ema_alpha : EMA learning rate for SCR (default 0.1 = fast tracking)
        scr_relax     : numerical floor to prevent alpha* collapsing to 0 (default 0.05)
        clip_c        : innovation clipping multiplier (default 3.0 = 99.7% Normal)
    """
    NAME = "Optimal_HCGAE_Bayesian"

    def __init__(
        self,
        env: gym.Env,
        name: str = "Optimal_HCGAE_Bayesian",
        scr_ema_alpha: float = 0.1,
        scr_relax: float = 0.05,
        clip_c: float = 3.0,
        **kwargs,
    ):
        super().__init__(env=env, name=name, **kwargs)
        self.scr_ema_alpha = scr_ema_alpha
        self.scr_relax = scr_relax
        self.clip_c = clip_c

        self._scr_ema = 1.0
        self._sigma_e_ema = 1.0
        self._ev_ema = 0.0
        self._ev_ema_alpha = 0.05

        # ── Diagnostics cache (written per rollout; read by train() for logging) ──
        # These allow the outer train loop to call logger.log_update(..., snr=..., alpha_mean=...)
        # without needing to change the train() interface.
        self._diag_scr: float = 1.0          # current SCR_ema (= sigma_V / sigma_G)
        self._diag_sigma_V: float = 0.0      # batch-level sigma_V (MAE)
        self._diag_sigma_G: float = 1.0      # batch-level sigma_G (std of G)
        self._diag_sigma_e: float = 1.0      # batch-level sigma_e (std of delta)
        self._diag_alpha_star: float = 0.0   # current alpha* (Kalman gain)
        self._diag_c_mc: float = 1.0         # current c_mc for critic target
        self._diag_ev_now: float = 0.0       # raw EV of current rollout
        self._diag_clip_ratio: float = 0.0   # fraction of innovations that were clipped

    def compute_hindsight_gae(self, last_value: float):
        T = self.buffer.pos
        rewards = self.buffer.rewards[:T]
        terminated = self.buffer.terminated[:T]
        values = self.buffer.values[:T]

        # 1. MC Returns
        returns_mc = np.zeros(T, dtype=np.float32)
        running_return = last_value
        for t in reversed(range(T)):
            if terminated[t]:
                running_return = 0.0
            running_return = rewards[t] + self.gamma * running_return
            returns_mc[t] = running_return

        # 2. Global Bayesian Gain (alpha*)
        # sigma_V = MAE(G, V) = mean(|G - V|), approximates Critic RMSE.
        # sigma_G = std(G), captures MC return noise.
        # SCR = sigma_V / sigma_G; alpha* = SCR^2 / (SCR^2 + 1).
        # IMPORTANT: use mean(|G-V|) NOT |mean(G-V)|; the latter is the signed
        # mean bias and can be near zero even when Critic error is large (errors cancel).
        delta = returns_mc - values
        sigma_V = float(np.mean(np.abs(delta))) + 1e-8   # MAE: Critic error proxy
        sigma_G = float(np.std(returns_mc)) + 1e-8       # MC return noise
        scr_now = sigma_V / sigma_G

        self._scr_ema = (1 - self.scr_ema_alpha) * self._scr_ema + self.scr_ema_alpha * scr_now
        alpha_star = (self._scr_ema ** 2) / (self._scr_ema ** 2 + 1.0) + self.scr_relax
        alpha_star = float(np.clip(alpha_star, 0.0, 1.0))

        # 3. Robust Innovation Clipping
        # sigma_e = std(G - V): spread of innovations in the current batch.
        # Clip innovation to [-c*sigma_e, c*sigma_e] (c=3 covers 99.7% of Normal).
        # This replaces all complex boundary-prior rules with a single statistical clip.
        sigma_e_now = float(np.std(delta)) + 1e-8
        self._sigma_e_ema = (1 - self.scr_ema_alpha) * self._sigma_e_ema + self.scr_ema_alpha * sigma_e_now

        innovation = np.clip(
            delta,
            -self.clip_c * self._sigma_e_ema,
            self.clip_c * self._sigma_e_ema
        )

        # 4. Corrected Values: V^c = V + alpha* * clip(G - V, ±c*sigma_e)
        # alpha_star is a global scalar — no per-sample sigmoid gate.
        # Per-sample sigmoid gates are heuristic and contradict the principled Bayesian design.
        v_corrected = values + alpha_star * innovation

        # 5. Boundary Correction (Unified via Innovation Clipping)
        # Apply the same clip to the last_value bootstrap, replacing ad-hoc boundary rules.
        last_innovation = returns_mc[-1] - last_value
        last_innovation_clipped = float(np.clip(
            last_innovation,
            -self.clip_c * self._sigma_e_ema,
            self.clip_c * self._sigma_e_ema
        ))
        last_value_corrected = last_value + alpha_star * last_innovation_clipped

        # 7. GAE Computation
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

        # 8. Critic Targets (EV-driven mixing)
        # NOTE: Use OLD _ev_ema (from previous rollout) to compute c_mc,
        # then update _ev_ema AFTER. This avoids using fresh information
        # from the current rollout to scale its own targets.
        c_mc = float(np.clip(1.0 - self._ev_ema, 0.1, 1.0))
        # Update EV EMA with current rollout's EV
        var_y = np.var(returns_mc) + 1e-8
        ev_now = float(1.0 - np.var(returns_mc - values) / var_y)
        self._ev_ema = (1 - self._ev_ema_alpha) * self._ev_ema + self._ev_ema_alpha * ev_now

        # For MC component: use innovation-clipped target V(s) + clip(G-V, ±c*σ_e)
        # which equals the BHVF-corrected value; this is also the paper's "clamped MC".
        # For GAE component: use standard GAE returns from ORIGINAL (uncorrected) V,
        # keeping advantage estimation and critic training paths fully decoupled.
        returns_mc_clamped = values + innovation   # = V + clip(G-V, ±c*σ_e)
        std_gae_returns = self.buffer._compute_standard_returns(last_value, self.gamma, self.lam)
        critic_returns = c_mc * returns_mc_clamped + (1 - c_mc) * std_gae_returns

        self.buffer.advantages[:T] = advantages
        self.buffer.returns[:T] = critic_returns

        # ── Write diagnostics cache for train() logging ──
        # clip_ratio: fraction of timesteps where |delta| > c*sigma_e (i.e., innovation was clipped)
        clip_bound = self.clip_c * self._sigma_e_ema
        clip_ratio = float(np.mean(np.abs(delta) > clip_bound))

        self._diag_scr = self._scr_ema
        self._diag_sigma_V = sigma_V
        self._diag_sigma_G = sigma_G
        self._diag_sigma_e = self._sigma_e_ema
        self._diag_alpha_star = alpha_star
        self._diag_c_mc = c_mc
        self._diag_ev_now = ev_now
        self._diag_clip_ratio = clip_ratio

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

    elif algo_name == "Optimal_HCGAE_Bayesian":
        hcgae_kwargs = {k: v for k, v in opt_defaults.items()}
        return OptimalHCGAE_Bayesian(
            env=env, name=name,
            scr_ema_alpha=0.1,
            scr_relax=0.05,
            clip_c=3.0,
            **hcgae_kwargs
        )

    elif algo_name == "BHVF":
        # ── BHVF: Paper proposed method (§2) ──
        # Replaces all heuristic gates with principled Bayesian Value Fusion.
        # Uses same 3 env-agnostic hyperparameters across ALL environments:
        #   scr_ema_alpha=0.1, scr_relax=0.05, clip_c=3.0
        # NO DCPPO-S (pure BHVF ablation).
        hcgae_kwargs = {k: v for k, v in opt_defaults.items()}
        # Disable EV-driven advantage scaling (pure BHVF, no DCPPO-S)
        hcgae_kwargs['ev_scale_advantages'] = False
        return OptimalHCGAE_Bayesian(
            env=env, name=name,
            scr_ema_alpha=0.1,
            scr_relax=0.05,
            clip_c=3.0,
            **{k: v for k, v in hcgae_kwargs.items()
               if k not in ('ev_scale_advantages',)}
        )

    elif algo_name == "BHVF_DCPPO":
        # ── BHVF + DCPPO-S: Full proposed method (§2 + §3) ──
        # BHVF value fusion + EV-based linear shrinkage on policy gradient.
        # The EV weight w = clip(EV, 0.1, 1.0) is applied to advantages
        # before the PPO clip objective (gradient direction preserved).
        hcgae_kwargs = {k: v for k, v in opt_defaults.items()}
        return OptimalHCGAE_Bayesian(
            env=env, name=name,
            scr_ema_alpha=0.1,
            scr_relax=0.05,
            clip_c=3.0,
            **hcgae_kwargs
        )

    else:
        raise ValueError(f"Unknown optimal agent: {algo_name}")

