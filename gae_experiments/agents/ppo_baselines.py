"""
PPO Baseline Variants for Comparison
=====================================

Implements several published PPO improvement methods as baselines
for fair comparison with HCGAE and DCPPO:

1. PPO-KL (KLPEN) - Schulman et al. (2017)
   Original paper's KL penalty variant: L = L_clip - beta * KL(old || new)
   Adaptive beta control via dual KL threshold.

2. PPO-Anneal - Linear LR annealing (common practice, e.g., OpenAI baseline)
   Linearly decays learning rate from lr_init to 0 over training.

3. PPO-EntropyDecay - Entropy bonus with annealing
   Starts with higher entropy bonus (exploration) then decays.

4. PPO-VClip - Value clipping (Engstrom et al. 2020, Spinning Up implementation)
   Clips the value loss to prevent Critic from changing too much in one update.

5. PPO-Dual - Dual optimizer with separate gradient schedules for actor/critic
   Actor: Adam with gradient clipping = 0.5
   Critic: Adam with more aggressive LR and gradient clipping = 1.0

All variants use the same network architecture and rollout procedure as HCGAE
for fair comparison (no cherry-picking).

References:
- Schulman et al. (2017): "Proximal Policy Optimization Algorithms"
  https://arxiv.org/abs/1707.06347
- Engstrom et al. (2020): "Implementation Matters in Deep RL: A Case Study on PPO and TRPO"
  https://openreview.net/forum?id=r1etN1rtPB
- Andrychowicz et al. (2021): "What Matters for On-Policy Deep Actor-Critic Methods? A Large-Scale Study"
  https://arxiv.org/abs/2006.05990
"""
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from ..utils.logger import MetricLogger
from ..utils.networks import ActorNetwork, CriticNetwork
from ..utils.rollout_buffer import RolloutBuffer


class PPOBaseline:
    """
    Configurable PPO baseline with published improvement techniques.
    Shares rollout/network architecture with HCGAE for fair comparison.
    """

    def __init__(
        self,
        env: gym.Env,
        name: str = "Standard_PPO",
        # ── Improvement switches ──
        use_klpen: bool = False,       # PPO-KL penalty (instead of clip)
        use_lr_anneal: bool = False,   # Linear LR annealing
        use_ent_decay: bool = False,   # Entropy bonus with annealing
        use_vclip: bool = False,       # Value function clipping
        use_dual_lr: bool = False,     # Dual optimizer with separate LR schedule
        # ── Standard PPO hyperparams ──
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
        # ── KL penalty params ──
        kl_coef_init: float = 0.2,    # initial beta for KL penalty
        kl_target: float = 0.01,      # target KL for adaptive beta
        kl_coef_max: float = 2.0,
        kl_coef_min: float = 0.0001,
        # ── Entropy decay params ──
        ent_coef_init: float = 0.01,  # starting entropy coefficient
        ent_decay_steps: int = 200000, # steps over which to decay to ent_coef
        # ── Value clip params ──
        vclip_eps: float = 0.2,        # value clip epsilon (same as eps_clip)
        # ── Dual LR params ──
        lr_critic_dual: float = 3e-3,  # more aggressive critic LR in dual mode
        # ── Other ──
        device: str = "cpu",
        save_dir: str = "results",
    ):
        self.NAME = name
        self.use_klpen = use_klpen
        self.use_lr_anneal = use_lr_anneal
        self.use_ent_decay = use_ent_decay
        self.use_vclip = use_vclip
        self.use_dual_lr = use_dual_lr

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

        # KL penalty
        self.kl_coef = kl_coef_init
        self.kl_target = kl_target
        self.kl_coef_max = kl_coef_max
        self.kl_coef_min = kl_coef_min

        # Entropy decay
        self.ent_coef_init = ent_coef_init
        self.ent_coef_final = ent_coef
        self.ent_decay_steps = ent_decay_steps

        # Value clip
        self.vclip_eps = vclip_eps

        # LR scheduling
        self.lr_actor_init = lr_actor
        self.lr_critic_init = lr_critic if not use_dual_lr else lr_critic_dual
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

        self.actor = ActorNetwork(obs_dim, action_dim, hidden_dim, self.continuous).to(self.device)
        self.critic = CriticNetwork(obs_dim, hidden_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                  lr=self.lr_critic_init)

        self.buffer = RolloutBuffer(n_steps, obs_dim, action_dim, self.device, self.continuous)

        self.logger = MetricLogger(self.NAME, save_dir)
        self.total_steps = 0

    # ────────────────────────────────────────────────────────────────
    # Data collection (same as HCGAE for fair comparison)
    # ────────────────────────────────────────────────────────────────
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

    # ────────────────────────────────────────────────────────────────
    # Standard GAE computation
    # ────────────────────────────────────────────────────────────────
    def compute_gae(self, last_value: float):
        self.buffer.compute_standard_gae(last_value, self.gamma, self.lam)

    # ────────────────────────────────────────────────────────────────
    # PPO Update with optional improvements
    # ────────────────────────────────────────────────────────────────
    def update(self) -> dict:
        obs, actions, old_log_probs, advantages, returns, old_values = self.buffer.get_batch()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute current entropy coef (with annealing if enabled)
        if self.use_ent_decay:
            progress = min(self.total_steps / max(self._total_timesteps, 1), 1.0)
            # Linear decay from ent_coef_init to ent_coef_final
            current_ent_coef = (
                self.ent_coef_init
                + (self.ent_coef_final - self.ent_coef_init) * progress
            )
        else:
            current_ent_coef = self.ent_coef

        # Compute current LR (with annealing if enabled)
        if self.use_lr_anneal:
            progress = 1.0 - min(self.total_steps / max(self._total_timesteps, 1), 1.0)
            current_lr_actor = self.lr_actor_init * progress
            current_lr_critic = self.lr_critic_init * progress
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = max(current_lr_actor, 1e-6)
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = max(current_lr_critic, 1e-6)

        T = self.buffer.pos
        indices = np.arange(T)
        metrics = {
            "value_loss": 0.0,
            "policy_loss": 0.0,
            "entropy_loss": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
            "kl_coef": self.kl_coef,
        }
        update_count = 0
        total_kl = 0.0

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

                new_log_probs, entropy = self.actor.evaluate_actions(batch_obs, batch_actions)
                new_values = self.critic(batch_obs)

                ratio = torch.exp(new_log_probs - batch_old_lp)

                # ── Policy loss: KL penalty or Clip ──────────────────────
                if self.use_klpen:
                    # PPO-KLPEN: L = L_PG - beta * KL
                    # KL approx: (ratio - 1) - log(ratio)
                    approx_kl_val = ((ratio - 1) - torch.log(ratio)).mean()
                    policy_loss = -(ratio * batch_adv).mean() + self.kl_coef * approx_kl_val
                    clip_frac_val = 0.0  # no clipping in KLPEN
                else:
                    # Standard PPO clip
                    surr1 = ratio * batch_adv
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_adv
                    policy_loss = -torch.min(surr1, surr2).mean()
                    clip_frac_val = ((ratio - 1).abs() > self.eps_clip).float().mean().item()

                # ── Value loss: standard or clipped ──────────────────────
                if self.use_vclip:
                    # PPO value clipping (Engstrom et al. 2020)
                    v_clipped = batch_old_val + torch.clamp(
                        new_values - batch_old_val,
                        -self.vclip_eps,
                        self.vclip_eps,
                    )
                    value_loss = 0.5 * torch.max(
                        (new_values - batch_ret) ** 2,
                        (v_clipped - batch_ret) ** 2,
                    ).mean()
                else:
                    value_loss = 0.5 * ((new_values - batch_ret) ** 2).mean()

                entropy_loss = -entropy.mean()

                # Actor update
                self.actor_optimizer.zero_grad()
                (policy_loss + current_ent_coef * entropy_loss).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Critic update
                self.critic_optimizer.zero_grad()
                (self.vf_coef * value_loss).backward()
                if self.use_dual_lr:
                    # More aggressive gradient clipping for critic in dual mode
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                else:
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    total_kl += approx_kl

                metrics["value_loss"] += value_loss.item()
                metrics["policy_loss"] += policy_loss.item()
                metrics["entropy_loss"] += entropy_loss.item()
                metrics["approx_kl"] += approx_kl
                metrics["clip_frac"] += clip_frac_val
                update_count += 1

        if update_count > 0:
            for k in ["value_loss", "policy_loss", "entropy_loss", "approx_kl", "clip_frac"]:
                metrics[k] /= update_count

        # ── KL adaptive control (PPO-KLPEN) ──────────────────────────
        if self.use_klpen:
            mean_kl = total_kl / max(update_count, 1)
            # Dual-threshold adaptive beta (Schulman 2017)
            if mean_kl < self.kl_target / 1.5:
                self.kl_coef = max(self.kl_coef / 2.0, self.kl_coef_min)
            elif mean_kl > self.kl_target * 1.5:
                self.kl_coef = min(self.kl_coef * 2.0, self.kl_coef_max)
            metrics["kl_coef"] = self.kl_coef

        # Explained variance
        with torch.no_grad():
            y_pred = old_values.cpu().numpy()
            y_true = returns.cpu().numpy()
            var_y = np.var(y_true)
            ev = 1 - np.var(y_true - y_pred) / (var_y + 1e-8)
            metrics["explained_variance"] = float(ev)

        return metrics

    # ────────────────────────────────────────────────────────────────
    # Training loop
    # ────────────────────────────────────────────────────────────────
    def train(
        self,
        total_timesteps: int,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        verbose: bool = True,
    ):
        import time
        self._total_timesteps = total_timesteps
        train_start = time.time()
        last_eval_reward = float("nan")
        update_idx = 0

        while self.total_steps < total_timesteps:
            last_value = self.collect_rollout()
            self.compute_gae(last_value)
            metrics = self.update()
            update_idx += 1

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
                fps = int(self.total_steps / (elapsed + 1e-8))
                eval_str = f"{last_eval_reward:7.1f}" if not np.isnan(last_eval_reward) else "    N/A"
                print(
                    f"  [{self.NAME:<20}] "
                    f"{self.total_steps:7d}/{total_timesteps} "
                    f"| Eval={eval_str} Rec={recent:6.1f} "
                    f"| VL={metrics['value_loss']:.3f} EV={metrics['explained_variance']:+.3f} "
                    f"| KL={metrics['approx_kl']:.4f} clip={metrics['clip_frac']:.3f} "
                    f"| {fps:5d}fps {elapsed:5.0f}s"
                )

        if verbose:
            elapsed = time.time() - train_start
            final_r = np.mean(self.logger.eval_rewards[-5:]) if self.logger.eval_rewards else 0.0
            best_r = max(self.logger.eval_rewards) if self.logger.eval_rewards else 0.0
            print(f"  Done [{self.NAME}] | time={elapsed:.1f}s | final={final_r:.1f} | best={best_r:.1f}")

        self.logger.save()
        return self.logger

    def evaluate(self, eval_env: gym.Env, n_episodes: int = 10) -> float:
        total_reward = 0.0
        for _ in range(n_episodes):
            obs, _ = eval_env.reset()
            done = False
            ep_r = 0.0
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
                done = terminated or truncated
                obs = next_obs
            total_reward += ep_r
        return total_reward / n_episodes


# ──────────────────────────────────────────────────────────────────────────
# Variant registry
# ──────────────────────────────────────────────────────────────────────────

PPO_BASELINE_VARIANTS = {
    # Standard PPO (vanilla, no improvements)
    "Standard_PPO": dict(
        use_klpen=False, use_lr_anneal=False, use_ent_decay=False,
        use_vclip=False, use_dual_lr=False,
    ),
    # PPO-KLPEN: Original KL penalty variant (Schulman 2017)
    "PPO_KLPEN": dict(
        use_klpen=True, use_lr_anneal=False, use_ent_decay=False,
        use_vclip=False, use_dual_lr=False,
    ),
    # PPO-Anneal: Linear LR annealing (OpenAI baseline / Stable-Baselines3)
    "PPO_Anneal": dict(
        use_klpen=False, use_lr_anneal=True, use_ent_decay=False,
        use_vclip=False, use_dual_lr=False,
    ),
    # PPO-EntDecay: Entropy coefficient decay (exploration → exploitation)
    "PPO_EntDecay": dict(
        use_klpen=False, use_lr_anneal=False, use_ent_decay=True,
        use_vclip=False, use_dual_lr=False,
    ),
    # PPO-VClip: Value function clipping (Engstrom et al. 2020)
    "PPO_VClip": dict(
        use_klpen=False, use_lr_anneal=False, use_ent_decay=False,
        use_vclip=True, use_dual_lr=False,
    ),
    # PPO-Full: All improvements combined (best practical baseline)
    "PPO_Full_Baseline": dict(
        use_klpen=False, use_lr_anneal=True, use_ent_decay=True,
        use_vclip=True, use_dual_lr=False,
        ent_coef_init=0.01,
    ),
}


def build_ppo_baseline(
    variant_name: str,
    env: gym.Env,
    save_dir: str = "results",
    **kwargs,
) -> PPOBaseline:
    """Factory function to build a PPO baseline variant."""
    if variant_name not in PPO_BASELINE_VARIANTS:
        raise ValueError(
            f"Unknown baseline variant '{variant_name}'. "
            f"Available: {list(PPO_BASELINE_VARIANTS.keys())}"
        )
    flags = PPO_BASELINE_VARIANTS[variant_name]
    # kwargs can override flags but cannot override name/save_dir
    merged = {**flags, **kwargs}
    # 'name' key in kwargs should not be passed separately (already set above)
    name = merged.pop("name", variant_name)
    return PPOBaseline(env=env, name=name, save_dir=save_dir, **merged)


def get_all_baseline_names():
    return list(PPO_BASELINE_VARIANTS.keys())

