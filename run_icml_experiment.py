#!/usr/bin/env python3
"""
ICML-Level Experiment: Optimal PPO vs HCGAE
=============================================

Experiment Design:
  - Environments: Hopper-v4, Walker2d-v4, HalfCheetah-v4, Ant-v4
  - Algorithms:
      * Standard_PPO      (our previous naive baseline, for reference)
      * Optimal_PPO       (obs norm + adv norm + LR anneal, best practices)
      * Optimal_PPO_VClip (Optimal_PPO + value clipping, for ablation)
      * Optimal_HCGAE     (HCGAE built on Optimal_PPO, same tricks)
      * Optimal_HCGAE_SCR (HCGAE_SCR built on Optimal_PPO)
  - Seeds: 10 independent seeds [0,1,2,3,4,5,6,7,8,9]
  - Steps: 1,000,000 per run
  - Eval: every 10,240 steps, 10 deterministic episodes, last-5-mean

Results saved to: results/ICMLExperiment/
"""

import json
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from gae_experiments.agents.optimal_ppo import build_optimal_agent
from gae_experiments.agents.ppo_baselines import build_ppo_baseline

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
# ── Scaled-down experiment (manageable on single CPU) ──
# 3 envs × 4 algos × 5 seeds × 500k steps  →  ~60 runs, ~3–6h total
ENVS = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]
SEEDS = list(range(5))    # 5 seeds: 0..4
TOTAL_TIMESTEPS = 500_000
EVAL_FREQ = 10_240
N_EVAL_EPISODES = 10
RESULTS_DIR = Path("results/ICMLExperiment")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Algorithms to compare (4 core algorithms)
ALGORITHMS = [
    "Standard_PPO",        # Naive PPO baseline (no obs norm etc.) — reference point
    "Optimal_PPO",         # Best-practice PPO (Andrychowicz 2021)
    "Optimal_HCGAE",       # HCGAE on top of Optimal_PPO  ← key comparison
    "Optimal_HCGAE_SCR",   # HCGAE-SCR on top of Optimal_PPO
]

# Shared hyperparameters for Standard_PPO (aligned with literature)
STANDARD_PPO_KWARGS = dict(
    hidden_dim=256,       # ★ Updated: 64➒56×56→256×256 MLP (aligned with literature)
    lr_actor=3e-4,
    lr_critic=1e-3,
    gamma=0.99,
    lam=0.95,
    eps_clip=0.2,
    n_epochs=10,
    batch_size=64,
    n_steps=2048,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    use_obs_norm=True,    # ★ Obs normalization (running mean/std)
    use_adv_norm=True,    # ★ Per-minibatch advantage normalization
    device="cpu",
)

# Optimal PPO kwargs
OPTIMAL_PPO_KWARGS = dict(
    hidden_dim=256,       # ★ Updated: 64➒56×56→256×256 MLP (aligned with literature)
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


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_policy(agent, eval_env, n_episodes=10):
    """Deterministic evaluation using policy mean (no stochastic sampling)."""
    rewards = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        # Apply obs normalization if agent has it
        if hasattr(agent, 'normalize_obs'):
            obs = agent.normalize_obs(obs)
        total_reward = 0.0
        done = False
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                dist = agent.actor.forward(obs_t)
                # Deterministic: use mean for continuous, argmax for discrete
                if agent.continuous:
                    action_np = dist.mean.squeeze(0).detach().cpu().numpy()
                else:
                    action_np = int(dist.probs.argmax(dim=-1).squeeze(0).detach().cpu().numpy())
            obs, r, terminated, truncated, _ = eval_env.step(action_np)
            if hasattr(agent, 'normalize_obs'):
                obs = agent.normalize_obs(obs)
            total_reward += r
            done = terminated or truncated
        rewards.append(total_reward)
    return float(np.mean(rewards))


# ─────────────────────────────────────────────────────────────────────────────
# Single run
# ─────────────────────────────────────────────────────────────────────────────
def run_single(
    env_name: str,
    algo_name: str,
    seed: int,
    total_timesteps: int = TOTAL_TIMESTEPS,
) -> dict:
    """Run one (env, algo, seed) combination."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    save_dir = str(RESULTS_DIR / env_name / algo_name)
    os.makedirs(save_dir, exist_ok=True)
    out_path = Path(save_dir) / f"{algo_name}_s{seed}.json"

    if out_path.exists():
        print(f"    [SKIP] {env_name}/{algo_name}/s{seed} already done")
        return json.load(open(out_path))

    # Create environments
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 50000)

    # Build agent
    if algo_name == "Standard_PPO":
        kw = dict(**STANDARD_PPO_KWARGS, save_dir=save_dir)
        agent = build_ppo_baseline("Standard_PPO", env, name=f"{algo_name}_s{seed}", **kw)
    else:
        kw = dict(**OPTIMAL_PPO_KWARGS, save_dir=save_dir)
        agent = build_optimal_agent(algo_name, env, name=f"{algo_name}_s{seed}", **kw)

    # Training loop
    eval_rewards = []
    eval_steps = []
    episode_rewards = []
    episode_lengths = []

    obs, _ = env.reset()
    if hasattr(agent, 'update_obs_rms'):
        agent.update_obs_rms(obs)
    if hasattr(agent, 'normalize_obs'):
        obs = agent.normalize_obs(obs)

    ep_reward = 0.0
    ep_length = 0
    total_steps = 0
    last_eval_step = 0

    t0 = time.time()

    while total_steps < total_timesteps:
        # Collect rollout
        agent.buffer.reset()
        for _ in range(agent.n_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, log_prob = agent.actor.get_action_and_logprob(obs_t)
                value = agent.critic(obs_t)

            action_np = action.squeeze(0).cpu().numpy()
            if not agent.continuous:
                action_np = int(action_np)

            next_obs, reward, terminated, truncated, _ = env.step(action_np)

            # Update obs running stats BEFORE normalizing
            if hasattr(agent, 'update_obs_rms'):
                agent.update_obs_rms(next_obs)
            if hasattr(agent, 'normalize_obs'):
                next_obs_norm = agent.normalize_obs(next_obs)
            else:
                next_obs_norm = next_obs

            ep_reward += reward
            ep_length += 1

            agent.buffer.add(
                obs, action_np, float(reward), float(terminated),
                log_prob.item(), value.item()
            )
            obs = next_obs_norm
            total_steps += 1

            if terminated or truncated:
                episode_rewards.append(ep_reward)
                episode_lengths.append(ep_length)
                ep_reward = 0.0
                ep_length = 0
                next_obs, _ = env.reset()
                if hasattr(agent, 'update_obs_rms'):
                    agent.update_obs_rms(next_obs)
                if hasattr(agent, 'normalize_obs'):
                    obs = agent.normalize_obs(next_obs)
                else:
                    obs = next_obs

            # Evaluation
            if total_steps - last_eval_step >= EVAL_FREQ:
                eval_r = evaluate_policy(agent, eval_env, N_EVAL_EPISODES)
                eval_rewards.append(eval_r)
                eval_steps.append(total_steps)
                last_eval_step = total_steps

        # Update: compute GAE and optimize
        with torch.no_grad():
            last_obs_t = torch.FloatTensor(obs).unsqueeze(0)
            last_val = agent.critic(last_obs_t).item()

        # Set total_timesteps for LR annealing
        agent._total_timesteps = total_timesteps
        agent.total_steps = total_steps

        if hasattr(agent, 'compute_hindsight_gae'):
            agent.compute_hindsight_gae(last_val)
        else:
            agent.compute_gae(last_val)

        agent.update()

    elapsed = time.time() - t0
    final_mean = float(np.mean(eval_rewards[-5:])) if len(eval_rewards) >= 5 else (
        float(np.mean(eval_rewards)) if eval_rewards else 0.0
    )

    result = {
        "env": env_name,
        "agent": algo_name,
        "seed": seed,
        "total_steps": total_steps,
        "final_reward": final_mean,
        "eval_rewards": eval_rewards,
        "eval_steps": eval_steps,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "elapsed_s": elapsed,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    env.close()
    eval_env.close()
    print(f"    Done: {algo_name} {env_name} s{seed} → final={final_mean:.1f} ({elapsed:.0f}s)")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
def compute_summary(results_dir=RESULTS_DIR, envs=ENVS, algos=ALGORITHMS, seeds=SEEDS):
    summary = {}
    for env_name in envs:
        summary[env_name] = {}
        for algo in algos:
            seed_means = []
            for seed in seeds:
                fp = results_dir / env_name / algo / f"{algo}_s{seed}.json"
                if fp.exists():
                    try:
                        d = json.load(open(fp))
                        er = d.get("eval_rewards", [])
                        if er:
                            val = float(np.mean(er[-5:])) if len(er) >= 5 else float(np.mean(er))
                            seed_means.append(val)
                    except Exception:
                        pass
            if seed_means:
                summary[env_name][algo] = {
                    "mean": float(np.mean(seed_means)),
                    "std": float(np.std(seed_means)),
                    "sem": float(np.std(seed_means) / np.sqrt(len(seed_means))),
                    "n": len(seed_means),
                    "seeds": seed_means,
                }
    return summary


def print_table(summary, envs=ENVS, algos=ALGORITHMS):
    # Header
    print(f"\n{'Algorithm':<25}", end="")
    for env in envs:
        short = env.split('-')[0]
        print(f"  {short:>18}", end="")
    print()
    print(f"{'─'*25}", end="")
    for _ in envs:
        print(f"  {'─'*18}", end="")
    print()

    for algo in algos:
        marker = " ◄ OURS" if "HCGAE" in algo else ""
        print(f"{algo:<25}", end="")
        for env in envs:
            info = summary.get(env, {}).get(algo, {})
            if info:
                m, s, n = info["mean"], info["sem"], info["n"]
                print(f"  {m:>8.0f}±{s:>5.0f}(n={n})", end="")
            else:
                print(f"  {'pending':>18}", end="")
        print(marker)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="ICML PPO Experiment")
    parser.add_argument("--envs", nargs="+", default=ENVS)
    parser.add_argument("--algos", nargs="+", default=ALGORITHMS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--env", type=str, default=None, help="Single env (shortcut)")
    parser.add_argument("--algo", type=str, default=None, help="Single algo (shortcut)")
    parser.add_argument("--seed", type=int, default=None, help="Single seed (shortcut)")
    args = parser.parse_args()

    # Single-run shortcuts
    if args.env:
        args.envs = [args.env]
    if args.algo:
        args.algos = [args.algo]
    if args.seed is not None:
        args.seeds = [args.seed]

    if args.summary_only:
        summary = compute_summary(RESULTS_DIR, args.envs, args.algos, args.seeds)
        print_table(summary, args.envs, args.algos)
        sp = RESULTS_DIR / "icml_summary.json"
        with open(sp, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved: {sp}")
        return

    total_runs = len(args.envs) * len(args.algos) * len(args.seeds)
    print(f"\n{'='*80}")
    print(f"  ICML PPO Experiment")
    print(f"  Envs:   {args.envs}")
    print(f"  Algos:  {args.algos}")
    print(f"  Seeds:  {args.seeds}")
    print(f"  Steps:  {args.timesteps:,}")
    print(f"  Total runs: {total_runs}")
    print(f"{'='*80}\n")

    t_global = time.time()
    run_idx = 0
    for env_name in args.envs:
        for algo_name in args.algos:
            for seed in args.seeds:
                run_idx += 1
                print(f"\n[{run_idx}/{total_runs}] {env_name} | {algo_name} | seed={seed}")
                run_single(env_name, algo_name, seed, total_timesteps=args.timesteps)

    # Final summary
    summary = compute_summary(RESULTS_DIR, args.envs, args.algos, args.seeds)
    sp = RESULTS_DIR / "icml_summary.json"
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n\nTotal elapsed: {time.time()-t_global:.0f}s")
    print_table(summary, args.envs, args.algos)
    print(f"\nSaved: {sp}")


if __name__ == "__main__":
    main()

