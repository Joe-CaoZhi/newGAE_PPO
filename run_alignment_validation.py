#!/usr/bin/env python3
"""
Literature Alignment Validation Experiment
===========================================
Validates that our PPO implementations match published results:
  - Schulman et al. (2017): Hopper-v4 ~2300, HalfCheetah-v4 ~1800 at 1M steps
  - Uses Standard PPO (256x256 MLP + obs norm + adv norm + LR anneal)

This script runs:
  1. Standard_PPO   (now with obs norm, adv norm, 256x256 MLP)
  2. Optimal_PPO    (same + LR annealing, single optimizer)

For quick validation: 1M steps, 3 seeds, 2 environments
Results saved to: results/AlignmentValidation/

Expected targets:
  - Hopper-v4:       Standard_PPO ~2300+, Optimal_PPO ~2500-3000+
  - HalfCheetah-v4: Standard_PPO ~1800+, Optimal_PPO ~4000+
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
# Config
# ─────────────────────────────────────────────────────────────────────────────
ENVS = ["Hopper-v4", "HalfCheetah-v4"]
SEEDS = [0, 1, 2]
TOTAL_TIMESTEPS = 1_000_000   # Full 1M steps for literature comparison
EVAL_FREQ = 20_480             # Evaluate every ~20K steps
N_EVAL_EPISODES = 10

ALGORITHMS = ["Standard_PPO", "Optimal_PPO"]

RESULTS_DIR = Path("results/AlignmentValidation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Literature targets (from Schulman et al. 2017 and common implementations)
LITERATURE_TARGETS = {
    "Hopper-v4": {
        "Standard_PPO": 2300,    # Basic PPO with implementation tricks
        "Optimal_PPO": 2500,     # PPO with all best practices
    },
    "HalfCheetah-v4": {
        "Standard_PPO": 1800,    # Basic PPO
        "Optimal_PPO": 4000,     # PPO with all best practices
    }
}

# Kwargs for Standard PPO (now with obs norm + adv norm + 256x256 MLP)
STANDARD_PPO_KWARGS = dict(
    hidden_dim=256,       # ★ 256x256 MLP
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
    use_obs_norm=True,    # ★ Running obs normalization
    use_adv_norm=True,    # ★ Per-minibatch advantage normalization
    device="cpu",
)

# Kwargs for Optimal PPO (all best practices)
OPTIMAL_PPO_KWARGS = dict(
    hidden_dim=256,       # ★ 256x256 MLP
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
    use_lr_anneal=True,   # ★ LR annealing to 0
    use_vclip=False,
    device="cpu",
)


def evaluate_policy(agent, eval_env, n_episodes=10):
    """Deterministic policy evaluation."""
    rewards = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        if hasattr(agent, 'normalize_obs'):
            obs = agent.normalize_obs(obs)
        total_reward = 0.0
        done = False
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                dist = agent.actor(obs_t)
                if agent.continuous:
                    action = dist.mean
                else:
                    action = dist.probs.argmax(dim=-1)
            action_np = action.squeeze(0).cpu().numpy()
            if agent.continuous:
                obs, r, terminated, truncated, _ = eval_env.step(action_np)
            else:
                obs, r, terminated, truncated, _ = eval_env.step(int(action_np))
            if hasattr(agent, 'normalize_obs'):
                obs = agent.normalize_obs(obs)
            total_reward += r
            done = terminated or truncated
        rewards.append(total_reward)
    return float(np.mean(rewards))


def run_single(env_name, algo_name, seed):
    """Run a single (env, algo, seed) combination."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    save_dir = str(RESULTS_DIR / env_name / algo_name)
    os.makedirs(save_dir, exist_ok=True)
    out_path = Path(save_dir) / f"{algo_name}_s{seed}.json"

    if out_path.exists():
        print(f"  [SKIP] {env_name}/{algo_name}/s{seed} already done")
        d = json.load(open(out_path))
        return d.get("final_reward", 0.0)

    print(f"  Running: {env_name}/{algo_name}/seed={seed}")
    t0 = time.time()

    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 50000)

    # Build agent
    if algo_name == "Standard_PPO":
        kw = dict(**STANDARD_PPO_KWARGS, save_dir=save_dir)
        agent = build_ppo_baseline("Standard_PPO", env, name=f"Standard_PPO_s{seed}", **kw)
    elif algo_name == "Optimal_PPO":
        kw = dict(**OPTIMAL_PPO_KWARGS, save_dir=save_dir)
        agent = build_optimal_agent("Optimal_PPO", env, name=f"Optimal_PPO_s{seed}", **kw)
    else:
        raise ValueError(f"Unknown algo: {algo_name}")

    # Training loop — ★ must set _total_timesteps before first update() for LR annealing
    agent._total_timesteps = TOTAL_TIMESTEPS
    eval_rewards = []
    eval_steps = []
    episode_rewards = []

    obs, _ = env.reset()
    ep_reward = 0.0
    ep_length = 0
    total_steps = 0
    last_eval_step = 0

    while total_steps < TOTAL_TIMESTEPS:
        # Collect rollout
        agent.buffer.reset()
        obs_start = obs

        for _ in range(agent.n_steps):
            if hasattr(agent, 'update_obs_rms'):
                agent.update_obs_rms(obs_start)
            obs_norm = agent.normalize_obs(obs_start) if hasattr(agent, 'normalize_obs') else obs_start
            obs_t = torch.FloatTensor(obs_norm).unsqueeze(0)
            with torch.no_grad():
                action, log_prob = agent.actor.get_action_and_logprob(obs_t)
                value = agent.critic(obs_t)

            action_np = action.squeeze(0).cpu().numpy()
            if not agent.continuous:
                action_np = int(action_np)

            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            ep_reward += reward
            ep_length += 1

            agent.buffer.add(obs_norm, action_np, float(reward), float(terminated),
                             log_prob.item(), value.item())
            total_steps += 1
            obs_start = next_obs

            if terminated or truncated:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                ep_length = 0
                obs_start, _ = env.reset()

        # Last value for GAE bootstrap
        if hasattr(agent, 'update_obs_rms'):
            agent.update_obs_rms(obs_start)
        last_obs_norm = agent.normalize_obs(obs_start) if hasattr(agent, 'normalize_obs') else obs_start
        last_obs_t = torch.FloatTensor(last_obs_norm).unsqueeze(0)
        with torch.no_grad():
            last_val = agent.critic(last_obs_t).item()

        obs = obs_start

        # Set total_steps for LR annealing
        agent.total_steps = total_steps

        # Compute GAE and update
        agent.compute_gae(last_val)
        agent.update()

        # Evaluate
        if total_steps - last_eval_step >= EVAL_FREQ:
            eval_r = evaluate_policy(agent, eval_env, N_EVAL_EPISODES)
            eval_rewards.append(eval_r)
            eval_steps.append(total_steps)
            last_eval_step = total_steps
            elapsed = time.time() - t0
            print(f"    [{algo_name} s{seed}] step={total_steps:,}/{TOTAL_TIMESTEPS:,} "
                  f"eval={eval_r:.1f} elapsed={elapsed:.0f}s")

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
        "elapsed_s": elapsed,
        "config": {
            "hidden_dim": 256,
            "use_obs_norm": True,
            "use_adv_norm": True,
            "use_lr_anneal": (algo_name == "Optimal_PPO"),
        }
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    env.close()
    eval_env.close()
    print(f"  ✓ Done: {algo_name} {env_name} s{seed} → final={final_mean:.1f} ({elapsed:.0f}s)")
    return final_mean


def print_alignment_summary(results):
    """Print comparison table vs literature values."""
    print("\n" + "="*70)
    print("  LITERATURE ALIGNMENT VALIDATION RESULTS")
    print("="*70)
    print(f"{'Algorithm':<20} {'Environment':<20} {'Our Result':>12} {'Target':>10} {'Status':>10}")
    print("-"*70)

    for env_name in ENVS:
        for algo_name in ALGORITHMS:
            if env_name in results and algo_name in results[env_name]:
                vals = results[env_name][algo_name]
                our_mean = np.mean(vals)
                our_std = np.std(vals)
                target = LITERATURE_TARGETS[env_name][algo_name]
                ratio = our_mean / target
                status = "✓ OK" if ratio >= 0.9 else ("⚠ LOW" if ratio >= 0.7 else "✗ FAIL")
                print(f"  {algo_name:<18} {env_name:<20} {our_mean:>7.0f}±{our_std:>4.0f} {target:>10} {status:>10}")
            else:
                target = LITERATURE_TARGETS[env_name][algo_name]
                print(f"  {algo_name:<18} {env_name:<20} {'pending':>12} {target:>10} {'–':>10}")

    print("="*70)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Literature Alignment Validation")
    parser.add_argument("--envs", nargs="+", default=ENVS)
    parser.add_argument("--algos", nargs="+", default=ALGORITHMS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    # Collect results
    results = {env: {algo: [] for algo in args.algos} for env in args.envs}

    if not args.summary_only:
        total_runs = len(args.envs) * len(args.algos) * len(args.seeds)
        run_idx = 0
        print(f"\n{'='*70}")
        print(f"  Literature Alignment Validation")
        print(f"  Config: {args.envs} × {args.algos} × {len(args.seeds)} seeds × {args.timesteps:,} steps")
        print(f"  Target: Hopper ~2300+, HalfCheetah ~1800+")
        print(f"{'='*70}\n")

        t_global = time.time()
        for env_name in args.envs:
            for algo_name in args.algos:
                for seed in args.seeds:
                    run_idx += 1
                    print(f"\n[{run_idx}/{total_runs}] {env_name} | {algo_name} | seed={seed}")
                    final_r = run_single(env_name, algo_name, seed)
                    results[env_name][algo_name].append(final_r)

        print(f"\nTotal elapsed: {time.time()-t_global:.0f}s")

    # Load existing results if summary-only
    else:
        for env_name in args.envs:
            for algo_name in args.algos:
                for seed in args.seeds:
                    fp = RESULTS_DIR / env_name / algo_name / f"{algo_name}_s{seed}.json"
                    if fp.exists():
                        d = json.load(open(fp))
                        er = d.get("eval_rewards", [])
                        if er:
                            results[env_name][algo_name].append(
                                float(np.mean(er[-5:])) if len(er) >= 5 else float(np.mean(er))
                            )

    # Save summary
    summary = {}
    for env_name in args.envs:
        summary[env_name] = {}
        for algo_name in args.algos:
            vals = results[env_name][algo_name]
            if vals:
                summary[env_name][algo_name] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n": len(vals),
                    "target": LITERATURE_TARGETS.get(env_name, {}).get(algo_name, 0),
                    "seeds": vals,
                }
    with open(RESULTS_DIR / "alignment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print_alignment_summary(results)
    print(f"\nSaved to: {RESULTS_DIR}/alignment_summary.json")


if __name__ == "__main__":
    main()

