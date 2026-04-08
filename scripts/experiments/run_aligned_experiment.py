#!/usr/bin/env python3
"""
Aligned Baseline Experiment: HCGAE v2 vs Optimal PPO vs Standard PPO
=====================================================================

All agents use the **aligned best-practice configuration**:
  - 256×256 MLP (hidden_dim=256)
  - Observation normalization (running mean/std)
  - Per-minibatch advantage normalization
  - LR annealing (Optimal_PPO and HCGAE_v2 only)

Experiment Design:
  Envs:   Hopper-v4, Walker2d-v4, HalfCheetah-v4
  Algos:  Standard_PPO, Optimal_PPO, Optimal_HCGAE_v2
  Seeds:  0..4 (5 seeds)
  Steps:  500,000 per run
  Eval:   every 10,240 steps, 10 deterministic episodes (policy mean)
  Total:  3 × 3 × 5 = 45 runs

Eval fix:  uses dist.mean (deterministic greedy) instead of dist.sample()
Results:   results/AlignedExperiment/
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
ENVS = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]
SEEDS = list(range(5))          # 5 seeds: 0..4
TOTAL_TIMESTEPS = 500_000
EVAL_FREQ = 10_240
N_EVAL_EPISODES = 10
RESULTS_DIR = Path("results/AlignedExperiment")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALGORITHMS = [
    "Standard_PPO",              # 256×256, obs_norm, adv_norm (no LR anneal — separate actor/critic LR)
    "Optimal_PPO",               # 256×256, obs_norm, adv_norm, LR anneal
    "Optimal_HCGAE_v2",          # HCGAE v2 on Optimal_PPO base ← our method (full: boundary+gate)
    "Optimal_HCGAE_v2_NoBdry",   # Ablation: HCGAE v2 without boundary bootstrap correction
    "Optimal_HCGAE_v2_NoGate",   # Ablation: HCGAE v2 without EV growth-rate gate
]

# For the primary comparison, only the first 3 algorithms
PRIMARY_ALGORITHMS = ALGORITHMS[:3]

# ── Shared best-practice kwargs for Standard_PPO (PPOBaseline) ──────────────
STANDARD_PPO_KWARGS = dict(
    hidden_dim=256,
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
    use_obs_norm=True,      # ★ aligned
    use_adv_norm=True,      # ★ aligned
    device="cpu",
)

# ── Shared best-practice kwargs for OptimalPPO / HCGAE_v2 ───────────────────
OPTIMAL_PPO_KWARGS = dict(
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
    use_obs_norm=True,      # ★ aligned
    use_adv_norm=True,      # ★ aligned
    use_lr_anneal=True,     # ★ aligned
    use_vclip=False,
    device="cpu",
)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation — deterministic (policy mean)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_policy(agent, eval_env, n_episodes=10):
    """Deterministic evaluation: use dist.mean (greedy), not dist.sample()."""
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
                dist = agent.actor.forward(obs_t)
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
def run_single(env_name, algo_name, seed, total_timesteps=TOTAL_TIMESTEPS):
    np.random.seed(seed)
    torch.manual_seed(seed)

    save_dir = str(RESULTS_DIR / env_name / algo_name)
    os.makedirs(save_dir, exist_ok=True)
    out_path = Path(save_dir) / f"{algo_name}_s{seed}.json"

    if out_path.exists():
        print(f"    [SKIP] {env_name}/{algo_name}/s{seed} already done")
        return json.load(open(out_path))

    # Environments
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

    # Training state
    eval_rewards, eval_steps = [], []
    episode_rewards, episode_lengths = [], []
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
        # ── Collect rollout ──────────────────────────────────────────────────
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

            if hasattr(agent, 'update_obs_rms'):
                agent.update_obs_rms(next_obs)
            next_obs_norm = agent.normalize_obs(next_obs) if hasattr(agent, 'normalize_obs') else next_obs

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
                obs = agent.normalize_obs(next_obs) if hasattr(agent, 'normalize_obs') else next_obs

            # ── Evaluation ──────────────────────────────────────────────────
            if total_steps - last_eval_step >= EVAL_FREQ:
                eval_r = evaluate_policy(agent, eval_env, N_EVAL_EPISODES)
                eval_rewards.append(eval_r)
                eval_steps.append(total_steps)
                last_eval_step = total_steps

        # ── GAE + Update ─────────────────────────────────────────────────────
        with torch.no_grad():
            last_val = agent.critic(torch.FloatTensor(obs).unsqueeze(0)).item()

        # CRITICAL: set _total_timesteps for LR annealing
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

    # Also record max and best-10 for Hopper stability analysis
    max_reward = float(max(eval_rewards)) if eval_rewards else 0.0
    best10_mean = float(np.mean(sorted(eval_rewards)[-10:])) if len(eval_rewards) >= 10 else final_mean

    result = {
        "env": env_name,
        "agent": algo_name,
        "seed": seed,
        "config": {
            "hidden_dim": 256,
            "use_obs_norm": True,
            "use_adv_norm": True,
            "use_lr_anneal": algo_name != "Standard_PPO",
            "eval_mode": "deterministic_mean",
        },
        "total_steps": total_steps,
        "final_reward": final_mean,
        "max_reward": max_reward,
        "best10_mean": best10_mean,
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
    print(f"    Done: {algo_name} {env_name} s{seed} → final={final_mean:.1f} "
          f"max={max_reward:.1f} best10={best10_mean:.1f} ({elapsed:.0f}s)")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
def compute_summary(results_dir=RESULTS_DIR, envs=ENVS, algos=ALGORITHMS, seeds=SEEDS):
    summary = {}
    for env_name in envs:
        summary[env_name] = {}
        for algo in algos:
            seed_final5, seed_max, seed_best10 = [], [], []
            for seed in seeds:
                fp = results_dir / env_name / algo / f"{algo}_s{seed}.json"
                if fp.exists():
                    try:
                        d = json.load(open(fp))
                        er = d.get("eval_rewards", [])
                        if er:
                            final5 = float(np.mean(er[-5:])) if len(er) >= 5 else float(np.mean(er))
                            seed_final5.append(final5)
                            seed_max.append(d.get("max_reward", float(max(er))))
                            seed_best10.append(d.get("best10_mean", float(np.mean(sorted(er)[-10:]))))
                    except Exception as e:
                        print(f"Warning: {fp}: {e}")
            if seed_final5:
                n = len(seed_final5)
                summary[env_name][algo] = {
                    "mean_final5": float(np.mean(seed_final5)),
                    "std_final5": float(np.std(seed_final5)),
                    "sem_final5": float(np.std(seed_final5) / np.sqrt(n)),
                    "mean_max": float(np.mean(seed_max)),
                    "mean_best10": float(np.mean(seed_best10)),
                    "n": n,
                    "seeds_final5": seed_final5,
                    "seeds_max": seed_max,
                }
    return summary


def print_table(summary, envs=ENVS, algos=ALGORITHMS):
    print(f"\n{'='*90}")
    print(f"  Aligned Experiment Results (256×256, obs_norm, adv_norm, deterministic eval)")
    print(f"{'='*90}")
    header = f"{'Algorithm':<25}"
    for env in envs:
        short = env.split('-')[0]
        header += f"  {short+' final5':>20}  {short+' max':>12}"
    print(header)
    print("─" * 90)

    for algo in algos:
        marker = " ◄ OURS" if "HCGAE" in algo else ""
        row = f"{algo:<25}"
        for env in envs:
            info = summary.get(env, {}).get(algo, {})
            if info:
                mf, sf = info["mean_final5"], info["sem_final5"]
                mx = info["mean_max"]
                n = info["n"]
                row += f"  {mf:>7.0f}±{sf:>4.0f}(n={n})  {mx:>10.0f}"
            else:
                row += f"  {'pending':>20}  {'---':>12}"
        print(row + marker)

    # Pct gains
    print("\n  Relative gain of HCGAE_v2 over Optimal_PPO:")
    for env in envs:
        opt = summary.get(env, {}).get("Optimal_PPO", {})
        hcg = summary.get(env, {}).get("Optimal_HCGAE_v2", {})
        if opt and hcg:
            gain = (hcg["mean_final5"] - opt["mean_final5"]) / (abs(opt["mean_final5"]) + 1e-8) * 100
            gain_max = (hcg["mean_max"] - opt["mean_max"]) / (abs(opt["mean_max"]) + 1e-8) * 100
            short = env.split('-')[0]
            print(f"    {short:<15}: final5 {gain:+.1f}%   max {gain_max:+.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Aligned Baseline Experiment")
    parser.add_argument("--envs", nargs="+", default=ENVS)
    parser.add_argument("--algos", nargs="+", default=ALGORITHMS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--algo", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.env:
        args.envs = [args.env]
    if args.algo:
        args.algos = [args.algo]
    if args.seed is not None:
        args.seeds = [args.seed]

    if args.summary_only:
        summary = compute_summary(RESULTS_DIR, args.envs, args.algos, args.seeds)
        print_table(summary, args.envs, args.algos)
        sp = RESULTS_DIR / "aligned_summary.json"
        with open(sp, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved: {sp}")
        return

    total_runs = len(args.envs) * len(args.algos) * len(args.seeds)
    print(f"\n{'='*80}")
    print(f"  Aligned Baseline Experiment")
    print(f"  All agents: hidden=256, obs_norm=True, adv_norm=True")
    print(f"  Eval: deterministic (policy mean)")
    print(f"  Envs:  {args.envs}")
    print(f"  Algos: {args.algos}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Steps: {args.timesteps:,}")
    print(f"  Total runs: {total_runs}")
    print(f"{'='*80}\n")

    t_global = time.time()
    run_idx = 0
    for env_name in args.envs:
        for algo_name in args.algos:
            for seed in args.seeds:
                run_idx += 1
                print(f"\n[{run_idx}/{total_runs}] {env_name} | {algo_name} | seed={seed}")
                try:
                    run_single(env_name, algo_name, seed, total_timesteps=args.timesteps)
                except Exception as e:
                    print(f"    [ERROR] {e}")
                    import traceback; traceback.print_exc()

    # Final summary
    summary = compute_summary(RESULTS_DIR, args.envs, args.algos, args.seeds)
    sp = RESULTS_DIR / "aligned_summary.json"
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)

    elapsed_total = time.time() - t_global
    print(f"\n\nTotal elapsed: {elapsed_total:.0f}s ({elapsed_total/3600:.2f}h)")
    print_table(summary, args.envs, args.algos)
    print(f"\nSaved: {sp}")


if __name__ == "__main__":
    main()

