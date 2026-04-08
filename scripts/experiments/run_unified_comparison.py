"""
Unified ICML-ready Comparison Experiment
==========================================

Fair comparison between HCGAE and PPO variants using IDENTICAL:
  - Network architecture (64-64 MLP actor/critic)
  - Hyperparameters (lr=3e-4, gamma=0.99, lambda=0.95, eps=0.2, n_epochs=10)
  - Rollout length (n_steps=2048)
  - Evaluation protocol (every 10240 steps, 10 deterministic rollouts, last-5 mean)
  - Total timesteps (1,000,000 steps for long / 300,000 for short)
  - Seeds: [42, 123, 456, 789, 1234]

Results saved to: results/UnifiedComparison/
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

from gae_experiments.agents.ppo_baselines import build_ppo_baseline
from gae_experiments.agents.hindsight_ablation import build_ablation_agent

# ─────────────────────────────────────────────────────────────────────────────
# Experiment configuration
# ─────────────────────────────────────────────────────────────────────────────
ENVS = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]
SEEDS = [42, 123, 456, 789, 1234]
TOTAL_TIMESTEPS = 1_000_000     # 1M steps for proper comparison
EVAL_FREQ = 10_240
N_EVAL_EPISODES = 10

# Algorithms to run
ALGORITHMS = [
    "Standard_PPO",      # Pure PPO (no VClip), Schulman 2017
    "PPO_KLPEN",         # KL-penalty variant, Schulman 2017
    "PPO_Anneal",        # LR annealing, OpenAI Baselines
    "PPO_EntDecay",      # Entropy decay, Andrychowicz 2021
    "PPO_VClip",         # Value clipping, Engstrom 2020
    "HCGAE_Imp12",       # Our method
]

RESULTS_DIR = Path("results/UnifiedComparison")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Shared hyperparameters (identical for all algorithms, aligned with literature)
SHARED_KWARGS = dict(
    hidden_dim=256,       # ★ Updated: 256x256 MLP (literature alignment)
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
    use_obs_norm=True,    # ★ Obs normalization
    use_adv_norm=True,    # ★ Per-minibatch advantage normalization
    device="cpu",
)


# ─────────────────────────────────────────────────────────────────────────────
# Single run
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_policy(agent, eval_env, n_episodes=10):
    """Deterministic evaluation."""
    rewards = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        total_reward = 0.0
        done = False
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, _ = agent.actor.get_action_and_logprob(obs_t)
                if agent.continuous:
                    action_np = action.squeeze(0).cpu().numpy()
                else:
                    action_np = int(action.squeeze(0).cpu().numpy())
            obs, r, terminated, truncated, _ = eval_env.step(action_np)
            total_reward += r
            done = terminated or truncated
        rewards.append(total_reward)
    return float(np.mean(rewards))


def run_single(env_name: str, algo_name: str, seed: int,
               total_timesteps: int = TOTAL_TIMESTEPS) -> dict:
    """
    Run one (env, algo, seed) combination.
    Returns dict with eval_rewards, eval_steps, episode_rewards.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    save_dir = str(RESULTS_DIR / env_name / algo_name)
    os.makedirs(save_dir, exist_ok=True)
    out_path = Path(save_dir) / f"{algo_name}_s{seed}.json"
    if out_path.exists():
        print(f"    [SKIP] {env_name}/{algo_name}/s{seed} already done")
        return json.load(open(out_path))

    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 10000)

    kw = dict(**SHARED_KWARGS, save_dir=save_dir)
    name = f"{algo_name}_s{seed}"

    # Build agent
    if algo_name == "HCGAE_Imp12":
        agent = build_ablation_agent("HCGAE_Imp12", env, name=name,
                                     **{k: v for k, v in kw.items()})
    else:
        bl_kw = {k: v for k, v in kw.items()}
        agent = build_ppo_baseline(algo_name, env, name=name, **bl_kw)

    # Training loop
    eval_rewards = []
    eval_steps = []
    episode_rewards = []
    episode_lengths = []

    obs, _ = env.reset()
    ep_reward = 0.0
    ep_length = 0
    total_steps = 0
    last_eval_step = 0

    t0 = time.time()

    while total_steps < total_timesteps:
        # --- collect rollout ---
        agent.buffer.reset()
        for _ in range(agent.n_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, log_prob = agent.actor.get_action_and_logprob(obs_t)
                value = agent.critic(obs_t)

            action_np = action.squeeze(0).cpu().numpy()
            if agent.continuous:
                next_obs, reward, terminated, truncated, _ = env.step(action_np)
            else:
                next_obs, reward, terminated, truncated, _ = env.step(int(action_np))

            ep_reward += reward
            ep_length += 1
            agent.buffer.add(obs, action_np, float(reward), float(terminated),
                              log_prob.item(), value.item())
            obs = next_obs
            total_steps += 1

            if terminated or truncated:
                episode_rewards.append(ep_reward)
                episode_lengths.append(ep_length)
                ep_reward = 0.0
                ep_length = 0
                obs, _ = env.reset()

            if total_steps - last_eval_step >= EVAL_FREQ:
                eval_r = evaluate_policy(agent, eval_env, N_EVAL_EPISODES)
                eval_rewards.append(eval_r)
                eval_steps.append(total_steps)
                last_eval_step = total_steps

        # --- update ---
        with torch.no_grad():
            last_obs_t = torch.FloatTensor(obs).unsqueeze(0)
            last_val = agent.critic(last_obs_t).item()

        # Handle different GAE methods
        if hasattr(agent, 'compute_hindsight_gae'):
            agent.compute_hindsight_gae(last_val)
        else:
            agent.compute_gae(last_val)

        # Set total_timesteps for LR annealing
        agent._total_timesteps = total_timesteps
        agent.total_steps = total_steps
        agent.update()

    elapsed = time.time() - t0
    final_mean = float(np.mean(eval_rewards[-5:])) if len(eval_rewards) >= 5 else float(np.mean(eval_rewards)) if eval_rewards else 0.0

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

    print(f"    Done: {algo_name} {env_name} s{seed} -> final={final_mean:.1f} ({elapsed:.0f}s)")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Summary utilities
# ─────────────────────────────────────────────────────────────────────────────
def compute_summary(results_dir: Path, envs=ENVS, algos=ALGORITHMS, seeds=SEEDS) -> dict:
    """Load all seed results and compute mean/std per (env, algo)."""
    summary = {}
    for env_name in envs:
        summary[env_name] = {}
        for algo in algos:
            seed_means = []
            for seed in seeds:
                fp = results_dir / env_name / algo / f"{algo}_s{seed}.json"
                if fp.exists():
                    d = json.load(open(fp))
                    er = d.get("eval_rewards", [])
                    if er:
                        val = float(np.mean(er[-5:])) if len(er) >= 5 else float(np.mean(er))
                        seed_means.append(val)
            if seed_means:
                summary[env_name][algo] = {
                    "mean": float(np.mean(seed_means)),
                    "std": float(np.std(seed_means)),
                    "n": len(seed_means),
                    "seeds": seed_means,
                }
    return summary


def print_table(summary: dict, envs=ENVS, algos=ALGORITHMS):
    print(f"\n{'Algorithm':<25} | {'Hopper-v4':>18} | {'Walker2d-v4':>18} | {'HalfCheetah-v4':>18}")
    print(f"{'-'*25} | {'-'*18} | {'-'*18} | {'-'*18}")
    for algo in algos:
        row = f"{algo:<25}"
        for env in envs:
            info = summary.get(env, {}).get(algo, {})
            if info:
                m, s, n = info["mean"], info["std"], info["n"]
                row += f" | {m:>8.0f}±{s:>5.0f} (n={n})"
            else:
                row += f" | {'pending':>18}"
        marker = " <-- OURS" if algo == "HCGAE_Imp12" else ""
        print(row + marker)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", default=ENVS)
    parser.add_argument("--algos", nargs="+", default=ALGORITHMS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    if args.summary_only:
        summary = compute_summary(RESULTS_DIR, args.envs, args.algos, args.seeds)
        print_table(summary, args.envs, args.algos)
        summary_path = RESULTS_DIR / "unified_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved: {summary_path}")
        return summary

    total_runs = len(args.envs) * len(args.algos) * len(args.seeds)
    print(f"\n{'='*80}")
    print(f"  ICML Unified Comparison")
    print(f"  Envs: {args.envs}")
    print(f"  Algos: {args.algos}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Timesteps: {args.timesteps:,}")
    print(f"  Total runs: {total_runs}")
    print(f"{'='*80}\n")

    run_idx = 0
    t_global = time.time()
    for env_name in args.envs:
        for algo_name in args.algos:
            for seed in args.seeds:
                run_idx += 1
                print(f"\n[{run_idx}/{total_runs}] {env_name} | {algo_name} | seed={seed}")
                run_single(env_name, algo_name, seed, total_timesteps=args.timesteps)

    summary = compute_summary(RESULTS_DIR, args.envs, args.algos, args.seeds)
    summary_path = RESULTS_DIR / "unified_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n\nTotal elapsed: {time.time()-t_global:.0f}s")
    print_table(summary, args.envs, args.algos)
    print(f"\nSaved: {summary_path}")
    return summary


if __name__ == "__main__":
    main()

