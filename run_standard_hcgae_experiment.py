#!/usr/bin/env python3
"""
Standard-Base HCGAE Experiment: Decoupling HCGAE from Optimal PPO Tricks
=========================================================================

Scientific Question
-------------------
Table 1 compares Optimal HCGAE v2 against Optimal PPO — but *both* use
Andrychowicz et al. (2021) best-practice tricks (obs normalisation, per-minibatch
advantage normalisation, LR annealing).  Does the HCGAE improvement reflect the
GAE correction itself, or is it entangled with those implementation tricks?

Experiment Design
-----------------
To isolate the HCGAE contribution we add a *matched* pair that runs on a
Standard-PPO base (NO obs_norm, NO adv_norm, NO LR anneal):

    Standard_PPO_Optimal   — same OptimalPPO codebase, all tricks disabled
    Standard_HCGAE_v2      — same HCGAE v2 plugin, all Optimal tricks disabled

If HCGAE v2 improves over Standard_PPO_Optimal with a similar relative gain as
it does over Optimal_PPO, the improvement is attributable to the GAE correction
itself.  If the gain collapses or reverses, the improvement is entangled with the
implementation tricks.

Conditions (aligned with ICMLExperiment)
-----------------------------------------
  Environments : Hopper-v4, Walker2d-v4, HalfCheetah-v4
  Seeds        : 5  (seeds 0-4, matching ICMLExperiment)
  Steps        : 500,000 per run
  Eval freq    : 10,240 steps, 10 episodes, last-5-eval mean (identical protocol)

Results are saved to: results/StandardHCGAEExperiment/
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

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
ENVS = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]
SEEDS = list(range(5))        # 5 seeds: 0..4  (same as ICMLExperiment)
TOTAL_TIMESTEPS = 500_000     # 500K steps (same as ICMLExperiment)
EVAL_FREQ = 10_240
N_EVAL_EPISODES = 10
RESULTS_DIR = Path("results/StandardHCGAEExperiment")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Two algorithms: Standard PPO base and Standard PPO + HCGAE v2 plugin
ALGORITHMS = [
    "Standard_PPO_Optimal",   # OptimalPPO framework, all tricks disabled (= Standard PPO)
    "Standard_HCGAE_v2",      # HCGAE v2 plugin on Standard PPO base
]

# Shared kwargs — same architecture/hyperparams as ICMLExperiment's OPTIMAL_PPO_KWARGS
# except use_obs_norm / use_adv_norm / use_lr_anneal are forced off inside build_optimal_agent
BASE_KWARGS = dict(
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
    device="cpu",
)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_policy(agent, eval_env, n_episodes=10):
    """Deterministic evaluation (no exploration noise)."""
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
                action, _ = agent.actor.get_action_and_logprob(obs_t)
                if agent.continuous:
                    action_np = action.squeeze(0).cpu().numpy()
                else:
                    action_np = int(action.squeeze(0).cpu().numpy())
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

    # Build agent (both use build_optimal_agent; flags are set inside)
    kw = dict(**BASE_KWARGS, save_dir=save_dir)
    agent = build_optimal_agent(algo_name, env, name=f"{algo_name}_s{seed}", **kw)

    # Training loop (identical to run_icml_experiment.py)
    eval_rewards = []
    eval_steps = []
    episode_rewards = []
    episode_lengths = []

    obs, _ = env.reset()
    # Standard base: obs normalisation disabled — update_obs_rms / normalize_obs are
    # no-ops because use_obs_norm=False means the methods still exist in OptimalPPO
    # but normalize_obs returns the raw observation unchanged.
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

        # Update: compute GAE and optimise
        with torch.no_grad():
            last_obs_t = torch.FloatTensor(obs).unsqueeze(0)
            last_val = agent.critic(last_obs_t).item()

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
# Summary + comparison with ICMLExperiment
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


def load_icml_reference(icml_dir="results/ICMLExperiment"):
    """Load the Optimal PPO / Optimal HCGAE v2 reference data from ICMLExperiment."""
    icml_path = Path(icml_dir)
    ref = {}
    for env_name in ENVS:
        ref[env_name] = {}
        for algo in ["Optimal_PPO", "Optimal_HCGAE_v2"]:
            seed_means = []
            for seed in SEEDS:
                fp = icml_path / env_name / algo / f"{algo}_s{seed}.json"
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
                ref[env_name][algo] = {
                    "mean": float(np.mean(seed_means)),
                    "std": float(np.std(seed_means)),
                    "n": len(seed_means),
                }
    return ref


def print_combined_table(summary, icml_ref):
    """Print a combined table comparing Standard base and Optimal base results."""
    print(f"\n{'='*100}")
    print(f"  COMBINED COMPARISON: Standard Base vs Optimal Base (5 seeds, 500K steps)")
    print(f"{'='*100}")

    envs_short = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]

    print(f"\n{'Method':<30}", end="")
    for env in envs_short:
        short = env.split('-')[0]
        print(f"  {short:>20}", end="")
    print()
    print("-" * 100)

    rows = [
        ("Standard_PPO_Optimal", summary, "Standard PPO (no tricks)"),
        ("Standard_HCGAE_v2", summary, "Standard HCGAE v2 (no tricks)"),
        ("Optimal_PPO", icml_ref, "Optimal PPO (best practices)"),
        ("Optimal_HCGAE_v2", icml_ref, "Optimal HCGAE v2 (best practices)"),
    ]

    results_store = {}
    for algo_key, data_src, label in rows:
        print(f"{label:<30}", end="")
        for env in envs_short:
            info = data_src.get(env, {}).get(algo_key, {})
            if info:
                m, s, n = info["mean"], info["std"], info["n"]
                print(f"  {m:>10.0f} ± {s:>5.0f}(n={n})", end="")
                results_store[(algo_key, env)] = m
            else:
                print(f"  {'pending':>20}", end="")
        print()

    # Delta rows
    print("-" * 100)
    print(f"\nRelative gain of HCGAE v2 plugin (HCGAE / PPO - 1):")
    for base_ppo, base_hcgae, label in [
        ("Standard_PPO_Optimal", "Standard_HCGAE_v2", "Standard base Δ"),
        ("Optimal_PPO", "Optimal_HCGAE_v2", "Optimal  base Δ"),
    ]:
        print(f"  {label:<30}", end="")
        for env in envs_short:
            ppo_v = results_store.get((base_ppo, env))
            hcg_v = results_store.get((base_hcgae, env))
            if ppo_v and hcg_v and ppo_v > 0:
                delta = (hcg_v - ppo_v) / abs(ppo_v) * 100
                sign = "+" if delta >= 0 else ""
                print(f"  {sign}{delta:>+8.1f}%{'':>10}", end="")
            else:
                print(f"  {'pending':>20}", end="")
        print()

    print(f"\n{'='*100}")
    print("Interpretation:")
    print("  If Standard-base Δ ≈ Optimal-base Δ  →  HCGAE gain is independent of Optimal tricks")
    print("  If Standard-base Δ ≪ Optimal-base Δ  →  gain is entangled with implementation tricks")
    print(f"{'='*100}\n")


def print_table(summary, envs=ENVS, algos=ALGORITHMS):
    print(f"\n{'Algorithm':<30}", end="")
    for env in envs:
        short = env.split('-')[0]
        print(f"  {short:>20}", end="")
    print()
    print(f"{'─'*30}", end="")
    for _ in envs:
        print(f"  {'─'*20}", end="")
    print()
    for algo in algos:
        print(f"{algo:<30}", end="")
        for env in envs:
            info = summary.get(env, {}).get(algo, {})
            if info:
                m, s, n = info["mean"], info["std"], info["n"]
                print(f"  {m:>10.0f} ± {s:>5.0f}(n={n})", end="")
            else:
                print(f"  {'pending':>20}", end="")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Standard-Base HCGAE Experiment (decoupling from Optimal tricks)"
    )
    parser.add_argument("--envs", nargs="+", default=ENVS)
    parser.add_argument("--algos", nargs="+", default=ALGORITHMS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--summary-only", action="store_true",
                        help="Only compute and print summary, no training")
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
        icml_ref = load_icml_reference()
        print_table(summary, args.envs, args.algos)
        print_combined_table(summary, icml_ref)
        sp = RESULTS_DIR / "standard_hcgae_summary.json"
        with open(sp, "w") as f:
            json.dump({"standard_base": summary, "optimal_base_ref": icml_ref}, f, indent=2)
        print(f"\nSaved: {sp}")
        return

    total_runs = len(args.envs) * len(args.algos) * len(args.seeds)
    print(f"\n{'='*80}")
    print(f"  Standard-Base HCGAE Experiment")
    print(f"  Envs:        {args.envs}")
    print(f"  Algos:       {args.algos}")
    print(f"  Seeds:       {args.seeds}")
    print(f"  Steps:       {args.timesteps:,}")
    print(f"  Total runs:  {total_runs}")
    print(f"  Results dir: {RESULTS_DIR}")
    print(f"{'='*80}\n")

    t_global = time.time()
    run_idx = 0
    for env_name in args.envs:
        for algo_name in args.algos:
            for seed in args.seeds:
                run_idx += 1
                print(f"\n[{run_idx}/{total_runs}] {env_name} | {algo_name} | seed={seed}")
                run_single(env_name, algo_name, seed, total_timesteps=args.timesteps)

    # Final summary + combined comparison
    summary = compute_summary(RESULTS_DIR, args.envs, args.algos, args.seeds)
    icml_ref = load_icml_reference()
    print_table(summary, args.envs, args.algos)
    print_combined_table(summary, icml_ref)

    sp = RESULTS_DIR / "standard_hcgae_summary.json"
    with open(sp, "w") as f:
        json.dump({"standard_base": summary, "optimal_base_ref": icml_ref}, f, indent=2)

    print(f"\nTotal elapsed: {time.time()-t_global:.0f}s")
    print(f"Saved: {sp}")


if __name__ == "__main__":
    main()

