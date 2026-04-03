"""
Baseline Comparison Experiment
================================

Compare HCGAE (ours) against several published PPO improvement variants:

Baselines:
- Standard_PPO       : Vanilla PPO (Schulman et al. 2017)
- PPO_KLPEN          : KL penalty variant with adaptive beta (Schulman 2017)
- PPO_Anneal         : PPO with linear LR annealing (OpenAI baselines / SB3)
- PPO_EntDecay       : PPO with entropy coefficient annealing
- PPO_VClip          : PPO + value function clipping (Engstrom et al. 2020)
- PPO_Full_Baseline  : Best practical PPO baseline (Anneal + EntDecay + VClip)

Ours:
- HCGAE_Imp12        : Our proposed method (batch-normalized hindsight + EV-driven mixing)

Protocol:
- Environments: Hopper-v4, Walker2d-v4, HalfCheetah-v4
- Seeds: 5 seeds per condition (42, 123, 456, 789, 1234)
- Timesteps: 300,000 per run
- Eval: every 10,240 steps, 10 deterministic rollouts
"""

import json
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from gae_experiments.agents.ppo_baselines import build_ppo_baseline
from gae_experiments.agents.hindsight_ablation import build_ablation_agent

# ──────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────
ENVS = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]
SEEDS = [42, 123, 456, 789, 1234]
TOTAL_TIMESTEPS = 300_000
EVAL_FREQ = 10_240
N_EVAL_EPISODES = 10

BASELINES = [
    "Standard_PPO",
    "PPO_KLPEN",
    "PPO_Anneal",
    "PPO_EntDecay",
    "PPO_VClip",
    "PPO_Full_Baseline",
]

# Our method to compare
OUR_METHOD = "HCGAE_Imp12"

RESULTS_DIR = Path("results/BaselineComparison")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────
def run_single(env_name: str, algo_name: str, seed: int) -> float:
    """Run one (env, algo, seed) combination, return final eval reward."""
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)

    save_dir = str(RESULTS_DIR / env_name / algo_name)
    os.makedirs(save_dir, exist_ok=True)

    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 10000)

    kwargs = dict(
        hidden_dim=64,
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
        device="cpu",
        save_dir=save_dir,
    )

    try:
        if algo_name == OUR_METHOD:
            agent = build_ablation_agent(
                algo_name, env,
                name=f"{algo_name}_s{seed}",
                **kwargs,
            )
        else:
            # kwargs already contains save_dir; pass it via kwargs not explicitly
            bl_kwargs = {k: v for k, v in kwargs.items() if k != "save_dir"}
            agent = build_ppo_baseline(
                algo_name, env,
                save_dir=save_dir,
                name=f"{algo_name}_s{seed}",
                **bl_kwargs,
            )

        logger = agent.train(
            total_timesteps=TOTAL_TIMESTEPS,
            eval_env=eval_env,
            eval_freq=EVAL_FREQ,
            n_eval_episodes=N_EVAL_EPISODES,
            verbose=True,
        )

        final_rewards = logger.eval_rewards[-5:] if logger.eval_rewards else [0.0]
        final_mean = float(np.mean(final_rewards))

    except Exception as e:
        print(f"  ERROR {algo_name} {env_name} seed={seed}: {e}")
        import traceback
        traceback.print_exc()
        final_mean = 0.0
    finally:
        env.close()
        eval_env.close()

    return final_mean


# ──────────────────────────────────────────────────────────────────────────
# Aggregate results
# ──────────────────────────────────────────────────────────────────────────
def load_existing_results(summary_path: Path) -> dict:
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return {}


def save_summary(summary: dict, summary_path: Path):
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary -> {summary_path}")


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main():
    summary_path = RESULTS_DIR / "baseline_comparison_summary.json"
    summary = load_existing_results(summary_path)

    all_algos = BASELINES + [OUR_METHOD]
    total_runs = len(ENVS) * len(all_algos) * len(SEEDS)
    run_idx = 0

    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"  Baseline Comparison Experiment")
    print(f"  Envs: {ENVS}")
    print(f"  Algorithms: {all_algos}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Total runs: {total_runs}")
    print(f"{'='*70}\n")

    for env_name in ENVS:
        if env_name not in summary:
            summary[env_name] = {}

        for algo_name in all_algos:
            if algo_name not in summary[env_name]:
                summary[env_name][algo_name] = {"seeds": [], "mean": None, "std": None}

            for seed in SEEDS:
                run_idx += 1

                # Skip if already done
                existing = summary[env_name][algo_name]["seeds"]
                if len(existing) >= SEEDS.index(seed) + 1:
                    print(f"  [{run_idx}/{total_runs}] SKIP {env_name} {algo_name} seed={seed}")
                    continue

                print(f"\n  [{run_idx}/{total_runs}] {env_name} | {algo_name} | seed={seed}")
                t1 = time.time()
                result = run_single(env_name, algo_name, seed)
                elapsed = time.time() - t1

                summary[env_name][algo_name]["seeds"].append(result)
                seeds_so_far = summary[env_name][algo_name]["seeds"]
                summary[env_name][algo_name]["mean"] = float(np.mean(seeds_so_far))
                summary[env_name][algo_name]["std"] = float(np.std(seeds_so_far))

                print(f"  -> {algo_name} {env_name} s{seed}: {result:.1f} ({elapsed:.0f}s)")
                save_summary(summary, summary_path)

    # ── Print final table ──────────────────────────────────────────
    total_elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Final Results (5-seed mean ± std)  [total time: {total_elapsed:.0f}s]")
    print(f"{'='*70}")
    print(f"{'Algorithm':<25} | {'Hopper-v4':>15} | {'Walker2d-v4':>15} | {'HalfCheetah-v4':>16}")
    print(f"{'-'*25} | {'-'*15} | {'-'*15} | {'-'*16}")

    for algo_name in all_algos:
        row = f"{algo_name:<25}"
        for env_name in ENVS:
            info = summary.get(env_name, {}).get(algo_name, {})
            m = info.get("mean", 0.0) or 0.0
            s = info.get("std", 0.0) or 0.0
            row += f" | {m:>8.0f}±{s:>5.0f}"
        marker = " <-- OURS" if algo_name == OUR_METHOD else ""
        print(row + marker)

    print(f"{'='*70}\n")
    print(f"  Summary saved: {summary_path}")

    return summary


if __name__ == "__main__":
    main()

