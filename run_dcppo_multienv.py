#!/usr/bin/env python3
"""
GAP1: DCPPO-S Multi-Environment × 5-Seed Experiment
====================================================
Runs DCPPO-S (best variant: HCGAE_Imp12 GAE + SNR scaling) against
Standard_PPO baseline across 4 environments × 5 seeds × 300K steps.

Saves results to: results/MultiEnv_DCPPO/
"""
import json
import os
import random
import sys
import time

import gymnasium as gym
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gae_experiments.agents.dcppo import DCPPO
from gae_experiments.agents.base_ppo import BasePPO

SAVE_ROOT      = "results/MultiEnv_DCPPO"
ENVS           = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4", "Ant-v4"]
SEEDS          = [42, 123, 456, 789, 1234]
TOTAL_STEPS    = 300_000
EVAL_FREQ      = 10_240
N_EVAL_EP      = 10

COMMON_KWARGS  = dict(
    hidden_dim=64, lr_actor=3e-4, lr_critic=1e-3,
    gamma=0.99, lam=0.95, eps_clip=0.2,
    n_epochs=10, batch_size=64, n_steps=2048,
    ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
)
DCPPO_EXTRA = dict(
    use_imp_g=False, use_imp_a=False, use_imp_s=True,
    use_hcgae=True,
    snr_target=0.3, snr_gamma=0.5, snr_min_weight=0.2,
)

os.makedirs(SAVE_ROOT, exist_ok=True)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    import torch; torch.manual_seed(seed)


def run_one(env_id, agent_cls, agent_name, seed, extra_kwargs=None):
    save_dir = os.path.join(SAVE_ROOT, env_id)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{agent_name}_s{seed}.json")
    if os.path.exists(out_path):
        print(f"    [skip] {agent_name} on {env_id} seed={seed} already exists")
        return json.load(open(out_path))

    set_seed(seed)
    train_env = gym.make(env_id); train_env.reset(seed=seed)
    eval_env  = gym.make(env_id); eval_env.reset(seed=seed + 9999)

    kwargs = dict(**COMMON_KWARGS)
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    if agent_cls is DCPPO:
        agent = DCPPO(env=train_env, name=agent_name, save_dir=save_dir,
                      **kwargs)
    else:
        agent = agent_cls(env=train_env, save_dir=save_dir, **kwargs)

    t0 = time.time()
    logger = agent.train(
        total_timesteps=TOTAL_STEPS,
        eval_env=eval_env, eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EP, verbose=False,
    )
    elapsed = time.time() - t0
    train_env.close(); eval_env.close()

    rews  = logger.eval_rewards if logger.eval_rewards else [0.0]
    steps = logger.eval_steps   if logger.eval_steps   else [0]

    result = dict(
        env=env_id, agent=agent_name, seed=seed,
        final_reward=float(np.mean(rews[-5:])) if len(rews) >= 5 else float(np.mean(rews)),
        best_reward=float(np.max(rews)),
        all_eval_rewards=[float(r) for r in rews],
        eval_steps=[int(s) for s in steps],
        elapsed_s=round(elapsed, 1),
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"    Done: {agent_name} {env_id} s={seed}  "
          f"final={result['final_reward']:.0f}  elapsed={elapsed:.0f}s")
    return result


def build_global_summary(results_by_env):
    summary = {"configs": {"envs": ENVS, "seeds": SEEDS,
                            "algorithms": ["Standard_PPO", "DCPPO_S"],
                            "total_timesteps": TOTAL_STEPS},
               "results": {}}
    for env, algo_dict in results_by_env.items():
        summary["results"][env] = {}
        for algo, runs in algo_dict.items():
            vals = [r["final_reward"] for r in runs if r]
            summary["results"][env][algo] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals)),
                "seeds": vals,
            }
    return summary


def main():
    print("=" * 68)
    print("  GAP1: DCPPO-S Multi-Environment × 5-Seed")
    print(f"  Envs: {ENVS}  Seeds: {SEEDS}  Steps: {TOTAL_STEPS//1000}K")
    print("=" * 68)

    results_by_env = {}

    for env_id in ENVS:
        results_by_env[env_id] = {"Standard_PPO": [], "DCPPO_S": []}
        print(f"\n  Environment: {env_id}")

        for seed in SEEDS:
            # Standard PPO baseline
            r = run_one(env_id, BasePPO, "Standard_PPO", seed)
            results_by_env[env_id]["Standard_PPO"].append(r)

            # DCPPO-S
            r = run_one(env_id, DCPPO, "DCPPO_S", seed,
                        extra_kwargs=dict(**DCPPO_EXTRA))
            results_by_env[env_id]["DCPPO_S"].append(r)

    summary = build_global_summary(results_by_env)
    spath = os.path.join(SAVE_ROOT, "global_summary.json")
    with open(spath, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print(f"\n{'='*68}")
    print(f"  Results Summary (mean +/- std over {len(SEEDS)} seeds)")
    print(f"  {'Env':<18} {'Standard_PPO':>14} {'DCPPO_S':>14} {'Delta':>10}")
    print(f"  {'-'*56}")
    for env in ENVS:
        sp = summary["results"][env]["Standard_PPO"]
        ds = summary["results"][env]["DCPPO_S"]
        delta = ds["mean"] - sp["mean"]
        pct   = 100 * delta / (abs(sp["mean"]) + 1e-8)
        print(f"  {env:<18} {sp['mean']:>8.0f}±{sp['std']:<5.0f} "
              f"{ds['mean']:>8.0f}±{ds['std']:<5.0f} "
              f"{delta:>+8.0f} ({pct:+.0f}%)")
    print(f"  Summary saved: {spath}")


if __name__ == "__main__":
    main()

