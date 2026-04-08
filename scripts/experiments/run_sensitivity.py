#!/usr/bin/env python3
"""
GAP3: Real Hyperparameter Sensitivity Experiments
==================================================
Tests HCGAE beta and alpha_max sensitivity on Hopper-v4 (seed=42, 300K steps).
Tests DCPPO-S SNR_target sensitivity on Hopper-v4 (seed=42, 300K steps).

Saves to: results/Sensitivity/
"""
import json
import os
import random
import sys
import time

import gymnasium as gym
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gae_experiments.agents.hindsight_ablation import HindsightAblation
from gae_experiments.agents.dcppo import DCPPO

SAVE_ROOT   = "results/Sensitivity"
ENV_ID      = "Hopper-v4"
SEED        = 42
TOTAL_STEPS = 300_000
EVAL_FREQ   = 10_240
N_EVAL_EP   = 10

COMMON_KWARGS = dict(
    hidden_dim=64, lr_actor=3e-4, lr_critic=1e-3,
    gamma=0.99, lam=0.95, eps_clip=0.2,
    n_epochs=10, batch_size=64, n_steps=2048,
)

os.makedirs(SAVE_ROOT, exist_ok=True)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    import torch; torch.manual_seed(seed)


def run_hcgae_variant(name, beta, alpha_max0, seed=SEED):
    out_path = os.path.join(SAVE_ROOT, f"{name}_s{seed}.json")
    if os.path.exists(out_path):
        print(f"    [skip] {name}")
        return json.load(open(out_path))

    set_seed(seed)
    train_env = gym.make(ENV_ID); train_env.reset(seed=seed)
    eval_env  = gym.make(ENV_ID); eval_env.reset(seed=seed + 9999)

    # Patch the hindsight agent to use custom beta/alpha_max0
    agent = HindsightAblation(
        env=train_env, name=name,
        use_imp1=True, use_imp2=True, use_imp3=False, use_imp4=False,
        hindsight_beta=float(beta),
        hindsight_alpha_max=float(alpha_max0),
        save_dir=SAVE_ROOT, **COMMON_KWARGS,
    )

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
        name=name, param_type="hcgae", beta=beta, alpha_max0=alpha_max0, seed=seed,
        final_reward=float(np.mean(rews[-5:])) if len(rews) >= 5 else float(np.mean(rews)),
        best_reward=float(np.max(rews)),
        all_eval_rewards=[float(r) for r in rews],
        eval_steps=[int(s) for s in steps],
        elapsed_s=round(elapsed, 1),
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"    Done: {name}  final={result['final_reward']:.0f}  elapsed={elapsed:.0f}s")
    return result


def run_dcppo_snr(snr_target, seed=SEED):
    name = f"DCPPO_S_snr{snr_target:.2f}".replace(".", "p")
    out_path = os.path.join(SAVE_ROOT, f"{name}_s{seed}.json")
    if os.path.exists(out_path):
        print(f"    [skip] {name}")
        return json.load(open(out_path))

    set_seed(seed)
    train_env = gym.make(ENV_ID); train_env.reset(seed=seed)
    eval_env  = gym.make(ENV_ID); eval_env.reset(seed=seed + 9999)

    agent = DCPPO(
        env=train_env, name=name, save_dir=SAVE_ROOT,
        use_imp_g=False, use_imp_a=False, use_imp_s=True,
        use_hcgae=True,
        snr_target=float(snr_target), snr_gamma=0.5, snr_min_weight=0.2,
        **COMMON_KWARGS,
    )

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
        name=name, param_type="dcppo_snr", snr_target=snr_target, seed=seed,
        final_reward=float(np.mean(rews[-5:])) if len(rews) >= 5 else float(np.mean(rews)),
        best_reward=float(np.max(rews)),
        all_eval_rewards=[float(r) for r in rews],
        eval_steps=[int(s) for s in steps],
        elapsed_s=round(elapsed, 1),
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"    Done: {name}  snr*={snr_target}  final={result['final_reward']:.0f}")
    return result


def main():
    print("=" * 68)
    print(f"  GAP3: Hyperparameter Sensitivity ({ENV_ID}, seed={SEED})")
    print("=" * 68)

    # --- HCGAE beta sensitivity [1.0, 2.0, 3.0, 4.0, 5.0] ---
    print(f"\n  [1/3] HCGAE beta sensitivity (alpha_max=0.7 fixed)")
    beta_results = []
    default_alpha = 0.7
    for beta in [1.0, 2.0, 3.0, 4.0, 5.0]:
        r = run_hcgae_variant(f"HCGAE_beta{beta:.0f}", beta, default_alpha)
        beta_results.append(r)

    # --- HCGAE alpha_max sensitivity [0.3, 0.5, 0.7, 0.9] ---
    print(f"\n  [2/3] HCGAE alpha_max sensitivity (beta=3.0 fixed)")
    amax_results = []
    default_beta = 3.0
    for amax in [0.3, 0.5, 0.7, 0.9]:
        r = run_hcgae_variant(f"HCGAE_amax{amax:.1f}", default_beta, amax)
        amax_results.append(r)

    # --- DCPPO-S SNR target sensitivity [0.1, 0.2, 0.3, 0.5, 0.7] ---
    print(f"\n  [3/3] DCPPO-S SNR* sensitivity")
    snr_results = []
    for snr in [0.1, 0.2, 0.3, 0.5, 0.7]:
        r = run_dcppo_snr(snr)
        snr_results.append(r)

    # Save consolidated summary
    summary = {
        "beta_sensitivity":  beta_results,
        "amax_sensitivity":  amax_results,
        "snr_sensitivity":   snr_results,
    }
    spath = os.path.join(SAVE_ROOT, "sensitivity_summary.json")
    with open(spath, "w") as f:
        json.dump(summary, f, indent=2)

    # Print tables
    print(f"\n{'='*68}")
    print("  HCGAE Beta Sensitivity:")
    print(f"  {'beta':>5}  {'final_reward':>13}")
    for r in beta_results:
        marker = " <-- default" if r["beta"] == 3.0 else ""
        print(f"  {r['beta']:>5.1f}  {r['final_reward']:>13.0f}{marker}")

    print("\n  HCGAE alpha_max Sensitivity:")
    print(f"  {'alpha_max':>9}  {'final_reward':>13}")
    for r in amax_results:
        marker = " <-- default" if r["alpha_max0"] == 0.7 else ""
        print(f"  {r['alpha_max0']:>9.1f}  {r['final_reward']:>13.0f}{marker}")

    print("\n  DCPPO-S SNR* Sensitivity:")
    print(f"  {'SNR*':>5}  {'final_reward':>13}")
    for r in snr_results:
        marker = " <-- default" if r["snr_target"] == 0.3 else ""
        print(f"  {r['snr_target']:>5.1f}  {r['final_reward']:>13.0f}{marker}")

    print(f"\n  Summary saved: {spath}")


if __name__ == "__main__":
    main()

