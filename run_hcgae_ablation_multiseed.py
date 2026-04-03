#!/usr/bin/env python3
"""
GAP2: HCGAE Ablation Multi-Seed Validation
==========================================
Runs HCGAE ablation (Base / Imp1 / Imp2 / Imp12) across 5 seeds
on Hopper-v4 to verify synergy stability.

Also records EV trajectory to metrics files (GAP4).

Saves to: results/Hopper-v4-Ablation-MultiSeed/
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

SAVE_ROOT   = "results/Hopper-v4-Ablation-MultiSeed"
ENV_ID      = "Hopper-v4"
SEEDS       = [42, 123, 456, 789, 1234]
TOTAL_STEPS = 300_000
EVAL_FREQ   = 10_240
N_EVAL_EP   = 10

VARIANTS = [
    dict(name="HCGAE_Base",  use_imp1=False, use_imp2=False,
         use_imp3=False, use_imp4=False),
    dict(name="HCGAE_Imp1",  use_imp1=True,  use_imp2=False,
         use_imp3=False, use_imp4=False),
    dict(name="HCGAE_Imp2",  use_imp1=False, use_imp2=True,
         use_imp3=False, use_imp4=False),
    dict(name="HCGAE_Imp12", use_imp1=True,  use_imp2=True,
         use_imp3=False, use_imp4=False),
]

COMMON_KWARGS = dict(
    hidden_dim=64, lr_actor=3e-4, lr_critic=1e-3,
    gamma=0.99, lam=0.95, eps_clip=0.2,
    n_epochs=10, batch_size=64, n_steps=2048,
)

os.makedirs(SAVE_ROOT, exist_ok=True)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    import torch; torch.manual_seed(seed)


def run_one(variant_cfg, seed):
    vname = variant_cfg["name"]
    out_path = os.path.join(SAVE_ROOT, f"{vname}_s{seed}.json")
    if os.path.exists(out_path):
        print(f"    [skip] {vname} seed={seed}")
        return json.load(open(out_path))

    set_seed(seed)
    train_env = gym.make(ENV_ID); train_env.reset(seed=seed)
    eval_env  = gym.make(ENV_ID); eval_env.reset(seed=seed + 9999)

    agent = HindsightAblation(
        env=train_env,
        name=vname,
        use_imp1=variant_cfg["use_imp1"],
        use_imp2=variant_cfg["use_imp2"],
        use_imp3=variant_cfg["use_imp3"],
        use_imp4=variant_cfg["use_imp4"],
        save_dir=SAVE_ROOT,
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
    ev_vals = getattr(logger, "ev_ema_history", [])

    result = dict(
        variant=vname, seed=seed,
        final_reward=float(np.mean(rews[-5:])) if len(rews) >= 5 else float(np.mean(rews)),
        best_reward=float(np.max(rews)),
        stability_std=float(np.std(rews[-10:])) if len(rews) >= 10 else float(np.std(rews)),
        all_eval_rewards=[float(r) for r in rews],
        eval_steps=[int(s) for s in steps],
        ev_trajectory=[float(v) for v in ev_vals],
        elapsed_s=round(elapsed, 1),
        use_imp1=variant_cfg["use_imp1"],
        use_imp2=variant_cfg["use_imp2"],
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"    Done: {vname} s={seed}  "
          f"final={result['final_reward']:.0f}  best={result['best_reward']:.0f}  "
          f"elapsed={elapsed:.0f}s")
    return result


def build_summary(all_results):
    from collections import defaultdict
    by_variant = defaultdict(list)
    for r in all_results:
        by_variant[r["variant"]].append(r)

    summary = {}
    for vname, runs in by_variant.items():
        vals = [r["final_reward"] for r in runs]
        summary[vname] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals)),
            "seeds": {r["seed"]: r["final_reward"] for r in runs},
        }
    return summary


def main():
    print("=" * 68)
    print(f"  GAP2: HCGAE Ablation Multi-Seed ({ENV_ID})")
    print(f"  Seeds: {SEEDS}   Steps: {TOTAL_STEPS//1000}K")
    print("=" * 68)

    all_results = []
    for vcfg in VARIANTS:
        print(f"\n  Variant: {vcfg['name']}")
        for seed in SEEDS:
            r = run_one(vcfg, seed)
            all_results.append(r)

    summary = build_summary(all_results)

    spath = os.path.join(SAVE_ROOT, "multiseed_summary.json")
    with open(spath, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*68}")
    print(f"  Multi-Seed Ablation Summary (Hopper-v4, 5 seeds)")
    print(f"  {'Variant':<16} {'Mean':>8} {'Std':>7} {'Delta vs Base':>14}")
    print(f"  {'-'*50}")
    base_mean = summary.get("HCGAE_Base", {}).get("mean", 0)
    for vname in ["HCGAE_Base", "HCGAE_Imp1", "HCGAE_Imp2", "HCGAE_Imp12"]:
        if vname in summary:
            m, s = summary[vname]["mean"], summary[vname]["std"]
            delta = m - base_mean
            print(f"  {vname:<16} {m:>8.0f} {s:>7.0f} {delta:>+14.0f}")

    if "HCGAE_Imp1" in summary and "HCGAE_Imp2" in summary and "HCGAE_Imp12" in summary:
        d1    = summary["HCGAE_Imp1"]["mean"]  - base_mean
        d2    = summary["HCGAE_Imp2"]["mean"]  - base_mean
        d12   = summary["HCGAE_Imp12"]["mean"] - base_mean
        syn   = d12 - d1 - d2
        print(f"\n  Synergy (Imp1+Imp2 vs additive): {syn:+.0f} pts")

    print(f"  Summary saved: {spath}")


if __name__ == "__main__":
    main()

