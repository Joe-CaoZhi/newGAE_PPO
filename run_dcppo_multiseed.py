"""
DCPPO-S 多种子实验
==================

目标：在 Hopper-v4 / Walker2d-v4 上运行 DCPPO-S (仅改进S)
和 DCPPO_Full 的5种子完整实验，验证 SNR 自适应梯度缩放的稳定改进效果。

实验设计：
- Variants: DCPPO_Base, DCPPO_ImpS (S only), DCPPO_Full (G+A+S)
- Envs: Hopper-v4, Walker2d-v4
- Seeds: [42, 123, 456, 789, 1234] (5 seeds)
- Timesteps: 500,000
- Eval: every 10,240 steps, 10 episodes

关键修复：每个 seed 独立保存文件，name=f"{variant}_s{seed}"
MetricLogger 保存路径: {save_dir}/{agent_name}_metrics.json
= {save_dir}/{variant}_s{seed}_metrics.json
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
from gae_experiments.agents.dcppo import build_dcppo_agent

# ──────────────────────────────────────────────────────────────────────────
ENVS = ["Hopper-v4", "Walker2d-v4"]
SEEDS = [42, 123, 456, 789, 1234]
TOTAL_TIMESTEPS = 500_000
EVAL_FREQ = 10_240
N_EVAL_EPISODES = 10

VARIANTS = ["DCPPO_Base", "DCPPO_ImpS", "DCPPO_Full"]

RESULTS_DIR = Path("results/MultiEnv_DCPPO")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = RESULTS_DIR / "dcppo_multiseed_summary.json"


def metrics_path(env_name: str, variant: str, seed: int) -> Path:
    # MetricLogger saves as {save_dir}/{agent_name}_metrics.json
    # We pass name=f"{variant}_s{seed}" so agent_name = f"{variant}_s{seed}"
    # => file = {variant}_s{seed}_metrics.json
    return RESULTS_DIR / env_name / variant / f"{variant}_s{seed}_metrics.json"


def is_done(env_name: str, variant: str, seed: int) -> bool:
    p = metrics_path(env_name, variant, seed)
    if not p.exists():
        return False
    try:
        d = json.load(open(p))
        return len(d.get("eval_rewards", [])) >= 48  # 500K / 10240 ≈ 48 evals
    except Exception:
        return False


def get_final_reward(env_name: str, variant: str, seed: int) -> float:
    p = metrics_path(env_name, variant, seed)
    if not p.exists():
        return 0.0
    try:
        d = json.load(open(p))
        evr = d.get("eval_rewards", [])
        return float(np.mean(evr[-5:])) if len(evr) >= 5 else 0.0
    except Exception:
        return 0.0


def run_single(env_name: str, variant: str, seed: int) -> float:
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Each variant's seeds share the same directory, but have distinct file names
    save_dir = str(RESULTS_DIR / env_name / variant)
    os.makedirs(save_dir, exist_ok=True)

    # Unique name per seed so MetricLogger creates separate files
    agent_name = f"{variant}_s{seed}"

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
    )

    try:
        # Pass name=agent_name so each seed gets its own metrics file
        agent = build_dcppo_agent(
            variant, env,
            save_dir=save_dir,
            name=agent_name,
            **kwargs,
        )

        logger = agent.train(
            total_timesteps=TOTAL_TIMESTEPS,
            eval_env=eval_env,
            eval_freq=EVAL_FREQ,
            n_eval_episodes=N_EVAL_EPISODES,
            verbose=True,
        )

        evr = logger.eval_rewards
        return float(np.mean(evr[-5:])) if len(evr) >= 5 else 0.0

    except Exception as e:
        print(f"  ERROR {variant} {env_name} seed={seed}: {e}")
        import traceback; traceback.print_exc()
        return 0.0
    finally:
        env.close()
        eval_env.close()


def rebuild_summary():
    summary = {}
    for env_name in ENVS:
        summary[env_name] = {}
        for variant in VARIANTS:
            seed_rewards = []
            for seed in SEEDS:
                if is_done(env_name, variant, seed):
                    seed_rewards.append(get_final_reward(env_name, variant, seed))
                else:
                    seed_rewards.append(None)
            valid = [r for r in seed_rewards if r is not None and r > 0]
            summary[env_name][variant] = {
                "seeds": seed_rewards,
                "mean": float(np.mean(valid)) if valid else None,
                "std":  float(np.std(valid)) if valid else None,
                "n_seeds": len(valid),
            }
    return summary


def main():
    missing = [(e, v, s) for e in ENVS for v in VARIANTS for s in SEEDS
               if not is_done(e, v, s)]

    print(f"\n{'='*65}")
    print(f"  DCPPO Multi-Seed Experiment")
    print(f"  Missing: {len(missing)} runs")
    for e, v, s in missing:
        print(f"    {e:20s}  {v:15s}  seed={s}")
    print(f"{'='*65}\n")

    if not missing:
        print("  All done! Rebuilding summary.")
    else:
        t0 = time.time()
        for idx, (env_name, variant, seed) in enumerate(missing, 1):
            print(f"\n  [{idx}/{len(missing)}] {env_name} | {variant} | seed={seed}")
            t1 = time.time()
            r = run_single(env_name, variant, seed)
            elapsed = time.time() - t1
            print(f"  -> {variant} {env_name} s{seed}: {r:.1f} ({elapsed:.0f}s)")
        print(f"\n  All done in {time.time()-t0:.0f}s")

    summary = rebuild_summary()
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved -> {SUMMARY_PATH}")

    # Print table
    print(f"\n{'='*65}")
    print(f"  {'Variant':<18} | {'Hopper-v4':>16} | {'Walker2d-v4':>16}")
    print(f"  {'-'*18} | {'-'*16} | {'-'*16}")
    for v in VARIANTS:
        row = f"  {v:<18}"
        for env_name in ENVS:
            info = summary.get(env_name, {}).get(v, {})
            m, s, n = info.get("mean"), info.get("std"), info.get("n_seeds", 0)
            if m is not None:
                row += f" | {m:>8.0f}±{s:>5.0f}(n={n})"
            else:
                row += f" | {'N/A':>16}"
        print(row)
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()

