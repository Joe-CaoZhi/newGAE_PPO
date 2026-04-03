"""
缺失实验补跑脚本
=================

基于文件级 skip 逻辑，只运行尚未完成的 (env, algo, seed) 组合。
目标：完成 3 环境 x 7 算法 x 5 种子 = 105 runs 的完整 BaselineComparison 矩阵。

当前缺口（截至分析时）：
- HalfCheetah-v4: PPO_Anneal×5, PPO_EntDecay×5, PPO_VClip×5,
                   PPO_Full_Baseline×5, HCGAE_Imp12×5, PPO_KLPEN×1(s1234)
- Walker2d-v4: 完整 (35 runs)
- Hopper-v4: 完整 (35 runs)

实验协议（与 run_baseline_comparison.py 完全一致）：
- Timesteps: 300,000 per run
- Seeds: [42, 123, 456, 789, 1234]
- Eval: every 10,240 steps, 10 deterministic rollouts
- Network: MLP 64×64, same hyperparams for all algorithms
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

# ──────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────
ENVS = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]
SEEDS = [42, 123, 456, 789, 1234]
TOTAL_TIMESTEPS = 300_000
EVAL_FREQ = 10_240
N_EVAL_EPISODES = 10
OUR_METHOD = "HCGAE_Imp12"

BASELINES = [
    "Standard_PPO",
    "PPO_KLPEN",
    "PPO_Anneal",
    "PPO_EntDecay",
    "PPO_VClip",
    "PPO_Full_Baseline",
]
ALL_ALGOS = BASELINES + [OUR_METHOD]

RESULTS_DIR = Path("results/BaselineComparison")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = RESULTS_DIR / "baseline_comparison_summary.json"


# ──────────────────────────────────────────────────────────────────────────
# File-level skip: check if metrics JSON exists and is valid
# ──────────────────────────────────────────────────────────────────────────
def metrics_path(env_name: str, algo_name: str, seed: int) -> Path:
    return RESULTS_DIR / env_name / algo_name / f"{algo_name}_s{seed}_metrics.json"


def is_done(env_name: str, algo_name: str, seed: int) -> bool:
    p = metrics_path(env_name, algo_name, seed)
    if not p.exists():
        return False
    try:
        with open(p) as f:
            d = json.load(f)
        evr = d.get("eval_rewards", [])
        # Consider valid if at least 10 eval points (= ~100K steps covered)
        return len(evr) >= 10
    except Exception:
        return False


def get_final_reward(env_name: str, algo_name: str, seed: int) -> float:
    """Load final mean reward from metrics JSON (last 5 evals)."""
    p = metrics_path(env_name, algo_name, seed)
    if not p.exists():
        return 0.0
    try:
        with open(p) as f:
            d = json.load(f)
        evr = d.get("eval_rewards", [])
        if not evr:
            return 0.0
        last5 = evr[-5:]
        return float(np.mean(last5))
    except Exception:
        return 0.0


# ──────────────────────────────────────────────────────────────────────────
# Single run
# ──────────────────────────────────────────────────────────────────────────
def run_single(env_name: str, algo_name: str, seed: int) -> float:
    np.random.seed(seed)
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

        evr = logger.eval_rewards
        final_mean = float(np.mean(evr[-5:])) if len(evr) >= 5 else (float(np.mean(evr)) if evr else 0.0)

    except Exception as e:
        print(f"  ERROR {algo_name} {env_name} seed={seed}: {e}")
        import traceback; traceback.print_exc()
        final_mean = 0.0
    finally:
        env.close()
        eval_env.close()

    return final_mean


# ──────────────────────────────────────────────────────────────────────────
# Rebuild summary from all metrics files
# ──────────────────────────────────────────────────────────────────────────
def rebuild_summary() -> dict:
    """Rebuild summary JSON from individual metrics files."""
    summary = {}
    for env_name in ENVS:
        summary[env_name] = {}
        for algo_name in ALL_ALGOS:
            seed_rewards = []
            for seed in SEEDS:
                r = get_final_reward(env_name, algo_name, seed)
                if is_done(env_name, algo_name, seed):
                    seed_rewards.append(r)
                else:
                    seed_rewards.append(None)
            valid = [r for r in seed_rewards if r is not None and r > 0]
            summary[env_name][algo_name] = {
                "seeds": seed_rewards,
                "mean": float(np.mean(valid)) if valid else None,
                "std":  float(np.std(valid)) if valid else None,
                "n_seeds": len(valid),
            }
    return summary


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main():
    # ── Enumerate missing runs ──────────────────────────────────────────
    missing = []
    for env_name in ENVS:
        for algo_name in ALL_ALGOS:
            for seed in SEEDS:
                if not is_done(env_name, algo_name, seed):
                    missing.append((env_name, algo_name, seed))

    print(f"\n{'='*70}")
    print(f"  Baseline Comparison – Missing Experiments Runner")
    print(f"  Total missing: {len(missing)} runs")
    for env_name, algo_name, seed in missing:
        print(f"    {env_name:20s}  {algo_name:20s}  seed={seed}")
    print(f"{'='*70}\n")

    if not missing:
        print("  All experiments already complete! Rebuilding summary only.")
        summary = rebuild_summary()
        with open(SUMMARY_PATH, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary saved -> {SUMMARY_PATH}")
        return summary

    t0 = time.time()
    for idx, (env_name, algo_name, seed) in enumerate(missing, 1):
        print(f"\n  [{idx}/{len(missing)}] {env_name} | {algo_name} | seed={seed}")
        t1 = time.time()
        result = run_single(env_name, algo_name, seed)
        elapsed = time.time() - t1
        print(f"  -> {algo_name} {env_name} s{seed}: final={result:.1f}  ({elapsed:.0f}s)")

    total_elapsed = time.time() - t0
    print(f"\n  All missing runs done in {total_elapsed:.0f}s")

    # ── Rebuild summary from files ──────────────────────────────────────
    summary = rebuild_summary()
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved -> {SUMMARY_PATH}")

    # ── Print final table ──────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  Final Results (5-seed mean ± std)")
    print(f"{'='*75}")
    print(f"{'Algorithm':<22} | {'Hopper-v4':>15} | {'Walker2d-v4':>15} | {'HalfCheetah-v4':>16}")
    print(f"{'-'*22} | {'-'*15} | {'-'*15} | {'-'*16}")
    for algo_name in ALL_ALGOS:
        row = f"{algo_name:<22}"
        for env_name in ENVS:
            info = summary.get(env_name, {}).get(algo_name, {})
            m = info.get("mean")
            s = info.get("std")
            n = info.get("n_seeds", 0)
            if m is not None:
                row += f" | {m:>8.0f}±{s:>5.0f}(n={n})"
            else:
                row += f" | {'N/A':>16}"
        marker = " ← OURS" if algo_name == OUR_METHOD else ""
        print(row + marker)
    print(f"{'='*75}\n")
    return summary


if __name__ == "__main__":
    main()

