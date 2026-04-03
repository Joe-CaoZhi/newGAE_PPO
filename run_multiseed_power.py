"""
Multi-Seed Statistical Power Experiment
=========================================

目标：通过增加种子数量（n=10 快速验证 / n=25 完整版）来提升统计功效，
解决 n=5 种子下统计功效不足（Hopper p=0.841, d=0.28）的问题。

实验设计：
-----------
1. HCGAE_Imp12 vs Standard_PPO（主要对比）
   - 环境：Hopper-v4, Walker2d-v4, HalfCheetah-v4
   - 目的：验证 HCGAE 效果是否统计显著

2. HCGAE_Imp12_SCR vs HCGAE_Imp12（SCR 自适应消融）
   - 环境：Hopper-v4, Walker2d-v4, HalfCheetah-v4（重点验证 HalfCheetah）
   - 目的：验证 SCR 自适应能否改善 HalfCheetah 上的表现

统计分析：
-----------
- Mann-Whitney U test（非参数检验）
- Cohen's d（效应量）
- Bootstrap 95% CI
- 功效分析（power analysis）

运行模式：
-----------
--quick : n=10 种子，约 50 分钟
--full  : n=25 种子，约 2 小时
--scr   : 同时运行 SCR 变体对比
"""

import argparse
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
TOTAL_TIMESTEPS = 300_000
EVAL_FREQ = 10_240
N_EVAL_EPISODES = 10

RESULTS_DIR = Path("results/MultiSeedPower")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────
def run_single(env_name: str, algo_name: str, seed: int, save_dir: str,
               use_scr_adapt: bool = False) -> dict:
    """Run one (env, algo, seed) combination, return result dict."""
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)

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
        if algo_name in ("Standard_PPO", "PPO_Anneal", "PPO_KLPEN", "PPO_EntDecay",
                         "PPO_VClip", "PPO_Full_Baseline"):
            bl_kwargs = {k: v for k, v in kwargs.items() if k != "save_dir"}
            agent = build_ppo_baseline(
                algo_name, env,
                save_dir=save_dir,
                name=f"{algo_name}_s{seed}",
                **bl_kwargs,
            )
        else:
            # HCGAE 变体
            if use_scr_adapt and algo_name == "HCGAE_Imp12":
                # 通过 use_scr_adapt 覆盖
                kwargs["use_scr_adapt"] = True
                kwargs["scr_threshold"] = 1.0
                kwargs["scr_min_scale"] = 0.1
            agent = build_ablation_agent(
                algo_name, env,
                name=f"{algo_name}_s{seed}",
                **kwargs,
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
        best_reward = float(max(logger.eval_rewards)) if logger.eval_rewards else 0.0

        # 收集 SCR 诊断信息（如果存在）
        scr_info = {}
        if hasattr(agent, '_scr_ema'):
            scr_info = {
                "scr_ema_final": float(agent._scr_ema),
                "scr_history_mean": float(np.mean(agent._scr_history)) if agent._scr_history else 0.0,
                "scr_history_std": float(np.std(agent._scr_history)) if len(agent._scr_history) > 1 else 0.0,
            }

        result = {
            "final_mean": final_mean,
            "best_reward": best_reward,
            "ev_final": float(getattr(agent, '_ev_ema', 0.0)),
            **scr_info,
        }

    except Exception as e:
        print(f"  ERROR {algo_name} {env_name} seed={seed}: {e}")
        import traceback
        traceback.print_exc()
        result = {"final_mean": 0.0, "best_reward": 0.0, "ev_final": 0.0}
    finally:
        env.close()
        eval_env.close()

    return result


# ──────────────────────────────────────────────────────────────────────────
# Statistical Analysis
# ──────────────────────────────────────────────────────────────────────────
def compute_statistics(scores_a: list, scores_b: list, name_a: str, name_b: str) -> dict:
    """Compute statistical comparison between two sets of scores."""
    try:
        from scipy import stats
    except ImportError:
        print("  WARNING: scipy not available, using simple statistics")
        return {
            "mean_a": float(np.mean(scores_a)),
            "mean_b": float(np.mean(scores_b)),
            "std_a": float(np.std(scores_a)),
            "std_b": float(np.std(scores_b)),
            "n_a": len(scores_a),
            "n_b": len(scores_b),
        }

    n_a, n_b = len(scores_a), len(scores_b)
    mean_a, mean_b = np.mean(scores_a), np.mean(scores_b)
    std_a, std_b = np.std(scores_a, ddof=1), np.std(scores_b, ddof=1)

    # Mann-Whitney U test（非参数检验）
    u_stat, p_value = stats.mannwhitneyu(scores_a, scores_b, alternative='two-sided')

    # Cohen's d（效应量）
    pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
    cohens_d = (mean_a - mean_b) / (pooled_std + 1e-8)

    # Bootstrap 95% CI for difference
    n_bootstrap = 2000
    diff_boot = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        boot_a = rng.choice(scores_a, size=n_a, replace=True)
        boot_b = rng.choice(scores_b, size=n_b, replace=True)
        diff_boot.append(np.mean(boot_a) - np.mean(boot_b))
    ci_low = float(np.percentile(diff_boot, 2.5))
    ci_high = float(np.percentile(diff_boot, 97.5))

    # 功效分析（事后检验，基于观测效应量）
    # 使用 t-test 近似功效（统计学上的近似，Mann-Whitney 精确功效需要模拟）
    try:
        import importlib
        sm_spec = importlib.util.find_spec("statsmodels")
        if sm_spec is not None:
            from statsmodels.stats.power import TTestIndPower  # type: ignore[import]
            analysis = TTestIndPower()
            power = analysis.solve_power(effect_size=abs(cohens_d), nobs1=n_a,
                                         ratio=n_b/n_a, alpha=0.05)
        else:
            # 简单近似：基于 Cohen's d 和 n 的经验公式
            # power ≈ Φ(|d|*sqrt(n/2) - z_alpha/2)，其中 z_alpha/2=1.96
            from scipy.stats import norm
            lambda_nc = abs(cohens_d) * np.sqrt(n_a / 2.0)
            power = float(1 - norm.cdf(1.96 - lambda_nc) + norm.cdf(-1.96 - lambda_nc))
    except Exception:
        power = float("nan")

    pct_improvement = (mean_a - mean_b) / (abs(mean_b) + 1e-8) * 100

    return {
        f"mean_{name_a}": float(mean_a),
        f"mean_{name_b}": float(mean_b),
        f"std_{name_a}": float(std_a),
        f"std_{name_b}": float(std_b),
        f"n_{name_a}": n_a,
        f"n_{name_b}": n_b,
        "mann_whitney_u": float(u_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "power_estimate": float(power),
        "pct_improvement": float(pct_improvement),
        "significant_p05": p_value < 0.05,
        "significant_p10": p_value < 0.10,
    }


# ──────────────────────────────────────────────────────────────────────────
# Load / Save
# ──────────────────────────────────────────────────────────────────────────
def load_existing(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_results(data: dict, path: Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved -> {path}")


# ──────────────────────────────────────────────────────────────────────────
# Print Summary Table
# ──────────────────────────────────────────────────────────────────────────
def print_stat_summary(summary: dict, comparisons: list):
    """Print formatted statistical summary."""
    print(f"\n{'='*80}")
    print(f"  Statistical Summary")
    print(f"{'='*80}")

    for env_name in ENVS:
        if env_name not in summary:
            continue
        print(f"\n  {env_name}:")
        for algo in summary[env_name]:
            data = summary[env_name][algo]
            seeds = data.get("seeds", [])
            if seeds:
                m, s = np.mean(seeds), np.std(seeds)
                n = len(seeds)
                print(f"    {algo:<25} n={n:2d}  {m:8.1f} ± {s:6.1f}")

    print(f"\n{'='*80}")
    print(f"  Pairwise Comparisons")
    print(f"{'='*80}")
    for comp in comparisons:
        print(f"\n  {comp['name_a']} vs {comp['name_b']} | {comp['env']}:")
        stats_d = comp.get("stats", {})
        if stats_d:
            ma = stats_d.get(f"mean_{comp['name_a']}", 0)
            mb = stats_d.get(f"mean_{comp['name_b']}", 0)
            p = stats_d.get("p_value", 1.0)
            d = stats_d.get("cohens_d", 0)
            power = stats_d.get("power_estimate", float("nan"))
            ci_low = stats_d.get("ci_low", 0)
            ci_high = stats_d.get("ci_high", 0)
            pct = stats_d.get("pct_improvement", 0)
            sig = "✓ p<0.05" if p < 0.05 else ("~ p<0.10" if p < 0.10 else "✗ ns")
            print(f"    Mean: {ma:.1f} vs {mb:.1f}  (+{pct:.1f}%)")
            print(f"    Mann-Whitney p={p:.4f} {sig} | Cohen's d={d:.3f}")
            print(f"    95% CI for diff: [{ci_low:.1f}, {ci_high:.1f}]")
            print(f"    Statistical power: {power:.3f}")


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: n=10 seeds")
    parser.add_argument("--full", action="store_true",
                        help="Full mode: n=25 seeds")
    parser.add_argument("--scr", action="store_true",
                        help="Also compare SCR-adaptive variant")
    parser.add_argument("--envs", nargs="+", default=None,
                        help="Specific environments to run (default: all 3)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing results")
    args = parser.parse_args()

    # 确定种子数量
    if args.full:
        n_seeds = 25
        seeds = list(range(1, 26))   # seeds 1..25
    else:
        n_seeds = 10
        seeds = list(range(1, 11))   # seeds 1..10 (快速验证)

    envs = args.envs if args.envs else ENVS

    # 确定要运行的算法
    algos = ["Standard_PPO", "HCGAE_Imp12"]
    if args.scr:
        algos.append("HCGAE_Imp12_SCR")

    summary_path = RESULTS_DIR / f"multiseed_summary_n{n_seeds}.json"
    if args.resume:
        summary = load_existing(summary_path)
    else:
        summary = {}

    total_runs = len(envs) * len(algos) * n_seeds
    run_idx = 0
    t0 = time.time()

    print(f"\n{'='*70}")
    print(f"  Multi-Seed Statistical Power Experiment")
    print(f"  Mode: {'full' if args.full else 'quick'} | n_seeds={n_seeds}")
    print(f"  Envs: {envs}")
    print(f"  Algorithms: {algos}")
    print(f"  Total runs: {total_runs}")
    print(f"  Estimated time: ~{total_runs * 5 / 60:.0f} min (5 min/run)")
    print(f"{'='*70}\n")

    for env_name in envs:
        if env_name not in summary:
            summary[env_name] = {}

        for algo_name in algos:
            if algo_name not in summary[env_name]:
                summary[env_name][algo_name] = {
                    "seeds": [],
                    "seed_list": [],
                    "scr_ema_list": [],
                }

            for seed in seeds:
                run_idx += 1

                # Skip if already done
                seed_list = summary[env_name][algo_name].get("seed_list", [])
                if seed in seed_list:
                    print(f"  [{run_idx}/{total_runs}] SKIP {env_name} {algo_name} seed={seed}")
                    continue

                print(f"\n  [{run_idx}/{total_runs}] {env_name} | {algo_name} | seed={seed}")
                t1 = time.time()

                save_dir = str(RESULTS_DIR / env_name / algo_name)
                use_scr = (algo_name == "HCGAE_Imp12_SCR")
                algo_key = "HCGAE_Imp12" if algo_name == "HCGAE_Imp12_SCR" else algo_name
                result = run_single(env_name, algo_key, seed, save_dir, use_scr_adapt=use_scr)
                elapsed = time.time() - t1

                summary[env_name][algo_name]["seeds"].append(result["final_mean"])
                summary[env_name][algo_name]["seed_list"].append(seed)
                if "scr_ema_final" in result:
                    summary[env_name][algo_name]["scr_ema_list"].append(result["scr_ema_final"])

                # Update running stats
                seeds_so_far = summary[env_name][algo_name]["seeds"]
                summary[env_name][algo_name]["mean"] = float(np.mean(seeds_so_far))
                summary[env_name][algo_name]["std"] = float(np.std(seeds_so_far))
                summary[env_name][algo_name]["n_seeds"] = len(seeds_so_far)

                print(f"  -> {algo_name} {env_name} s{seed}: {result['final_mean']:.1f} ({elapsed:.0f}s)")
                print(f"     Running: mean={summary[env_name][algo_name]['mean']:.1f} ± "
                      f"{summary[env_name][algo_name]['std']:.1f} (n={len(seeds_so_far)})")
                save_results(summary, summary_path)

    # ── Statistical analysis ──────────────────────────────────────────
    print(f"\n  Computing statistical tests...")
    comparisons = []

    for env_name in envs:
        if "Standard_PPO" not in summary.get(env_name, {}):
            continue
        if "HCGAE_Imp12" not in summary.get(env_name, {}):
            continue

        scores_std = summary[env_name]["Standard_PPO"]["seeds"]
        scores_hcgae = summary[env_name]["HCGAE_Imp12"]["seeds"]

        if len(scores_std) >= 5 and len(scores_hcgae) >= 5:
            stats_result = compute_statistics(
                scores_hcgae, scores_std,
                "HCGAE_Imp12", "Standard_PPO"
            )
            comparisons.append({
                "env": env_name,
                "name_a": "HCGAE_Imp12",
                "name_b": "Standard_PPO",
                "stats": stats_result,
            })

        # SCR vs 无 SCR
        if args.scr and "HCGAE_Imp12_SCR" in summary.get(env_name, {}):
            scores_scr = summary[env_name]["HCGAE_Imp12_SCR"]["seeds"]
            if len(scores_scr) >= 5 and len(scores_hcgae) >= 5:
                stats_result = compute_statistics(
                    scores_scr, scores_hcgae,
                    "HCGAE_Imp12_SCR", "HCGAE_Imp12"
                )
                comparisons.append({
                    "env": env_name,
                    "name_a": "HCGAE_Imp12_SCR",
                    "name_b": "HCGAE_Imp12",
                    "stats": stats_result,
                })

    # Print summary
    print_stat_summary(summary, comparisons)

    # Save with stats
    summary["_comparisons"] = comparisons
    summary["_meta"] = {
        "n_seeds": n_seeds,
        "total_time_s": time.time() - t0,
        "envs": envs,
        "algos": algos,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_results(summary, summary_path)

    # Also save a stats-only file
    stats_path = RESULTS_DIR / f"statistical_analysis_n{n_seeds}.json"
    save_results({
        "comparisons": comparisons,
        "meta": summary["_meta"],
    }, stats_path)

    print(f"\n  Total time: {(time.time() - t0)/60:.1f} min")
    print(f"  Results: {summary_path}")
    print(f"  Stats:   {stats_path}")

    return summary


if __name__ == "__main__":
    main()

