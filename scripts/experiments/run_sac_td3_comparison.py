"""
SAC / TD3 vs HCGAE 跨方法对比实验
=====================================

将离线策略方法（SAC、TD3）与在线策略 HCGAE 进行公平对比。
对比说明：
  - SAC/TD3 使用 1M 环境步（离线策略的标准 sample-efficiency 配置）
  - HCGAE/Standard_PPO 复用已有 300K 结果（在 BaselineComparison 目录中）
  - 评测协议：每 10,240 步评测一次，10 次确定性回合
  - 种子：42, 123, 456, 789, 1234（与 PPO 实验对齐）

参考文献：
  - SAC: Haarnoja et al. (2018) https://arxiv.org/abs/1801.01290
  - TD3: Fujimoto et al. (2018) https://arxiv.org/abs/1802.09477
"""

import json
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from gae_experiments.agents.sac_td3 import SACAgent, TD3Agent

# ─────────────────────────────────────────────────────────────────────────────
# 实验配置
# ─────────────────────────────────────────────────────────────────────────────
ENVS = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]
SEEDS = [42, 123, 456, 789, 1234]

# SAC/TD3 使用 1M 步（对离线策略更公平）
SAC_TOTAL_STEPS = 1_000_000
TD3_TOTAL_STEPS = 1_000_000

EVAL_FREQ = 10_240
N_EVAL_EPISODES = 10
LEARNING_STARTS = 10_000   # 预热步数（随机动作填充 buffer）

# 结果保存目录
RESULTS_DIR = Path("results/OffPolicyComparison")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 设备（离线策略因频繁采样收益有限，保持 CPU 与 PPO 一致）
DEVICE = "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# 单次运行函数
# ─────────────────────────────────────────────────────────────────────────────
def run_single(env_name: str, algo_name: str, seed: int) -> dict:
    """运行一次 (env, algo, seed)，返回包含完整曲线的 metrics dict。"""
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)

    save_dir = str(RESULTS_DIR / env_name / algo_name)
    os.makedirs(save_dir, exist_ok=True)

    env = gym.make(env_name)
    env.reset(seed=seed)

    # ── SAC ────────────────────────────────────────────────────────────────
    if algo_name == "SAC":
        agent = SACAgent(
            env=env,
            hidden_dim=256,
            lr_actor=3e-4,
            lr_critic=3e-4,
            lr_alpha=3e-4,
            gamma=0.99,
            tau=0.005,
            alpha_init=0.2,
            auto_alpha=True,
            batch_size=256,
            buffer_size=1_000_000,
            learning_starts=LEARNING_STARTS,
            update_freq=1,
            n_updates=1,
            eval_freq=EVAL_FREQ,
            n_eval_episodes=N_EVAL_EPISODES,
            device=DEVICE,
            save_dir=save_dir,
        )
        metrics = agent.train(total_steps=SAC_TOTAL_STEPS, seed=seed)

    # ── TD3 ────────────────────────────────────────────────────────────────
    elif algo_name == "TD3":
        agent = TD3Agent(
            env=env,
            hidden_dim=256,
            lr_actor=3e-4,
            lr_critic=3e-4,
            gamma=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_delay=2,
            expl_noise=0.1,
            batch_size=256,
            buffer_size=1_000_000,
            learning_starts=LEARNING_STARTS,
            update_freq=1,
            n_updates=1,
            eval_freq=EVAL_FREQ,
            n_eval_episodes=N_EVAL_EPISODES,
            device=DEVICE,
            save_dir=save_dir,
        )
        metrics = agent.train(total_steps=TD3_TOTAL_STEPS, seed=seed)

    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    env.close()

    # 保存单次 seed 结果
    out_path = Path(save_dir) / f"{algo_name}_s{seed}_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  -> Saved: {out_path}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 汇总辅助
# ─────────────────────────────────────────────────────────────────────────────
def load_summary(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_summary(summary: dict, path: Path):
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary -> {path}")


def compute_final_mean(metrics: dict, last_k: int = 5) -> float:
    """取最后 k 次 eval 的均值作为最终性能指标。"""
    evals = metrics.get("eval_rewards", [])
    if not evals:
        return metrics.get("final_reward", 0.0)
    return float(np.mean(evals[-last_k:])) if len(evals) >= last_k else float(np.mean(evals))


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────
def main():
    summary_path = RESULTS_DIR / "offpolicy_comparison_summary.json"
    summary = load_summary(summary_path)

    algos = ["SAC", "TD3"]
    total_runs = len(ENVS) * len(algos) * len(SEEDS)
    run_idx = 0

    t0 = time.time()
    print(f"\n{'='*72}")
    print(f"  Off-Policy vs On-Policy Comparison")
    print(f"  Envs:  {ENVS}")
    print(f"  Algos: {algos}  (SAC/TD3: {SAC_TOTAL_STEPS:,} steps)")
    print(f"  Seeds: {SEEDS}")
    print(f"  Total runs: {total_runs}")
    print(f"{'='*72}\n")

    for env_name in ENVS:
        if env_name not in summary:
            summary[env_name] = {}

        for algo_name in algos:
            if algo_name not in summary[env_name]:
                summary[env_name][algo_name] = {"seeds": [], "mean": None, "std": None}

            for seed in SEEDS:
                run_idx += 1
                seed_results = summary[env_name][algo_name]["seeds"]

                # 检查是否已有该 seed 结果（通过文件存在性判断，更可靠）
                save_dir = RESULTS_DIR / env_name / algo_name
                out_path = save_dir / f"{algo_name}_s{seed}_metrics.json"

                if out_path.exists() and len(seed_results) >= SEEDS.index(seed) + 1:
                    print(f"  [{run_idx}/{total_runs}] SKIP {env_name} {algo_name} seed={seed} (already done)")
                    continue

                print(f"\n  [{run_idx}/{total_runs}] {env_name} | {algo_name} | seed={seed}")
                t1 = time.time()

                try:
                    metrics = run_single(env_name, algo_name, seed)
                    final_mean = compute_final_mean(metrics)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    final_mean = 0.0

                elapsed = time.time() - t1
                print(f"  -> {algo_name} {env_name} s{seed}: {final_mean:.1f} ({elapsed:.0f}s)")

                # 如果已从文件加载过，避免重复 append
                if len(seed_results) < SEEDS.index(seed) + 1:
                    seed_results.append(final_mean)
                else:
                    seed_results[SEEDS.index(seed)] = final_mean

                summary[env_name][algo_name]["mean"] = float(np.mean(seed_results))
                summary[env_name][algo_name]["std"] = float(np.std(seed_results))
                save_summary(summary, summary_path)

    # ── 打印最终汇总表 ──────────────────────────────────────────────────────
    total_elapsed = time.time() - t0
    print(f"\n{'='*72}")
    print(f"  Off-Policy Comparison Final Results (last-5 eval mean ± std)")
    print(f"  Total wall time: {total_elapsed/3600:.2f}h")
    print(f"{'='*72}")
    print(f"{'Algorithm':<12} | {'Hopper-v4':>18} | {'Walker2d-v4':>18} | {'HalfCheetah-v4':>18}")
    print(f"{'-'*12} | {'-'*18} | {'-'*18} | {'-'*18}")

    for algo_name in algos:
        row = f"{algo_name:<12}"
        for env_name in ENVS:
            info = summary.get(env_name, {}).get(algo_name, {})
            m = info.get("mean") or 0.0
            s = info.get("std") or 0.0
            row += f" | {m:>10.0f} ± {s:>5.0f}"
        print(row)

    # 也打印已有的 PPO 数据作参考（从 BaselineComparison 中读取）
    print(f"\n  Reference: On-Policy baselines (from BaselineComparison, 300K steps, 5 seeds)")
    baseline_summary_path = Path("results/BaselineComparison/baseline_comparison_summary.json")
    if baseline_summary_path.exists():
        with open(baseline_summary_path) as f:
            bl = json.load(f)
        for ref_algo in ["Standard_PPO", "HCGAE_Imp12"]:
            row = f"{ref_algo:<25}"
            for env_name in ENVS:
                info = bl.get(env_name, {}).get(ref_algo, {})
                m = info.get("mean") or 0.0
                s = info.get("std") or 0.0
                row += f" | {m:>10.0f} ± {s:>5.0f}"
            print(row)

    print(f"\n{'='*72}")
    print(f"  Full summary: {summary_path}")


if __name__ == "__main__":
    main()

