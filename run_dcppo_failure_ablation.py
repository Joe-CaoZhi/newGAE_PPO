"""
DCPPO_Full 失效机制消融实验
============================

问题：DCPPO_Full (G+A+S) 相比 DCPPO_ImpS 下降 -61%（p=0.008，d=-4.23），
      是一个重要的负面结果，但干扰源不明。

实验设计（成对消融精确定位）：
─────────────────────────────────────────────────
变体         G   A   S   作用
─────────────────────────────────────────────────
DCPPO_Base   ✗   ✗   ✗   纯 HCGAE_Imp12（已有数据）
DCPPO_ImpG   ✓   ✗   ✗   仅 G（几何均值归一化）
DCPPO_ImpA   ✗   ✓   ✗   仅 A（非对称 Clip）
DCPPO_ImpS   ✗   ✗   ✓   仅 S（SNR 梯度缩放）—— 最优单项
DCPPO_ImpGA  ✓   ✓   ✗   G+A 组合（无 S）
DCPPO_ImpGS  ✓   ✗   ✓   G+S 组合（无 A）
DCPPO_ImpAS  ✗   ✓   ✓   A+S 组合（无 G）
DCPPO_Full   ✓   ✓   ✓   全量（已有数据）
─────────────────────────────────────────────────

分析逻辑：
- 若 DCPPO_ImpGS < DCPPO_ImpS → G 干扰 S
- 若 DCPPO_ImpAS < DCPPO_ImpS → A 干扰 S
- 若 DCPPO_ImpGA ≈ DCPPO_Base → G 和 A 互相抵消
- 若 DCPPO_ImpGS ≈ DCPPO_ImpS 且 DCPPO_ImpAS < DCPPO_ImpS → A 是主要干扰源

统计方法：n=5 种子，Mann-Whitney U test，Cohen's d，95% CI

运行方式：
    python run_dcppo_failure_ablation.py [--env Hopper-v4] [--resume]
"""

import argparse
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
# 实验配置
# ──────────────────────────────────────────────────────────────────────────
ENVS = ["Hopper-v4", "Walker2d-v4"]
SEEDS = [42, 123, 456, 789, 1234]
TOTAL_TIMESTEPS = 500_000
EVAL_FREQ = 10_240
N_EVAL_EPISODES = 10

# 所有需要测试的变体（DCPPO_Base/ImpS/Full 已有数据，将被复用）
ALL_VARIANTS = [
    "DCPPO_Base",   # 已有
    "DCPPO_ImpG",   # 新增
    "DCPPO_ImpA",   # 新增
    "DCPPO_ImpS",   # 已有
    "DCPPO_ImpGA",  # 新增
    "DCPPO_ImpGS",  # 新增
    "DCPPO_ImpAS",  # 新增
    "DCPPO_Full",   # 已有
]

RESULTS_DIR = Path("results/MultiEnv_DCPPO")
ABLATION_DIR = Path("results/DCPPO_Failure_Ablation")
ABLATION_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────────────────
def get_existing_result(env_name: str, variant: str, seed: int):
    """尝试从已有的 MultiEnv_DCPPO 结果中读取数据（复用 DCPPO_Base/ImpS/Full）"""
    p = RESULTS_DIR / env_name / variant / f"{variant}_s{seed}_metrics.json"
    if p.exists():
        try:
            d = json.load(open(p))
            evr = d.get("eval_rewards", [])
            if len(evr) >= 5:
                return float(np.mean(evr[-5:]))
        except Exception:
            pass
    return None


def metrics_path_new(env_name: str, variant: str, seed: int) -> Path:
    return ABLATION_DIR / env_name / variant / f"{variant}_s{seed}_metrics.json"


def is_done_new(env_name: str, variant: str, seed: int) -> bool:
    p = metrics_path_new(env_name, variant, seed)
    if not p.exists():
        return False
    try:
        d = json.load(open(p))
        return len(d.get("eval_rewards", [])) >= 48  # 500K / 10240 ≈ 48 evals
    except Exception:
        return False


def get_final_reward_new(env_name: str, variant: str, seed: int) -> float:
    p = metrics_path_new(env_name, variant, seed)
    if not p.exists():
        return 0.0
    try:
        d = json.load(open(p))
        evr = d.get("eval_rewards", [])
        return float(np.mean(evr[-5:])) if len(evr) >= 5 else 0.0
    except Exception:
        return 0.0


def run_single(env_name: str, variant: str, seed: int) -> float:
    """运行单个 (env, variant, seed) 实验"""
    np.random.seed(seed)
    torch.manual_seed(seed)

    save_dir = str(ABLATION_DIR / env_name / variant)
    os.makedirs(save_dir, exist_ok=True)
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


# ──────────────────────────────────────────────────────────────────────────
# 统计分析
# ──────────────────────────────────────────────────────────────────────────
def compute_stats(scores_a: list, scores_b: list, name_a: str, name_b: str) -> dict:
    """计算两组数据的统计对比"""
    try:
        from scipy import stats
    except ImportError:
        return {"mean_a": float(np.mean(scores_a)), "mean_b": float(np.mean(scores_b))}

    n_a, n_b = len(scores_a), len(scores_b)
    mean_a, mean_b = float(np.mean(scores_a)), float(np.mean(scores_b))
    std_a = float(np.std(scores_a, ddof=1)) if n_a > 1 else 0.0
    std_b = float(np.std(scores_b, ddof=1)) if n_b > 1 else 0.0

    if n_a >= 2 and n_b >= 2:
        u_stat, p_value = stats.mannwhitneyu(scores_a, scores_b, alternative='two-sided')
        pooled_std = np.sqrt((std_a**2 + std_b**2) / 2) + 1e-8
        cohens_d = (mean_a - mean_b) / pooled_std
    else:
        u_stat, p_value, cohens_d = 0.0, 1.0, 0.0

    pct = (mean_a - mean_b) / (abs(mean_b) + 1e-8) * 100

    return {
        f"mean_{name_a}": mean_a,
        f"mean_{name_b}": mean_b,
        f"std_{name_a}": std_a,
        f"std_{name_b}": std_b,
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "pct_change": float(pct),
        "significant_p05": bool(p_value < 0.05),
        "significant_p10": bool(p_value < 0.10),
        "n_a": int(n_a),
        "n_b": int(n_b),
    }


def print_failure_analysis(results: dict, envs: list):
    """打印失效机制分析报告"""
    print(f"\n{'═'*80}")
    print(f"  DCPPO_Full 失效机制分析")
    print(f"{'═'*80}")

    for env_name in envs:
        if env_name not in results:
            continue
        env_results = results[env_name]
        print(f"\n  ── {env_name} ──")
        print(f"  {'变体':<16} {'均值':>8} {'±std':>7} {'n':>3}")
        print(f"  {'-'*16} {'-'*8} {'-'*7} {'-'*3}")

        for v in ALL_VARIANTS:
            if v not in env_results:
                continue
            scores = env_results[v]
            if scores:
                m = float(np.mean(scores))
                s = float(np.std(scores)) if len(scores) > 1 else 0.0
                print(f"  {v:<16} {m:>8.1f} {s:>7.1f} {len(scores):>3d}")
            else:
                print(f"  {v:<16} {'N/A':>8}")

        # 核心对比：相对 DCPPO_ImpS（最优单项）的变化
        ref = env_results.get("DCPPO_ImpS", [])
        if not ref:
            continue
        ref_mean = float(np.mean(ref))
        print(f"\n  相对 DCPPO_ImpS（{ref_mean:.1f}）的变化：")
        for v in ["DCPPO_ImpGS", "DCPPO_ImpAS", "DCPPO_Full"]:
            scores = env_results.get(v, [])
            if scores:
                m = float(np.mean(scores))
                pct = (m - ref_mean) / (abs(ref_mean) + 1e-8) * 100
                stats_r = compute_stats(scores, ref, v, "ImpS")
                sig = "★p<0.05" if stats_r["significant_p05"] else ("·p<0.10" if stats_r["significant_p10"] else "  ns")
                print(f"    {v:<16} {m:>8.1f}  ({pct:+.1f}%)  d={stats_r['cohens_d']:+.2f}  {sig}")

        # 诊断：是 G 还是 A 破坏了 S？
        print(f"\n  诊断（干扰源定位）：")
        igs = env_results.get("DCPPO_ImpGS", [])
        ias = env_results.get("DCPPO_ImpAS", [])
        if igs and ias and ref:
            m_gs = float(np.mean(igs))
            m_as = float(np.mean(ias))
            g_interference = (ref_mean - m_gs) / (abs(ref_mean) + 1e-8) * 100
            a_interference = (ref_mean - m_as) / (abs(ref_mean) + 1e-8) * 100
            print(f"    G 对 S 的干扰：ImpS({ref_mean:.0f}) → GS({m_gs:.0f}) = {-g_interference:+.1f}%")
            print(f"    A 对 S 的干扰：ImpS({ref_mean:.0f}) → AS({m_as:.0f}) = {-a_interference:+.1f}%")
            if abs(g_interference) > abs(a_interference):
                print(f"    → 主要干扰源：G（几何均值归一化 Ratio）")
            elif abs(a_interference) > abs(g_interference):
                print(f"    → 主要干扰源：A（方向感知非对称 Clip）")
            else:
                print(f"    → G 和 A 对 S 的干扰程度相近")

    print(f"\n{'═'*80}")


# ──────────────────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="DCPPO_Full 失效机制消融实验")
    parser.add_argument("--envs", nargs="+", default=ENVS)
    parser.add_argument("--resume", action="store_true", help="从已有结果继续")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--analyze-only", action="store_true",
                        help="仅分析已有数据，不运行新实验")
    args = parser.parse_args()

    summary_path = ABLATION_DIR / "failure_ablation_summary.json"

    # 收集所有数据（优先从已有结果读取）
    results: dict = {}

    print(f"\n{'═'*70}")
    print(f"  DCPPO_Full 失效机制消融实验")
    print(f"  环境: {args.envs}")
    print(f"  变体: {ALL_VARIANTS}")
    print(f"  种子: {args.seeds}")
    print(f"{'═'*70}\n")

    # 构建待运行任务列表
    missing = []
    for env_name in args.envs:
        if env_name not in results:
            results[env_name] = {}
        for variant in ALL_VARIANTS:
            if variant not in results[env_name]:
                results[env_name][variant] = []
            for seed in args.seeds:
                # 先尝试从已有数据读取
                r = get_existing_result(env_name, variant, seed)
                if r is not None:
                    results[env_name][variant].append(r)
                    print(f"  [CACHED] {env_name} {variant} s{seed}: {r:.1f}")
                elif is_done_new(env_name, variant, seed):
                    r = get_final_reward_new(env_name, variant, seed)
                    results[env_name][variant].append(r)
                    print(f"  [CACHED_NEW] {env_name} {variant} s{seed}: {r:.1f}")
                else:
                    missing.append((env_name, variant, seed))

    print(f"\n  已有数据：{sum(len(v) for env in results.values() for v in env.values())} 条")
    print(f"  待运行：{len(missing)} 条")

    if not args.analyze_only:
        t0 = time.time()
        for idx, (env_name, variant, seed) in enumerate(missing, 1):
            print(f"\n  [{idx}/{len(missing)}] {env_name} | {variant} | seed={seed}")
            t1 = time.time()
            r = run_single(env_name, variant, seed)
            elapsed = time.time() - t1
            results[env_name][variant].append(r)
            print(f"  -> {r:.1f} ({elapsed:.0f}s)")

        if missing:
            print(f"\n  全部完成，耗时 {(time.time()-t0)/60:.1f} min")

    # 保存汇总
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  已保存 -> {summary_path}")

    # 打印失效机制分析
    print_failure_analysis(results, args.envs)

    # 保存带统计的详细报告
    detailed_report = {}
    for env_name in args.envs:
        if env_name not in results:
            continue
        detailed_report[env_name] = {}
        ref = results[env_name].get("DCPPO_ImpS", [])
        for variant in ALL_VARIANTS:
            scores = results[env_name].get(variant, [])
            if scores and ref:
                stats_r = compute_stats(scores, ref, variant, "DCPPO_ImpS")
                detailed_report[env_name][variant] = {
                    "scores": scores,
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)) if len(scores) > 1 else 0.0,
                    "n": len(scores),
                    "vs_ImpS": stats_r,
                }

    report_path = ABLATION_DIR / "failure_ablation_stats.json"
    with open(report_path, "w") as f:
        json.dump(detailed_report, f, indent=2)
    print(f"  统计报告 -> {report_path}")


if __name__ == "__main__":
    main()

