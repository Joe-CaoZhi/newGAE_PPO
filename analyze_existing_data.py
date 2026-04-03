#!/usr/bin/env python3
"""
分析现有实验数据，提取统计信息
利用已有的 n=5 种子实验数据 + 当前运行中的 n=10 种子实验数据
进行全面的统计分析
"""
import json
import os
import glob
import numpy as np
from pathlib import Path
from scipy import stats


def extract_final_reward(fpath, last_n=5):
    """从 metrics.json 文件中提取最终评估奖励"""
    with open(fpath) as f:
        d = json.load(f)
    eval_rewards = d.get('eval_rewards', [])
    if not eval_rewards:
        return None
    final = float(np.mean(eval_rewards[-last_n:]))
    return final


def compute_statistics(scores_a, scores_b, name_a, name_b):
    """计算两组得分之间的统计量"""
    n_a, n_b = len(scores_a), len(scores_b)
    mean_a, mean_b = np.mean(scores_a), np.mean(scores_b)
    std_a, std_b = np.std(scores_a, ddof=1), np.std(scores_b, ddof=1)

    # Mann-Whitney U test
    u_stat, p_value = stats.mannwhitneyu(scores_a, scores_b, alternative='two-sided')

    # Cohen's d
    pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
    cohens_d = (mean_a - mean_b) / (pooled_std + 1e-8)

    # Bootstrap 95% CI
    n_bootstrap = 5000
    diff_boot = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        boot_a = rng.choice(scores_a, size=n_a, replace=True)
        boot_b = rng.choice(scores_b, size=n_b, replace=True)
        diff_boot.append(np.mean(boot_a) - np.mean(boot_b))
    ci_low = float(np.percentile(diff_boot, 2.5))
    ci_high = float(np.percentile(diff_boot, 97.5))

    # Power analysis
    try:
        from statsmodels.stats.power import TTestIndPower
        analysis = TTestIndPower()
        power = analysis.solve_power(
            effect_size=abs(cohens_d), nobs1=n_a, ratio=n_b/n_a, alpha=0.05
        )
    except Exception:
        from scipy.stats import norm
        lambda_nc = abs(cohens_d) * np.sqrt(n_a / 2.0)
        power = float(1 - norm.cdf(1.96 - lambda_nc) + norm.cdf(-1.96 - lambda_nc))

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
        "significant_p05": bool(p_value < 0.05),
        "significant_p10": bool(p_value < 0.10),
    }


def collect_from_metrics_files(env, algo, base_dir="results/BaselineComparison"):
    """从 metrics.json 文件中收集得分"""
    pattern = f"{base_dir}/{env}/{algo}/{algo}_s*_metrics.json"
    files = sorted(glob.glob(pattern))
    scores = []
    for fp in files:
        r = extract_final_reward(fp)
        if r is not None:
            scores.append(r)
    return scores


def collect_from_ablation_multiseed(env="Hopper-v4", algo="HCGAE_Imp12"):
    """从消融多种子文件中收集得分"""
    pattern = f"results/Hopper-v4-Ablation-MultiSeed/{algo}_s*.json"
    files = sorted(glob.glob(pattern))
    scores = []
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        er = d.get('eval_rewards', [])
        if er:
            scores.append(float(np.mean(er[-5:])))
    return scores


def main():
    print("=" * 80)
    print("  统计功效分析报告 (基于已有实验数据)")
    print("=" * 80)

    envs = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']
    all_results = {}

    for env in envs:
        print(f"\n{'─'*60}")
        print(f"  环境: {env}")
        print(f"{'─'*60}")

        # 从 BaselineComparison 目录收集数据
        scores_std = collect_from_metrics_files(env, 'Standard_PPO')
        scores_hcgae = collect_from_metrics_files(env, 'HCGAE_Imp12')

        print(f"  Standard_PPO  (n={len(scores_std)}): "
              f"{np.mean(scores_std):.1f} ± {np.std(scores_std):.1f}")
        print(f"    seeds: {[f'{s:.0f}' for s in scores_std]}")
        print(f"  HCGAE_Imp12   (n={len(scores_hcgae)}): "
              f"{np.mean(scores_hcgae):.1f} ± {np.std(scores_hcgae):.1f}")
        print(f"    seeds: {[f'{s:.0f}' for s in scores_hcgae]}")

        if len(scores_std) >= 3 and len(scores_hcgae) >= 3:
            stat = compute_statistics(scores_hcgae, scores_std, 'HCGAE_Imp12', 'Standard_PPO')
            all_results[env] = {
                'HCGAE_Imp12': {'scores': scores_std, 'mean': np.mean(scores_hcgae), 'std': np.std(scores_hcgae)},
                'Standard_PPO': {'scores': scores_std, 'mean': np.mean(scores_std), 'std': np.std(scores_std)},
                'stats': stat,
            }

            sig = "✓ p<0.05" if stat['p_value'] < 0.05 else ("~ p<0.10" if stat['p_value'] < 0.10 else "✗ 不显著")
            print(f"\n  【统计检验结果】")
            print(f"    HCGAE_Imp12 vs Standard_PPO: {stat['pct_improvement']:+.1f}%")
            print(f"    Mann-Whitney p={stat['p_value']:.4f}  {sig}")
            print(f"    Cohen's d = {stat['cohens_d']:.3f}  (|d|>0.8: large, >0.5: medium, >0.2: small)")
            print(f"    95% Bootstrap CI: [{stat['ci_low']:.1f}, {stat['ci_high']:.1f}]")
            print(f"    统计功效 (power): {stat['power_estimate']:.3f}")

            # 判断效应大小
            d_abs = abs(stat['cohens_d'])
            if d_abs >= 0.8:
                effect_size = "大效应 (large)"
            elif d_abs >= 0.5:
                effect_size = "中效应 (medium)"
            elif d_abs >= 0.2:
                effect_size = "小效应 (small)"
            else:
                effect_size = "可忽略 (negligible)"
            print(f"    效应大小: {effect_size}")

    # 读取 MultiSeedPower 中已完成的运行
    mseed_path = Path("results/MultiSeedPower/multiseed_summary_n10.json")
    if mseed_path.exists():
        with open(mseed_path) as f:
            mseed = json.load(f)
        print(f"\n\n{'='*80}")
        print("  在线实验进度 (MultiSeedPower n=10)")
        print(f"{'='*80}")
        for env_name, env_data in mseed.items():
            if env_name.startswith('_'):
                continue
            print(f"\n  {env_name}:")
            for algo_name, algo_data in env_data.items():
                seeds = algo_data.get('seeds', [])
                if seeds:
                    print(f"    {algo_name:<25} n={len(seeds):2d}  "
                          f"{np.mean(seeds):8.1f} ± {np.std(seeds):6.1f}")

    # 整合分析
    print(f"\n\n{'='*80}")
    print("  综合统计分析摘要")
    print(f"{'='*80}")
    print("\n  已有数据 (n=5 seeds, from BaselineComparison):")
    print(f"  {'环境':<20} {'HCGAE_Imp12':>12} {'Standard_PPO':>14} {'改进%':>8} {'p值':>8} {'Cohen d':>9} {'功效':>7}")
    print(f"  {'─'*80}")
    for env in envs:
        if env not in all_results:
            continue
        stat = all_results[env]['stats']
        ma = stat['mean_HCGAE_Imp12']
        mb = stat['mean_Standard_PPO']
        pct = stat['pct_improvement']
        p = stat['p_value']
        d = stat['cohens_d']
        pw = stat['power_estimate']
        sig_mark = "✓" if p < 0.05 else ("~" if p < 0.10 else " ")
        print(f"  {env:<20} {ma:>12.1f} {mb:>14.1f} {pct:>+7.1f}% "
              f"{p:>8.4f}{sig_mark} {d:>9.3f} {pw:>7.3f}")

    # 需要多少种子才能达到统计功效 80%
    print(f"\n\n{'='*80}")
    print("  功效分析：需要多少种子才能达到 80% 功效？")
    print(f"{'='*80}")
    for env in envs:
        if env not in all_results:
            continue
        stat = all_results[env]['stats']
        d = abs(stat['cohens_d'])
        if d < 1e-3:
            print(f"  {env}: 效应量接近 0，无法估计")
            continue
        # 使用 t-test 近似：n_required ≈ 2 * (z_alpha + z_beta)^2 / d^2
        # 其中 z_0.025=1.96, z_0.20=0.842 (80% power)
        n_req_80 = max(5, int(np.ceil(2 * (1.96 + 0.842)**2 / d**2)))
        n_req_90 = max(5, int(np.ceil(2 * (1.96 + 1.282)**2 / d**2)))
        print(f"  {env}: d={d:.3f}  →  80%功效需 n≥{n_req_80}  |  90%功效需 n≥{n_req_90}")

    # 保存分析结果
    out = {
        "summary": {env: {
            "HCGAE_Imp12_mean": all_results[env]['stats']['mean_HCGAE_Imp12'],
            "Standard_PPO_mean": all_results[env]['stats']['mean_Standard_PPO'],
            "stats": all_results[env]['stats'],
        } for env in envs if env in all_results},
    }
    with open("results/existing_data_analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  分析结果已保存到 results/existing_data_analysis.json")


if __name__ == "__main__":
    main()

