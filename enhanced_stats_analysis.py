#!/usr/bin/env python3
"""
增强版统计分析脚本
整合 n=5 历史数据 + 进行中的 n=10 多种子实验数据
生成完整的统计报告，供论文更新使用
"""
import json
import os
import glob
import numpy as np
from pathlib import Path


def extract_final_reward(fpath, last_n=5):
    """从 metrics.json 文件中提取最终评估奖励"""
    try:
        with open(fpath) as f:
            d = json.load(f)
        eval_rewards = d.get('eval_rewards', [])
        if not eval_rewards:
            return None
        return float(np.mean(eval_rewards[-last_n:]))
    except Exception as e:
        print(f"  Warning: could not read {fpath}: {e}")
        return None


def compute_stats(scores_a, scores_b, name_a="A", name_b="B"):
    """完整统计分析"""
    from scipy import stats as sp_stats

    scores_a = np.array(scores_a, dtype=float)
    scores_b = np.array(scores_b, dtype=float)
    n_a, n_b = len(scores_a), len(scores_b)

    mean_a, mean_b = np.mean(scores_a), np.mean(scores_b)
    std_a = np.std(scores_a, ddof=1) if n_a > 1 else 0.0
    std_b = np.std(scores_b, ddof=1) if n_b > 1 else 0.0
    sem_a = std_a / np.sqrt(n_a) if n_a > 1 else 0.0
    sem_b = std_b / np.sqrt(n_b) if n_b > 1 else 0.0

    # Mann-Whitney U test
    if n_a >= 3 and n_b >= 3:
        u_stat, p_value = sp_stats.mannwhitneyu(scores_a, scores_b, alternative='two-sided')
    else:
        u_stat, p_value = 0.0, 1.0

    # Cohen's d (pooled std)
    pooled_std = np.sqrt(
        (std_a**2 * (n_a - 1) + std_b**2 * (n_b - 1)) / (n_a + n_b - 2)
    ) if (n_a + n_b > 2) else 1.0
    cohens_d = (mean_a - mean_b) / (pooled_std + 1e-8)

    # Bootstrap 95% CI (5000 replications)
    if n_a >= 2 and n_b >= 2:
        rng = np.random.default_rng(42)
        diff_boot = []
        for _ in range(5000):
            ba = rng.choice(scores_a, size=n_a, replace=True)
            bb = rng.choice(scores_b, size=n_b, replace=True)
            diff_boot.append(np.mean(ba) - np.mean(bb))
        ci_low = float(np.percentile(diff_boot, 2.5))
        ci_high = float(np.percentile(diff_boot, 97.5))
    else:
        ci_low = ci_high = float(mean_a - mean_b)

    # Power analysis (t-test approximation, two-sided alpha=0.05)
    d_abs = abs(cohens_d)
    if d_abs > 0.01 and n_a >= 3:
        lambda_nc = d_abs * np.sqrt(n_a / 2.0)
        from scipy.stats import norm
        power = float(1 - norm.cdf(1.96 - lambda_nc) + norm.cdf(-1.96 - lambda_nc))
        # Required sample size for 80% power: n = 2*(z_alpha/2 + z_beta)^2 / d^2
        n_needed_80 = max(5, int(np.ceil(2 * (1.96 + 0.842)**2 / d_abs**2)))
    else:
        power = 0.0
        n_needed_80 = 9999

    pct = (mean_a - mean_b) / (abs(mean_b) + 1e-8) * 100

    return {
        "mean_a": float(mean_a), "mean_b": float(mean_b),
        "std_a": float(std_a), "std_b": float(std_b),
        "sem_a": float(sem_a), "sem_b": float(sem_b),
        "n_a": int(n_a), "n_b": int(n_b),
        "mann_whitney_u": float(u_stat), "p_value": float(p_value),
        "cohens_d": float(cohens_d), "pooled_std": float(pooled_std),
        "ci_low": float(ci_low), "ci_high": float(ci_high),
        "power_at_current_n": float(power),
        "n_needed_for_80pct_power": int(n_needed_80),
        "pct_improvement": float(pct),
        "significant_p05": bool(p_value < 0.05),
        "significant_p10": bool(p_value < 0.10),
        "name_a": name_a, "name_b": name_b,
    }


def power_at_n(cohens_d, n):
    """计算指定 n 下的统计功效"""
    from scipy.stats import norm
    d_abs = abs(cohens_d)
    if d_abs < 0.01:
        return 0.05
    lambda_nc = d_abs * np.sqrt(n / 2.0)
    return float(1 - norm.cdf(1.96 - lambda_nc) + norm.cdf(-1.96 - lambda_nc))


def collect_baseline_scores(env, algo, base_dir="results/BaselineComparison"):
    pattern = f"{base_dir}/{env}/{algo}/{algo}_s*_metrics.json"
    files = sorted(glob.glob(pattern))
    scores = []
    for fp in files:
        r = extract_final_reward(fp)
        if r is not None:
            scores.append(r)
    return scores


def collect_multiseed_progress():
    """读取正在进行的 MultiSeedPower 实验数据"""
    path = "results/MultiSeedPower/multiseed_summary_n10.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def main():
    print("=" * 80)
    print("  HCGAE Enhanced Statistical Analysis Report")
    print("  (Based on n=5 historical data + in-progress n=10 data)")
    print("=" * 80)

    envs = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']
    all_algos = [
        'Standard_PPO', 'HCGAE_Imp12', 'PPO_KLPEN', 'PPO_Anneal',
        'PPO_EntDecay', 'PPO_VClip', 'PPO_Full_Baseline'
    ]

    # =========================================================================
    # 1. 收集所有历史 n=5 数据
    # =========================================================================
    baseline_data = {}
    for env in envs:
        baseline_data[env] = {}
        for algo in all_algos:
            scores = collect_baseline_scores(env, algo)
            if scores:
                baseline_data[env][algo] = scores

    # 收集正在进行的多种子实验数据
    mseed_data = collect_multiseed_progress()

    # =========================================================================
    # 2. 核心结果：HCGAE vs Standard PPO (n=5)
    # =========================================================================
    print("\n" + "─" * 80)
    print("  [Section 1] Core Results: HCGAE vs Standard PPO (n=5, historical)")
    print("─" * 80)

    core_stats = {}
    for env in envs:
        hcgae_s = baseline_data[env].get('HCGAE_Imp12', [])
        std_s = baseline_data[env].get('Standard_PPO', [])
        if not hcgae_s or not std_s:
            print(f"  {env}: 缺少数据")
            continue

        stat = compute_stats(hcgae_s, std_s, 'HCGAE_Imp12', 'Standard_PPO')
        core_stats[env] = stat

        sig = "✓ p<0.05" if stat['p_value'] < 0.05 else (
              "~ p<0.10" if stat['p_value'] < 0.10 else "✗ n.s.")

        print(f"\n  {env}:")
        print(f"    HCGAE_Imp12:  {stat['mean_a']:8.1f} ± {stat['std_a']:6.1f} (SEM={stat['sem_a']:.0f}, n={stat['n_a']})")
        print(f"    Standard_PPO: {stat['mean_b']:8.1f} ± {stat['std_b']:6.1f} (SEM={stat['sem_b']:.0f}, n={stat['n_b']})")
        print(f"    Δ = {stat['pct_improvement']:+.1f}%  {sig}")
        print(f"    Mann-Whitney U={stat['mann_whitney_u']:.0f}, p={stat['p_value']:.4f}")
        print(f"    Cohen's d = {stat['cohens_d']:.3f}")
        print(f"    Bootstrap 95% CI: [{stat['ci_low']:.0f}, {stat['ci_high']:.0f}]")
        print(f"    Power at n=5: {stat['power_at_current_n']:.3f}")
        print(f"    Need n ≥ {stat['n_needed_for_80pct_power']} for 80% power")

    # =========================================================================
    # 3. 功效分析：不同 n 值下的统计功效
    # =========================================================================
    print("\n" + "─" * 80)
    print("  [Section 2] Power Analysis: Power at Different n Values")
    print("─" * 80)

    print(f"\n  {'Environment':<20} {'d':>7} {'n=5':>7} {'n=10':>7} {'n=15':>7} "
          f"{'n=20':>7} {'n=25':>7} {'n=50':>7} {'n80%':>8}")
    print(f"  {'─'*75}")

    power_table = {}
    for env in envs:
        if env not in core_stats:
            continue
        d = core_stats[env]['cohens_d']
        d_abs = abs(d)
        powers = {n: power_at_n(d_abs, n) for n in [5, 10, 15, 20, 25, 50]}
        n80 = core_stats[env]['n_needed_for_80pct_power']
        power_table[env] = {
            "cohens_d": d,
            "powers": powers,
            "n_needed_80pct": n80
        }
        print(f"  {env:<20} {d:>+7.3f} {powers[5]:>7.3f} {powers[10]:>7.3f} "
              f"{powers[15]:>7.3f} {powers[20]:>7.3f} {powers[25]:>7.3f} "
              f"{powers[50]:>7.3f} {n80:>8}")

    # =========================================================================
    # 4. 正在进行的多种子实验进度
    # =========================================================================
    if mseed_data:
        print("\n" + "─" * 80)
        print(f"  [Section 3] MultiSeedPower Experiment Progress (in-progress n=10)")
        print("─" * 80)
        for env_name, env_data in mseed_data.items():
            if env_name.startswith('_'):
                continue
            print(f"\n  {env_name}:")
            for algo_name, algo_data in env_data.items():
                seeds = algo_data.get('seeds', [])
                if seeds:
                    m = np.mean(seeds)
                    s = np.std(seeds, ddof=1) if len(seeds) > 1 else 0.0
                    sem = s / np.sqrt(len(seeds))
                    print(f"    {algo_name:<25} n={len(seeds):2d}  {m:8.1f} ± {s:6.1f} (SEM={sem:.0f})")

        # 尝试整合 n=5 历史数据 + 新多种子数据（如果存在匹配）
        print("\n  [注] 尝试整合历史n=5数据与新多种子实验数据:")
        for env_name, env_data in mseed_data.items():
            if env_name not in envs:
                continue
            for algo_name, algo_data in env_data.items():
                new_seeds = algo_data.get('seeds', [])
                hist_scores = baseline_data.get(env_name, {}).get(algo_name, [])
                if new_seeds and hist_scores and len(new_seeds) >= 5:
                    # 合并数据
                    combined = list(hist_scores) + list(new_seeds)
                    m_comb = np.mean(combined)
                    s_comb = np.std(combined, ddof=1)
                    print(f"    {env_name} | {algo_name}: "
                          f"hist_n={len(hist_scores)} + new_n={len(new_seeds)} → "
                          f"combined_n={len(combined)}: {m_comb:.1f} ± {s_comb:.1f}")

    # =========================================================================
    # 5. SCR 分析（Signal-to-Correction Ratio）
    # =========================================================================
    print("\n" + "─" * 80)
    print("  [Section 4] SCR (Signal-to-Correction Ratio) Analysis")
    print("─" * 80)
    print("""
  理论框架（§5.1）:
  SCR = |Critic Bias| / sqrt(Var[MC Return])

  SCR > 1.0 → HCGAE 有益（Critic偏差 > MC方差 → 校正信号净正）
  SCR < 1.0 → HCGAE 有害（MC方差 > Critic偏差 → 噪声超过校正）

  从实验数据推断的 SCR 水平:
  ┌─────────────────────┬──────────────────┬──────────────────┬──────────────────┐
  │ 环境                │ Hopper-v4        │ Walker2d-v4      │ HalfCheetah-v4   │
  ├─────────────────────┼──────────────────┼──────────────────┼──────────────────┤
  │ HCGAE Δ%           │ +5.1%            │ +9.0%            │ -8.2%            │
  │ EV @ 50K steps      │ ~0.45            │ ~0.50            │ ~0.75 (fast)     │
  │ α_late (correction) │ 0.081 (moderate) │ 0.083 (moderate) │ <0.05 (suppressed)│
  │ 推断 SCR            │ > 1.0 (有益)    │ > 1.0 (有益)    │ < 1.0 (有害)    │
  │ CV(episode reward)  │ 0.57             │ 0.72             │ 0.76             │
  └─────────────────────┴──────────────────┴──────────────────┴──────────────────┘

  结论：HalfCheetah-v4 的高密度奖励使 Critic 快速收敛（EV@50K=0.75），
  导致 MC 方差超过 Critic 偏差（SCR < 1.0），HCGAE 校正反而有害。

  SCR 自适应变体（§2.4）：在线估计 SCR，当 SCR < threshold 时关闭 HCGAE，
  可实现跨环境安全部署。
""")

    # =========================================================================
    # 6. 综合表格（论文 Table 1 更新）
    # =========================================================================
    print("─" * 80)
    print("  [Section 5] Paper Table 1 Update Data")
    print("─" * 80)

    print(f"\n  {'Method':<22} {'Hopper-v4':>16} {'Walker2d-v4':>16} {'HalfCheetah-v4':>17}")
    print(f"  {'─'*73}")
    for algo in all_algos:
        cols = []
        for env in envs:
            scores = baseline_data.get(env, {}).get(algo, [])
            if scores:
                m = np.mean(scores)
                sem = np.std(scores, ddof=1) / np.sqrt(len(scores))
                cols.append(f"{m:.0f} ± {sem:.0f}")
            else:
                cols.append("  N/A")
        display = algo.replace('_', ' ')
        if algo == 'HCGAE_Imp12':
            display = "**HCGAE (Ours)**"
        print(f"  {display:<22} {cols[0]:>16} {cols[1]:>16} {cols[2]:>17}")

    # =========================================================================
    # 7. DCPPO 结果摘要
    # =========================================================================
    dcppo_path = "results/MultiEnv_DCPPO/dcppo_multiseed_summary.json"
    if os.path.exists(dcppo_path):
        print("\n" + "─" * 80)
        print("  [Section 6] DCPPO Results (500K steps, 5 seeds)")
        print("─" * 80)
        with open(dcppo_path) as f:
            dcppo = json.load(f)
        for env_name, env_data in dcppo.items():
            if not isinstance(env_data, dict):
                continue
            print(f"\n  {env_name}:")
            for variant in ['DCPPO_Base', 'DCPPO_ImpS', 'DCPPO_Full']:
                d = env_data.get(variant, {})
                seeds = d.get('seeds', []) if isinstance(d, dict) else []
                if seeds:
                    m = np.mean(seeds)
                    s = np.std(seeds, ddof=1) if len(seeds) > 1 else 0.0
                    print(f"    {variant:<20}  n={len(seeds)}  {m:.1f} ± {s:.1f}")

    # =========================================================================
    # 8. 论文更新建议
    # =========================================================================
    print("\n" + "=" * 80)
    print("  [Section 7] Paper Update Recommendations")
    print("=" * 80)
    print("""
  ABSTRACT 要点（已验证数据）:
  ─────────────────────────────
  1. Hopper-v4: HCGAE 2873±220 vs PPO 2735±228 (+5.1%, d=+0.247, p=0.841, n.s.)
     → Post-hoc power: n=5 功效仅 6.8%，80% 功效需 n≥258

  2. Walker2d-v4: HCGAE 1290±305 vs PPO 1184±263 (+9.0%, d=+0.149, p=0.690, n.s.)
     → n=5 功效仅 5.6%，80% 功效需 n≥705

  3. HalfCheetah-v4: HCGAE 828±113 vs PPO 902±90 (-8.2%, d=-0.290, p=0.690, n.s.)
     → SCR < 1.0 导致 MC 校正有害；EV 门控部分自校正但不充分

  4. DCPPO_ImpS: 3056±420 on Hopper (+11.7%, d=+0.69, p=0.31, n.s.)；
     Walker2d +60.0% (d=+1.16, p=0.095, marginal)

  5. PPO-VClip/Full: catastrophic failure (d>6.0, p=0.008**)

  §7.2 LIMITATIONS 要点:
  ─────────────────────────────
  - 主要局限：n=5 样本统计功效极低（6.8%~7.4%）
  - 80% 功效需 n=187~705，远超单机实验能力
  - 诚实声明：不能声称 HCGAE vs Standard PPO 统计显著
  - 可以声称的：
    (a) PPO-VClip/Full 灾难性失败（d>6.0，**）
    (b) 效应量方向一致（Hopper/Walker d>0）
    (c) SCR<1.0 解释 HalfCheetah 失效（理论预测一致）
    (d) HCGAE I+II 协同效应 +661 pts（5 seeds consistent）
    (e) DCPPO-S 中等效应（d≈0.69 Hopper, d≈1.16 Walker）
""")

    # =========================================================================
    # 9. 保存报告
    # =========================================================================
    def clean_json(obj):
        if isinstance(obj, dict):
            return {k: clean_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_json(x) for x in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    report = {
        "core_stats_hcgae_vs_std_n5": clean_json(core_stats),
        "power_analysis": clean_json(power_table),
        "multiseed_progress": clean_json(mseed_data),
        "baseline_table_summary": clean_json({
            env: {
                algo: {
                    "mean": float(np.mean(scores)),
                    "sem": float(np.std(scores, ddof=1) / np.sqrt(len(scores))),
                    "n": len(scores),
                    "scores": [float(x) for x in scores]
                }
                for algo, scores in algo_dict.items()
            }
            for env, algo_dict in baseline_data.items()
        }),
    }

    out_path = "results/enhanced_stats_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  报告已保存至: {out_path}")
    print("\n" + "=" * 80)
    print("  分析完成")
    print("=" * 80)

    return report


if __name__ == "__main__":
    main()

