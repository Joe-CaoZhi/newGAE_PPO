#!/usr/bin/env python3
"""
综合统计分析：整合所有已有数据，计算完整的统计报告
用于论文更新
"""
import glob
import json
import os

import numpy as np


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

    # Cohen's d
    pooled_std = np.sqrt((std_a**2 * (n_a-1) + std_b**2 * (n_b-1)) / (n_a + n_b - 2)) if (n_a + n_b > 2) else 1.0
    cohens_d = (mean_a - mean_b) / (pooled_std + 1e-8)

    # Bootstrap 95% CI
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
        ci_low = ci_high = mean_a - mean_b

    # Power analysis (t-test approximation)
    d_abs = abs(cohens_d)
    if d_abs > 0.01 and n_a >= 3:
        lambda_nc = d_abs * np.sqrt(n_a / 2.0)
        power = float(1 - sp_stats.norm.cdf(1.96 - lambda_nc) + sp_stats.norm.cdf(-1.96 - lambda_nc))
        # Needed sample size for 80% power
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


def collect_baseline_data():
    """收集 BaselineComparison 目录中的数据"""
    envs = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']
    algos = ['Standard_PPO', 'HCGAE_Imp12', 'PPO_KLPEN', 'PPO_Anneal',
             'PPO_EntDecay', 'PPO_VClip', 'PPO_Full_Baseline']

    data = {}
    for env in envs:
        data[env] = {}
        for algo in algos:
            pattern = f"results/BaselineComparison/{env}/{algo}/{algo}_s*_metrics.json"
            files = sorted(glob.glob(pattern))
            scores = []
            for fp in files:
                r = extract_final_reward(fp)
                if r is not None:
                    scores.append(r)
            if scores:
                data[env][algo] = scores
    return data


def collect_ablation_data():
    """收集消融实验数据"""
    ablation_algos = ['HCGAE_Base', 'HCGAE_Imp1', 'HCGAE_Imp2', 'HCGAE_Imp12']
    data = {}
    for algo in ablation_algos:
        pattern = f"results/Hopper-v4-Ablation-MultiSeed/{algo}_s*.json"
        files = sorted(glob.glob(pattern))
        scores = []
        for fp in files:
            r = extract_final_reward(fp)
            if r is not None:
                scores.append(r)
        if scores:
            data[algo] = scores
    return data


def collect_dcppo_data():
    """收集 DCPPO 多种子实验数据"""
    dcppo_path = "results/MultiEnv_DCPPO/dcppo_multiseed_summary.json"
    if not os.path.exists(dcppo_path):
        return {}
    with open(dcppo_path) as f:
        return json.load(f)


def collect_multiseed_power_data():
    """收集新的多种子统计功效实验数据"""
    path = "results/MultiSeedPower/multiseed_summary_n10.json"
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def main():
    print("=" * 80)
    print("  HCGAE Paper: Comprehensive Statistical Analysis Report")
    print("=" * 80)

    # 1. 收集数据
    baseline_data = collect_baseline_data()
    ablation_data = collect_ablation_data()
    dcppo_data = collect_dcppo_data()
    mseed_data = collect_multiseed_power_data()

    report = {}

    # =========================================================================
    # 2. 主要比较：HCGAE vs Standard PPO (3 environments, n=5)
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("  Section 1: Main Results — HCGAE vs Standard PPO (n=5)")
    print("=" * 80)

    envs = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']
    main_results = {}

    for env in envs:
        std_scores = baseline_data.get(env, {}).get('Standard_PPO', [])
        hcgae_scores = baseline_data.get(env, {}).get('HCGAE_Imp12', [])

        if not std_scores or not hcgae_scores:
            print(f"  {env}: Missing data")
            continue

        stat = compute_stats(hcgae_scores, std_scores, "HCGAE_Imp12", "Standard_PPO")
        main_results[env] = stat

        sig = "✓ p<0.05" if stat['p_value'] < 0.05 else ("~ p<0.10" if stat['p_value'] < 0.10 else "✗ n.s.")
        print(f"\n  {env}:")
        print(f"    HCGAE_Imp12:  {stat['mean_a']:8.1f} ± {stat['std_a']:6.1f}  (n={stat['n_a']})")
        print(f"    Standard_PPO: {stat['mean_b']:8.1f} ± {stat['std_b']:6.1f}  (n={stat['n_b']})")
        print(f"    Δ = {stat['pct_improvement']:+.1f}%   {sig}")
        print(f"    p = {stat['p_value']:.4f}   Cohen's d = {stat['cohens_d']:.3f}")
        print(f"    Bootstrap 95% CI: [{stat['ci_low']:.0f}, {stat['ci_high']:.0f}]")
        print(f"    Power at n=5: {stat['power_at_current_n']:.3f}")
        print(f"    Need n≥{stat['n_needed_for_80pct_power']} for 80% power")

    report['main_hcgae_vs_std'] = main_results

    # =========================================================================
    # 3. Walker2d HalfCheetah 详细分析
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("  Section 2: Per-Environment Detailed Stats Table")
    print("=" * 80)
    print(f"\n  {'Method':<22} {'Hopper-v4':>15} {'Walker2d-v4':>15} {'HalfCheetah-v4':>17}")
    print(f"  {'─'*75}")

    all_algos = ['Standard_PPO', 'HCGAE_Imp12', 'PPO_KLPEN', 'PPO_Anneal',
                 'PPO_EntDecay', 'PPO_VClip', 'PPO_Full_Baseline']
    table_data = {}
    for algo in all_algos:
        row = {}
        cols = []
        for env in envs:
            scores = baseline_data.get(env, {}).get(algo, [])
            if scores:
                m = np.mean(scores)
                s = np.std(scores) / np.sqrt(len(scores))  # SEM
                cols.append(f"{m:.0f} ± {s:.0f}")
                row[env] = {"mean": float(m), "sem": float(s), "n": len(scores), "scores": scores}
            else:
                cols.append("   N/A")
                row[env] = None
        table_data[algo] = row
        print(f"  {algo:<22} {cols[0]:>15} {cols[1]:>15} {cols[2]:>17}")

    report['table1_data'] = {
        algo: {env: table_data[algo].get(env) for env in envs}
        for algo in all_algos
    }

    # =========================================================================
    # 4. Mann-Whitney 全矩阵
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("  Section 3: HCGAE vs All Baselines — Mann-Whitney Tests")
    print("=" * 80)

    all_pairwise = {}
    for env in envs:
        print(f"\n  {env}:")
        hcgae_s = baseline_data.get(env, {}).get('HCGAE_Imp12', [])
        if not hcgae_s:
            continue
        pairwise = {}
        for algo in all_algos:
            if algo == 'HCGAE_Imp12':
                continue
            scores = baseline_data.get(env, {}).get(algo, [])
            if not scores:
                continue
            stat = compute_stats(hcgae_s, scores, 'HCGAE_Imp12', algo)
            pairwise[algo] = stat
            sig_str = "**" if stat['p_value'] < 0.01 else ("*" if stat['p_value'] < 0.05 else ("." if stat['p_value'] < 0.10 else "n.s."))
            print(f"    vs {algo:<22} U={stat['mann_whitney_u']:4.0f}  "
                  f"p={stat['p_value']:.4f} {sig_str:4s}  d={stat['cohens_d']:+.3f}")
        all_pairwise[env] = pairwise

    report['pairwise_comparisons'] = all_pairwise

    # =========================================================================
    # 5. 消融实验统计
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("  Section 4: Ablation Study Statistics (Hopper-v4, n=5)")
    print("=" * 80)

    ablation_report = {}
    ablation_order = ['HCGAE_Base', 'HCGAE_Imp1', 'HCGAE_Imp2', 'HCGAE_Imp12']
    for abl in ablation_order:
        scores = ablation_data.get(abl, [])
        if scores:
            m = np.mean(scores)
            s = np.std(scores, ddof=1) if len(scores) > 1 else 0.0
            sem = s / np.sqrt(len(scores))
            ablation_report[abl] = {"mean": float(m), "std": float(s), "sem": float(sem),
                                    "n": len(scores), "scores": [float(x) for x in scores]}
            print(f"  {abl:<20}  {m:8.1f} ± {s:6.1f}  (n={len(scores)})")
    report['ablation'] = ablation_report

    # 消融协同效应计算
    if all(a in ablation_report for a in ablation_order):
        base = ablation_report['HCGAE_Base']['mean']
        imp1 = ablation_report['HCGAE_Imp1']['mean']
        imp2 = ablation_report['HCGAE_Imp2']['mean']
        imp12 = ablation_report['HCGAE_Imp12']['mean']
        d1 = imp1 - base
        d2 = imp2 - base
        additive_pred = base + d1 + d2
        synergy = imp12 - additive_pred
        print(f"\n  Synergy analysis:")
        print(f"    Δ_Imp1 = {d1:+.0f},  Δ_Imp2 = {d2:+.0f}")
        print(f"    Additive prediction = {additive_pred:.0f}")
        print(f"    Actual HCGAE_Imp12  = {imp12:.0f}")
        print(f"    Synergy = {synergy:+.0f} points above additive expectation")
        report['ablation']['synergy'] = float(synergy)
        report['ablation']['additive_prediction'] = float(additive_pred)

    # =========================================================================
    # 6. DCPPO 结果
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("  Section 5: DCPPO Multi-seed Results")
    print("=" * 80)
    dcppo_report = {}
    if dcppo_data:
        for env_name, env_data in dcppo_data.items():
            if isinstance(env_data, dict) and 'DCPPO_ImpS' in env_data:
                print(f"\n  {env_name}:")
                for variant in ['DCPPO_Base', 'DCPPO_ImpS', 'DCPPO_Full']:
                    if variant in env_data:
                        d = env_data[variant]
                        scores = d.get('seeds', []) if isinstance(d, dict) else []
                        if not scores and isinstance(d, dict):
                            scores = list(d.get('seeds', {}).values()) if isinstance(d.get('seeds'), dict) else []
                        if scores:
                            m = np.mean(scores)
                            s = np.std(scores, ddof=1) if len(scores) > 1 else 0.0
                            print(f"    {variant:<20}  {m:8.1f} ± {s:6.1f}  (n={len(scores)})")
    report['dcppo'] = dcppo_data if dcppo_data else {}

    # =========================================================================
    # 7. 统计功效分析摘要（论文更新用）
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("  Section 6: Power Analysis Summary (for paper update)")
    print("=" * 80)

    power_summary = {}
    for env in envs:
        std_scores = baseline_data.get(env, {}).get('Standard_PPO', [])
        hcgae_scores = baseline_data.get(env, {}).get('HCGAE_Imp12', [])
        if not std_scores or not hcgae_scores:
            continue
        stat = compute_stats(hcgae_scores, std_scores, "HCGAE_Imp12", "Standard_PPO")
        d = abs(stat['cohens_d'])
        n5_power = stat['power_at_current_n']
        n_needed = stat['n_needed_for_80pct_power']

        # Compute power at different n values
        powers = {}
        for n_val in [5, 10, 15, 20, 25, 50]:
            if d > 0.01:
                lnc = d * np.sqrt(n_val / 2.0)
                from scipy.stats import norm
                pw = float(1 - norm.cdf(1.96 - lnc) + norm.cdf(-1.96 - lnc))
                powers[n_val] = round(pw, 3)
        power_summary[env] = {
            "cohens_d": round(stat['cohens_d'], 3),
            "power_curve": powers,
            "n_needed_80pct": n_needed,
            "current_power_n5": round(n5_power, 3),
            "current_p_value": round(stat['p_value'], 4),
        }
        print(f"\n  {env}:")
        print(f"    Cohen's d = {stat['cohens_d']:.3f}")
        print(f"    Power at n=5:  {powers.get(5, 0):.3f}")
        print(f"    Power at n=10: {powers.get(10, 0):.3f}")
        print(f"    Power at n=25: {powers.get(25, 0):.3f}")
        print(f"    Power at n=50: {powers.get(50, 0):.3f}")
        print(f"    Need n={n_needed} for 80% power")

    report['power_analysis'] = power_summary

    # =========================================================================
    # 8. 新增多种子实验进度（MultiSeedPower）
    # =========================================================================
    if mseed_data:
        print("\n\n" + "=" * 80)
        print("  Section 7: MultiSeedPower Experiment Progress (n=10, in progress)")
        print("=" * 80)
        for env_name, env_data in mseed_data.items():
            if env_name.startswith('_'):
                continue
            print(f"\n  {env_name}:")
            for algo, algo_data in env_data.items():
                seeds = algo_data.get('seeds', [])
                if seeds:
                    m = np.mean(seeds)
                    s = np.std(seeds, ddof=1) if len(seeds) > 1 else 0.0
                    print(f"    {algo:<25} n={len(seeds):2d}  {m:8.1f} ± {s:6.1f}")
                    # If we have combined data (existing n=5 + new seeds)
                    existing_key = env_data.get(algo, {})

    report['multiseed_progress'] = mseed_data

    # =========================================================================
    # 9. 保存
    # =========================================================================
    out_path = "results/comprehensive_stats_report.json"
    # Convert numpy types for JSON
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(x) for x in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    report_clean = clean_for_json(report)
    with open(out_path, "w") as f:
        json.dump(report_clean, f, indent=2)
    print(f"\n\n  Full report saved to {out_path}")

    # =========================================================================
    # 10. LATEX TABLE 格式（论文用）
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("  LaTeX Table Draft")
    print("=" * 80)
    print(r"""
\begin{table}[h]
\caption{Performance comparison --- mean $\pm$ SEM (5 seeds, 300K steps).
Mann-Whitney U test vs.\ Standard PPO: $^{**}p<0.01$, $^*p<0.05$, $^\dagger p<0.10$.}
\begin{tabular}{lccc}
\toprule
Method & Hopper-v4 & Walker2d-v4 & HalfCheetah-v4 \\
\midrule""")

    for algo in all_algos:
        cols = []
        for env in envs:
            d = table_data.get(algo, {}).get(env)
            if d:
                m, sem = d['mean'], d['sem']
                # Get stat vs HCGAE
                if algo != 'HCGAE_Imp12':
                    hcgae_vs = all_pairwise.get(env, {}).get(algo, {})
                    p = hcgae_vs.get('p_value', 1.0)
                    # Note: hcgae_vs is HCGAE vs algo (HCGAE perspective)
                    sig = ""
                else:
                    sig = r"\textbf"
                cols.append(f"${m:.0f} \\pm {sem:.0f}${sig}")
            else:
                cols.append("N/A")
        algo_display = algo.replace("_", r"\_")
        if algo == 'HCGAE_Imp12':
            algo_display = r"\textbf{HCGAE (Ours)}"
        print(f"  {algo_display} & {' & '.join(cols)} \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")

    print("\n\nDone.")
    return report


if __name__ == "__main__":
    main()

