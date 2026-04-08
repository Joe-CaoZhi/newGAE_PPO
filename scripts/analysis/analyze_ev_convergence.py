"""
EV 收敛研究分析脚本（§4.6 定理 1 实证验证）
=============================================
读取 results/EVConvergenceStudy/ 中的所有结果，计算：
  1. 每个算法每个环境的 steps_to_ev09 均值 ± 标准差
  2. HCGAE vs PPO 的加速比（实测 vs 理论预测 1.87×）
  3. 各关键时间点 EV 快照（80K, 150K）
  4. AULC（面积下面积）对比
  5. 统计显著性（Mann-Whitney U）
  6. 输出可直接引用的论文表格
"""
import json
import os

import numpy as np
from scipy import stats

RESULT_DIR = "results/EVConvergenceStudy"
ENVS = ["Hopper-v4", "Walker2d-v4"]
ALGOS = [
    "Optimal_PPO",
    "Optimal_HCGAE_v2",
    "Optimal_HCGAE_v2_NoBdry",
    "Optimal_HCGAE_v2_NoGate",
    "Optimal_HCGAE",
]
SEEDS = [0, 1, 2, 3, 4]

THEOREM1_PREDICTIONS = {
    "PPO_steps_to_ev09":   149_504,
    "HCGAE_steps_to_ev09":  79_872,
    "speedup_ratio":          1.87,
    "speedup_pct":            47.0,
}

ALGO_LABELS = {
    "Optimal_PPO":             "Optimal PPO (baseline)",
    "Optimal_HCGAE_v2":        "HCGAE v2 (full)",
    "Optimal_HCGAE_v2_NoBdry": "HCGAE v2 NoBdry (Imp-I+II only)",
    "Optimal_HCGAE_v2_NoGate": "HCGAE v2 NoGate (Imp-I+III only)",
    "Optimal_HCGAE":           "HCGAE v1 (baseline)",
}


def load_results():
    """加载所有已完成的实验结果。"""
    results = {}
    for env in ENVS:
        for algo in ALGOS:
            for seed in SEEDS:
                fname = os.path.join(RESULT_DIR, f"{env}_{algo}_s{seed}.json")
                if os.path.exists(fname):
                    with open(fname) as f:
                        results[f"{env}/{algo}/s{seed}"] = json.load(f)
    return results


def collect_metric(results, env, algo, metric, seeds=SEEDS):
    """收集某算法在某环境的指定指标的所有 seed 值。"""
    vals = []
    for seed in seeds:
        key = f"{env}/{algo}/s{seed}"
        r = results.get(key, {})
        v = r.get(metric, None)
        if v is not None and v != -1 and not (isinstance(v, float) and np.isnan(v)):
            vals.append(v)
    return vals


def fmt_mean_std(vals, unit="", scale=1.0, fmt=".0f"):
    if not vals:
        return "N/A"
    m = np.mean(vals) * scale
    s = np.std(vals) * scale
    return f"{m:{fmt}} ± {s:{fmt}}{unit}"


def fmt_steps(vals):
    """格式化步数为 XK 形式。"""
    if not vals:
        return "N/A"
    m = np.mean(vals) / 1000
    s = np.std(vals) / 1000
    return f"{m:.0f}K ± {s:.0f}K"


def mw_test(a, b):
    """Mann-Whitney U 双边检验，返回 (U, p)。"""
    if len(a) < 2 or len(b) < 2:
        return None, None
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return u, p


def cohens_d(a, b):
    """Cohen's d 效应量。"""
    if len(a) < 2 or len(b) < 2:
        return float('nan')
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    if pooled_std == 0:
        return float('nan')
    return (np.mean(a) - np.mean(b)) / pooled_std


def analyze(results):
    print("\n" + "=" * 90)
    print("§4.6 定理 1 实证验证 — EV 收敛速度分析结果")
    print(f"理论预测（推论 1）：PPO ~{THEOREM1_PREDICTIONS['PPO_steps_to_ev09']//1000}K 步，"
          f"HCGAE ~{THEOREM1_PREDICTIONS['HCGAE_steps_to_ev09']//1000}K 步，"
          f"加速比 ~{THEOREM1_PREDICTIONS['speedup_ratio']:.2f}×")
    print("=" * 90)

    # ── 完成度统计 ─────────────────────────────────────────────────────────
    total = len(ENVS) * len(ALGOS) * len(SEEDS)
    done = len(results)
    print(f"\n完成进度: {done}/{total} runs")

    summary = {}

    for env in ENVS:
        print(f"\n{'─'*90}")
        print(f"环境: {env}")
        print(f"{'─'*90}")

        # 表格标题
        print(f"\n{'算法':<45} {'步→EV>0.9':>14} {'EV@80K':>8} {'EV@150K':>9} "
              f"{'AULC_200K':>11} {'n':>3}")
        print(f"{'─'*95}")

        ppo_steps = collect_metric(results, env, "Optimal_PPO", "steps_to_ev09")
        ppo_ev80  = collect_metric(results, env, "Optimal_PPO", "ev_at_80k")
        ppo_ev150 = collect_metric(results, env, "Optimal_PPO", "ev_at_150k")
        ppo_aulc  = collect_metric(results, env, "Optimal_PPO", "aulc_200k")

        summary[env] = {}

        for algo in ALGOS:
            steps = collect_metric(results, env, algo, "steps_to_ev09")
            ev80  = collect_metric(results, env, algo, "ev_at_80k")
            ev150 = collect_metric(results, env, algo, "ev_at_150k")
            aulc  = collect_metric(results, env, algo, "aulc_200k")

            steps_str = fmt_steps(steps) if steps else "N/A (>500K)"
            ev80_str  = f"{np.mean(ev80):.3f}" if ev80 else "N/A"
            ev150_str = f"{np.mean(ev150):.3f}" if ev150 else "N/A"
            aulc_str  = f"{np.mean(aulc):.4f}" if aulc else "N/A"
            n = min(len(steps), len(ev80), len(ev150), len(aulc))

            label = ALGO_LABELS.get(algo, algo)
            print(f"  {label:<43} {steps_str:>14} {ev80_str:>8} {ev150_str:>9} {aulc_str:>11} {n:>3}")

            summary[env][algo] = {
                "steps_mean": np.mean(steps) if steps else None,
                "steps_std":  np.std(steps) if steps else None,
                "ev80_mean":  np.mean(ev80) if ev80 else None,
                "ev150_mean": np.mean(ev150) if ev150 else None,
                "aulc_mean":  np.mean(aulc) if aulc else None,
                "n": n,
            }

        # 理论预测对照行
        print(f"  {'[Corollary 1 预测 PPO]':<43} "
              f"{'~150K':>14} {'<0.9':>8} {'≥0.9':>9} {'—':>11}")
        print(f"  {'[Corollary 1 预测 HCGAE]':<43} "
              f"{'~80K':>14} {'≥0.9':>8} {'≥0.9':>9} {'—':>11}")

        # ── 核心对比：HCGAE v2 vs PPO ──────────────────────────────────────
        hcgae_steps = collect_metric(results, env, "Optimal_HCGAE_v2", "steps_to_ev09")
        hcgae_aulc  = collect_metric(results, env, "Optimal_HCGAE_v2", "aulc_200k")

        if ppo_steps and hcgae_steps:
            observed_ratio = np.mean(ppo_steps) / np.mean(hcgae_steps)
            pct_reduction  = (1 - np.mean(hcgae_steps) / np.mean(ppo_steps)) * 100
            pred_ratio     = THEOREM1_PREDICTIONS["speedup_ratio"]
            agreement_pct  = abs(observed_ratio - pred_ratio) / pred_ratio * 100

            _, p_steps = mw_test(hcgae_steps, ppo_steps)
            d_steps = cohens_d(ppo_steps, hcgae_steps)  # positive = PPO takes more steps
            _, p_aulc = mw_test(hcgae_aulc, ppo_aulc) if hcgae_aulc and ppo_aulc else (None, None)

            print(f"\n  ── 核心指标：HCGAE v2 vs Optimal PPO ──")
            print(f"    PPO   步→EV>0.9: {np.mean(ppo_steps)/1000:.1f}K ± {np.std(ppo_steps)/1000:.1f}K"
                  f"  (n={len(ppo_steps)}, Corollary 1: ~150K)")
            print(f"    HCGAE 步→EV>0.9: {np.mean(hcgae_steps)/1000:.1f}K ± {np.std(hcgae_steps)/1000:.1f}K"
                  f"  (n={len(hcgae_steps)}, Corollary 1: ~80K)")
            print(f"    观测加速比: {observed_ratio:.2f}×  ({pct_reduction:.1f}% 步数减少)")
            print(f"    理论预测:   {pred_ratio:.2f}×  (47% 步数减少)")
            print(f"    预测偏差:   {agreement_pct:.1f}%  ({'✓ 符合' if agreement_pct < 30 else '△ 偏差>30%'})")
            if p_steps is not None:
                sig = "*" if p_steps < 0.05 else "n.s."
                print(f"    Mann-Whitney U: p={p_steps:.4f} {sig}, Cohen's d={d_steps:.2f}")
            if ppo_aulc and hcgae_aulc:
                aulc_diff = (np.mean(hcgae_aulc) - np.mean(ppo_aulc)) / np.mean(ppo_aulc) * 100
                print(f"    AULC_200K 提升: {aulc_diff:+.1f}%"
                      f"  (HCGAE={np.mean(hcgae_aulc):.4f} vs PPO={np.mean(ppo_aulc):.4f})")

            summary[env]["speedup_ratio_observed"] = observed_ratio
            summary[env]["speedup_pct_observed"]   = pct_reduction
            summary[env]["p_steps"]  = p_steps
            summary[env]["cohens_d_steps"] = d_steps

    # ── 论文级别的结论声明 ──────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("论文可引用结论（基于实测数据）")
    print("=" * 90)

    for env in ENVS:
        ppo_steps   = collect_metric(results, env, "Optimal_PPO", "steps_to_ev09")
        hcgae_steps = collect_metric(results, env, "Optimal_HCGAE_v2", "steps_to_ev09")
        ppo_ev80    = collect_metric(results, env, "Optimal_PPO", "ev_at_80k")
        hcgae_ev80  = collect_metric(results, env, "Optimal_HCGAE_v2", "ev_at_80k")
        ppo_ev150   = collect_metric(results, env, "Optimal_PPO", "ev_at_150k")
        hcgae_ev150 = collect_metric(results, env, "Optimal_HCGAE_v2", "ev_at_150k")

        if not (ppo_steps and hcgae_steps):
            print(f"\n{env}: 数据不足，跳过")
            continue

        ratio = np.mean(ppo_steps) / np.mean(hcgae_steps)
        pct   = (1 - np.mean(hcgae_steps) / np.mean(ppo_steps)) * 100
        n_ppo = len(ppo_steps)
        n_h   = len(hcgae_steps)
        _, p  = mw_test(hcgae_steps, ppo_steps)
        d     = cohens_d(ppo_steps, hcgae_steps)

        print(f"\n{env}:")
        print(f"  Optimal PPO   : EV>0.9 at {np.mean(ppo_steps)/1000:.0f}K ± {np.std(ppo_steps)/1000:.0f}K steps (n={n_ppo})")
        print(f"  HCGAE v2      : EV>0.9 at {np.mean(hcgae_steps)/1000:.0f}K ± {np.std(hcgae_steps)/1000:.0f}K steps (n={n_h})")
        print(f"  Speedup       : {ratio:.2f}× ({pct:.0f}% fewer steps)")
        if p is not None:
            p_str = f"p={p:.3f}" + ("*" if p < 0.05 else ", n.s.")
            print(f"  Statistics    : {p_str}, Cohen's d={d:.2f}")
        if ppo_ev80 and hcgae_ev80:
            print(f"  EV @ 80K      : PPO={np.mean(ppo_ev80):.3f}, HCGAE={np.mean(hcgae_ev80):.3f}")
        if ppo_ev150 and hcgae_ev150:
            print(f"  EV @ 150K     : PPO={np.mean(ppo_ev150):.3f}, HCGAE={np.mean(hcgae_ev150):.3f}")

    return summary


def generate_latex_table(results, summary):
    """生成可直接插入论文的 LaTeX 表格。"""
    print("\n" + "=" * 90)
    print("LaTeX 表格（可直接插入论文）")
    print("=" * 90)
    print(r"""
\begin{table}[h]
\centering
\caption{Critic EV convergence speed comparison: steps to reach EV > 0.9 (mean ± std, $n$=5 seeds, 500K steps). Corollary~1 predicts PPO $\approx$150K steps, HCGAE $\approx$80K steps (1.87$\times$ speedup).}
\label{tab:ev-convergence}
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{Steps to EV>0.9} & \textbf{EV@80K} & \textbf{EV@150K} & \textbf{AULC\_200K} \\
\midrule""")

    for env in ENVS:
        print(f"\\multicolumn{{5}}{{l}}{{\\textit{{{env}}}}} \\\\")
        for algo in ALGOS:
            steps = collect_metric(results, env, algo, "steps_to_ev09")
            ev80  = collect_metric(results, env, algo, "ev_at_80k")
            ev150 = collect_metric(results, env, algo, "ev_at_150k")
            aulc  = collect_metric(results, env, algo, "aulc_200k")

            steps_str = f"{np.mean(steps)/1000:.0f}K $\\pm$ {np.std(steps)/1000:.0f}K" if steps else "---"
            ev80_str  = f"{np.mean(ev80):.3f}" if ev80 else "---"
            ev150_str = f"{np.mean(ev150):.3f}" if ev150 else "---"
            aulc_str  = f"{np.mean(aulc):.4f}" if aulc else "---"

            label = algo.replace("Optimal_", "").replace("_v2", " v2").replace("_", " ")
            bold = "\\textbf" if "HCGAE_v2" in algo and "NoBdry" not in algo and "NoGate" not in algo else ""
            if bold:
                print(f"\\quad {label} & {bold}{{{steps_str}}} & {bold}{{{ev80_str}}} & "
                      f"{bold}{{{ev150_str}}} & {bold}{{{aulc_str}}} \\\\")
            else:
                print(f"\\quad {label} & {steps_str} & {ev80_str} & {ev150_str} & {aulc_str} \\\\")

    print(r"""\midrule
\quad \textit{Corollary 1 pred. (PPO)} & \textit{$\sim$150K} & \textit{$<$0.9} & \textit{$\geq$0.9} & --- \\
\quad \textit{Corollary 1 pred. (HCGAE)} & \textit{$\sim$80K} & \textit{$\geq$0.9} & \textit{$\geq$0.9} & --- \\
\bottomrule
\end{tabular}
\end{table}""")


def main():
    results = load_results()
    print(f"加载了 {len(results)} 个结果文件")

    if len(results) == 0:
        print("没有找到结果文件，请先运行 run_ev_convergence_study.py")
        return

    summary = analyze(results)
    generate_latex_table(results, summary)

    # 保存汇总到 JSON
    summary_path = os.path.join(RESULT_DIR, "analysis_summary.json")
    with open(summary_path, "w") as f:
        # 将 numpy 类型转换为 Python 原生类型
        def convert(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj
        json.dump(summary, f, indent=2, default=convert)
    print(f"\n汇总已保存至: {summary_path}")


if __name__ == "__main__":
    main()

