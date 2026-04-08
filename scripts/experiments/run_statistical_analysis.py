"""
Statistical Analysis for Baseline Comparison
==============================================

计算所有可用实验数据的完整统计检验。
读取 results/BaselineComparison/ 的所有 seed 结果，
进行 Mann-Whitney U 检验，生成完整的统计表格。

输出：
1. 统计分析结果 JSON
2. 控制台输出汇总表格
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
RESULTS_BASE = Path("results/BaselineComparison")

ENVS = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]

METHODS = [
    "Standard_PPO",
    "PPO_KLPEN",
    "PPO_Anneal",
    "PPO_EntDecay",
    "PPO_VClip",
    "PPO_Full_Baseline",
    "HCGAE_Imp12",
]

SEEDS = [42, 123, 456, 789, 1234]

METHOD_LABELS = {
    "Standard_PPO":      "Standard PPO",
    "PPO_KLPEN":         "PPO-KLPEN",
    "PPO_Anneal":        "PPO-Anneal",
    "PPO_EntDecay":      "PPO-EntDecay",
    "PPO_VClip":         "PPO-VClip",
    "PPO_Full_Baseline": "PPO-Full",
    "HCGAE_Imp12":       "HCGAE (Ours)",
}


# ─────────────────────────────────────────────────────────────────────────────
def load_seed_results(env: str, method: str) -> List[float]:
    """读取某个方法在某个环境下的所有种子结果"""
    rewards = []
    method_dir = RESULTS_BASE / env / method
    if not method_dir.exists():
        return rewards

    for seed in SEEDS:
        fpath = method_dir / f"{method}_s{seed}_metrics.json"
        if not fpath.exists():
            continue
        try:
            with open(fpath) as f:
                data = json.load(f)
            evr = data.get("eval_rewards", [])
            if len(evr) >= 5:
                final = float(np.mean(evr[-5:]))
                rewards.append(final)
        except Exception as e:
            print(f"  Warning: Cannot read {fpath}: {e}")

    return rewards


def cohens_d(a: List[float], b: List[float]) -> float:
    """计算 Cohen's d 效果量"""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_std = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
                         / (na + nb - 2))
    if pooled_std < 1e-8:
        return float("nan")
    return (np.mean(a) - np.mean(b)) / pooled_std


def mann_whitney_test(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Mann-Whitney U 检验，返回 (U 统计量, p 值)"""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    try:
        result = stats.mannwhitneyu(a, b, alternative="two-sided")
        return float(result.statistic), float(result.pvalue)
    except Exception:
        return float("nan"), float("nan")


def significance_stars(p: float) -> str:
    """转换 p 值为显著性标记"""
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return "n.s."


def sem(values: List[float]) -> float:
    """标准误"""
    if len(values) < 2:
        return float("nan")
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 80)
    print("  Statistical Analysis — Baseline Comparison")
    print("=" * 80)

    # ── Load all data ──────────────────────────────────────────────────────
    data: Dict[str, Dict[str, List[float]]] = {}
    for env in ENVS:
        data[env] = {}
        for method in METHODS:
            data[env][method] = load_seed_results(env, method)

    # ── Print data summary ────────────────────────────────────────────────
    print("\n  Data availability:")
    for env in ENVS:
        print(f"\n  {env}:")
        for method in METHODS:
            vals = data[env][method]
            label = METHOD_LABELS[method]
            if vals:
                print(f"    {label:<20} n={len(vals):2d}  "
                      f"mean={np.mean(vals):8.1f} ± {sem(vals):6.1f}  "
                      f"[{min(vals):.0f}, {max(vals):.0f}]")
            else:
                print(f"    {label:<20} n= 0  (no data)")

    # ── Main results table ────────────────────────────────────────────────
    print("\n\n  Table 1: Mean ± SEM (5 seeds, last 5 evals of 300K steps)")
    print("  " + "-" * 80)
    header = f"  {'Method':<22} | {'Hopper-v4':>18} | {'Walker2d-v4':>18} | {'HalfCheetah-v4':>18}"
    print(header)
    print("  " + "-" * 80)
    for method in METHODS:
        label = METHOD_LABELS[method]
        row = f"  {label:<22}"
        for env in ENVS:
            vals = data[env][method]
            if len(vals) >= 3:
                row += f" | {np.mean(vals):8.0f} ± {sem(vals):5.0f}(n={len(vals)})"
            elif len(vals) > 0:
                row += f" | {np.mean(vals):8.0f} ± {sem(vals):5.0f}(n={len(vals)},⚠)"
            else:
                row += f" | {'—':>18}"
        print(row)
    print("  " + "-" * 80)

    # ── Mann-Whitney tests: HCGAE vs each method ─────────────────────────
    print("\n\n  Statistical Significance: HCGAE (Ours) vs. each baseline")
    print("  " + "=" * 80)

    stat_results = {}
    for env in ENVS:
        hcgae_vals = data[env]["HCGAE_Imp12"]
        if not hcgae_vals:
            print(f"\n  {env}: No HCGAE data available")
            continue

        print(f"\n  {env} (HCGAE n={len(hcgae_vals)}, mean={np.mean(hcgae_vals):.1f}):")
        print(f"  {'Baseline':<22} | {'n':>4} | {'Mean±SEM':>18} | "
              f"{'U stat':>8} | {'p-value':>8} | {'Cohen_d':>8} | Sig")
        print("  " + "-" * 85)

        stat_results[env] = {}
        for method in METHODS:
            if method == "HCGAE_Imp12":
                continue
            vals = data[env][method]
            label = METHOD_LABELS[method]
            if len(vals) < 2:
                print(f"  {label:<22} | {'—':>4} | {'—':>18} | {'—':>8} | "
                      f"{'—':>8} | {'—':>8} | —")
                continue
            u_stat, p_val = mann_whitney_test(hcgae_vals, vals)
            d = cohens_d(hcgae_vals, vals)
            sig = significance_stars(p_val)
            mean_val = np.mean(vals)
            sem_val = sem(vals)
            d_str = f"{d:+.2f}" if not np.isnan(d) else "n/a"
            p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "n/a"
            u_str = f"{u_stat:.0f}" if not np.isnan(u_stat) else "n/a"

            print(f"  {label:<22} | {len(vals):>4} | "
                  f"{mean_val:8.0f} ± {sem_val:5.0f} | "
                  f"{u_str:>8} | {p_str:>8} | {d_str:>8} | {sig}")

            stat_results[env][method] = {
                "n": len(vals),
                "mean": float(mean_val),
                "sem": float(sem_val),
                "u_stat": float(u_stat),
                "p_value": float(p_val),
                "cohens_d": float(d),
                "significance": sig,
            }

    # ── Pairwise comparison matrix ─────────────────────────────────────────
    print("\n\n  Pairwise Mann-Whitney p-values (all methods vs HCGAE, all envs)")
    for env in ENVS:
        hcgae_vals = data[env]["HCGAE_Imp12"]
        if not hcgae_vals:
            continue
        print(f"\n  {env}:")
        for method in METHODS:
            if method == "HCGAE_Imp12":
                continue
            vals = data[env][method]
            if len(vals) < 2:
                continue
            _, p_val = mann_whitney_test(hcgae_vals, vals)
            d = cohens_d(hcgae_vals, vals)
            direction = "HCGAE>" if np.mean(hcgae_vals) > np.mean(vals) else "HCGAE<"
            print(f"    HCGAE vs {METHOD_LABELS[method]:<20}: p={p_val:.4f} "
                  f"  d={d:+.2f}  {direction}  {significance_stars(p_val)}")

    # ── Save results ──────────────────────────────────────────────────────
    output = {
        "description": "Statistical analysis: Mann-Whitney U tests, HCGAE vs each baseline",
        "methods": METHODS,
        "environments": ENVS,
        "data_counts": {
            env: {m: len(data[env][m]) for m in METHODS}
            for env in ENVS
        },
        "means_sems": {
            env: {
                m: {"mean": float(np.mean(data[env][m])) if data[env][m] else None,
                    "sem": float(sem(data[env][m])) if len(data[env][m]) >= 2 else None,
                    "n": len(data[env][m]),
                    "seeds": data[env][m]}
                for m in METHODS
            }
            for env in ENVS
        },
        "hcgae_vs_baselines": stat_results,
    }

    out_path = RESULTS_BASE / "statistical_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved -> {out_path}")

    # ── Summary for paper update ──────────────────────────────────────────
    print("\n\n  === PAPER-READY TABLE (copy to paper_draft.md) ===")
    print()
    print("  | Method | Hopper-v4 | Walker2d-v4 | HalfCheetah-v4 |")
    print("  |---|:---:|:---:|:---:|")
    for method in METHODS:
        label = METHOD_LABELS[method]
        row = f"  | {label}"
        for env in ENVS:
            vals = data[env][method]
            if len(vals) >= 3:
                n_note = "" if len(vals) == 5 else f" (n={len(vals)}⚠)"
                row += f" | {np.mean(vals):.0f} ± {sem(vals):.0f}{n_note}"
            elif len(vals) > 0:
                row += f" | {np.mean(vals):.0f} ± {sem(vals):.0f} (n={len(vals)}⚠)"
            else:
                row += " | — (pending)"
        row += " |"
        print(row)

    # ── Check completeness ─────────────────────────────────────────────────
    print("\n\n  Data completeness:")
    for env in ENVS:
        for method in METHODS:
            n = len(data[env][method])
            status = "✓" if n == 5 else (f"⚠ {n}/5" if n > 0 else "✗ missing")
            print(f"    {env:20s} {METHOD_LABELS[method]:22s}: {status}")

    print()


if __name__ == "__main__":
    main()

