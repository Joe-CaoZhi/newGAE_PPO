"""
Multi-Seed Final Statistical Analysis (n=10)
=============================================

读取所有10个种子的实验数据，执行完整统计分析：
- Mann-Whitney U 检验
- Cohen's d 效应量
- Bootstrap 95% CI
- 统计功效分析
- 生成可直接用于论文的表格

使用方法：
  python3 analyze_multiseed_final.py
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional

RESULTS_DIR = Path("results/MultiSeedPower")
OUTPUT_PATH = RESULTS_DIR / "final_statistical_report_n10.json"


# ──────────────────────────────────────────────────────────
# 统计分析函数
# ──────────────────────────────────────────────────────────

def compute_statistics(scores_a: list, scores_b: list,
                       name_a: str, name_b: str) -> dict:
    """完整统计对比。"""
    try:
        from scipy import stats
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False

    n_a, n_b = len(scores_a), len(scores_b)
    arr_a = np.array(scores_a, dtype=float)
    arr_b = np.array(scores_b, dtype=float)
    mean_a, mean_b = float(np.mean(arr_a)), float(np.mean(arr_b))
    std_a = float(np.std(arr_a, ddof=1)) if n_a > 1 else 0.0
    std_b = float(np.std(arr_b, ddof=1)) if n_b > 1 else 0.0
    sem_a = std_a / np.sqrt(n_a) if n_a > 0 else 0.0
    sem_b = std_b / np.sqrt(n_b) if n_b > 0 else 0.0

    result = {
        f"mean_{name_a}": mean_a,
        f"mean_{name_b}": mean_b,
        f"std_{name_a}": std_a,
        f"std_{name_b}": std_b,
        f"sem_{name_a}": sem_a,
        f"sem_{name_b}": sem_b,
        f"n_{name_a}": n_a,
        f"n_{name_b}": n_b,
    }

    if n_a < 2 or n_b < 2:
        result.update({"p_value": float("nan"), "cohens_d": float("nan"),
                       "ci_low": float("nan"), "ci_high": float("nan"),
                       "power_estimate": float("nan"),
                       "pct_improvement": float("nan"),
                       "significant_p05": False})
        return result

    # Mann-Whitney U 检验
    if HAS_SCIPY:
        u_stat, p_value = stats.mannwhitneyu(arr_a, arr_b, alternative='two-sided')
    else:
        # 简单近似
        u_stat = float(np.sum(arr_a > arr_b[:, None]))
        p_value = float("nan")

    # Cohen's d
    pooled_std = np.sqrt((std_a**2 + std_b**2) / 2.0)
    cohens_d = (mean_a - mean_b) / (pooled_std + 1e-8)

    # Bootstrap 95% CI（差值）
    rng = np.random.default_rng(42)
    n_boot = 5000
    diff_boot = []
    for _ in range(n_boot):
        ba = rng.choice(arr_a, size=n_a, replace=True)
        bb = rng.choice(arr_b, size=n_b, replace=True)
        diff_boot.append(float(np.mean(ba) - np.mean(bb)))
    ci_low  = float(np.percentile(diff_boot, 2.5))
    ci_high = float(np.percentile(diff_boot, 97.5))

    # 功效分析
    try:
        if HAS_SCIPY:
            import importlib
            sm_spec = importlib.util.find_spec("statsmodels")
            if sm_spec is not None:
                from statsmodels.stats.power import TTestIndPower
                analysis = TTestIndPower()
                power = float(analysis.solve_power(
                    effect_size=abs(cohens_d), nobs1=n_a,
                    ratio=n_b / n_a, alpha=0.05))
            else:
                from scipy.stats import norm
                lnc = abs(cohens_d) * np.sqrt(n_a / 2.0)
                power = float(1 - norm.cdf(1.96 - lnc) + norm.cdf(-1.96 - lnc))
        else:
            power = float("nan")
    except Exception:
        power = float("nan")

    pct = (mean_a - mean_b) / (abs(mean_b) + 1e-8) * 100.0

    result.update({
        "mann_whitney_u": float(u_stat) if HAS_SCIPY else float("nan"),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "power_estimate": power,
        "pct_improvement": float(pct),
        "significant_p05": bool(p_value < 0.05) if not np.isnan(p_value) else False,
        "significant_p10": bool(p_value < 0.10) if not np.isnan(p_value) else False,
    })
    return result


# ──────────────────────────────────────────────────────────
# 加载结果
# ──────────────────────────────────────────────────────────

def load_summary() -> dict:
    """从 JSON 汇总文件加载数据。"""
    summary_path = RESULTS_DIR / "multiseed_summary_n10.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    with open(summary_path) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────
# 格式化输出
# ──────────────────────────────────────────────────────────

def sig_star(p: float) -> str:
    if np.isnan(p): return "n/a"
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "."
    return "n.s."


def print_report(summary: dict, comparisons: list):
    print("\n" + "="*80)
    print("  MULTI-SEED STATISTICAL POWER ANALYSIS  (n=10 seeds)")
    print("="*80)

    ENVS = [k for k in summary if not k.startswith("_")]
    for env in ENVS:
        print(f"\n{'─'*70}")
        print(f"  Environment: {env}")
        print(f"{'─'*70}")
        for algo, data in summary[env].items():
            seeds = data.get("seeds", [])
            n = len(seeds)
            if n == 0: continue
            m = np.mean(seeds)
            s = np.std(seeds, ddof=1) if n > 1 else 0.0
            sem = s / np.sqrt(n)
            scr_list = data.get("scr_ema_list", [])
            scr_str = f"  SCR={np.mean(scr_list):.3f}±{np.std(scr_list):.3f}" if scr_list else ""
            print(f"    {algo:<28} n={n:2d}  {m:8.1f} ± {sem:6.1f} (SEM)  std={s:6.1f}{scr_str}")

    print("\n" + "="*80)
    print("  PAIRWISE COMPARISONS")
    print("="*80)
    for comp in comparisons:
        st = comp.get("stats", {})
        if not st: continue
        na = comp["name_a"]; nb = comp["name_b"]
        ma = st.get(f"mean_{na}", 0); mb = st.get(f"mean_{nb}", 0)
        p  = st.get("p_value", float("nan"))
        d  = st.get("cohens_d", float("nan"))
        pw = st.get("power_estimate", float("nan"))
        ci_l = st.get("ci_low", float("nan"))
        ci_h = st.get("ci_high", float("nan"))
        pct  = st.get("pct_improvement", float("nan"))
        sig  = sig_star(p)
        print(f"\n  {na} vs {nb}  [{comp['env']}]")
        print(f"    Mean:  {ma:.1f} vs {mb:.1f}   Δ={ma-mb:+.1f}  ({pct:+.1f}%)")
        print(f"    Mann-Whitney p={p:.4f} {sig}  |  Cohen's d={d:+.3f}")
        print(f"    95% Bootstrap CI for (A−B): [{ci_l:.1f}, {ci_h:.1f}]")
        print(f"    Statistical power (estimated): {pw:.3f}")
        if ci_l > 0:
            print(f"    → Δ > 0 consistent across bootstrap (CI entirely positive)")
        elif ci_h < 0:
            print(f"    → Δ < 0 consistent across bootstrap (CI entirely negative)")
        else:
            print(f"    → CI spans zero; direction uncertain")


# ──────────────────────────────────────────────────────────
# 论文数字汇总（直接可用于 paper）
# ──────────────────────────────────────────────────────────

def print_paper_numbers(summary: dict, comparisons: list):
    print("\n" + "="*80)
    print("  NUMBERS FOR PAPER (copy-paste ready)")
    print("="*80)

    ENVS = [k for k in summary if not k.startswith("_")]
    for env in ENVS:
        if env not in summary: continue
        std_ppo  = summary[env].get("Standard_PPO", {})
        hcgae    = summary[env].get("HCGAE_Imp12", {})
        hcgae_s  = summary[env].get("HCGAE_Imp12_SCR", {})

        std_seeds  = std_ppo.get("seeds", [])
        hc_seeds   = hcgae.get("seeds", [])
        scr_seeds  = hcgae_s.get("seeds", [])

        print(f"\n  {env}:")
        for label, seeds in [("Standard_PPO", std_seeds),
                              ("HCGAE_Imp12", hc_seeds),
                              ("HCGAE_Imp12_SCR", scr_seeds)]:
            if seeds:
                n = len(seeds)
                m = np.mean(seeds)
                std = np.std(seeds, ddof=1)
                sem = std / np.sqrt(n)
                print(f"    {label}: {m:.0f} ± {sem:.0f} (SEM, n={n}),  std={std:.0f}")

    # 关键比较表
    print("\n  Key comparison table:")
    print(f"  {'Env':<18} {'Comparison':<45} {'p-val':>8} {'d':>7} {'power':>7} {'CI':>20}")
    print(f"  {'-'*18} {'-'*45} {'-'*8} {'-'*7} {'-'*7} {'-'*20}")
    for comp in comparisons:
        st = comp.get("stats", {})
        if not st: continue
        na = comp["name_a"]; nb = comp["name_b"]
        p  = st.get("p_value", float("nan"))
        d  = st.get("cohens_d", float("nan"))
        pw = st.get("power_estimate", float("nan"))
        ci_l = st.get("ci_low", float("nan"))
        ci_h = st.get("ci_high", float("nan"))
        sig = sig_star(p)
        print(f"  {comp['env']:<18} {na:<22} vs {nb:<20} {p:8.4f}{sig:>4} {d:7.3f} {pw:7.3f} [{ci_l:6.0f},{ci_h:6.0f}]")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    summary = load_summary()

    ENVS = [k for k in summary if not k.startswith("_")]

    # 构建比较列表
    comparisons = []

    PAIRS = [
        ("HCGAE_Imp12",     "Standard_PPO",    "HCGAE_Imp12 vs Standard_PPO"),
        ("HCGAE_Imp12_SCR", "Standard_PPO",    "HCGAE_Imp12_SCR vs Standard_PPO"),
        ("HCGAE_Imp12_SCR", "HCGAE_Imp12",     "HCGAE_Imp12_SCR vs HCGAE_Imp12"),
    ]

    for env in ENVS:
        env_data = summary.get(env, {})
        for name_a, name_b, label in PAIRS:
            if name_a not in env_data or name_b not in env_data:
                continue
            seeds_a = env_data[name_a].get("seeds", [])
            seeds_b = env_data[name_b].get("seeds", [])
            if len(seeds_a) < 3 or len(seeds_b) < 3:
                print(f"  [SKIP] {label} in {env}: too few seeds (n_a={len(seeds_a)}, n_b={len(seeds_b)})")
                continue
            st = compute_statistics(seeds_a, seeds_b, name_a, name_b)
            comparisons.append({
                "env": env,
                "name_a": name_a,
                "name_b": name_b,
                "label": label,
                "stats": st,
            })

    print_report(summary, comparisons)
    print_paper_numbers(summary, comparisons)

    # 保存
    report = {
        "summary": {
            env: {
                algo: {
                    "mean": float(np.mean(summary[env][algo]["seeds"])),
                    "std": float(np.std(summary[env][algo]["seeds"], ddof=1)) if len(summary[env][algo]["seeds"]) > 1 else 0.0,
                    "sem": float(np.std(summary[env][algo]["seeds"], ddof=1) / np.sqrt(len(summary[env][algo]["seeds"]))) if len(summary[env][algo]["seeds"]) > 1 else 0.0,
                    "n_seeds": len(summary[env][algo]["seeds"]),
                    "seeds": summary[env][algo]["seeds"],
                    "scr_ema_list": summary[env][algo].get("scr_ema_list", []),
                }
                for algo in summary[env]
            }
            for env in ENVS
        },
        "comparisons": comparisons,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved: {OUTPUT_PATH}")

    return report


if __name__ == "__main__":
    main()

