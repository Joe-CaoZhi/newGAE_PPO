#!/usr/bin/env python3
"""
Statistical Analysis for ICML Experiment Results
=================================================

Analyzes results from: results/ICMLExperiment/
Algorithms: Standard_PPO, Optimal_PPO, Optimal_HCGAE, Optimal_HCGAE_SCR
Environments: Hopper-v4, Walker2d-v4, HalfCheetah-v4

Tests:
  - Mann-Whitney U (non-parametric, robust to small n)
  - Bootstrap 95% CI for mean difference
  - Cohen's d (effect size)
  - Statistical power estimate
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

BASE = Path("results/ICMLExperiment")
ENVS = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]
ALGOS = ["Standard_PPO", "Optimal_PPO", "Optimal_HCGAE", "Optimal_HCGAE_SCR"]


# ─────────────────────────────────────────────────────────────────────────────
def load_results() -> dict:
    """Load all per-seed JSON results into {env: {algo: [final_reward, ...]}}"""
    data = {}
    for env in ENVS:
        data[env] = {}
        for algo in ALGOS:
            seeds = []
            algo_dir = BASE / env / algo
            if not algo_dir.exists():
                continue
            for fp in sorted(algo_dir.glob("*.json")):
                try:
                    d = json.load(open(fp))
                    er = d.get("eval_rewards", [])
                    if er:
                        val = float(np.mean(er[-5:])) if len(er) >= 5 else float(np.mean(er))
                        seeds.append(val)
                    elif "final_reward" in d:
                        seeds.append(float(d["final_reward"]))
                except Exception as e:
                    print(f"  [WARN] Failed to load {fp}: {e}")
            if seeds:
                data[env][algo] = seeds
    return data


# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_ci_diff(a, b, n_boot=10000, alpha=0.05):
    rng = np.random.default_rng(42)
    diffs = [rng.choice(a, len(a), replace=True).mean() -
             rng.choice(b, len(b), replace=True).mean()
             for _ in range(n_boot)]
    return np.percentile(diffs, 100*alpha/2), np.percentile(diffs, 100*(1-alpha/2))


def cohens_d(a, b):
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return (np.mean(a) - np.mean(b)) / (pooled + 1e-9)


def power_approx(a, b):
    from scipy import stats
    d = abs(cohens_d(a, b))
    n = (len(a) + len(b)) / 2
    ncp = d * np.sqrt(n / 2)
    df = len(a) + len(b) - 2
    t_crit = stats.t.ppf(0.975, df)
    pwr = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
    return float(np.clip(pwr, 0, 1))


def compare(name_a, a, name_b, b, env):
    a, b = np.array(a, float), np.array(b, float)
    u, p = mannwhitneyu(a, b, alternative='two-sided')
    ci_lo, ci_hi = bootstrap_ci_diff(a, b)
    d = cohens_d(a, b)
    pwr = power_approx(a, b)
    pct = (a.mean() - b.mean()) / (abs(b.mean()) + 1e-9) * 100
    return dict(
        env=env, a=name_a, b=name_b,
        mean_a=float(a.mean()), std_a=float(a.std(ddof=1)) if len(a)>1 else 0,
        mean_b=float(b.mean()), std_b=float(b.std(ddof=1)) if len(b)>1 else 0,
        n_a=len(a), n_b=len(b),
        pct_improvement=pct,
        p_value=float(p),
        cohens_d=float(d),
        ci_95=(float(ci_lo), float(ci_hi)),
        power=pwr,
        significant=bool(p < 0.05),
        direction="better" if pct > 0 else "worse",
        effect_label=("neg." if abs(d)<0.2 else "small" if abs(d)<0.5
                      else "medium" if abs(d)<0.8 else "large"),
    )


# ─────────────────────────────────────────────────────────────────────────────
def main():
    data = load_results()

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY  (mean ± std, last-5-eval)")
    print("="*80)
    print(f"{'Algorithm':<25} {'Hopper':>20} {'Walker2d':>20} {'HalfCheetah':>20}")
    print("─"*85)
    for algo in ALGOS:
        tag = "  ← OURS" if "HCGAE" in algo else ""
        row = f"{algo:<25}"
        for env in ENVS:
            seeds = data.get(env, {}).get(algo, [])
            if seeds:
                a = np.array(seeds)
                row += f"  {a.mean():>8.0f}±{a.std(ddof=1) if len(a)>1 else 0:>5.0f}(n={len(a)})"
            else:
                row += f"  {'---':>18}"
        print(row + tag)

    # ── Key comparisons ────────────────────────────────────────────────────
    comparisons = []
    for env in ENVS:
        envd = data.get(env, {})
        # HCGAE vs Optimal_PPO (the critical comparison — same base)
        for hcgae_name in ["Optimal_HCGAE", "Optimal_HCGAE_SCR"]:
            if "Optimal_PPO" in envd and hcgae_name in envd:
                comparisons.append(compare(hcgae_name, envd[hcgae_name],
                                           "Optimal_PPO", envd["Optimal_PPO"], env))
        # HCGAE vs Standard_PPO (reference)
        for hcgae_name in ["Optimal_HCGAE", "Optimal_HCGAE_SCR"]:
            if "Standard_PPO" in envd and hcgae_name in envd:
                comparisons.append(compare(hcgae_name, envd[hcgae_name],
                                           "Standard_PPO", envd["Standard_PPO"], env))
        # Optimal_PPO vs Standard_PPO (sanity check)
        if "Standard_PPO" in envd and "Optimal_PPO" in envd:
            comparisons.append(compare("Optimal_PPO", envd["Optimal_PPO"],
                                       "Standard_PPO", envd["Standard_PPO"], env))

    print("\n" + "="*80)
    print("PAIRWISE COMPARISONS")
    print("="*80)
    print(f"{'Env':<14} {'A vs B':<45} {'Δ%':>7} {'p':>7} {'sig':>5} {'d':>6} {'power':>7}")
    print("─"*90)
    for c in comparisons:
        sig_str = "***" if c['p_value'] < 0.001 else ("**" if c['p_value'] < 0.01
                  else ("*" if c['p_value'] < 0.05 else "ns"))
        print(f"{c['env']:<14} {c['a']:<22} vs {c['b']:<22} "
              f"{c['pct_improvement']:>+7.1f}% "
              f"{c['p_value']:>7.3f} "
              f"{sig_str:>5} "
              f"{c['cohens_d']:>+6.2f} "
              f"{c['power']:>7.2f}")

    # ── Save to JSON ────────────────────────────────────────────────────────
    out_path = BASE / "icml_stats_report.json"
    report = {
        "summary": {
            env: {
                algo: {
                    "seeds": data[env][algo],
                    "mean": float(np.mean(data[env][algo])),
                    "std": float(np.std(data[env][algo], ddof=1)) if len(data[env][algo])>1 else 0,
                    "n": len(data[env][algo]),
                }
                for algo in data.get(env, {})
            }
            for env in ENVS
        },
        "comparisons": comparisons,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {out_path}")

    # ── Interpretation ──────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    hcgae_vs_opt = [c for c in comparisons if c['b'] == 'Optimal_PPO']
    n_sig_better = sum(1 for c in hcgae_vs_opt if c['significant'] and c['pct_improvement'] > 0)
    n_sig_worse  = sum(1 for c in hcgae_vs_opt if c['significant'] and c['pct_improvement'] < 0)
    n_total = len(hcgae_vs_opt)
    print(f"  HCGAE vs Optimal_PPO: {n_total} comparisons")
    print(f"    Significantly better: {n_sig_better}")
    print(f"    Significantly worse:  {n_sig_worse}")
    print(f"    Not significant:      {n_total - n_sig_better - n_sig_worse}")
    for c in hcgae_vs_opt:
        sym = "✓" if (c['significant'] and c['pct_improvement']>0) else \
              "✗" if (c['significant'] and c['pct_improvement']<0) else "~"
        print(f"  {sym} {c['env']:<14} {c['a']:<25}: "
              f"{c['pct_improvement']:+.1f}%  p={c['p_value']:.3f}  "
              f"d={c['cohens_d']:+.2f}  [{c['effect_label']}]")


if __name__ == "__main__":
    main()

