#!/usr/bin/env python3
"""
Full statistical analysis of ICMLExperiment data.
Computes: final performance, AULC, steps-to-threshold, Mann-Whitney U tests,
Cohen's d effect sizes, and generates a comprehensive report.
"""

import glob
import json
import os

import numpy as np
from scipy import stats

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'results', 'ICMLExperiment')
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'results', 'ICMLExperiment')

ENVS = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4', 'Ant-v4']
ALGOS = [
    'Standard_PPO',
    'Optimal_PPO',
    'Optimal_HCGAE',
    'Optimal_HCGAE_v2',
    'Optimal_HCGAE_SCR',
    'Optimal_HCGAE_v2_NoBdry',
    'Optimal_HCGAE_v2_NoGate',
]

ALGO_DISPLAY = {
    'Standard_PPO':           'Standard PPO',
    'Optimal_PPO':            'Optimal PPO',
    'Optimal_HCGAE':          'HCGAE v1',
    'Optimal_HCGAE_v2':       'HCGAE v2',
    'Optimal_HCGAE_SCR':      'HCGAE v1+SCR',
    'Optimal_HCGAE_v2_NoBdry':'HCGAE v2 -Bdry',
    'Optimal_HCGAE_v2_NoGate':'HCGAE v2 -Gate',
}


def load_env_algo(env, algo):
    """Load all seed data for (env, algo). Returns list of dicts."""
    pattern = os.path.join(BASE, env, algo, '*.json')
    files = sorted(glob.glob(pattern))
    results = []
    for f in files:
        with open(f) as fp:
            results.append(json.load(fp))
    return results


def get_curves(data_list):
    """Return (steps, matrix) where matrix[i] = eval curve for seed i."""
    curves = [d['eval_rewards'] for d in data_list]
    steps  = data_list[0]['eval_steps']
    min_len = min(len(c) for c in curves)
    steps = steps[:min_len]
    curves = np.array([c[:min_len] for c in curves])  # (n_seeds, T)
    return steps, curves


def compute_aulc(steps, curves):
    """Area under learning curve (normalized by total steps)."""
    steps_arr = np.array(steps, dtype=float)
    total_area = steps_arr[-1] - steps_arr[0]
    # trapezoid integration - ensure steps and curve have same length
    aulcs = []
    for c in curves:
        min_len = min(len(steps_arr), len(c))
        aulcs.append(np.trapz(c[:min_len], steps_arr[:min_len]) / total_area)
    return np.array(aulcs)


def compute_final_perf(curves, last_n=5):
    """Mean of last_n checkpoints per seed."""
    return curves[:, -last_n:].mean(axis=1)


def compute_steps_to_threshold(steps, curves, threshold):
    """Steps needed to first exceed threshold (NaN if never reached)."""
    result = []
    for c in curves:
        idx = np.where(c >= threshold)[0]
        if len(idx) == 0:
            result.append(np.nan)
        else:
            result.append(steps[idx[0]])
    return np.array(result)


def mann_whitney(a, b):
    """Mann-Whitney U test, returns (U, p_value, effect_size_r)."""
    u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    n1, n2 = len(a), len(b)
    # effect size r = U / (n1*n2)  (normalized)
    r = u / (n1 * n2)
    return u, p, r


def cohens_d(a, b):
    """Cohen's d effect size."""
    na, nb = len(a), len(b)
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled_std = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std


def analyze():
    # ------------------------------------------------------------------
    # 1. Load all data
    # ------------------------------------------------------------------
    data = {}
    for env in ENVS:
        data[env] = {}
        for algo in ALGOS:
            seeds = load_env_algo(env, algo)
            if seeds:
                data[env][algo] = seeds

    # ------------------------------------------------------------------
    # 2. Per-env-algo statistics
    # ------------------------------------------------------------------
    stats_table = {}
    for env in ENVS:
        stats_table[env] = {}
        for algo in ALGOS:
            if algo not in data[env]:
                continue
            steps, curves = get_curves(data[env][algo])
            final   = compute_final_perf(curves)
            aulcs   = compute_aulc(steps, curves)
            stats_table[env][algo] = {
                'n_seeds':    len(curves),
                'final_mean': float(np.mean(final)),
                'final_std':  float(np.std(final, ddof=1)),
                'final_all':  final.tolist(),
                'aulc_mean':  float(np.mean(aulcs)),
                'aulc_std':   float(np.std(aulcs, ddof=1)),
                'aulc_all':   aulcs.tolist(),
                'total_steps': steps[-1],
            }

    # ------------------------------------------------------------------
    # 3. Pairwise significance tests (vs Optimal_PPO baseline)
    # ------------------------------------------------------------------
    sig_tests = {}
    for env in ENVS:
        sig_tests[env] = {}
        if 'Optimal_PPO' not in stats_table[env]:
            continue
        baseline_final = np.array(stats_table[env]['Optimal_PPO']['final_all'])
        baseline_aulc  = np.array(stats_table[env]['Optimal_PPO']['aulc_all'])
        for algo in ALGOS:
            if algo == 'Optimal_PPO' or algo not in stats_table[env]:
                continue
            algo_final = np.array(stats_table[env][algo]['final_all'])
            algo_aulc  = np.array(stats_table[env][algo]['aulc_all'])
            u_f, p_f, r_f = mann_whitney(algo_final, baseline_final)
            d_f = cohens_d(algo_final, baseline_final)
            u_a, p_a, r_a = mann_whitney(algo_aulc, baseline_aulc)
            d_a = cohens_d(algo_aulc, baseline_aulc)
            sig_tests[env][algo] = {
                'final_p':   float(p_f),
                'final_d':   float(d_f),
                'final_sig': bool(p_f < 0.05),
                'aulc_p':    float(p_a),
                'aulc_d':    float(d_a),
                'aulc_sig':  bool(p_a < 0.05),
            }

    # ------------------------------------------------------------------
    # 4. Print summary
    # ------------------------------------------------------------------
    print("=" * 90)
    print("ICML EXPERIMENT FULL STATISTICAL REPORT")
    print("=" * 90)

    for env in ENVS:
        print(f"\n{'─'*80}")
        print(f"Environment: {env}")
        print(f"{'─'*80}")
        print(f"{'Algorithm':<28} {'Final Mean':>12} {'Final Std':>10} {'AULC Mean':>12} {'AULC Std':>10}")
        print(f"{'─'*74}")
        for algo in ALGOS:
            if algo not in stats_table[env]:
                continue
            s = stats_table[env][algo]
            print(f"{ALGO_DISPLAY[algo]:<28} {s['final_mean']:>12.1f} {s['final_std']:>10.1f} "
                  f"{s['aulc_mean']:>12.1f} {s['aulc_std']:>10.1f}")
        print()
        print("Significance vs Optimal PPO:")
        print(f"{'Algorithm':<28} {'Δ Final':>10} {'p-val(F)':>10} {'d(F)':>8} {'p-val(A)':>10} {'d(A)':>8} {'sig':>5}")
        print(f"{'─'*84}")
        for algo in ALGOS:
            if algo not in sig_tests[env]:
                continue
            s_test = sig_tests[env][algo]
            s_algo = stats_table[env][algo]
            s_base = stats_table[env]['Optimal_PPO']
            delta = s_algo['final_mean'] - s_base['final_mean']
            sig_str = "✓" if s_test['final_sig'] else ""
            print(f"{ALGO_DISPLAY[algo]:<28} {delta:>+10.1f} {s_test['final_p']:>10.4f} "
                  f"{s_test['final_d']:>8.3f} {s_test['aulc_p']:>10.4f} "
                  f"{s_test['aulc_d']:>8.3f} {sig_str:>5}")

    # ------------------------------------------------------------------
    # 5. Cross-environment summary table (for paper Table 1)
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 90)
    print("PAPER TABLE 1: FINAL PERFORMANCE (Mean ± Std, 5 seeds)")
    print("=" * 90)
    header = f"{'Algorithm':<22}" + "".join(f" {e:>22}" for e in ENVS)
    print(header)
    print("─" * (22 + 23 * len(ENVS)))
    for algo in ALGOS:
        row = f"{ALGO_DISPLAY[algo]:<22}"
        for env in ENVS:
            if algo in stats_table[env]:
                s = stats_table[env][algo]
                row += f" {s['final_mean']:>9.1f}±{s['final_std']:<9.1f}"
            else:
                row += f" {'N/A':>22}"
        print(row)

    # ------------------------------------------------------------------
    # 6. AULC table (for paper Table 2)
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 90)
    print("PAPER TABLE 2: AREA UNDER LEARNING CURVE (Mean ± Std)")
    print("=" * 90)
    print(header)
    print("─" * (22 + 23 * len(ENVS)))
    for algo in ALGOS:
        row = f"{ALGO_DISPLAY[algo]:<22}"
        for env in ENVS:
            if algo in stats_table[env]:
                s = stats_table[env][algo]
                row += f" {s['aulc_mean']:>9.1f}±{s['aulc_std']:<9.1f}"
            else:
                row += f" {'N/A':>22}"
        print(row)

    # ------------------------------------------------------------------
    # 7. Save full report as JSON
    # ------------------------------------------------------------------
    report = {
        'stats_table': stats_table,
        'sig_tests':   sig_tests,
    }
    out_path = os.path.join(OUT_DIR, 'full_analysis_report.json')
    with open(out_path, 'w') as fp:
        json.dump(report, fp, indent=2)
    print(f"\n\nFull report saved to: {out_path}")

    return report


if __name__ == '__main__':
    report = analyze()

