#!/usr/bin/env python3
"""
全面数据溯源验证脚本
验证论文中所有引用数据的真实性和准确性
"""
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

BASE = Path('/Users/joe-caozhi/newGAE_ppo')
RESULTS = BASE / 'results'

PASS = '✓'
FAIL = '✗'
WARN = '⚠'

def load_icml_seeds(env, algo):
    """从 ICMLExperiment 目录加载种子数据"""
    d = RESULTS / 'ICMLExperiment' / env / algo
    vals = []
    for fp in sorted(d.glob('*.json')) if d.exists() else []:
        data = json.load(open(fp))
        er = data.get('eval_rewards', [])
        if er:
            vals.append(float(np.mean(er[-5:])))
    return vals

def load_baseline_comparison(env, algo):
    """从 BaselineComparison 目录加载数据"""
    pattern = str(RESULTS / 'BaselineComparison' / env / algo / f'{algo}_s*_metrics.json')
    vals = []
    for fp in sorted(glob.glob(pattern)):
        data = json.load(open(fp))
        er = data.get('eval_rewards', [])
        if er:
            vals.append(float(np.mean(er[-5:])))
    return vals

def load_unified_comparison(env, algo):
    """从 UnifiedComparison 目录加载数据"""
    pattern = str(RESULTS / 'UnifiedComparison' / env / algo / f'{algo}_s*_metrics.json')
    vals = []
    for fp in sorted(glob.glob(pattern)):
        data = json.load(open(fp))
        er = data.get('eval_rewards', [])
        if er:
            vals.append(float(np.mean(er[-5:])))
    return vals

def load_ablation_multiseed(algo):
    """从 Hopper-v4-Ablation-MultiSeed 目录加载数据"""
    d = RESULTS / 'Hopper-v4-Ablation-MultiSeed'
    vals = []
    for fp in sorted(d.glob(f'{algo}_s*.json')) if d.exists() else []:
        data = json.load(open(fp))
        er = data.get('eval_rewards', [])
        if er:
            vals.append(float(np.mean(er[-5:])))
    return vals

def load_dcppo_multiseed(env, algo):
    """从 MultiEnv_DCPPO 目录加载数据"""
    pattern = str(RESULTS / 'MultiEnv_DCPPO' / env / algo / f'{algo}_s*_metrics.json')
    vals = []
    for fp in sorted(glob.glob(pattern)):
        data = json.load(open(fp))
        er = data.get('eval_rewards', [])
        if er:
            vals.append(float(np.mean(er[-5:])))
    return vals

def load_multiseed_power(env, algo):
    """从 MultiSeedPower 目录加载数据"""
    pattern = str(RESULTS / 'MultiSeedPower' / env / algo / f'{algo}_s*_metrics.json')
    vals = []
    for fp in sorted(glob.glob(pattern)):
        data = json.load(open(fp))
        er = data.get('eval_rewards', [])
        if er:
            vals.append(float(np.mean(er[-5:])))
    return vals

def load_sensitivity(param_type):
    """从 Sensitivity 目录加载数据"""
    pattern = str(RESULTS / 'Sensitivity' / f'*{param_type}*.json')
    files = sorted(glob.glob(pattern))
    return files

def check(condition, msg):
    icon = PASS if condition else FAIL
    print(f'  {icon} {msg}')
    return condition

def compare_stat(vals_a, vals_b, name_a, name_b):
    if len(vals_a) < 2 or len(vals_b) < 2:
        return None
    a, b = np.array(vals_a), np.array(vals_b)
    _, p = mannwhitneyu(a, b, alternative='two-sided')
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    d = (a.mean() - b.mean()) / (pooled + 1e-9)
    pct = (a.mean() - b.mean()) / (abs(b.mean()) + 1e-9) * 100
    return {'p': float(p), 'd': float(d), 'pct': float(pct),
            'mean_a': float(a.mean()), 'mean_b': float(b.mean()),
            'std_a': float(a.std(ddof=1)), 'std_b': float(b.std(ddof=1))}

errors = []

print('=' * 80)
print('数据溯源验证报告')
print('=' * 80)

# ─────────────────────────────────────────────────────────────────────────────
print('\n【1】TABLE 1: ICMLExperiment (3 envs × 4 algos × 5 seeds × 500K steps)')
print('─' * 70)
# Paper claims:
# Standard PPO: Hopper 1804±69, Walker 1425±223, HC 1051±134
# Optimal PPO:  Hopper 1598±149, Walker 1596±417, HC 1487±61
# Opt HCGAE:    Hopper 1752±81,  Walker 1872±547, HC 1250±53
# Opt HCGAE-SCR: Hopper 1366±133, Walker 1896±682, HC 1254±78
paper_table1 = {
    'Hopper-v4': {
        'Standard_PPO': (1804, 69),
        'Optimal_PPO': (1598, 149),
        'Optimal_HCGAE': (1752, 81),
        'Optimal_HCGAE_SCR': (1366, 133),
    },
    'Walker2d-v4': {
        'Standard_PPO': (1425, 223),
        'Optimal_PPO': (1596, 417),
        'Optimal_HCGAE': (1872, 547),
        'Optimal_HCGAE_SCR': (1896, 682),
    },
    'HalfCheetah-v4': {
        'Standard_PPO': (1051, 134),
        'Optimal_PPO': (1487, 61),
        'Optimal_HCGAE': (1250, 53),
        'Optimal_HCGAE_SCR': (1254, 78),
    },
}

table1_data = {}
for env in ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']:
    table1_data[env] = {}
    for algo in ['Standard_PPO', 'Optimal_PPO', 'Optimal_HCGAE', 'Optimal_HCGAE_SCR']:
        vals = load_icml_seeds(env, algo)
        table1_data[env][algo] = vals
        paper_mean, paper_std = paper_table1[env][algo]
        if not vals:
            print(f'  {FAIL} MISSING: {env}/{algo} (0 files)')
            errors.append(f'Table1 MISSING: {env}/{algo}')
            continue
        actual_mean = round(float(np.mean(vals)))
        actual_std = round(float(np.std(vals, ddof=1))) if len(vals) > 1 else 0
        mean_ok = abs(actual_mean - paper_mean) <= 5
        std_ok = abs(actual_std - paper_std) <= 5
        n_ok = len(vals) >= 5
        icon = PASS if (mean_ok and std_ok and n_ok) else WARN
        print(f'  {icon} {env}/{algo}: paper={paper_mean}±{paper_std}  actual={actual_mean}±{actual_std}  n={len(vals)}  seeds={[round(v) for v in vals]}')
        if not mean_ok:
            errors.append(f'Table1 MISMATCH mean: {env}/{algo} paper={paper_mean} actual={actual_mean}')
        if not std_ok:
            errors.append(f'Table1 MISMATCH std: {env}/{algo} paper={paper_std} actual={actual_std}')

# ─────────────────────────────────────────────────────────────────────────────
print('\n【2】STATISTICAL CLAIMS (Key comparisons)')
print('─' * 70)
# Paper claims in Key Statistical Comparisons table
stat_claims = [
    ('Hopper-v4', 'Optimal_HCGAE', 'Optimal_PPO', +9.6, 0.222, +1.28),
    ('Walker2d-v4', 'Optimal_HCGAE', 'Optimal_PPO', +17.3, 0.841, +0.57),
    ('HalfCheetah-v4', 'Optimal_HCGAE', 'Optimal_PPO', -16.0, 0.008, -4.14),
    ('Hopper-v4', 'Optimal_HCGAE', 'Standard_PPO', -2.9, 0.421, -0.69),
    ('Walker2d-v4', 'Optimal_HCGAE', 'Standard_PPO', +31.4, 0.151, +1.07),
    ('HalfCheetah-v4', 'Optimal_HCGAE', 'Standard_PPO', +18.9, 0.008, +1.95),
    ('Hopper-v4', 'Optimal_PPO', 'Standard_PPO', -11.4, 0.032, -1.78),
    ('Walker2d-v4', 'Optimal_PPO', 'Standard_PPO', +12.0, 0.548, +0.51),
    ('HalfCheetah-v4', 'Optimal_PPO', 'Standard_PPO', +41.5, 0.008, +4.18),
]

for env, algo_a, algo_b, paper_pct, paper_p, paper_d in stat_claims:
    vals_a = table1_data.get(env, {}).get(algo_a, [])
    vals_b = table1_data.get(env, {}).get(algo_b, [])
    if len(vals_a) < 2 or len(vals_b) < 2:
        print(f'  {FAIL} {env}: {algo_a} vs {algo_b} - INSUFFICIENT DATA')
        continue
    stat = compare_stat(vals_a, vals_b, algo_a, algo_b)
    pct_ok = abs(stat['pct'] - paper_pct) < 1.0
    p_ok = abs(stat['p'] - paper_p) < 0.05
    d_ok = abs(stat['d'] - paper_d) < 0.20
    icon = PASS if (pct_ok and p_ok and d_ok) else WARN
    print(f'  {icon} {env}: {algo_a} vs {algo_b}')
    print(f'       paper: Δ%={paper_pct:+.1f}  p={paper_p:.3f}  d={paper_d:+.2f}')
    print(f'       actual: Δ%={stat["pct"]:+.1f}  p={stat["p"]:.3f}  d={stat["d"]:+.2f}')
    if not pct_ok:
        errors.append(f'Stat MISMATCH pct: {env} {algo_a} vs {algo_b}: paper={paper_pct} actual={stat["pct"]:.1f}')

# ─────────────────────────────────────────────────────────────────────────────
print('\n【3】TABLE 3: HCGAE Ablation (Hopper-v4, 5 seeds, 300K steps)')
print('─' * 70)
# Paper claims:
# HCGAE_Base: 2653±627, Imp1-only: 2406±787, Imp2-only: 2425±615, Imp12: 2839±543
paper_ablation = {
    'HCGAE_Base': (2653, 627),
    'HCGAE_Imp1': (2406, 787),
    'HCGAE_Imp2': (2425, 615),
    'HCGAE_Imp12': (2839, 543),
}
ablation_data = {}
for algo, (paper_mean, paper_std) in paper_ablation.items():
    vals = load_ablation_multiseed(algo)
    ablation_data[algo] = vals
    if not vals:
        print(f'  {FAIL} MISSING: {algo} (0 files in Hopper-v4-Ablation-MultiSeed)')
        errors.append(f'Table3 MISSING: {algo}')
        continue
    actual_mean = round(float(np.mean(vals)))
    actual_std = round(float(np.std(vals, ddof=1))) if len(vals) > 1 else 0
    mean_ok = abs(actual_mean - paper_mean) <= 50
    icon = PASS if (mean_ok and len(vals) >= 5) else WARN
    print(f'  {icon} {algo}: paper={paper_mean}±{paper_std}  actual={actual_mean}±{actual_std}  n={len(vals)}  seeds={[round(v) for v in vals]}')
    if not mean_ok:
        errors.append(f'Table3 MISMATCH: {algo} paper={paper_mean} actual={actual_mean}')

# ─────────────────────────────────────────────────────────────────────────────
print('\n【4】TABLE 2/4: DCPPO MultiEnv (Hopper, Walker, 5 seeds, 500K steps)')
print('─' * 70)
# Paper claims:
# DCPPO_Base: Hopper 2958±397, Walker 1895±632
# DCPPO_ImpS: Hopper 3056±420, Walker 1895±632
# DCPPO_Full: Hopper 1192±461, Walker 610±205
paper_dcppo = {
    'Hopper-v4': {'DCPPO_Base': (2958, 397), 'DCPPO_ImpS': (3056, 420), 'DCPPO_Full': (1192, 461)},
    'Walker2d-v4': {'DCPPO_Base': (1895, 632), 'DCPPO_ImpS': (1895, 632), 'DCPPO_Full': (610, 205)},
}
dcppo_data = {}
for env, algos in paper_dcppo.items():
    dcppo_data[env] = {}
    for algo, (paper_mean, paper_std) in algos.items():
        vals = load_dcppo_multiseed(env, algo)
        dcppo_data[env][algo] = vals
        if not vals:
            # Also try without _metrics suffix
            pattern2 = str(RESULTS / 'MultiEnv_DCPPO' / env / algo / f'{algo}_s*.json')
            files2 = sorted(glob.glob(pattern2))
            for fp in files2:
                if not fp.endswith('_metrics.json'):
                    try:
                        data = json.load(open(fp))
                        er = data.get('eval_rewards', [])
                        if er:
                            vals.append(float(np.mean(er[-5:])))
                    except:
                        pass
        if not vals:
            print(f'  {FAIL} MISSING: {env}/{algo}')
            errors.append(f'Table2/4 MISSING: {env}/{algo}')
            continue
        actual_mean = round(float(np.mean(vals)))
        actual_std = round(float(np.std(vals, ddof=1))) if len(vals) > 1 else 0
        mean_ok = abs(actual_mean - paper_mean) <= 100
        icon = PASS if (mean_ok and len(vals) >= 5) else WARN
        print(f'  {icon} {env}/{algo}: paper={paper_mean}±{paper_std}  actual={actual_mean}±{actual_std}  n={len(vals)}')
        if not mean_ok:
            errors.append(f'Table2/4 MISMATCH: {env}/{algo} paper={paper_mean} actual={actual_mean}')

# ─────────────────────────────────────────────────────────────────────────────
print('\n【5】TABLE 6: MultiSeedPower n=10 (300K steps, 3 envs)')
print('─' * 70)
# Paper claims for n=10:
# Hopper: Std=2524, HCGAE=2663, SCR=2834
# Walker: Std=1252, HCGAE=1063, SCR=1516
# HC: Std=950, HCGAE=757, SCR=709
paper_multiseed = {
    'Hopper-v4': {'Standard_PPO': (2524, 167), 'HCGAE_Imp12': (2663, 150), 'HCGAE_Imp12_SCR': (2834, 155)},
    'Walker2d-v4': {'Standard_PPO': (1252, 228), 'HCGAE_Imp12': (1063, 212), 'HCGAE_Imp12_SCR': (1516, 298)},
    'HalfCheetah-v4': {'Standard_PPO': (950, 56), 'HCGAE_Imp12': (757, 47), 'HCGAE_Imp12_SCR': (709, 59)},
}
for env, algos in paper_multiseed.items():
    for algo, (paper_mean, paper_sem) in algos.items():
        vals = load_multiseed_power(env, algo)
        if not vals:
            # Try summary JSON
            summary_path = RESULTS / 'MultiSeedPower' / 'multiseed_summary_n10.json'
            if summary_path.exists():
                summary = json.load(open(summary_path))
                seeds = summary.get(env, {}).get(algo, {}).get('seeds', [])
                if seeds:
                    vals = [float(s) for s in seeds]
        if not vals:
            print(f'  {FAIL} MISSING: {env}/{algo} (MultiSeedPower)')
            errors.append(f'Table6 MISSING: {env}/{algo}')
            continue
        actual_mean = round(float(np.mean(vals)))
        # SEM not std
        actual_sem = round(float(np.std(vals, ddof=1) / np.sqrt(len(vals))))
        mean_ok = abs(actual_mean - paper_mean) <= 100
        icon = PASS if (mean_ok and len(vals) >= 8) else WARN
        print(f'  {icon} {env}/{algo}: paper={paper_mean}±{paper_sem}(SEM)  actual={actual_mean}±{actual_sem}(SEM)  n={len(vals)}  seeds=[{", ".join(str(round(v)) for v in vals[:5])}{"..." if len(vals)>5 else ""}]')
        if not mean_ok:
            errors.append(f'Table6 MISMATCH: {env}/{algo} paper={paper_mean} actual={actual_mean}')

# ─────────────────────────────────────────────────────────────────────────────
print('\n【6】TABLE 5: Computational Overhead')
print('─' * 70)
overhead_path = RESULTS / 'overhead_measurement.json'
if overhead_path.exists():
    oh = json.load(open(overhead_path))
    print(f'  {PASS} overhead_measurement.json exists')
    print(f'       Contents: {list(oh.keys())[:5]}')
else:
    print(f'  {FAIL} MISSING: results/overhead_measurement.json')
    errors.append('Table5 MISSING: overhead_measurement.json')

# ─────────────────────────────────────────────────────────────────────────────
print('\n【7】SENSITIVITY TABLES (B1/B2/B3, S1/S2/S3)')
print('─' * 70)
sensitivity_dir = RESULTS / 'Sensitivity'
if sensitivity_dir.exists():
    sens_files = list(sensitivity_dir.glob('*.json'))
    print(f'  {PASS if sens_files else FAIL} Sensitivity dir has {len(sens_files)} JSON files')
    if sens_files:
        print(f'       Files: {[f.name for f in sens_files[:8]]}')
else:
    print(f'  {FAIL} MISSING: results/Sensitivity/ directory')
    errors.append('Sensitivity MISSING: directory')

# ─────────────────────────────────────────────────────────────────────────────
print('\n【8】Consistency: OLD BaselineComparison data vs Paper Abstract')
print('─' * 70)
# Paper Abstract (pre-update) mentioned 2873, 1290, 828 for HCGAE — these are from BaselineComparison
for env in ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']:
    for algo in ['Standard_PPO', 'HCGAE_Imp12']:
        vals = load_baseline_comparison(env, algo)
        if vals:
            print(f'  {PASS} BaselineComparison/{env}/{algo}: n={len(vals)} mean={round(np.mean(vals))} std={round(np.std(vals,ddof=1)) if len(vals)>1 else 0}  seeds={[round(v) for v in vals]}')
        else:
            print(f'  {WARN} BaselineComparison/{env}/{algo}: no data')

# ─────────────────────────────────────────────────────────────────────────────
print('\n【9】Old data in HalfCheetah-v4 Mann-Whitney table (lines 225-234 of paper)')
print('─' * 70)
# Lines 223-234 are OLD HalfCheetah Mann-Whitney from BaselineComparison!
# They reference: HCGAE=828, Standard=902, PPO-KLPEN, Anneal, EntDecay, VClip, Full
# These numbers are from OLD BaselineComparison, NOT from ICMLExperiment
print('  PROBLEM: Lines 223-234 cite old BaselineComparison data (828 vs 902)')
print('  But new ICMLExperiment shows: HCGAE=1250, Standard=1051, Optimal=1487')
print('  These OLD numbers are INCONSISTENT with Section 4.2 Table 1')
errors.append('INCONSISTENCY: HalfCheetah MW table (lines 223-234) uses OLD BaselineComparison data (HCGAE=828, Std=902) inconsistent with Table 1 (HCGAE=1250, Std=1051)')

# ─────────────────────────────────────────────────────────────────────────────
print('\n【10】Check MultiSeedPower summary JSON')
print('─' * 70)
mseed_path = RESULTS / 'MultiSeedPower' / 'multiseed_summary_n10.json'
mseed_report_path = RESULTS / 'MultiSeedPower' / 'final_statistical_report_n10.json'
if mseed_path.exists():
    mseed = json.load(open(mseed_path))
    print(f'  {PASS} multiseed_summary_n10.json exists')
    for env in ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']:
        env_data = mseed.get(env, {})
        for algo in ['Standard_PPO', 'HCGAE_Imp12', 'HCGAE_Imp12_SCR']:
            seeds = env_data.get(algo, {}).get('seeds', [])
            print(f'    {env}/{algo}: n={len(seeds)} seeds={[round(s) for s in seeds[:5]]}{"..." if len(seeds)>5 else ""}')
else:
    print(f'  {FAIL} MISSING: multiseed_summary_n10.json')
    errors.append('Table6 MISSING: multiseed_summary_n10.json')

if mseed_report_path.exists():
    print(f'  {PASS} final_statistical_report_n10.json exists')
else:
    print(f'  {WARN} MISSING: final_statistical_report_n10.json')

# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 80)
print('SUMMARY OF ERRORS/INCONSISTENCIES:')
print('=' * 80)
if errors:
    for i, e in enumerate(errors, 1):
        print(f'  [{i}] {e}')
else:
    print('  No errors found!')
print()
print(f'Total issues: {len(errors)}')

