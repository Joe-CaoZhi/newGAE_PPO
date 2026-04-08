#!/usr/bin/env python3
"""验证论文中所有关键数据与实验结果文件的一致性"""
import json
import os

import numpy as np


def load_final_perf(path):
    """从JSON文件加载最终性能（最后5次评估的均值）"""
    with open(path) as f:
        data = json.load(f)
    # 尝试各种可能的键名
    for key in ['eval_rewards', 'eval_returns', 'evaluations', 'eval_mean_rewards']:
        if key in data:
            evals = data[key]
            if evals:
                return float(np.mean(evals[-5:]))
    if 'final_reward' in data:
        return float(data['final_reward'])
    return None

def compute_stats(values):
    arr = np.array(values)
    return np.mean(arr), np.std(arr), np.std(arr)/np.sqrt(len(arr))

def mann_whitney_u(a, b):
    from scipy.stats import mannwhitneyu
    try:
        stat, p = mannwhitneyu(a, b, alternative='two-sided')
        return p
    except:
        return None

def cohens_d(a, b):
    a, b = np.array(a), np.array(b)
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    if pooled_std == 0:
        return 0
    return (np.mean(a) - np.mean(b)) / pooled_std

def load_dir_results(dir_path):
    vals = []
    if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
        return vals
    for fname in sorted(os.listdir(dir_path)):
        if fname.endswith('.json'):
            fpath = os.path.join(dir_path, fname)
            try:
                v = load_final_perf(fpath)
                if v is not None:
                    vals.append(v)
            except Exception as e:
                pass
    return vals

print("=" * 70)
print("论文数据验证报告")
print("=" * 70)

# ========== ICMLExperiment 验证 ==========
print("\n### ICMLExperiment 数据 (5 seeds, 500K steps, Optimal PPO base)")
icml_base = '/Users/joe-caozhi/newGAE_ppo/results/ICMLExperiment'
envs = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4', 'Ant-v4']

icml_data = {}
for env in envs:
    icml_data[env] = {}
    env_path = os.path.join(icml_base, env)
    if not os.path.exists(env_path):
        print(f"  {env}: 目录不存在")
        continue
    for method in sorted(os.listdir(env_path)):
        method_path = os.path.join(env_path, method)
        if not os.path.isdir(method_path):
            continue
        vals = load_dir_results(method_path)
        if vals:
            mean, std, sem = compute_stats(vals)
            icml_data[env][method] = {'vals': vals, 'mean': mean, 'std': std, 'sem': sem}

for env in envs:
    print(f"\n  {env}:")
    for method, d in sorted(icml_data[env].items()):
        seed_vals = [round(v, 1) for v in d['vals']]
        print(f"    {method}: n={len(d['vals'])}, mean={d['mean']:.1f}, std={d['std']:.1f}, sem={d['sem']:.1f}")
        print(f"      seeds={seed_vals}")

# 核心数字验证（中文论文中用std，英文论文中用SEM）
print("\n\n### 论文Table 1数字验证（按中文论文std口径）")
paper_claims_zh = {
    'Hopper-v4': {
        'Standard_PPO':   {'mean': 1804, 'std': 69,  'seeds': [None,None,None,None,None]},
        'Optimal_PPO':    {'mean': 1598, 'std': 149, 'seeds': [None,None,None,None,None]},
        'Optimal_HCGAE':  {'mean': 1752, 'std': 81,  'seeds': [None,None,None,None,None]},
        'Optimal_HCGAE_v2': {'mean': 1760, 'std': 380, 'seeds': [1241,2275,1603,1889,1794]},
    },
    'Walker2d-v4': {
        'Standard_PPO':   {'mean': 1425, 'std': 223, 'seeds': [None,None,None,None,None]},
        'Optimal_PPO':    {'mean': 1596, 'std': 418, 'seeds': [None,None,None,None,None]},
        'Optimal_HCGAE':  {'mean': 1872, 'std': 547, 'seeds': [None,None,None,None,None]},
        'Optimal_HCGAE_v2': {'mean': 1999, 'std': 785, 'seeds': [955,2760,2363,1383,2532]},
    },
    'HalfCheetah-v4': {
        'Standard_PPO':   {'mean': 1051, 'std': 134, 'seeds': [None,None,None,None,None]},
        'Optimal_PPO':    {'mean': 1487, 'std': 61,  'seeds': [None,None,None,None,None]},
        'Optimal_HCGAE':  {'mean': 1250, 'std': 53,  'seeds': [None,None,None,None,None]},
        'Optimal_HCGAE_v2': {'mean': 1550, 'std': 389, 'seeds': [2136,1347,1324,1589,1356]},
    },
    'Ant-v4': {
        'Standard_PPO':   {'mean': 747,  'std': 118, 'seeds': [None,None,None,None,None]},
        'Optimal_PPO':    {'mean': 793,  'std': 123, 'seeds': [None,None,None,None,None]},
        'Optimal_HCGAE':  {'mean': 562,  'std': 44,  'seeds': [None,None,None,None,None]},
        'Optimal_HCGAE_v2': {'mean': 677, 'std': 201, 'seeds': [987,693,513,484,709]},
    }
}

issues = []
for env in envs:
    print(f"\n  {env}:")
    for method, claim in paper_claims_zh.get(env, {}).items():
        if method not in icml_data.get(env, {}):
            print(f"    ? {method}: 数据目录不存在（{env}/{method}/）")
            continue
        actual = icml_data[env][method]
        mean_diff = abs(actual['mean'] - claim['mean'])
        std_diff = abs(actual['std'] - claim['std'])
        ok_mean = mean_diff < 5
        ok_std = std_diff < 30
        status = "✓" if ok_mean else "✗ MEAN"
        if not ok_std:
            status += "+STD"
        print(f"    {status} {method}:")
        print(f"      Paper: mean={claim['mean']}, std={claim['std']}")
        print(f"      Actual: mean={actual['mean']:.1f}, std={actual['std']:.1f}")
        if not ok_mean:
            issues.append(f"{env}/{method}: paper mean={claim['mean']}, actual={actual['mean']:.1f}")
        # 验证per-seed值
        if any(s is not None for s in claim['seeds']):
            paper_seeds = sorted([s for s in claim['seeds'] if s is not None])
            actual_seeds = sorted([round(v,0) for v in actual['vals']])
            print(f"      Paper seeds (sorted): {paper_seeds}")
            print(f"      Actual seeds (sorted): {actual_seeds}")

# ========== MultiSeedPower 验证 ==========
print("\n\n### MultiSeedPower 数据 (10 seeds, 300K steps, Standard PPO base)")
ms_base = '/Users/joe-caozhi/newGAE_ppo/results/MultiSeedPower'
ms_data = {}

for env in ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']:
    ms_data[env] = {}
    ms_data[env]['Standard_PPO'] = load_dir_results(os.path.join(ms_base, env, 'Standard_PPO'))
    ms_data[env]['HCGAE'] = load_dir_results(os.path.join(ms_base, env, 'HCGAE_Imp12'))
    ms_data[env]['HCGAE_SCR'] = load_dir_results(os.path.join(ms_base, env, 'HCGAE_Imp12_SCR'))

print()
for env in ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']:
    print(f"  {env}:")
    for method, vals in ms_data[env].items():
        if vals:
            mean, std, sem = compute_stats(vals)
            print(f"    {method}: n={len(vals)}, mean={mean:.1f}, std={std:.1f}, sem={sem:.1f}")
            print(f"      seeds=[{', '.join([f'{v:.1f}' for v in vals])}]")

paper_ms_claims = {
    'Hopper-v4': {
        'PPO':  {'mean': 2524, 'sem': 167},
        'HCGAE': {'mean': 2663, 'sem': 150},
        'SCR':   {'mean': 2834, 'sem': 155},
    },
    'Walker2d-v4': {
        'PPO':  {'mean': 1252, 'sem': 228},
        'HCGAE': {'mean': 1063, 'sem': 212},
        'SCR':   {'mean': 1516, 'sem': 298},
    },
    'HalfCheetah-v4': {
        'PPO':  {'mean': 950, 'sem': 56},
        'HCGAE': {'mean': 757, 'sem': 47},
        'SCR':   {'mean': 709, 'sem': 59},
    }
}

print("\n  统计检验（Mann-Whitney U, n=10）:")
ms_stat_claims = {
    'Hopper-v4': {
        'HCGAE_vs_PPO': {'delta_pct': 5.5,   'p': 0.571, 'd': +0.28},
        'SCR_vs_PPO':   {'delta_pct': 12.3,  'p': 0.241, 'd': +0.61},
    },
    'Walker2d-v4': {
        'HCGAE_vs_PPO': {'delta_pct': -15.1, 'p': 0.427, 'd': -0.27},
        'SCR_vs_PPO':   {'delta_pct': 21.1,  'p': 0.970, 'd': +0.31},
    },
    'HalfCheetah-v4': {
        'HCGAE_vs_PPO': {'delta_pct': -20.3, 'p': 0.026, 'd': -1.17},
        'SCR_vs_PPO':   {'delta_pct': -25.3, 'p': 0.011, 'd': -1.32},
    }
}

for env in ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']:
    print(f"\n  {env}:")
    ppo   = ms_data[env]['Standard_PPO']
    hcgae = ms_data[env]['HCGAE']
    scr   = ms_data[env]['HCGAE_SCR']
    if ppo:
        mean_ppo, _, sem_ppo = compute_stats(ppo)
        claim_ppo = paper_ms_claims[env]['PPO']
        ok = abs(mean_ppo - claim_ppo['mean']) < 10
        print(f"    {'✓' if ok else '✗'} PPO: actual={mean_ppo:.1f}±{sem_ppo:.1f}, paper={claim_ppo['mean']}±{claim_ppo['sem']}")
    if ppo and hcgae:
        p_val = mann_whitney_u(hcgae, ppo)
        d_val = cohens_d(hcgae, ppo)
        delta = (np.mean(hcgae) - np.mean(ppo)) / np.mean(ppo) * 100
        claim = ms_stat_claims[env]['HCGAE_vs_PPO']
        ok = abs(delta - claim['delta_pct']) < 3
        print(f"    {'✓' if ok else '✗'} HCGAE vs PPO: Δ={delta:.1f}%(paper={claim['delta_pct']}%), p={p_val:.3f}(paper={claim['p']}), d={d_val:.2f}(paper={claim['d']})")
    if ppo and scr:
        p_val = mann_whitney_u(scr, ppo)
        d_val = cohens_d(scr, ppo)
        delta = (np.mean(scr) - np.mean(ppo)) / np.mean(ppo) * 100
        claim = ms_stat_claims[env]['SCR_vs_PPO']
        ok = abs(delta - claim['delta_pct']) < 3
        print(f"    {'✓' if ok else '✗'} SCR vs PPO:   Δ={delta:.1f}%(paper={claim['delta_pct']}%), p={p_val:.3f}(paper={claim['p']}), d={d_val:.2f}(paper={claim['d']})")

# ========== 英文/中文论文关键数字不一致性 ==========
print("\n\n### 英文论文 vs 中文论文数字不一致检查")
print("  (英文表用SEM, 中文表用std -- 需统一)")
en_table1 = {
    'Hopper-v4':     {'Standard_PPO': (1804, 31),  'Optimal_PPO': (1598, 67),  'HCGAE_v2': (1760, 170)},
    'Walker2d-v4':   {'Standard_PPO': (1425, 100), 'Optimal_PPO': (1596, 187), 'HCGAE_v2': (1998, 351)},
    'HalfCheetah-v4':{'Standard_PPO': (1051, 60),  'Optimal_PPO': (1487, 27),  'HCGAE_v2': (1550, 174)},
}
zh_table1 = {
    'Hopper-v4':     {'Standard_PPO': (1804, 69),  'Optimal_PPO': (1598, 149), 'HCGAE_v2': (1760, 380)},
    'Walker2d-v4':   {'Standard_PPO': (1425, 223), 'Optimal_PPO': (1596, 418), 'HCGAE_v2': (1999, 785)},
    'HalfCheetah-v4':{'Standard_PPO': (1051, 134), 'Optimal_PPO': (1487, 61),  'HCGAE_v2': (1550, 389)},
}
for env in ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']:
    for method in ['Standard_PPO', 'Optimal_PPO', 'HCGAE_v2']:
        en = en_table1[env][method]
        zh = zh_table1[env][method]
        mean_match = en[0] == zh[0]
        status = "✓" if mean_match else "✗ MEAN"
        # SEM vs std - std应该是SEM*sqrt(n), n=5
        # EN is SEM, ZH is std, check ratio
        if en[1] > 0:
            ratio = zh[1] / en[1]
            expected_ratio = np.sqrt(5)  # sqrt(n=5)
            ratio_ok = abs(ratio - expected_ratio) < 0.5
            ratio_status = f"ratio={ratio:.2f}(expect √5={expected_ratio:.2f})"
        else:
            ratio_status = ""
        print(f"  {status} {env}/{method}: EN_SEM={en[1]}, ZH_std={zh[1]}, {ratio_status}")

# 实际数据计算SEM和std比较
print("\n  实际数据验证（从文件计算）:")
for env in ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']:
    for method_key, method_dir in [('Standard_PPO','Standard_PPO'), ('Optimal_PPO','Optimal_PPO'), ('HCGAE_v2','Optimal_HCGAE_v2')]:
        if method_dir in icml_data.get(env, {}):
            d = icml_data[env][method_dir]
            print(f"  {env}/{method_key}: mean={d['mean']:.1f}, std={d['std']:.1f}, sem={d['sem']:.1f} (n={len(d['vals'])})")

print("\n\n=== 主要问题汇总 ===")
if issues:
    for issue in issues:
        print(f"  ✗ {issue}")
else:
    print("  均值数据无重大偏差")

print("\n验证完成！")

