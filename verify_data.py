#!/usr/bin/env python3
"""验证论文中的实验数据准确性"""
import json, os, numpy as np

def get_score(base, env, algo, seed):
    path = f"{base}/{env}/{algo}/{algo}_s{seed}.json"
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    r = d.get('eval_rewards', d.get('eval_returns', []))
    return float(np.mean(r[-5:])) if len(r) >= 5 else float(np.mean(r)) if r else None

def stats(base, env, algo):
    scores = [get_score(base, env, algo, s) for s in range(5)]
    scores = [s for s in scores if s is not None]
    if not scores:
        return None, None, scores
    return float(np.mean(scores)), float(np.std(scores)), scores

base = '/Users/joe-caozhi/newGAE_ppo/results/ICMLExperiment'

print("=== 主要结果（表1）核查 ===")
rows = [
    ('Optimal_PPO', 'Optimal PPO', {
        'Hopper-v4': (1598, 149), 'Walker2d-v4': (1596, 418),
        'HalfCheetah-v4': (1487, 61), 'Ant-v4': (793, 123)}),
    ('Optimal_HCGAE', 'HCGAE v1', {
        'Hopper-v4': (1752, 81), 'Walker2d-v4': (1872, 547),
        'HalfCheetah-v4': (1250, 53), 'Ant-v4': (562, 44)}),
    ('Optimal_HCGAE_v2', 'HCGAE v2', {
        'Hopper-v4': (1760, 380), 'Walker2d-v4': (1999, 785),
        'HalfCheetah-v4': (1550, 389), 'Ant-v4': (677, 201)}),
]

issues = []
for algo, label, paper_vals in rows:
    print(f"\n{label} ({algo}):")
    for env in ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4', 'Ant-v4']:
        m, s, sc = stats(base, env, algo)
        paper_m, paper_s = paper_vals.get(env, (None, None))
        if m is not None and paper_m is not None:
            diff = abs(m - paper_m)
            ok = diff < 10  # 允许10以内的误差（浮点rounding）
            print(f"  {env}: 实测={m:.0f}±{s:.0f}  论文={paper_m}±{paper_s}  {'OK' if ok else f'DIFF(diff={diff:.0f})'}")
            if not ok:
                issues.append(f"{label} {env}: 实测{m:.0f} vs 论文{paper_m}")
            print(f"    seeds: {[f'{x:.0f}' for x in sc]}")
        elif m is None:
            print(f"  {env}: 数据缺失")
        else:
            print(f"  {env}: 实测={m:.0f}±{s:.0f}  (论文值未提供)")

print("\n=== 核查摘要 ===")
if issues:
    print(f"发现 {len(issues)} 处差异：")
    for iss in issues:
        print(f"  - {iss}")
else:
    print("所有核查数据与论文一致（误差<10分）")

# 额外核查：Hopper-v4各种子
print("\n=== Hopper-v4 HCGAE v2 各种子详情（论文声称：s0=1241,s1=2275,s2=1603,s3=1889,s4=1794）===")
for s in range(5):
    score = get_score(base, 'Hopper-v4', 'Optimal_HCGAE_v2', s)
    print(f"  s{s}: {score:.0f}" if score else f"  s{s}: 缺失")

print("\n=== Walker2d-v4 HCGAE v2 各种子详情（论文声称：s0=955,s1=2760,s2=2363,s3=1383,s4=2532）===")
for s in range(5):
    score = get_score(base, 'Walker2d-v4', 'Optimal_HCGAE_v2', s)
    print(f"  s{s}: {score:.0f}" if score else f"  s{s}: 缺失")

print("\n=== HalfCheetah-v4 HCGAE v2 各种子详情（论文声称：s0=2136,s1=1347,s2=1324,s3=1589,s4=1356）===")
for s in range(5):
    score = get_score(base, 'HalfCheetah-v4', 'Optimal_HCGAE_v2', s)
    print(f"  s{s}: {score:.0f}" if score else f"  s{s}: 缺失")

print("\n=== Ant-v4 HCGAE v2 各种子详情（论文声称：s0=987,s1=693,s2=513,s3=484,s4=709）===")
for s in range(5):
    score = get_score(base, 'Ant-v4', 'Optimal_HCGAE_v2', s)
    print(f"  s{s}: {score:.0f}" if score else f"  s{s}: 缺失")

