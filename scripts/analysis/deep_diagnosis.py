#!/usr/bin/env python3
"""深度诊断：找出新旧实验差距的根本原因"""
import json
import numpy as np
from pathlib import Path


def load_full(exp_dir, env, algo, seeds=range(5)):
    results = []
    for s in seeds:
        p = Path(f'results/{exp_dir}/{env}/{algo}/{algo}_s{s}.json')
        if p.exists():
            d = json.load(open(p))
            results.append(d)
    return results

# ============================================================
# 1. 确认配置差异
# ============================================================
print("="*70)
print("1. 配置差异分析")
print("="*70)

for env in ['HalfCheetah-v4', 'Hopper-v4']:
    for exp_dir in ['AlignedExperiment', 'ICMLExperiment']:
        p = Path(f'results/{exp_dir}/{env}/Optimal_PPO/Optimal_PPO_s0.json')
        if p.exists():
            d = json.load(open(p))
            cfg = d.get('config', {})
            print(f"{exp_dir} / {env}: config={cfg}")

# ============================================================
# 2. 早期学习轨迹分析（最关键）
# ============================================================
print()
print("="*70)
print("2. HalfCheetah 早期训练轨迹（前10个eval点）")
print("="*70)

env = 'HalfCheetah-v4'
for algo in ['Optimal_PPO', 'Optimal_HCGAE_v2']:
    print(f"\n{algo}:")
    for exp_dir in ['AlignedExperiment', 'ICMLExperiment']:
        results = load_full(exp_dir, env, algo)
        print(f"  [{exp_dir}]")
        all_early = []
        for d in results:
            ev = d.get('eval_rewards', [])
            if ev:
                early = ev[:10]
                all_early.append(early)
                print(f"    s{d['seed']}: {[round(v,0) for v in early[:5]]} -> ... -> {[round(v,0) for v in ev[-3:]]}")
        if all_early:
            mat = np.array([e[:min(len(e),10)] for e in all_early if len(e)>=10])
            if len(mat) > 0:
                print(f"    Mean early evals: {[round(v,1) for v in np.mean(mat, axis=0).tolist()]}")

# ============================================================
# 3. AlignedExperiment HalfCheetah 失效分析
# ============================================================
print()
print("="*70)
print("3. AlignedExperiment HalfCheetah HCGAE 失效分析")
print("   (Aligned HCGAE: -30.5% vs ICMLExp HCGAE: +4.3%)")
print("="*70)

env = 'HalfCheetah-v4'
results_aligned_hcgae = load_full('AlignedExperiment', env, 'Optimal_HCGAE_v2')
results_aligned_ppo = load_full('AlignedExperiment', env, 'Optimal_PPO')

print("\nAligned HCGAE vs PPO seed-by-seed:")
for d_hcgae in results_aligned_hcgae:
    s = d_hcgae['seed']
    d_ppo = next((d for d in results_aligned_ppo if d['seed'] == s), None)
    if d_ppo:
        hcgae_final = np.mean(d_hcgae['eval_rewards'][-5:])
        ppo_final = np.mean(d_ppo['eval_rewards'][-5:])
        hcgae_max = max(d_hcgae['eval_rewards'])
        ppo_max = max(d_ppo['eval_rewards'])
        print(f"  s{s}: HCGAE_final={hcgae_final:.1f} (max={hcgae_max:.1f}) | PPO_final={ppo_final:.1f} (max={ppo_max:.1f}) | Δ={hcgae_final-ppo_final:+.1f}")

print()
print("Key insight: seed 3 崩溃分析")
d3 = next(d for d in results_aligned_hcgae if d['seed'] == 3)
evals3 = d3['eval_rewards']
print(f"  HCGAE s3 全程 eval: n={len(evals3)}")
print(f"  前10: {[round(v,0) for v in evals3[:10]]}")
print(f"  最高点位置: step {np.argmax(evals3)}, value={max(evals3):.1f}")
print(f"  后10: {[round(v,0) for v in evals3[-10:]]}")

# ============================================================
# 4. 为什么 Optimal_PPO 在 AlignedExp 上分数大幅提升？
# ============================================================
print()
print("="*70)
print("4. PPO 基线本身在AlignedExp上大幅提升的原因分析")
print("   HalfCheetah: ICML=1486.8 → Aligned=2118.9 (+42.5%)")
print("   Hopper:      ICML=1598.1 → Aligned=2877.2 (+80.0%)")
print("="*70)

# 但HCGAE在HalfCheetah没有同等提升 - 找出为什么
print()
print("提升比较（Aligned vs ICML）:")
for env in ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']:
    for algo in ['Optimal_PPO', 'Optimal_HCGAE_v2']:
        aligned = load_full('AlignedExperiment', env, algo)
        icml = load_full('ICMLExperiment', env, algo)
        if aligned and icml:
            av = np.mean([np.mean(d['eval_rewards'][-5:]) for d in aligned])
            iv = np.mean([np.mean(d['eval_rewards'][-5:]) for d in icml])
            print(f"  {env} / {algo}: {iv:.1f} -> {av:.1f} ({(av-iv)/iv*100:+.1f}%)")

# ============================================================
# 5. 消融数据揭示：NoGate 在新实验中的表现
# ============================================================
print()
print("="*70)
print("5. 消融：NoGate版本在AlignedExp中的表现")
print("="*70)
for env in ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']:
    print(f"\n{env}:")
    for algo in ['Optimal_PPO', 'Optimal_HCGAE_v2', 'Optimal_HCGAE_v2_NoBdry', 'Optimal_HCGAE_v2_NoGate']:
        res = load_full('AlignedExperiment', env, algo)
        if res:
            vals = [np.mean(d['eval_rewards'][-5:]) for d in res]
            print(f"  {algo:<35}: mean={np.mean(vals):.1f}±{np.std(vals)/np.sqrt(len(vals)):.1f}")

# ============================================================
# 6. HalfCheetah 关键问题：PPO提升了但HCGAE没有提升
# ============================================================
print()
print("="*70)
print("6. 核心矛盾分析：为什么 HalfCheetah 的 PPO 提升了 42% 但 HCGAE 反而下降了？")
print("="*70)
print()
print("关键数据:")
print("  HalfCheetah Aligned: PPO=2118.9, HCGAE=1472.6 → HCGAE/PPO = 69.5%")
print("  HalfCheetah ICML:    PPO=1486.8, HCGAE=1550.1 → HCGAE/PPO = 104.3%")
print()
print("  Hopper Aligned:  PPO=2877.2, HCGAE=3166.2 → HCGAE/PPO = 110.0%")
print("  Hopper ICML:     PPO=1598.1, HCGAE=1760.1 → HCGAE/PPO = 110.1%")
print()
print("  Walker2d Aligned: PPO=3038.6, HCGAE=3269.2 → HCGAE/PPO = 107.6%")
print("  Walker2d ICML:    PPO=1596.4, HCGAE=1998.5 → HCGAE/PPO = 125.2%")
print()
print("=== 关键发现 ===")
print("1. Hopper: 新旧实验中HCGAE/PPO比值相同(~110%) → HCGAE改进稳定")
print("2. Walker2d: 新实验中HCGAE/PPO比值下降(125%→108%) → 改进被稀释")
print("3. HalfCheetah: 新实验中HCGAE/PPO比值崩溃(104%→70%) → 致命退化")
print()
print("=== 假设 ===")
print("新实验中HalfCheetah的PPO大幅提升(1487→2119, +42%)，")
print("表明新的PPO基线在HalfCheetah上的训练动态发生了根本性变化。")
print("而HCGAE没有同等提升，表明HCGAE在新的高性能PPO基线下反而产生了干扰。")
print()
print("这符合我们的SCR理论：PPO基线越好(EV更高)，MC校正就越有害。")
print("当PPO自身的EV很快收敛时，HCGAE的MC blending只会引入噪声。")

