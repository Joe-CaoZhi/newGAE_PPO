#!/usr/bin/env python3
"""验证图表数据的准确性"""
import json
from pathlib import Path
import numpy as np

ENVS = ['HalfCheetah-v4', 'Hopper-v4', 'Walker2d-v4', 'Ant-v4']

def load_scores(base_dir, env, algo, min_steps=0, last_n=10):
    d = Path(base_dir) / env / algo
    if not d.exists():
        return [], []
    scores, steps_list = [], []
    for f in sorted(d.glob("*.json")):
        try:
            data = json.load(open(f))
            if data.get('total_steps', 0) < min_steps:
                continue
            ev = data.get('eval_rewards', [])
            ts = data.get('total_steps', 0)
            if ev:
                scores.append(float(np.mean(ev[-last_n:])))
                steps_list.append(ts)
        except:
            pass
    return scores, steps_list

def load_curve_range(base_dir, env, algo, min_steps=0):
    """返回实际曲线的步数范围和末尾均值"""
    d = Path(base_dir) / env / algo
    if not d.exists():
        return None
    all_last_vals = []
    all_first_steps = []
    all_last_steps = []
    for f in sorted(d.glob("*.json")):
        try:
            data = json.load(open(f))
            if data.get('total_steps', 0) < min_steps:
                continue
            ev = data.get('eval_rewards', [])
            es = data.get('eval_steps', [])
            if ev and es:
                all_last_vals.append(ev[-1])
                all_first_steps.append(es[0])
                all_last_steps.append(es[-1])
        except:
            pass
    if not all_last_vals:
        return None
    return {
        'n': len(all_last_vals),
        'step_range': (int(np.mean(all_first_steps)), int(np.mean(all_last_steps))),
        'final_eval': (np.mean(all_last_vals), np.std(all_last_vals)),
    }

print("=" * 70)
print("  数据验证：FinalOptimal PPO (1M steps, 20 seeds)")
print("=" * 70)
for env in ENVS:
    print(f"\n  {env}:")
    for algo in ['Optimal_PPO', 'Optimal_HCGAE_Optimal']:
        scores, steps = load_scores('results/FinalOptimal', env, algo, min_steps=700_000)
        info = load_curve_range('results/FinalOptimal', env, algo, min_steps=700_000)
        if scores:
            m, s = np.mean(scores), np.std(scores)
            print(f"    {algo:<30} n={len(scores):2d}  "
                  f"final={m:7.0f}±{s:.0f}  "
                  f"step_range=[{info['step_range'][0]/1e6:.2f}M, {info['step_range'][1]/1e6:.2f}M]")
        else:
            print(f"    {algo:<30} NO DATA")

print("\n" + "=" * 70)
print("  数据验证：GRPO (1.5M steps, 15 seeds)")
print("=" * 70)
for env in ENVS:
    print(f"\n  {env}:")
    for algo in ['Optimal_GRPO', 'HCGAE_Optimal_GRPO']:
        scores, steps = load_scores('results/GRPO', env, algo, min_steps=1_000_000)
        info = load_curve_range('results/GRPO', env, algo, min_steps=1_000_000)
        if scores:
            m, s = np.mean(scores), np.std(scores)
            pct_vs_grpo = None
            print(f"    {algo:<30} n={len(scores):2d}  "
                  f"final={m:7.0f}±{s:.0f}  "
                  f"step_range=[{info['step_range'][0]/1e6:.2f}M, {info['step_range'][1]/1e6:.2f}M]")
        else:
            print(f"    {algo:<30} NO DATA")

print("\n" + "=" * 70)
print("  GRPO 提升百分比验证（对应 figA8 中的标注）")
print("=" * 70)
for env in ENVS:
    grpo_s, _  = load_scores('results/GRPO', env, 'Optimal_GRPO', min_steps=1_000_000)
    hcgae_s, _ = load_scores('results/GRPO', env, 'HCGAE_Optimal_GRPO', min_steps=1_000_000)
    if grpo_s and hcgae_s:
        grpo_m  = np.mean(grpo_s)
        hcgae_m = np.mean(hcgae_s)
        pct = (hcgae_m - grpo_m) / abs(grpo_m) * 100
        abs_diff = hcgae_m - grpo_m
        print(f"  {env:<20} GRPO={grpo_m:7.0f}  HCGAE-GRPO={hcgae_m:7.0f}  "
              f"Δ={abs_diff:+.0f}  ({pct:+.1f}%)")

print("\n" + "=" * 70)
print("  Ant-v4 GRPO 详细分析（了解初始 return=900 的来源）")
print("=" * 70)
d = Path('results/GRPO/Ant-v4/Optimal_GRPO')
for f in sorted(d.glob("*.json"))[:3]:
    data = json.load(open(f))
    ev = data.get('eval_rewards', [])
    es = data.get('eval_steps', [])
    print(f"  {f.name}: first_eval_step={es[0] if es else 'N/A'}  "
          f"first_eval_return={ev[0]:.0f if ev else 'N/A'  }  "
          f"last_eval_return={ev[-1]:.0f if ev else 'N/A'}")

print()

