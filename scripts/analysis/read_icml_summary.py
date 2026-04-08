#!/usr/bin/env python3
"""快速汇总 ICMLExperiment 的实验数据（5 seeds × 500K steps）"""
import json
from pathlib import Path

import numpy as np

ICML_DIR = Path("results/ICMLExperiment")
ENVS = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]
ALGOS = ["Optimal_PPO", "Optimal_HCGAE_v2", "Optimal_HCGAE_SCR",
         "Optimal_HCGAE_v2_NoBdry", "Optimal_HCGAE_v2_NoGate"]

def load_seeds(env, algo, seeds=range(5)):
    vals = []
    for s in seeds:
        p = ICML_DIR / env / algo / f"{algo}_s{s}.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        er = d.get('eval_rewards', [])
        if er:
            vals.append(float(np.mean(er[-5:])))
    return vals

print("="*70)
print("ICMLExperiment Summary (5 seeds × 500K steps, deterministic eval)")
print("="*70)

for env in ENVS:
    print(f"\n{env}:")
    ppo_mean = None
    for algo in ALGOS:
        vals = load_seeds(env, algo)
        if vals:
            m = np.mean(vals)
            sem = np.std(vals) / np.sqrt(len(vals))
            delta = ""
            if algo != "Optimal_PPO" and ppo_mean is not None:
                d_pct = (m - ppo_mean) / max(abs(ppo_mean), 1) * 100
                delta = f"  Δ={d_pct:+.1f}%"
            print(f"  {algo:35s}: {m:7.1f} ± {sem:5.1f}  (n={len(vals)}){delta}")
            if algo == "Optimal_PPO":
                ppo_mean = m
        else:
            print(f"  {algo:35s}: [no data]")

print("\n" + "="*70)
print("Per-seed values for Optimal_PPO and Optimal_HCGAE_v2:")
print("="*70)
for env in ENVS:
    print(f"\n{env}:")
    for algo in ["Optimal_PPO", "Optimal_HCGAE_v2"]:
        vals = []
        for s in range(5):
            p = ICML_DIR / env / algo / f"{algo}_s{s}.json"
            if p.exists():
                d = json.load(open(p))
                er = d.get('eval_rewards', [])
                if er:
                    v = float(np.mean(er[-5:]))
                    vals.append((s, v))
        if vals:
            vs = [v for _, v in vals]
            print(f"  {algo}: seeds={[f's{s}={v:.0f}' for s,v in vals]}")
            print(f"    mean={np.mean(vs):.1f}, std={np.std(vs):.1f}, min={np.min(vs):.1f}, max={np.max(vs):.1f}")

