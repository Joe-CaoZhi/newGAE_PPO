#!/usr/bin/env python3
"""实验进度检查 + 离群值过滤统计"""
import glob
import json
import numpy as np
from pathlib import Path

from scipy import stats as sc

RESULTS = Path("results/LargeScaleExperiment")

checks = [
    ("Hopper PPO",    "Hopper-v4/Optimal_PPO"),
    ("Hopper v4",     "Hopper-v4/Optimal_HCGAE_v4"),
    ("Walker PPO",    "Walker2d-v4/Optimal_PPO"),
    ("Walker v4",     "Walker2d-v4/Optimal_HCGAE_v4"),
    ("HC PPO",        "HalfCheetah-v4/Optimal_PPO"),
    ("HC v4",         "HalfCheetah-v4/Optimal_HCGAE_v4"),
    ("HC v6",         "HalfCheetah-v4/Optimal_HCGAE_v6"),
    ("Ant PPO",       "Ant-v4/Optimal_PPO"),
    ("Ant NoBdry",    "Ant-v4/Optimal_HCGAE_v2_NoBdry"),
    ("Ant v5",        "Ant-v4/Optimal_HCGAE_v5"),
    ("Swimmer PPO",   "Swimmer-v4/Optimal_PPO"),
    ("Swimmer v4",    "Swimmer-v4/Optimal_HCGAE_v4"),
    ("Swimmer v6",    "Swimmer-v4/Optimal_HCGAE_v6"),
]

def load_vals(subdir):
    files = sorted(glob.glob(str(RESULTS / subdir / "*.json")))
    vals, seeds = [], []
    for f in files:
        d = json.load(open(f))
        vals.append(d["final_reward"])
        seeds.append(d["seed"])
    return vals, seeds

def remove_outliers_iqr(vals, k=1.5):
    """IQR法去离群值"""
    if len(vals) < 4:
        return vals, []
    arr = np.array(vals)
    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    clean = [v for v in vals if lo <= v <= hi]
    outliers = [v for v in vals if v < lo or v > hi]
    return clean, outliers

# ── 原始统计 ──────────────────────────────────────────────────────────────────
print(f"\n{'='*100}")
print("  RAW (含离群值)")
print(f"{'='*100}")
print(f"{'Group':<18} {'n':>4}  {'mean':>8}  {'std':>7}  {'min':>7}  {'max':>7}  seeds")
print("-" * 100)
all_data = {}
for name, subdir in checks:
    vals, seeds = load_vals(subdir)
    all_data[name] = (vals, seeds)
    if not vals:
        print(f"{name:<18} {'0':>4}  {'pending':>8}")
        continue
    print(f"{name:<18} {len(vals):>4}  {np.mean(vals):>8.0f}  {np.std(vals):>7.0f}  {min(vals):>7.0f}  {max(vals):>7.0f}  {sorted(seeds)}")

# ── 去离群值统计 ───────────────────────────────────────────────────────────────
print(f"\n{'='*100}")
print("  CLEAN (IQR×1.5 去离群值)")
print(f"{'='*100}")
print(f"{'Group':<18} {'n_raw':>6}  {'n_clean':>8}  {'mean':>8}  {'std':>7}  removed")
print("-" * 100)
clean_data = {}
for name, subdir in checks:
    vals, seeds = all_data[name]
    if not vals:
        print(f"{name:<18}  pending")
        continue
    clean, outliers = remove_outliers_iqr(vals)
    clean_data[name] = clean
    removed = f"{[round(v) for v in sorted(outliers)]}" if outliers else "none"
    print(f"{name:<18} {len(vals):>6}  {len(clean):>8}  {np.mean(clean):>8.0f}  {np.std(clean):>7.0f}  removed={removed}")

# ── 去离群值后的对比 ──────────────────────────────────────────────────────────
print(f"\n{'='*100}")
print("  PAIRWISE (去离群值后，vs PPO baseline)")
print(f"{'='*100}")
pairs = [
    ("Hopper",      "Hopper PPO",   "Hopper v4",   "HCGAE_v4"),
    ("Walker2d",    "Walker PPO",   "Walker v4",   "HCGAE_v4"),
    ("HalfCheetah","HC PPO",       "HC v4",        "HCGAE_v4"),
    ("HalfCheetah","HC PPO",       "HC v6",        "HCGAE_v6"),
    ("Ant",         "Ant PPO",      "Ant NoBdry",   "NoBdry"),
    ("Ant",         "Ant PPO",      "Ant v5",       "v5"),
    ("Swimmer",     "Swimmer PPO",  "Swimmer v4",   "v4"),
    ("Swimmer",     "Swimmer PPO",  "Swimmer v6",   "v6"),
]

print(f"\n{'Env':<14} {'Method':<10} {'n_ppo':>6} {'n_hcg':>6} {'Δ%':>8}  {'p':>8}  {'sig':>5}  {'d':>6}  {'PPO_clean':>10}  {'HCG_clean':>10}")
print("-" * 100)
for env, ppo_name, hcg_name, tag in pairs:
    ppo_clean = clean_data.get(ppo_name, [])
    hcg_clean = clean_data.get(hcg_name, [])
    if not ppo_clean or not hcg_clean:
        status = "pending" if not hcg_clean else "no ppo"
        print(f"{env:<14} {tag:<10} {'—':>6} {'—':>6}  {status}")
        continue
    delta = (np.mean(hcg_clean) - np.mean(ppo_clean)) / abs(np.mean(ppo_clean)) * 100
    pooled = np.sqrt((np.std(hcg_clean)**2 + np.std(ppo_clean)**2) / 2)
    d = (np.mean(hcg_clean) - np.mean(ppo_clean)) / (pooled + 1e-8)
    if len(hcg_clean) >= 2:
        _, p = sc.mannwhitneyu(hcg_clean, ppo_clean, alternative='two-sided')
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    else:
        p, sig = 1.0, "—"
    ppo_str = f"{np.mean(ppo_clean):.0f}±{np.std(ppo_clean):.0f}"
    hcg_str = f"{np.mean(hcg_clean):.0f}±{np.std(hcg_clean):.0f}"
    arrow = "✅" if delta > 5 else ("⚠️" if delta > 0 else "❌")
    print(f"{env:<14} {tag:<10} {len(ppo_clean):>6} {len(hcg_clean):>6}  {delta:>+7.1f}%  {p:>8.3f}  {sig:>5}  {d:>+6.2f}  {ppo_str:>10}  {hcg_str:>10}  {arrow}")

# ── per-seed 详情 ─────────────────────────────────────────────────────────────
print(f"\n{'='*100}")
print("  PER-SEED DETAIL (HalfCheetah — 双峰分析)")
print(f"{'='*100}")
ppo_files = sorted(glob.glob(str(RESULTS / "HalfCheetah-v4/Optimal_PPO/*.json")))
v4_files  = sorted(glob.glob(str(RESULTS / "HalfCheetah-v4/Optimal_HCGAE_v4/*.json")))
v6_files  = sorted(glob.glob(str(RESULTS / "HalfCheetah-v4/Optimal_HCGAE_v6/*.json")))

ppo_by_seed = {json.load(open(f))["seed"]: json.load(open(f))["final_reward"] for f in ppo_files}
v4_by_seed  = {json.load(open(f))["seed"]: json.load(open(f))["final_reward"] for f in v4_files}
v6_by_seed  = {json.load(open(f))["seed"]: json.load(open(f))["final_reward"] for f in v6_files}

all_seeds = sorted(set(ppo_by_seed) | set(v4_by_seed) | set(v6_by_seed))
print(f"{'seed':>5}  {'PPO':>7}  {'v4':>7}  {'v6':>7}  {'Δv4':>8}  {'Δv6':>8}")
print("-" * 55)
for s in all_seeds:
    ppo = ppo_by_seed.get(s, None)
    v4  = v4_by_seed.get(s, None)
    v6  = v6_by_seed.get(s, None)
    ppo_s = f"{ppo:.0f}" if ppo else "—"
    v4_s  = f"{v4:.0f}"  if v4  else "—"
    v6_s  = f"{v6:.0f}"  if v6  else "—"
    dv4   = f"{v4-ppo:+.0f}" if (ppo and v4) else "—"
    dv6   = f"{v6-ppo:+.0f}" if (ppo and v6) else "—"
    print(f"   s{s:>2}  {ppo_s:>7}  {v4_s:>7}  {v6_s:>7}  {dv4:>8}  {dv6:>8}")

