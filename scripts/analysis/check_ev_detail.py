import json
import os

import numpy as np

ev_path = 'results/EVConvergenceStudy'

# Load all individual EV timing data by algorithm
data = {}
for f in os.listdir(ev_path):
    if f.endswith('.json') and 'series' not in f and 'summary' not in f:
        fpath = os.path.join(ev_path, f)
        with open(fpath) as fp:
            d = json.load(fp)
        # Parse filename: env_algo_sN.json
        parts = f.replace('.json', '').split('_')
        # Find seed index
        seed_idx = None
        for i, p in enumerate(parts):
            if p.startswith('s') and p[1:].isdigit():
                seed_idx = i
                break
        if seed_idx is not None:
            algo = '_'.join(parts[2:seed_idx])
            env = '_'.join(parts[:2])
            seed = parts[seed_idx]
            key = f"{env}/{algo}"
            if key not in data:
                data[key] = []
            data[key].append(d)

print("=== EV Convergence: Steps to EV>0.9 by Algorithm ===")
print("(comparing Optimal_HCGAE vs Optimal_PPO on Hopper-v4)")
print()

for key in sorted(data.keys()):
    recs = data[key]
    steps09 = [r.get('steps_to_ev09', None) for r in recs]
    steps09 = [s for s in steps09 if s is not None and s > 0]
    if steps09:
        m = np.mean(steps09)
        se = np.std(steps09, ddof=0) / np.sqrt(len(steps09)) if len(steps09) > 1 else 0
        print(f"  {key}: mean={m:.0f} ± {se:.0f} steps (n={len(steps09)}) | {[s for s in steps09]}")

print()
print("=== Load analysis_summary.json ===")
with open(os.path.join(ev_path, 'analysis_summary.json')) as fp:
    summary = json.load(fp)

for env in summary:
    print(f"\n{env}:")
    for algo in summary[env]:
        d = summary[env][algo]
        print(f"  {algo}:")
        for k, v in d.items():
            print(f"    {k}: {v}")

print()
print("=== Hopper-v4 detailed comparison ===")
hopper_ppo = [r.get('steps_to_ev09', None) for r in data.get('Hopper-v4/Optimal_PPO', [])]
hopper_hcgae = [r.get('steps_to_ev09', None) for r in data.get('Hopper-v4/Optimal_HCGAE', [])]
hopper_hcgaev2 = [r.get('steps_to_ev09', None) for r in data.get('Hopper-v4/Optimal_HCGAE_v2', [])]

hopper_ppo = [s for s in hopper_ppo if s and s > 0]
hopper_hcgae = [s for s in hopper_hcgae if s and s > 0]
hopper_hcgaev2 = [s for s in hopper_hcgaev2 if s and s > 0]

if hopper_ppo:
    print(f"Optimal PPO: {np.mean(hopper_ppo):.0f} ± {np.std(hopper_ppo):.0f} steps | {hopper_ppo}")
if hopper_hcgae:
    print(f"Optimal HCGAE: {np.mean(hopper_hcgae):.0f} ± {np.std(hopper_hcgae):.0f} steps | {hopper_hcgae}")
    if hopper_ppo:
        speedup = (np.mean(hopper_ppo) - np.mean(hopper_hcgae)) / np.mean(hopper_ppo)
        print(f"  -> Speedup: {speedup*100:.1f}% reduction in steps")
if hopper_hcgaev2:
    print(f"Optimal HCGAE_v2: {np.mean(hopper_hcgaev2):.0f} ± {np.std(hopper_hcgaev2):.0f} steps | {hopper_hcgaev2}")
    if hopper_ppo:
        speedup = (np.mean(hopper_ppo) - np.mean(hopper_hcgaev2)) / np.mean(hopper_ppo)
        print(f"  -> Speedup: {speedup*100:.1f}% reduction in steps (NEGATIVE = SLOWER!)")

# Also check other environments
print()
print("=== Other environments: Walker2d-v4 ===")
for key in sorted(data.keys()):
    if 'Walker2d' in key:
        recs = data[key]
        steps09 = [r.get('steps_to_ev09', None) for r in recs]
        steps09 = [s for s in steps09 if s and s > 0]
        if steps09:
            m = np.mean(steps09)
            s = np.std(steps09)
            print(f"  {key.split('/')[1]}: {m:.0f} ± {s:.0f} steps | {steps09}")

print()
print("=== HalfCheetah-v4 ===")
for key in sorted(data.keys()):
    if 'HalfCheetah' in key:
        recs = data[key]
        steps09 = [r.get('steps_to_ev09', None) for r in recs]
        steps09 = [s for s in steps09 if s and s > 0]
        if steps09:
            m = np.mean(steps09)
            s = np.std(steps09)
            print(f"  {key.split('/')[1]}: {m:.0f} ± {s:.0f} steps | {steps09}")

