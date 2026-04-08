import json
import os

import numpy as np

ev_path = 'results/EVConvergenceStudy'

# Load analysis_summary.json properly
with open(os.path.join(ev_path, 'analysis_summary.json')) as fp:
    summary = json.load(fp)

print("=== analysis_summary.json full content ===")
for env in summary:
    print(f"\n{env}:")
    for algo in summary[env]:
        v = summary[env][algo]
        if isinstance(v, dict):
            print(f"  {algo}: {v}")
        else:
            print(f"  {algo}: {v}")

print()
print("=== Key Finding: EV Convergence Comparison ===")
print()
print("Hopper-v4 - Steps to EV > 0.9:")
print(f"  Optimal PPO: mean=33997 steps (range: 30720-40960)")
print(f"  Optimal HCGAE: mean=48333 steps (range: 34816-63488)")
print(f"  Optimal HCGAE_v2: mean=45875 steps (range: 34816-57344)")
print()
print("  CONCLUSION: HCGAE is SLOWER than Optimal PPO at reaching EV>0.9!")
print("  Speedup relative to OptPPO: {:.1f}%".format((33997-48333)/33997*100))
print()
print("Walker2d-v4 - Steps to EV > 0.9:")
print(f"  Optimal PPO: mean=40141 steps (range: 32768-47104)")
print(f"  Optimal HCGAE: mean=49971 steps (range: 49152-53248)")
print(f"  Optimal HCGAE_v2: mean=52838 steps (range: 49152-59392)")
print()
print("  CONCLUSION: HCGAE is also SLOWER than Optimal PPO on Walker2d!")
print()
print("=== The 47% claim is from Theorem 1 Corollary 1 ===")
print("The 47% claim is a THEORETICAL result for Standard PPO (no obs normalization)")
print("Calibrated to eta=0.01 (Standard PPO without obs norm)")
print()
print("=== Checking if there are Standard PPO EV timing data ===")
# Load ev_convergence_summary.json to see if Standard PPO is there
with open(os.path.join(ev_path, 'ev_convergence_summary.json')) as fp:
    ev_sum = json.load(fp)

# Get unique environments and algos from keys
algos_seen = set()
for key in ev_sum:
    parts = key.split('/')
    if len(parts) >= 2:
        algos_seen.add(parts[1])
print(f"Algorithms in ev_convergence_summary: {sorted(algos_seen)}")

# Get env/algo/step data
from collections import defaultdict
by_algo = defaultdict(list)
for key, vals in ev_sum.items():
    parts = key.split('/')
    if len(parts) >= 2:
        env_algo = f"{parts[0]}/{parts[1]}"
        by_algo[env_algo].append(vals.get('steps_to_ev09', None))

print()
print("=== All Hopper-v4 algorithms ===")
for key, steps in sorted(by_algo.items()):
    if 'Hopper' in key:
        steps_clean = [s for s in steps if s and s > 0]
        if steps_clean:
            print(f"  {key}: mean={np.mean(steps_clean):.0f} ± {np.std(steps_clean):.0f} | n={len(steps_clean)}")

print()
print("=== Summary for paper ===")
print("The 47% claim (80K vs 150K steps) was from diagnostics on Standard PPO base,")
print("comparing Standard_PPO (no obs norm) vs HCGAE_Imp12 (no obs norm).")
print()
print("In our actual Optimal PPO experiments (with obs norm):")
print("  - Optimal PPO: ~34K steps to EV>0.9")
print("  - HCGAE: ~48K steps to EV>0.9")
print("  HCGAE is SLOWER, not faster, in the Optimal PPO setting!")
print()
print("This is because obs normalization helps PPO converge FAST,")
print("while HCGAE's MC correction can temporarily perturb EV in early training.")

