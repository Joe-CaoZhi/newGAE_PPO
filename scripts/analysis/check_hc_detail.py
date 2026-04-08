import glob
import json
import numpy as np

for algo in ['Optimal_PPO', 'Optimal_HCGAE_v4', 'Optimal_HCGAE_v6']:
    files = sorted(glob.glob(f'results/LargeScaleExperiment/HalfCheetah-v4/{algo}/*.json'))
    if not files:
        print(f'--- HC {algo}: no data yet ---')
        continue
    print(f'--- HC {algo} (n={len(files)}) ---')
    vals = []
    for f in files:
        d = json.load(open(f))
        vals.append(d['final_reward'])
        print(f"  s{d['seed']}: final={d['final_reward']:.0f}  max={d['max_reward']:.0f}  t={d['elapsed_s']/60:.1f}min")
    print(f"  => mean={np.mean(vals):.0f}  std={np.std(vals):.0f}")

print()
print("--- Hopper v4 ---")
for f in sorted(glob.glob('results/LargeScaleExperiment/Hopper-v4/Optimal_HCGAE_v4/*.json')):
    d = json.load(open(f))
    print(f"  s{d['seed']}: final={d['final_reward']:.0f}  t={d['elapsed_s']/60:.1f}min")

print()
print("--- Walker v4 ---")
for f in sorted(glob.glob('results/LargeScaleExperiment/Walker2d-v4/Optimal_HCGAE_v4/*.json')):
    d = json.load(open(f))
    print(f"  s{d['seed']}: final={d['final_reward']:.0f}  t={d['elapsed_s']/60:.1f}min")

