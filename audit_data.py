"""Audit script: understand all data discrepancies."""
import json, numpy as np, os

print("=" * 70)
print("AUDIT: Understanding MultiEnv vs BaselineComparison discrepancy")
print("=" * 70)

# MultiEnv HCGAE
d1 = json.load(open('results/MultiEnv/Hopper-v4/HCGAE_Imp12_s42.json'))
print("\n[MultiEnv] HCGAE_Imp12_s42:")
print(f"  keys: {list(d1.keys())}")
print(f"  n_eval_points: {len(d1['all_eval_rewards'])}")
print(f"  eval_steps last5: {d1['eval_steps'][-5:]}")
print(f"  last eval reward: {d1['all_eval_rewards'][-1]:.1f}")
print(f"  last5 eval mean: {np.mean(d1['all_eval_rewards'][-5:]):.1f}")

# BaselineComp HCGAE
d2 = json.load(open('results/BaselineComparison/Hopper-v4/HCGAE_Imp12/HCGAE_Imp12_s42_metrics.json'))
print("\n[BaselineComp] HCGAE_Imp12_s42_metrics:")
print(f"  keys: {list(d2.keys())}")
print(f"  n_episodes: {len(d2['episode_rewards'])}")
er = d2.get('eval_rewards', [])
print(f"  n_eval_rewards: {len(er)}")
if er:
    print(f"  eval_rewards last5: {[round(x,1) for x in er[-5:]]}")
    print(f"  last5 eval mean: {np.mean(er[-5:]):.1f}")
    print(f"  eval_steps last5: {d2.get('eval_steps',[])[-5:]}")

print()
print("=" * 70)
print("AUDIT: Standard_PPO in MultiEnv (uses BasePPO = contains VClip?)")
print("=" * 70)

d3 = json.load(open('results/MultiEnv/Hopper-v4/Standard_PPO_s42.json'))
print(f"\n[MultiEnv] Standard_PPO_s42 keys: {list(d3.keys())}")
print(f"  agent: {d3.get('agent','?')}")
print(f"  last5 eval mean: {np.mean(d3['all_eval_rewards'][-5:]):.1f}")
print(f"  last eval: {d3['all_eval_rewards'][-1]:.1f}")

d4 = json.load(open('results/BaselineComparison/Hopper-v4/Standard_PPO/Standard_PPO_s42_metrics.json'))
print(f"\n[BaselineComp] Standard_PPO_s42 keys: {list(d4.keys())}")
er4 = d4.get('eval_rewards', [])
if er4:
    print(f"  last5 eval mean: {np.mean(er4[-5:]):.1f}")

print()
print("=" * 70)
print("AUDIT: What eval protocol gives what numbers?")
print("=" * 70)
print("\nMultiEnv protocol: all_eval_rewards (periodic evaluation)")
print("BaselineComp protocol: eval_rewards (also periodic evaluation)")
print()

# Compare all algorithms in MultiEnv
print("MultiEnv Hopper-v4 ALL ALGORITHMS (last5 eval mean):")
env_path = 'results/MultiEnv/Hopper-v4'
results = {}
for fname in sorted(os.listdir(env_path)):
    if not fname.endswith('.json') or fname.endswith('_metrics.json') or fname == 'summary.json':
        continue
    fpath = f'{env_path}/{fname}'
    d = json.load(open(fpath))
    if not isinstance(d, dict) or 'all_eval_rewards' not in d:
        continue
    algo = d.get('agent', fname.split('_s')[0])
    val = np.mean(d['all_eval_rewards'][-5:])
    results.setdefault(algo, []).append(val)
for k in sorted(results, key=lambda x: -np.mean(results[x])):
    vs = results[k]
    print(f"  {k}: {np.mean(vs):.0f} +/- {np.std(vs):.0f} (n={len(vs)})")

print()
print("BaselineComp Hopper-v4 ALL (last5 eval_rewards mean):")
base2 = 'results/BaselineComparison/Hopper-v4'
for alg in sorted(os.listdir(base2)):
    apath = f'{base2}/{alg}'
    if not os.path.isdir(apath): continue
    vals = []
    for s in os.listdir(apath):
        if not s.endswith('.json'): continue
        d = json.load(open(f'{apath}/{s}'))
        er = d.get('eval_rewards', [])
        if er:
            vals.append(np.mean(er[-5:]))
    if vals:
        print(f"  {alg}: {np.mean(vals):.0f} +/- {np.std(vals):.0f} (n={len(vals)})")
    else:
        # Fallback to episode_rewards
        vals2 = []
        for s in os.listdir(apath):
            if not s.endswith('.json'): continue
            d = json.load(open(f'{apath}/{s}'))
            ep = d.get('episode_rewards', [])
            if ep:
                vals2.append(np.mean(ep[-20:]))
        if vals2:
            print(f"  {alg}: {np.mean(vals2):.0f} +/- {np.std(vals2):.0f} (n={len(vals2)}) [episode fallback]")

