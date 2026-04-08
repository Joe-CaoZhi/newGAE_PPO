#!/usr/bin/env python3
"""Show current experiment status using correct JSON keys."""
import glob
import json
import numpy as np

envs = ['Hopper-v4','Walker2d-v4','HalfCheetah-v4','Ant-v4']
algos = ['Standard_PPO','Optimal_PPO','Optimal_HCGAE','Optimal_HCGAE_v2',
         'Optimal_HCGAE_SCR','Optimal_HCGAE_v2_NoBdry','Optimal_HCGAE_v2_NoGate']

print("=" * 65)
print("EXPERIMENT STATUS (using eval_rewards, last 5 evaluations)")
print("=" * 65)

all_data = {}
for env in envs:
    print(f'\n=== {env} ===')
    all_data[env] = {}
    for algo in algos:
        files = glob.glob(f'results/ICMLExperiment/{env}/{algo}/*.json')
        if not files:
            continue
        vals = []
        for f in files:
            d = json.load(open(f))
            # Correct key: eval_rewards (list), take mean of last 5
            er = d.get('eval_rewards', None)
            if er and len(er) > 0:
                vals.append(float(np.mean(er[-5:])))
            elif 'final_reward' in d:
                vals.append(float(d['final_reward']))
        if vals:
            m, s = np.mean(vals), np.std(vals)
            all_data[env][algo] = {'mean': m, 'std': s, 'n': len(vals)}
            print(f'  {algo:<32s} ({len(vals)}/5): {m:7.0f} ± {s:.0f}')

# Summary table
print('\n' + '=' * 65)
print('SUMMARY TABLE (mean ± std, n=5 seeds)')
print('=' * 65)
header = f"{'Method':<30s} {'Hopper':>10s} {'Walker':>10s} {'HalfCheetah':>12s} {'Ant':>10s}"
print(header)
print('-' * 75)
for algo in algos:
    row = f'{algo:<30s}'
    for env in envs:
        d = all_data.get(env, {}).get(algo, None)
        if d:
            row += f" {d['mean']:7.0f}({d['n']})"
        else:
            row += f" {'--':>10s}"
    print(row)

# v2 improvements analysis
print('\n' + '=' * 65)
print('v2 IMPROVEMENT ANALYSIS vs v1')
print('=' * 65)
for env in ['HalfCheetah-v4','Hopper-v4','Walker2d-v4','Ant-v4']:
    v1 = all_data.get(env, {}).get('Optimal_HCGAE', None)
    v2 = all_data.get(env, {}).get('Optimal_HCGAE_v2', None)
    opt = all_data.get(env, {}).get('Optimal_PPO', None)
    if v1 and v2:
        pct_v1 = (v2['mean'] - v1['mean']) / abs(v1['mean']) * 100
        pct_opt = (v2['mean'] - opt['mean']) / abs(opt['mean']) * 100 if opt else 0
        print(f"  {env}: v1={v1['mean']:.0f}(n={v1['n']}) -> v2={v2['mean']:.0f}(n={v2['n']}) "
              f"[vs v1: {pct_v1:+.1f}%, vs OptPPO: {pct_opt:+.1f}%]")
    elif v1:
        print(f"  {env}: v1={v1['mean']:.0f}(n={v1['n']}) -> v2=PENDING")
    else:
        print(f"  {env}: v1=PENDING, v2=PENDING")

