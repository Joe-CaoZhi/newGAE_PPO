import json
import numpy as np
import os

print("=" * 65)
print("PART 1: MULTIENV 5-SEED RESULTS (all_eval_rewards[-5:] mean)")
print("=" * 65)

base = 'results/MultiEnv'
for env in ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']:
    env_path = f'{base}/{env}'
    if not os.path.exists(env_path):
        continue
    algos = {}
    # files directly in env folder
    for fname in os.listdir(env_path):
        fpath = f'{env_path}/{fname}'
        if not fname.endswith('.json') or fname.endswith('_metrics.json'):
            continue
        with open(fpath) as f:
            d = json.load(f)
        if not isinstance(d, dict):
            continue
        algo = d.get('agent', fname.split('_s')[0])
        r = d.get('all_eval_rewards', [])
        if r:
            val = np.mean(r[-5:]) if len(r) >= 5 else np.mean(r)
            if algo not in algos:
                algos[algo] = []
            algos[algo].append(val)
    print(f'\n--- {env} ---')
    for k in sorted(algos.keys(), key=lambda x: -np.mean(algos[x])):
        vs = algos[k]
        print(f'  {k}: {np.mean(vs):.0f} +/- {np.std(vs):.0f} (n={len(vs)})')

print("\n" + "=" * 65)
print("PART 2: BASELINE COMPARISON Hopper-v4 (episode_rewards[-10:] mean)")
print("=" * 65)

base2 = 'results/BaselineComparison/Hopper-v4'
for alg in sorted(os.listdir(base2)):
    apath = f'{base2}/{alg}'
    if not os.path.isdir(apath):
        continue
    seeds = [f for f in os.listdir(apath) if f.endswith('.json')]
    rews = []
    for s in seeds:
        with open(f'{apath}/{s}') as f:
            d = json.load(f)
        r = d.get('episode_rewards', [])
        if r:
            rews.append(np.mean(r[-20:]))
    if rews:
        print(f'  {alg}: {np.mean(rews):.0f} +/- {np.std(rews):.0f} (n={len(rews)})')

print("\n" + "=" * 65)
print("PART 3: DCPPO MULTI-SEED RESULTS")
print("=" * 65)
base3 = 'results/MultiEnv_DCPPO'
if os.path.exists(base3):
    for fname in sorted(os.listdir(base3)):
        fpath = f'{base3}/{fname}'
        if not fname.endswith('.json'):
            continue
        with open(fpath) as f:
            d = json.load(f)
        if isinstance(d, dict):
            print(f'  {fname}: {d}')

print("\n" + "=" * 65)
print("PART 4: ABLATION MULTI-SEED (Hopper-v4)")
print("=" * 65)
base4 = 'results/Hopper-v4-Ablation-MultiSeed'
if os.path.exists(base4):
    algos = {}
    for algo in os.listdir(base4):
        apath = f'{base4}/{algo}'
        if not os.path.isdir(apath):
            continue
        seeds = [f for f in os.listdir(apath) if f.endswith('.json')]
        rews = []
        for s in seeds:
            with open(f'{apath}/{s}') as fp:
                d = json.load(fp)
            r = d.get('episode_rewards', [])
            if r:
                rews.append(np.mean(r[-20:]))
        if rews:
            algos[algo] = (np.mean(rews), np.std(rews), len(rews))
    for k, (m, s, n) in sorted(algos.items(), key=lambda x: -x[1][0]):
        print(f'  {k}: {m:.0f} +/- {s:.0f} (n={n})')

