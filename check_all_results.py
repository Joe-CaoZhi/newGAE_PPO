import json, os, numpy as np

print("=" * 60)
print("MULTIENV 5-SEED RESULTS")
print("=" * 60)
base = 'results/MultiEnv'
for env in ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']:
    env_path = f'{base}/{env}'
    if not os.path.exists(env_path):
        continue
    algos = {}
    for algo in os.listdir(env_path):
        apath = f'{env_path}/{algo}'
        if not os.path.isdir(apath):
            continue
        seeds = [f for f in os.listdir(apath) if f.endswith('.json')]
        rews = []
        for s in seeds:
            with open(f'{apath}/{s}') as f:
                d = json.load(f)
            r = d.get('episode_rewards', [])
            if r:
                rews.append(np.mean(r[-5:]))
        if rews:
            algos[algo] = (np.mean(rews), np.std(rews), len(rews))
    print(f'\n--- {env} ---')
    for k, (m, s, n) in sorted(algos.items(), key=lambda x: -x[1][0]):
        print(f'  {k}: {m:.0f} +/- {s:.0f} (n={n})')

print("\n" + "=" * 60)
print("BASELINE COMPARISON Hopper-v4")
print("=" * 60)
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
            rews.append(np.mean(r[-5:]))
    if rews:
        print(f'{alg}: mean={np.mean(rews):.0f} +/- {np.std(rews):.0f} (n={len(rews)})')
    else:
        print(f'{alg}: no reward data')

print("\n" + "=" * 60)
print("DCPPO HOPPER RESULTS")
print("=" * 60)
base3 = 'results/Hopper-v4-DCPPO'
if os.path.exists(base3):
    for f in os.listdir(base3):
        if f.endswith('.json'):
            with open(f'{base3}/{f}') as fp:
                d = json.load(fp)
            r = d.get('episode_rewards', [])
            if r:
                print(f'{f}: last5_mean={np.mean(r[-5:]):.0f}, std={np.std(r[-5:]):.0f}')

print("\n" + "=" * 60)
print("ABLATION MULTI-SEED")
print("=" * 60)
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
                rews.append(np.mean(r[-5:]))
        if rews:
            algos[algo] = (np.mean(rews), np.std(rews), len(rews))
    for k, (m, s, n) in sorted(algos.items(), key=lambda x: -x[1][0]):
        print(f'  {k}: {m:.0f} +/- {s:.0f} (n={n})')

