#!/usr/bin/env python3
"""Check EV convergence data to validate the 47% speedup claim across environments"""
import json
import os

# The 47% claim: "HCGAE_Imp12 reaches EV > 0.9 by step ~80K, vs step ~150K for Standard PPO"
# This is based on single-seed (seed=42) diagnostic, Hopper-v4 only

# Check if there's any EV convergence study data
ev_dirs = [
    'results/MultiSeedPower',
    'results/Hopper-v4',
    'results/ICMLExperiment/Hopper-v4',
    'results/AlignedExperiment',
]

for d in ev_dirs:
    if os.path.exists(d):
        print(f"\n=== {d} ===")
        items = os.listdir(d)
        print(f"  Contents: {items[:10]}")
        # Check for any EV data files
        for item in items:
            fpath = f'{d}/{item}'
            if os.path.isfile(fpath) and fpath.endswith('.json'):
                try:
                    with open(fpath) as f:
                        d2 = json.load(f)
                    keys = list(d2.keys()) if isinstance(d2, dict) else type(d2)
                    print(f"  {item}: {keys}")
                except:
                    pass

# Check if any ICMLExperiment json has EV data
print("\n=== Checking ICMLExperiment Hopper HCGAE json for EV data ===")
f = 'results/ICMLExperiment/Hopper-v4/Optimal_HCGAE/Optimal_HCGAE_s0.json'
if os.path.exists(f):
    with open(f) as fp:
        d = json.load(fp)
    print(f"Keys: {list(d.keys())}")
    for key in d:
        val = d[key]
        if isinstance(val, list) and len(val) > 0:
            print(f"  {key}: list of {len(val)} items, first={val[0]}, last={val[-1]}")
        elif isinstance(val, (int, float, str)):
            print(f"  {key}: {val}")

# Similarly for Standard PPO
print("\n=== Checking ICMLExperiment Hopper Standard PPO json ===")
f2 = 'results/ICMLExperiment/Hopper-v4/Standard_PPO/Standard_PPO_s0.json'
if os.path.exists(f2):
    with open(f2) as fp:
        d = json.load(fp)
    print(f"Keys: {list(d.keys())}")
    for key in d:
        val = d[key]
        if isinstance(val, list) and len(val) > 0:
            print(f"  {key}: list of {len(val)} items, first={val[0]}, last={val[-1]}")
        elif isinstance(val, (int, float, str)):
            print(f"  {key}: {val}")

# Check MultiSeedPower for any EV data
print("\n=== Checking MultiSeedPower for EV data ===")
msp = 'results/MultiSeedPower'
for env_dir in ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']:
    for algo in ['HCGAE_Imp12', 'Standard_PPO']:
        fdir = f'{msp}/{env_dir}/{algo}'
        if os.path.exists(fdir):
            files = os.listdir(fdir)[:2]
            for fname in files:
                fpath = f'{fdir}/{fname}'
                if fpath.endswith('.json'):
                    try:
                        with open(fpath) as fp:
                            d = json.load(fp)
                        print(f"  {env_dir}/{algo}/{fname}: keys={list(d.keys())}")
                    except:
                        pass

