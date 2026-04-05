"""Analyze AutoSCR experiment progress."""
import json
import numpy as np
import os

# AutoSCR Experiment
base = '/Users/joe-caozhi/newGAE_ppo/results/AutoSCRExperiment'
print("=== AutoSCR Experiment Progress ===")
for env in sorted(os.listdir(base)):
    env_path = os.path.join(base, env)
    if os.path.isdir(env_path) and env not in ['test']:
        print(f"\n{env}:")
        for algo in sorted(os.listdir(env_path)):
            algo_path = os.path.join(env_path, algo)
            if os.path.isdir(algo_path):
                seeds = []
                for f in os.listdir(algo_path):
                    if f.endswith('.json'):
                        try:
                            d = json.load(open(os.path.join(algo_path, f)))
                            if 'eval_rewards' in d:
                                final = d['eval_rewards'][-5:]
                            elif 'eval_returns' in d:
                                final = d['eval_returns'][-5:]
                            else:
                                final = [d.get('final_reward', 0)]
                            seeds.append(sum(final)/len(final))
                        except Exception as e:
                            print(f"    Error reading {f}: {e}")
                if seeds:
                    print(f"  {algo}: n={len(seeds)}, mean={np.mean(seeds):.1f}, std={np.std(seeds):.1f}")

# HighPower Experiment
base = '/Users/joe-caozhi/newGAE_ppo/results/HighPowerExperiment'
print("\n=== HighPower Experiment Progress ===")
if os.path.exists(base):
    for env in sorted(os.listdir(base)):
        env_path = os.path.join(base, env)
        if os.path.isdir(env_path):
            print(f"\n{env}:")
            for algo in sorted(os.listdir(env_path)):
                algo_path = os.path.join(env_path, algo)
                if os.path.isdir(algo_path):
                    seeds = []
                    for f in os.listdir(algo_path):
                        if f.endswith('.json'):
                            try:
                                d = json.load(open(os.path.join(algo_path, f)))
                                if 'eval_rewards' in d:
                                    final = d['eval_rewards'][-5:]
                                elif 'eval_returns' in d:
                                    final = d['eval_returns'][-5:]
                                else:
                                    final = [d.get('final_reward', 0)]
                                seeds.append(sum(final)/len(final))
                            except Exception as e:
                                pass
                    if seeds:
                        print(f"  {algo}: n={len(seeds)}, mean={np.mean(seeds):.1f}, std={np.std(seeds):.1f}")

