#!/usr/bin/env python3
"""验证DCPPO数据文件的实际格式和内容"""
import json
import os
import numpy as np

BASE = "/Users/joe-caozhi/newGAE_ppo/results"

def get_final_reward(filepath):
    with open(filepath) as f:
        data = json.load(f)
    if "final_reward" in data:
        return data["final_reward"]
    if "all_eval_rewards" in data:
        r = data["all_eval_rewards"]
        return float(np.mean(r[-5:]))
    if "eval_rewards" in data:
        r = data["eval_rewards"]
        return float(np.mean(r[-5:]))
    return None

def load_dcppo_seeds(env, variant):
    env_dir = os.path.join(BASE, "MultiEnv_DCPPO", env, variant)
    seeds = [42, 123, 456, 789, 1234]
    rewards = []
    for s in seeds:
        fn = os.path.join(env_dir, f"{variant}_s{s}_metrics.json")
        if os.path.exists(fn):
            r = get_final_reward(fn)
            if r is not None:
                rewards.append(r)
                print(f"  {fn.split('/')[-1]}: {r:.1f}")
            else:
                print(f"  {fn.split('/')[-1]}: NO FINAL REWARD FOUND")
        else:
            print(f"  {fn.split('/')[-1]}: FILE NOT FOUND")
    return rewards

for env in ["Hopper-v4", "Walker2d-v4"]:
    for variant in ["DCPPO_Base", "DCPPO_ImpS", "DCPPO_Full"]:
        print(f"\n{env}/{variant}:")
        rewards = load_dcppo_seeds(env, variant)
        if rewards:
            print(f"  --> mean={np.mean(rewards):.0f} ± {np.std(rewards, ddof=1):.0f} (n={len(rewards)})")

