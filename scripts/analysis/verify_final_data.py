#!/usr/bin/env python3
"""Final data verification for ICML paper submission."""
import json
import os

import numpy as np

# Verify ICMLExperiment main results (Table 1)
envs = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4', 'Ant-v4']
methods_map = {
    'Optimal_PPO': 'Optimal PPO',
    'Optimal_HCGAE_v2': 'HCGAE',
}

print("=" * 70)
print("TABLE 1 VERIFICATION: HCGAE vs Optimal PPO (5 seeds, 500K steps)")
print("=" * 70)

paper_values = {
    'Hopper-v4': {'Optimal_PPO': (1598, 133), 'Optimal_HCGAE_v2': (1760, 340)},
    'Walker2d-v4': {'Optimal_PPO': (1596, 373), 'Optimal_HCGAE_v2': (1999, 702)},
    'HalfCheetah-v4': {'Optimal_PPO': (1487, 55), 'Optimal_HCGAE_v2': (1550, 348)},
}

for env in envs:
    print(f"\n--- {env} ---")
    for method_dir, method_name in methods_map.items():
        path = f'/Users/joe-caozhi/newGAE_ppo/results/ICMLExperiment/{env}/{method_dir}/'
        if not os.path.exists(path):
            print(f"  {method_name}: PATH NOT FOUND ({path})")
            continue

        rewards = []
        for fname in sorted(os.listdir(path)):
            if fname.endswith('.json'):
                with open(os.path.join(path, fname)) as f:
                    data = json.load(f)
                key = None
                for k in ['eval_rewards', 'eval_returns']:
                    if k in data:
                        key = k
                        break
                if key and data[key]:
                    last5 = data[key][-5:]
                    rewards.append(np.mean(last5))

        if rewards:
            actual_mean = np.mean(rewards)
            actual_std = np.std(rewards, ddof=1) if len(rewards) > 1 else 0.0
            actual_sem = actual_std / np.sqrt(len(rewards))

            paper_entry = paper_values.get(env, {}).get(method_dir)
            if paper_entry:
                paper_mean, paper_std = paper_entry
                mean_diff = abs(actual_mean - paper_mean)
                status = "OK" if mean_diff < 50 else "MISMATCH"
                print(f"  {method_name}: actual={actual_mean:.1f}±{actual_std:.1f}(std) / ±{actual_sem:.1f}(sem) | paper={paper_mean}±{paper_std} [{status}]")
            else:
                print(f"  {method_name}: actual={actual_mean:.1f}±{actual_std:.1f} (n={len(rewards)})")
        else:
            print(f"  {method_name}: NO DATA ({len(os.listdir(path))} files in dir)")

# Verify sensitivity data
print("\n" + "=" * 70)
print("SENSITIVITY DATA VERIFICATION (Table S1/S2/S3)")
print("=" * 70)
sens_path = '/Users/joe-caozhi/newGAE_ppo/results/Sensitivity/sensitivity_summary.json'
if os.path.exists(sens_path):
    with open(sens_path) as f:
        sens_data = json.load(f)

    print("\nBeta sensitivity (Table S1):")
    for item in sens_data.get('beta_sensitivity', []):
        beta = item.get('beta', '?')
        reward = item.get('final_reward', '?')
        print(f"  beta={beta}: {reward}")

    print("\nAlpha_max sensitivity (Table S2):")
    for item in sens_data.get('alpha_max_sensitivity', []):
        alpha = item.get('alpha_max', '?')
        reward = item.get('final_reward', '?')
        print(f"  alpha_max={alpha}: {reward}")

    print("\nSNR sensitivity (Table S3):")
    for item in sens_data.get('snr_sensitivity', []):
        snr = item.get('snr_target', '?')
        reward = item.get('final_reward', '?')
        print(f"  snr={snr}: {reward}")
else:
    print(f"  sensitivity_summary.json NOT FOUND at {sens_path}")

# Verify MultiSeedPower data (Table 2)
print("\n" + "=" * 70)
print("TABLE 2 VERIFICATION: Statistical Robustness (10 seeds, 300K steps)")
print("=" * 70)
report_path = '/Users/joe-caozhi/newGAE_ppo/results/MultiSeedPower/final_statistical_report_n10.json'
if os.path.exists(report_path):
    with open(report_path) as f:
        report = json.load(f)
    print("  Report found. Keys:", list(report.keys())[:10])
    # Show key values
    for env in ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']:
        if env in report:
            print(f"\n  {env}:")
            for method, vals in report[env].items():
                if isinstance(vals, dict):
                    mean = vals.get('mean', '?')
                    sem = vals.get('sem', vals.get('std', '?'))
                    print(f"    {method}: {mean} ± {sem}")
else:
    print(f"  Report NOT FOUND at {report_path}")
    # Try to find it
    ms_path = '/Users/joe-caozhi/newGAE_ppo/results/MultiSeedPower/'
    if os.path.exists(ms_path):
        print(f"  Files in {ms_path}: {os.listdir(ms_path)}")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)

