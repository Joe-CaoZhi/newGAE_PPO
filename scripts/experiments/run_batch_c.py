#!/usr/bin/env python3
"""Batch C: v2_NoGate ablation on HalfCheetah-v4 and Ant-v4 (all 5 seeds each)
   Also v2_NoBdry completions: Hopper, Walker2d (all 5 seeds), HalfCheetah s4, Ant s3,s4"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from run_missing_experiments import run_single

runs = [
    ("HalfCheetah-v4", "Optimal_HCGAE_v2_NoGate", [0, 1, 2, 3, 4]),
    ("Ant-v4", "Optimal_HCGAE_v2_NoGate", [0, 1, 2, 3, 4]),
    # NoBdry completions
    ("Hopper-v4", "Optimal_HCGAE_v2_NoBdry", [0, 1, 2, 3, 4]),
    ("Walker2d-v4", "Optimal_HCGAE_v2_NoBdry", [0, 1, 2, 3, 4]),
    ("HalfCheetah-v4", "Optimal_HCGAE_v2_NoBdry", [4]),
    ("Ant-v4", "Optimal_HCGAE_v2_NoBdry", [3, 4]),
]

print("=== Batch C: v2_NoGate (HC + Ant) + NoBdry completions ===")
total = sum(len(s) for _, _, s in runs)
done = 0
for env, algo, seeds in runs:
    print(f"\n--- {env}/{algo} ---")
    for seed in seeds:
        run_single(env, algo, seed)
        done += 1
        print(f"Progress: {done}/{total}")
print("\nBatch C complete!")

