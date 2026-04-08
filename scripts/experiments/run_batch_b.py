#!/usr/bin/env python3
"""Batch B: v2_NoGate ablation on Hopper-v4 and Walker2d-v4 (all 5 seeds each)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from run_missing_experiments import run_single

runs = [
    ("Hopper-v4", "Optimal_HCGAE_v2_NoGate", [0, 1, 2, 3, 4]),
    ("Walker2d-v4", "Optimal_HCGAE_v2_NoGate", [0, 1, 2, 3, 4]),
]

print("=== Batch B: v2_NoGate ablation (Hopper + Walker) ===")
total = sum(len(s) for _, _, s in runs)
done = 0
for env, algo, seeds in runs:
    print(f"\n--- {env}/{algo} ---")
    for seed in seeds:
        run_single(env, algo, seed)
        done += 1
        print(f"Progress: {done}/{total}")
print("\nBatch B complete!")

