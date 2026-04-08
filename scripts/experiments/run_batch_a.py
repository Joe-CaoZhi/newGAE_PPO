#!/usr/bin/env python3
"""Batch A: Complete Ant-v4 experiments (Optimal_HCGAE_SCR s3,s4 + Optimal_HCGAE_v2 s3,s4)
   and Walker2d-v4 Optimal_HCGAE_v2 (s1..s4)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from run_missing_experiments import run_single

runs = [
    # Ant-v4 completions
    ("Ant-v4", "Optimal_HCGAE_SCR", [3, 4]),
    ("Ant-v4", "Optimal_HCGAE_v2", [3, 4]),
    # Walker2d-v4 Optimal_HCGAE_v2
    ("Walker2d-v4", "Optimal_HCGAE_v2", [1, 2, 3, 4]),
]

print("=== Batch A: Core completions ===")
total = sum(len(s) for _, _, s in runs)
done = 0
for env, algo, seeds in runs:
    print(f"\n--- {env}/{algo} ---")
    for seed in seeds:
        run_single(env, algo, seed)
        done += 1
        print(f"Progress: {done}/{total}")
print("\nBatch A complete!")

