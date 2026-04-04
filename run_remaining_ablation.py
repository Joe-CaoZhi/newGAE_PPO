#!/usr/bin/env python3
"""
Run remaining ablation experiments:
- Hopper-v4 / Optimal_HCGAE_v2_NoBdry: seeds 2,3,4
- Walker2d-v4 / Optimal_HCGAE_v2_NoBdry: seeds 0,1,2,3,4
- Walker2d-v4 / Optimal_HCGAE_v2_NoGate: seeds 2,3,4
- Ant-v4 / Optimal_HCGAE_v2_NoGate: seed 4
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from run_missing_experiments import run_single

TASKS = [
    # Hopper NoBdry seeds 2,3,4
    ("Hopper-v4", "Optimal_HCGAE_v2_NoBdry", 2),
    ("Hopper-v4", "Optimal_HCGAE_v2_NoBdry", 3),
    ("Hopper-v4", "Optimal_HCGAE_v2_NoBdry", 4),
    # Walker NoBdry all seeds
    ("Walker2d-v4", "Optimal_HCGAE_v2_NoBdry", 0),
    ("Walker2d-v4", "Optimal_HCGAE_v2_NoBdry", 1),
    ("Walker2d-v4", "Optimal_HCGAE_v2_NoBdry", 2),
    ("Walker2d-v4", "Optimal_HCGAE_v2_NoBdry", 3),
    ("Walker2d-v4", "Optimal_HCGAE_v2_NoBdry", 4),
    # Walker NoGate seeds 2,3,4
    ("Walker2d-v4", "Optimal_HCGAE_v2_NoGate", 2),
    ("Walker2d-v4", "Optimal_HCGAE_v2_NoGate", 3),
    ("Walker2d-v4", "Optimal_HCGAE_v2_NoGate", 4),
    # Ant NoGate seed 4
    ("Ant-v4", "Optimal_HCGAE_v2_NoGate", 4),
]

print(f"Running {len(TASKS)} experiments...")
for env, algo, seed in TASKS:
    run_single(env, algo, seed)

print("All done!")

