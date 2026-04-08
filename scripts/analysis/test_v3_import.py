#!/usr/bin/env python3
"""Quick smoke test for HCGAE v3 agent instantiation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import gymnasium as gym
from gae_experiments.agents.optimal_ppo import build_optimal_agent

env = gym.make("Ant-v4")

# Test all v3 variants
variants = [
    "Optimal_HCGAE_v3",
    "Optimal_HCGAE_v3_NoClamp",
    "Optimal_HCGAE_v3_NoVWGate",
    "Optimal_HCGAE_v3_NoBdryPrior",
]

for v in variants:
    agent = build_optimal_agent(v, env)
    print(f"OK: {agent.NAME}  g_clamp={agent.use_g_clamp}  vw_gate={agent.use_vw_gate}  bdry_prior={agent.use_boundary_prior}")

env.close()
print("\nAll v3 variants instantiated successfully!")

