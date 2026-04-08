#!/usr/bin/env python3
import gymnasium as gym

from gae_experiments.agents.optimal_ppo import build_optimal_agent

env = gym.make('HalfCheetah-v4')
for name in ['Optimal_PPO', 'Optimal_HCGAE_v2', 'Optimal_HCGAE_v4', 'Optimal_HCGAE_v5', 'Optimal_HCGAE_v6']:
    try:
        agent = build_optimal_agent(name, env)
        print(f"OK: {agent.NAME}")
    except Exception as e:
        print(f"FAIL {name}: {e}")
env.close()
print("All done.")

