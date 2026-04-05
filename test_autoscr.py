"""Test that AutoSCR agent can be created and imported correctly."""
import sys
sys.path.insert(0, '/Users/joe-caozhi/newGAE_ppo')

from gae_experiments.agents.optimal_ppo import (
    build_optimal_agent
)
import gymnasium as gym

env = gym.make('Hopper-v4')
agent = build_optimal_agent('Optimal_HCGAE_v2_AutoSCR', env, name='test')
print(f'AutoSCR agent created: {agent.NAME}')
print(f'  scr_threshold={agent.scr_threshold}')
print(f'  scr_min_scale={agent.scr_min_scale}')
print(f'  scr_sharpness={agent.scr_sharpness}')
print(f'  use_auto_scr={agent.use_auto_scr}')
print(f'  use_ev_rate_gate={agent.use_ev_rate_gate}')
env.close()
print('Import test PASSED!')

