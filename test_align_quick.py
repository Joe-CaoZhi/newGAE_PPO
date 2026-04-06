#!/usr/bin/env python3
"""Quick test to verify literature alignment changes."""
import gymnasium as gym

# Test 1: BasePPO with obs norm
from gae_experiments.agents.base_ppo import BasePPO
env = gym.make('Hopper-v4')
agent = BasePPO(env, save_dir='/tmp/test_ppo')
assert agent.obs_rms is not None, 'BasePPO should have obs_rms'
assert agent.use_obs_norm == True
assert agent.use_adv_norm == True
obs, _ = env.reset()
agent.update_obs_rms(obs)
obs_norm = agent.normalize_obs(obs)
assert obs_norm.shape == obs.shape
env.close()
print('✓ BasePPO: obs_rms OK, use_obs_norm=True, use_adv_norm=True')

# Test 2: PPOBaseline with obs norm
from gae_experiments.agents.ppo_baselines import build_ppo_baseline
env2 = gym.make('Hopper-v4')
agent2 = build_ppo_baseline('Standard_PPO', env2, save_dir='/tmp/test_baseline')
assert agent2.obs_rms is not None, 'PPOBaseline should have obs_rms'
assert agent2.use_adv_norm == True
env2.close()
print('✓ PPOBaseline(Standard_PPO): obs_rms OK, use_adv_norm=True')

# Test 3: OptimalPPO
from gae_experiments.agents.optimal_ppo import build_optimal_agent
env3 = gym.make('Hopper-v4')
agent3 = build_optimal_agent('Optimal_PPO', env3, save_dir='/tmp/test_opt')
assert agent3.obs_rms is not None
env3.close()
print('✓ OptimalPPO: obs_rms OK')

# Test 4: OptimalHCGAE_v2
env4 = gym.make('Hopper-v4')
agent4 = build_optimal_agent('Optimal_HCGAE_v2', env4, save_dir='/tmp/test_v2')
assert agent4.obs_rms is not None
env4.close()
print('✓ OptimalHCGAE_v2: obs_rms OK (inherited from OptimalPPO)')

# Test 5: HindsightAblation
from gae_experiments.agents.hindsight_ablation import build_ablation_agent
env5 = gym.make('Hopper-v4')
agent5 = build_ablation_agent('HCGAE_Imp12', env5, save_dir='/tmp/test_abl')
assert agent5.obs_rms is not None, 'HindsightAblation should have obs_rms'
assert agent5.use_adv_norm == True
env5.close()
print('✓ HindsightAblation(HCGAE_Imp12): obs_rms OK, use_adv_norm=True')

# Test 6: Network size check
import torch
from gae_experiments.utils.networks import ActorNetwork, CriticNetwork
actor = ActorNetwork(11, 3, 256, True)
critic = CriticNetwork(11, 256)
# Count parameters
n_actor = sum(p.numel() for p in actor.parameters())
n_critic = sum(p.numel() for p in critic.parameters())
print(f'✓ Network: Actor params={n_actor:,}, Critic params={n_critic:,}')
# With 256 hidden dim, expect much more params than 64
assert n_actor > 10000, f'Expected >10k params, got {n_actor}'
print(f'  (For reference: 64-dim actor would have ~1500 params)')

print()
print('=== All tests PASSED! Literature alignment changes verified. ===')

