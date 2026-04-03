#!/usr/bin/env python3
"""Quick smoke test - 5K steps only"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gymnasium as gym

# Test 1: DCPPO-S init
from gae_experiments.agents.dcppo import DCPPO
env = gym.make('Hopper-v4')
env.reset(seed=42)
agent = DCPPO(
    env=env, name='test_dcppo',
    use_imp_g=False, use_imp_a=False, use_imp_s=True,
    use_hcgae=True, snr_target=0.3, snr_gamma=0.5, snr_min_weight=0.2,
    hidden_dim=64, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, lam=0.95,
    eps_clip=0.2, n_epochs=1, batch_size=64, n_steps=2048,
    save_dir='/tmp')
eval_env = gym.make('Hopper-v4')
logger = agent.train(
    total_timesteps=5000, eval_env=eval_env, eval_freq=2048,
    n_eval_episodes=2, verbose=True)
print(f"[PASS] DCPPO-S smoke test, final={logger.eval_rewards[-1]:.0f}")
env.close(); eval_env.close()

# Test 2: HindsightAblation sensitivity params
from gae_experiments.agents.hindsight_ablation import HindsightAblation
env2 = gym.make('Hopper-v4')
env2.reset(seed=42)
agent2 = HindsightAblation(
    env=env2, name='test_hcgae',
    use_imp1=True, use_imp2=True, use_imp3=False, use_imp4=False,
    hindsight_beta=4.0, hindsight_alpha_max=0.9,
    hidden_dim=64, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, lam=0.95,
    eps_clip=0.2, n_epochs=1, batch_size=64, n_steps=2048,
    save_dir='/tmp')
eval_env2 = gym.make('Hopper-v4')
logger2 = agent2.train(
    total_timesteps=5000, eval_env=eval_env2, eval_freq=2048,
    n_eval_episodes=2, verbose=False)
print(f"[PASS] HindsightAblation sensitivity smoke test, final={logger2.eval_rewards[-1] if logger2.eval_rewards else 'N/A'}")
env2.close(); eval_env2.close()

print("\nAll smoke tests passed!")

