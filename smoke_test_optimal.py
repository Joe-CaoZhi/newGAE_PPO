#!/usr/bin/env python3
"""Quick smoke test for OptimalPPO and OptimalHCGAE."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gymnasium as gym
import torch
from gae_experiments.agents.optimal_ppo import build_optimal_agent

env = gym.make('Hopper-v4')
for algo in ['Optimal_PPO', 'Optimal_HCGAE', 'Optimal_HCGAE_SCR']:
    agent = build_optimal_agent(algo, env, name=f'{algo}_test', save_dir='/tmp/test_opt')
    obs, _ = env.reset(seed=0)
    agent.update_obs_rms(obs)
    obs = agent.normalize_obs(obs)
    agent.buffer.reset()
    for _ in range(agent.n_steps):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action, lp = agent.actor.get_action_and_logprob(obs_t)
            val = agent.critic(obs_t)
        action_np = action.squeeze(0).cpu().numpy()
        next_obs, r, term, trunc, _ = env.step(action_np)
        agent.update_obs_rms(next_obs)
        next_obs = agent.normalize_obs(next_obs)
        agent.buffer.add(obs, action_np, float(r), float(term), lp.item(), val.item())
        obs = next_obs
        if term or trunc:
            obs, _ = env.reset()
            agent.update_obs_rms(obs)
            obs = agent.normalize_obs(obs)
    with torch.no_grad():
        last_v = agent.critic(torch.FloatTensor(obs).unsqueeze(0)).item()
    agent._total_timesteps = 500000
    agent.total_steps = 2048
    if hasattr(agent, 'compute_hindsight_gae'):
        agent.compute_hindsight_gae(last_v)
    else:
        agent.compute_gae(last_v)
    metrics = agent.update()
    print(f'[OK] {algo}: policy_loss={metrics["policy_loss"]:.4f}  value_loss={metrics["value_loss"]:.4f}  ev={metrics["explained_variance"]:.3f}')
env.close()
print('All smoke tests passed!')

