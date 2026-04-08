#!/usr/bin/env python3
"""Test that buffer.add() works correctly after fix."""
import sys; sys.path.insert(0, '.')
import gymnasium as gym
import torch
from gae_experiments.agents.optimal_ppo import build_optimal_agent

env = gym.make('HalfCheetah-v4')
env.reset(seed=0)
agent = build_optimal_agent('Optimal_PPO', env, name='test',
    hidden_dim=256, lr=3e-4, gamma=0.99, lam=0.95, eps_clip=0.2,
    n_epochs=10, batch_size=64, n_steps=2048, ent_coef=0.0, vf_coef=0.5,
    max_grad_norm=0.5, use_obs_norm=True, use_adv_norm=True,
    use_lr_anneal=True, use_vclip=False, device='cpu', save_dir='/tmp/test_agent')
obs, _ = env.reset()
obs_t = torch.FloatTensor(obs).unsqueeze(0)
with torch.no_grad():
    dist = agent.actor.forward(obs_t)
    act = dist.sample().squeeze(0).cpu().numpy()
    lp = dist.log_prob(torch.FloatTensor(act)).sum().item()
    val = agent.critic(obs_t).item()
nobs, r, term, trunc, _ = env.step(act)
agent.buffer.add(obs, act, r, float(term), lp, val)
print('SUCCESS: buffer.add() works correctly')
print(f'  act shape={act.shape}, lp={lp:.3f}, val={val:.3f}, r={r:.3f}, term={term}')

# Also test v4
agent_v4 = build_optimal_agent('Optimal_HCGAE_v4', env, name='test_v4',
    hidden_dim=256, lr=3e-4, gamma=0.99, lam=0.95, eps_clip=0.2,
    n_epochs=10, batch_size=64, n_steps=2048, ent_coef=0.0, vf_coef=0.5,
    max_grad_norm=0.5, use_obs_norm=True, use_adv_norm=True,
    use_lr_anneal=True, use_vclip=False, device='cpu', save_dir='/tmp/test_agent_v4')
obs2, _ = env.reset()
obs_t2 = torch.FloatTensor(obs2).unsqueeze(0)
with torch.no_grad():
    dist2 = agent_v4.actor.forward(obs_t2)
    act2 = dist2.sample().squeeze(0).cpu().numpy()
    lp2 = dist2.log_prob(torch.FloatTensor(act2)).sum().item()
    val2 = agent_v4.critic(obs_t2).item()
nobs2, r2, term2, trunc2, _ = env.step(act2)
agent_v4.buffer.add(obs2, act2, r2, float(term2), lp2, val2)
print('SUCCESS: HCGAE_v4 buffer.add() works correctly')
print(f'  v4 class={type(agent_v4).__name__}')

