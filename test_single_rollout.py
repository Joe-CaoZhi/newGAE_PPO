"""Quick test: run 1 rollout of AutoSCR to verify correctness."""
import os
import sys

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, '/Users/joe-caozhi/newGAE_ppo')
from gae_experiments.agents.optimal_ppo import build_optimal_agent

env_name = 'Hopper-v4'
algo_name = 'Optimal_HCGAE_v2_AutoSCR'
seed = 0
save_dir = '/Users/joe-caozhi/newGAE_ppo/results/AutoSCRExperiment/test'
os.makedirs(save_dir, exist_ok=True)

env = gym.make(env_name)
eval_env = gym.make(env_name)
env.reset(seed=seed)
eval_env.reset(seed=seed+50000)

kw = dict(hidden_dim=64, lr=3e-4, gamma=0.99, lam=0.95, eps_clip=0.2,
          n_epochs=10, batch_size=64, n_steps=2048, ent_coef=0.0, vf_coef=0.5,
          max_grad_norm=0.5, use_obs_norm=True, use_adv_norm=True,
          use_lr_anneal=True, use_vclip=False, device='cpu', save_dir=save_dir)

agent = build_optimal_agent(algo_name, env, name='test', **kw)
print(f'Agent created: {agent.NAME}')
print(f'  use_auto_scr={agent.use_auto_scr}, use_ev_rate_gate={agent.use_ev_rate_gate}')

obs, _ = env.reset()
if hasattr(agent, 'update_obs_rms'): agent.update_obs_rms(obs)
if hasattr(agent, 'normalize_obs'): obs = agent.normalize_obs(obs)

agent.buffer.reset()
for i in range(agent.n_steps):
    obs_t = torch.FloatTensor(obs).unsqueeze(0)
    with torch.no_grad():
        action, lp = agent.actor.get_action_and_logprob(obs_t)
        val = agent.critic(obs_t)
    a = action.squeeze(0).cpu().numpy()
    next_obs, r, term, trunc, _ = env.step(a)
    if hasattr(agent, 'update_obs_rms'): agent.update_obs_rms(next_obs)
    next_obs_n = agent.normalize_obs(next_obs) if hasattr(agent, 'normalize_obs') else next_obs
    agent.buffer.add(obs, a, float(r), float(term), lp.item(), val.item())
    obs = next_obs_n
    if term or trunc:
        next_obs, _ = env.reset()
        if hasattr(agent, 'update_obs_rms'): agent.update_obs_rms(next_obs)
        obs = agent.normalize_obs(next_obs) if hasattr(agent, 'normalize_obs') else next_obs

with torch.no_grad():
    last_val = agent.critic(torch.FloatTensor(obs).unsqueeze(0)).item()

agent._total_timesteps = 10000
agent.total_steps = agent.n_steps
agent.compute_hindsight_gae(last_val)
m = agent.update()

print(f'Update done. EV={agent._ev_ema:.3f}')
print(f'SCR history len: {len(agent._scr_history)}')
if agent._scr_history:
    print(f'SCR mean: {np.mean(agent._scr_history):.3f}')
    print(f'SCR scale mean: {np.mean(agent._scr_scale_history):.3f}')
print('Single rollout test PASSED!')
env.close()
eval_env.close()

