#!/usr/bin/env python3
"""
Ant-v4 experiment: Standard_PPO vs Optimal_PPO vs Optimal_HCGAE vs Optimal_HCGAE_SCR
4 algorithms × 5 seeds × 500K steps

Uses exact same training logic as run_icml_experiment.py for consistency.
"""
import json
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from gae_experiments.agents.optimal_ppo import build_optimal_agent
from gae_experiments.agents.ppo_baselines import build_ppo_baseline

SEEDS = list(range(5))
TOTAL_TIMESTEPS = 500_000
EVAL_FREQ = 10_240
N_EVAL_EPISODES = 10
RESULTS_DIR = Path("results/ICMLExperiment/Ant-v4")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALGORITHMS = ["Standard_PPO", "Optimal_PPO", "Optimal_HCGAE", "Optimal_HCGAE_SCR"]

STANDARD_PPO_KWARGS = dict(
    hidden_dim=64, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, lam=0.95,
    eps_clip=0.2, n_epochs=10, batch_size=64, n_steps=2048,
    ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, device="cpu",
)
OPTIMAL_PPO_KWARGS = dict(
    hidden_dim=64, lr=3e-4, gamma=0.99, lam=0.95, eps_clip=0.2,
    n_epochs=10, batch_size=64, n_steps=2048, ent_coef=0.0, vf_coef=0.5,
    max_grad_norm=0.5, use_obs_norm=True, use_adv_norm=True,
    use_lr_anneal=True, use_vclip=False, device="cpu",
)


def evaluate_policy(agent, eval_env, n_episodes=10):
    rewards = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        if hasattr(agent, 'normalize_obs'):
            obs = agent.normalize_obs(obs)
        total = 0.0
        done = False
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                act, _ = agent.actor.get_action_and_logprob(obs_t)
                act_np = act.squeeze(0).cpu().numpy()
            obs, r, terminated, truncated, _ = eval_env.step(act_np)
            if hasattr(agent, 'normalize_obs'):
                obs = agent.normalize_obs(obs)
            total += r
            done = terminated or truncated
        rewards.append(total)
    return float(np.mean(rewards))


def run_single(algo_name, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    save_dir = str(RESULTS_DIR / algo_name)
    os.makedirs(save_dir, exist_ok=True)
    out_path = Path(save_dir) / f"{algo_name}_s{seed}.json"

    if out_path.exists():
        print(f"  [SKIP] {algo_name} s{seed} already done")
        return

    print(f"  [{algo_name} s{seed}] starting …")
    t0 = time.time()

    env = gym.make("Ant-v4")
    eval_env = gym.make("Ant-v4")
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 50000)

    # Build agent
    if algo_name == "Standard_PPO":
        kw = dict(**STANDARD_PPO_KWARGS, save_dir=save_dir)
        agent = build_ppo_baseline("Standard_PPO", env, name=f"{algo_name}_s{seed}", **kw)
    else:
        kw = dict(**OPTIMAL_PPO_KWARGS, save_dir=save_dir)
        agent = build_optimal_agent(algo_name, env, name=f"{algo_name}_s{seed}", **kw)

    # Training loop (identical to run_icml_experiment.py)
    eval_rewards = []
    eval_steps = []
    episode_rewards = []

    obs, _ = env.reset()
    if hasattr(agent, 'update_obs_rms'):
        agent.update_obs_rms(obs)
    if hasattr(agent, 'normalize_obs'):
        obs = agent.normalize_obs(obs)

    ep_reward = 0.0
    total_steps = 0
    last_eval_step = 0

    while total_steps < TOTAL_TIMESTEPS:
        agent.buffer.reset()
        for _ in range(agent.n_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, log_prob = agent.actor.get_action_and_logprob(obs_t)
                value = agent.critic(obs_t)

            action_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, _ = env.step(action_np)

            if hasattr(agent, 'update_obs_rms'):
                agent.update_obs_rms(next_obs)
            if hasattr(agent, 'normalize_obs'):
                next_obs_norm = agent.normalize_obs(next_obs)
            else:
                next_obs_norm = next_obs

            ep_reward += reward
            agent.buffer.add(obs, action_np, float(reward), float(terminated),
                             log_prob.item(), value.item())
            obs = next_obs_norm
            total_steps += 1

            if terminated or truncated:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                next_obs, _ = env.reset()
                if hasattr(agent, 'update_obs_rms'):
                    agent.update_obs_rms(next_obs)
                if hasattr(agent, 'normalize_obs'):
                    obs = agent.normalize_obs(next_obs)
                else:
                    obs = next_obs

            if total_steps - last_eval_step >= EVAL_FREQ:
                er = evaluate_policy(agent, eval_env, N_EVAL_EPISODES)
                eval_rewards.append(er)
                eval_steps.append(total_steps)
                last_eval_step = total_steps

        # Update: compute GAE then call update()
        with torch.no_grad():
            last_obs_t = torch.FloatTensor(obs).unsqueeze(0)
            last_val = agent.critic(last_obs_t).item()

        agent._total_timesteps = TOTAL_TIMESTEPS
        agent.total_steps = total_steps

        if hasattr(agent, 'compute_hindsight_gae'):
            agent.compute_hindsight_gae(last_val)
        else:
            agent.compute_gae(last_val)

        agent.update()

    # Final evaluation
    er = evaluate_policy(agent, eval_env, N_EVAL_EPISODES)
    eval_rewards.append(er)

    final_mean = float(np.mean(eval_rewards[-5:])) if len(eval_rewards) >= 5 else float(np.mean(eval_rewards))
    elapsed = time.time() - t0

    result = {
        "env": "Ant-v4", "agent": algo_name, "seed": seed,
        "total_steps": total_steps, "final_reward": final_mean,
        "eval_rewards": eval_rewards, "eval_steps": eval_steps,
        "episode_rewards": episode_rewards, "elapsed_s": elapsed,
    }
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  [{algo_name} s{seed}] done → {final_mean:.1f} ({elapsed:.0f}s)")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    print("=== Ant-v4 Experiment: 4 algos × 5 seeds × 500K steps ===")
    for algo in ALGORITHMS:
        for seed in SEEDS:
            run_single(algo, seed)
    print("All done!")

