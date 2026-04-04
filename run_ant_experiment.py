#!/usr/bin/env python3
"""
Ant-v4 experiment: Standard_PPO vs Optimal_PPO vs Optimal_HCGAE vs Optimal_HCGAE_SCR
4 algorithms × 5 seeds × 500K steps
"""
import json
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gae_experiments.agents.optimal_ppo import build_optimal_agent
from gae_experiments.agents.ppo_baselines import build_ppo_baseline

SEEDS = list(range(5))      # 0..4
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
HCGAE_EXTRA = dict(hindsight_beta=3.0, hindsight_alpha_max=0.7, hindsight_alpha_min=0.1)

def build_agent(algo_name, env, seed, save_dir):
    np.random.seed(seed)
    import torch; torch.manual_seed(seed)
    if algo_name == "Standard_PPO":
        return build_ppo_baseline("Standard_PPO", env, save_dir=str(save_dir), **STANDARD_PPO_KWARGS)
    kw = dict(**OPTIMAL_PPO_KWARGS, save_dir=str(save_dir))
    if algo_name == "Optimal_PPO":
        return build_optimal_agent("Optimal_PPO", env, **kw)
    if algo_name == "Optimal_HCGAE":
        return build_optimal_agent("Optimal_HCGAE", env, **kw, **HCGAE_EXTRA)
    if algo_name == "Optimal_HCGAE_SCR":
        return build_optimal_agent("Optimal_HCGAE_SCR", env, **kw, **HCGAE_EXTRA, scr_threshold=1.0)
    raise ValueError(f"Unknown algo: {algo_name}")

def evaluate(agent, n_episodes=10):
    env = agent.env
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0.0
        done = False
        while not done:
            if hasattr(agent, 'obs_rms') and agent.use_obs_norm:
                obs_n = agent.obs_rms.normalize(obs[None])[0]
            else:
                obs_n = obs
            import torch
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs_n).unsqueeze(0)
                act, _ = agent.actor.get_action_and_logprob(obs_t)
                act_np = act.squeeze(0).cpu().numpy()
            obs, r, terminated, truncated, _ = env.step(act_np)
            total += r
            done = terminated or truncated
        rewards.append(total)
    return float(np.mean(rewards))

def run_single(algo_name, seed):
    out_dir = RESULTS_DIR / algo_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{algo_name}_s{seed}.json"
    if out_file.exists():
        print(f"  [SKIP] {algo_name} s{seed} already done")
        return

    print(f"  [{algo_name} s{seed}] starting …")
    t0 = time.time()
    env = gym.make("Ant-v4")
    env.reset(seed=seed)
    save_dir = out_dir
    agent = build_agent(algo_name, env, seed, save_dir)

    eval_rewards = []
    last_eval_step = 0

    while agent.total_steps < TOTAL_TIMESTEPS:
        last_value = agent.collect_rollout()
        gae_data   = agent.compute_gae(last_value)
        agent.update(gae_data)

        if agent.total_steps - last_eval_step >= EVAL_FREQ:
            r = evaluate(agent, N_EVAL_EPISODES)
            eval_rewards.append(r)
            last_eval_step = agent.total_steps

    # final eval
    r = evaluate(agent, N_EVAL_EPISODES)
    eval_rewards.append(r)
    final_reward = float(np.mean(eval_rewards[-5:])) if len(eval_rewards) >= 5 else float(np.mean(eval_rewards))

    result = {
        "algo": algo_name, "seed": seed, "env": "Ant-v4",
        "total_steps": agent.total_steps, "final_reward": final_reward,
        "eval_rewards": eval_rewards, "wall_time_s": time.time() - t0,
    }
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  [{algo_name} s{seed}] done → {final_reward:.1f}  ({time.time()-t0:.0f}s)")
    env.close()

if __name__ == "__main__":
    print("=== Ant-v4 Experiment: 4 algos × 5 seeds × 500K steps ===")
    for algo in ALGORITHMS:
        for seed in SEEDS:
            run_single(algo, seed)
    print("All done!")

