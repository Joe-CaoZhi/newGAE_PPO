#!/usr/bin/env python3
"""
HCGAE v2 Validation Experiment
================================
Compare Optimal_HCGAE_v2 vs Optimal_PPO vs Optimal_HCGAE (v1)

Target environments:
  - HalfCheetah-v4  (v1 was significantly worse, v2 aims to fix)
  - Hopper-v4       (v1 was +9.6%, v2 should maintain)
  - Walker2d-v4     (v1 was +17.3%, v2 should maintain)
  - Ant-v4          (supplement experiment)

5 seeds × 500K steps per algorithm
"""
import argparse
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

# ── Config ──────────────────────────────────────────────────────────
SEEDS = list(range(5))
TOTAL_TIMESTEPS = 500_000
EVAL_FREQ = 10_240
N_EVAL_EPISODES = 10
RESULTS_DIR = Path("results/ICMLExperiment")

ALGORITHMS = [
    "Optimal_PPO",
    "Optimal_HCGAE",
    "Optimal_HCGAE_v2",
]

OPTIMAL_PPO_KWARGS = dict(
    hidden_dim=256, lr=3e-4, gamma=0.99, lam=0.95, eps_clip=0.2,   # ★ 256x256 MLP
    n_epochs=10, batch_size=64, n_steps=2048, ent_coef=0.0, vf_coef=0.5,
    max_grad_norm=0.5, use_obs_norm=True, use_adv_norm=True,
    use_lr_anneal=True, use_vclip=False, device="cpu",
)

STANDARD_PPO_KWARGS = dict(
    hidden_dim=256, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, lam=0.95,  # ★ 256x256 MLP
    eps_clip=0.2, n_epochs=10, batch_size=64, n_steps=2048,
    ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
    use_obs_norm=True, use_adv_norm=True,  # ★ Obs norm + adv norm
    device="cpu",
)


def evaluate_policy(agent, eval_env, n_episodes=10):
    rewards = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        if hasattr(agent, 'normalize_obs') and agent.obs_rms is not None:
            obs = agent.normalize_obs(obs)
        total = 0.0
        done = False
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                act, _ = agent.actor.get_action_and_logprob(obs_t)
                act_np = act.squeeze(0).cpu().numpy()
            obs, r, terminated, truncated, _ = eval_env.step(act_np)
            if hasattr(agent, 'normalize_obs') and agent.obs_rms is not None:
                obs = agent.normalize_obs(obs)
            total += r
            done = terminated or truncated
        rewards.append(total)
    return float(np.mean(rewards))


def run_single(env_name, algo_name, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    save_dir = str(RESULTS_DIR / env_name / algo_name)
    os.makedirs(save_dir, exist_ok=True)
    out_path = Path(save_dir) / f"{algo_name}_s{seed}.json"

    if out_path.exists():
        print(f"  [SKIP] {env_name}/{algo_name} s{seed} already done")
        return

    print(f"  [{env_name}/{algo_name} s{seed}] starting …")
    t0 = time.time()

    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 100)

    if algo_name == "Standard_PPO":
        agent = build_ppo_baseline("Standard_PPO", env, name="Standard_PPO",
                                   save_dir=save_dir, **STANDARD_PPO_KWARGS)
    else:
        agent = build_optimal_agent(algo_name, env, name=algo_name,
                                    save_dir=save_dir, **OPTIMAL_PPO_KWARGS)

    agent._total_timesteps = TOTAL_TIMESTEPS

    obs, _ = env.reset(seed=seed)
    if hasattr(agent, 'normalize_obs') and agent.obs_rms is not None:
        agent.update_obs_rms(obs)
        obs = agent.normalize_obs(obs)

    eval_rewards = []
    eval_steps = []
    ep_rewards = []
    ep_rew_current = 0.0

    step = 0
    while step < TOTAL_TIMESTEPS:
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            if algo_name == "Standard_PPO":
                act, lp, val = agent.actor.get_action_logprob_value(obs_t)
            else:
                act, lp = agent.actor.get_action_and_logprob(obs_t)
                val = agent.critic(obs_t)
            act_np = act.squeeze(0).cpu().numpy()

        next_obs, reward, terminated, truncated, _ = env.step(act_np)
        done = terminated or truncated
        ep_rew_current += reward

        if hasattr(agent, 'normalize_obs') and agent.obs_rms is not None:
            agent.update_obs_rms(next_obs)
            next_obs_norm = agent.normalize_obs(next_obs)
        else:
            next_obs_norm = next_obs

        if algo_name == "Standard_PPO":
            agent.buffer.add(obs, act_np, lp.item(), reward, terminated, val.item())
        else:
            agent.buffer.add(obs, act_np, lp.item(), reward, terminated, val.item())

        obs = next_obs_norm

        if done:
            ep_rewards.append(ep_rew_current)
            ep_rew_current = 0.0
            obs, _ = env.reset()
            if hasattr(agent, 'normalize_obs') and agent.obs_rms is not None:
                agent.update_obs_rms(obs)
                obs = agent.normalize_obs(obs)

        step += 1
        agent.total_steps = step

        if agent.buffer.pos == agent.buffer.buffer_size:
            # Compute bootstrap value
            with torch.no_grad():
                obs_t_last = torch.FloatTensor(obs).unsqueeze(0)
                last_val = agent.critic(obs_t_last).item()
            if done:
                last_val = 0.0
            agent.compute_gae(last_val)
            agent.update()
            agent.buffer.reset()

        # Evaluate
        if step % EVAL_FREQ == 0:
            er = evaluate_policy(agent, eval_env, N_EVAL_EPISODES)
            eval_rewards.append(er)
            eval_steps.append(step)
            elapsed = time.time() - t0
            print(f"    step={step:>7d}  eval_r={er:>8.1f}  t={elapsed:.0f}s")

    total_time = time.time() - t0
    final_mean = float(np.mean(eval_rewards[-5:])) if len(eval_rewards) >= 5 else (
        float(np.mean(eval_rewards)) if eval_rewards else 0.0
    )

    result = {
        "env": env_name,
        "algo": algo_name,
        "seed": seed,
        "total_steps": step,
        "total_time_s": total_time,
        "eval_steps": eval_steps,
        "eval_rewards": eval_rewards,
        "episode_rewards": ep_rewards,
        "final_mean_return": final_mean,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  [{env_name}/{algo_name} s{seed}] DONE  final={final_mean:.1f}  "
          f"time={total_time:.0f}s → {out_path}")
    env.close()
    eval_env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+",
                        default=["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"])
    parser.add_argument("--algos", nargs="+", default=ALGORITHMS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    print(f"HCGAE v2 Validation Experiment")
    print(f"Environments: {args.envs}")
    print(f"Algorithms:   {args.algos}")
    print(f"Seeds:        {args.seeds}")
    print(f"Steps:        {TOTAL_TIMESTEPS}")
    print("=" * 60)

    total_start = time.time()
    for env_name in args.envs:
        for algo_name in args.algos:
            for seed in args.seeds:
                try:
                    run_single(env_name, algo_name, seed)
                except Exception as e:
                    print(f"  [ERROR] {env_name}/{algo_name} s{seed}: {e}")
                    import traceback
                    traceback.print_exc()

    print(f"\nAll done in {(time.time()-total_start)/60:.1f} min")

    # Quick summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for env_name in args.envs:
        print(f"\n{env_name}:")
        for algo_name in args.algos:
            save_dir = RESULTS_DIR / env_name / algo_name
            finals = []
            for seed in args.seeds:
                fp = save_dir / f"{algo_name}_s{seed}.json"
                if fp.exists():
                    with open(fp) as f:
                        d = json.load(f)
                    finals.append(d.get("final_mean_return", 0.0))
            if finals:
                print(f"  {algo_name:30s}: {np.mean(finals):.1f} ± {np.std(finals):.1f} "
                      f"(n={len(finals)})")
            else:
                print(f"  {algo_name:30s}: [NO DATA]")


if __name__ == "__main__":
    main()

