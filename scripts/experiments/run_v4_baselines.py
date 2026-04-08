#!/usr/bin/env python3
"""
在与V4FullExperiment相同的训练循环（update后eval）下
重新运行 Optimal_PPO 和 Optimal_HCGAE_v2 作为公平对比基准

这确保所有算法的eval timing完全一致：
  collect rollout → compute_gae → update → EVAL

结果保存到 results/V4FullExperiment/（与v4实验同目录）
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

# ─── Config（与run_v4_full.py完全相同）────────────────────────────
ENVS        = ["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4"]
ALGOS       = ["Optimal_PPO", "Optimal_HCGAE_v2"]   # 仅重新运行基准
SEEDS       = list(range(5))
TOTAL_STEPS = 500_000
EVAL_FREQ   = 10_240
N_EVAL_EPS  = 10
RESULTS_DIR = Path("results/V4FullExperiment")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AGENT_KWARGS = dict(
    hidden_dim=256, lr=3e-4, gamma=0.99, lam=0.95, eps_clip=0.2,
    n_epochs=10, batch_size=64, n_steps=2048, ent_coef=0.0,
    vf_coef=0.5, max_grad_norm=0.5, use_obs_norm=True,
    use_adv_norm=True, use_lr_anneal=True, use_vclip=False, device="cpu",
)

# ─── Evaluation（与run_v4_full.py完全相同）──────────────────────
def evaluate_policy(agent, eval_env, n_eps=N_EVAL_EPS):
    rewards = []
    for _ in range(n_eps):
        obs, _ = eval_env.reset()
        if hasattr(agent, 'normalize_obs'):
            obs = agent.normalize_obs(obs)
        total, done = 0.0, False
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                dist = agent.actor.forward(obs_t)
                if agent.continuous:
                    act = dist.mean.squeeze(0).detach().cpu().numpy()
                else:
                    act = int(dist.probs.argmax(dim=-1).squeeze(0).detach().cpu().numpy())
            obs, r, term, trunc, _ = eval_env.step(act)
            if hasattr(agent, 'normalize_obs'):
                obs = agent.normalize_obs(obs)
            total += r
            done = term or trunc
        rewards.append(total)
    return float(np.mean(rewards))

# ─── Single run（与run_v4_full.py完全相同的训练循环）─────────────
def run_single(env_name, algo_name, seed):
    save_dir = str(RESULTS_DIR / env_name / algo_name)
    os.makedirs(save_dir, exist_ok=True)
    out_path = Path(save_dir) / f"{algo_name}_s{seed}.json"
    if out_path.exists():
        print(f"  [SKIP] {env_name}/{algo_name}/s{seed}")
        return json.load(open(out_path))

    np.random.seed(seed); torch.manual_seed(seed)
    env      = gym.make(env_name); env.reset(seed=seed)
    eval_env = gym.make(env_name); eval_env.reset(seed=seed + 50000)

    kw = dict(**AGENT_KWARGS, save_dir=save_dir)
    agent = build_optimal_agent(algo_name, env, name=f"{algo_name}_s{seed}", **kw)
    agent._total_timesteps = TOTAL_STEPS

    eval_rewards, eval_steps = [], []
    obs, _ = env.reset()
    if hasattr(agent, 'update_obs_rms'): agent.update_obs_rms(obs)
    if hasattr(agent, 'normalize_obs'):  obs = agent.normalize_obs(obs)

    ep_reward, total_steps, last_eval = 0.0, 0, 0
    t0 = time.time()

    while total_steps < TOTAL_STEPS:
        # rollout
        agent.buffer.reset()
        for _ in range(agent.n_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                if agent.continuous:
                    dist = agent.actor.forward(obs_t)
                    act = dist.sample().squeeze(0).cpu().numpy()
                    lp  = dist.log_prob(torch.FloatTensor(act)).sum().item()
                else:
                    dist = agent.actor.forward(obs_t)
                    act_t = dist.sample()
                    act = int(act_t.item())
                    lp  = dist.log_prob(act_t).item()
                val = agent.critic(obs_t).item()

            nobs, r, term, trunc, _ = env.step(act)
            done = term or trunc
            ep_reward += r
            if hasattr(agent, 'update_obs_rms'): agent.update_obs_rms(nobs)
            if hasattr(agent, 'normalize_obs'):  nobs_n = agent.normalize_obs(nobs)
            else: nobs_n = nobs

            agent.buffer.add(obs, act, r, float(term), lp, val)
            obs = nobs_n
            total_steps += 1

            if done:
                obs_raw, _ = env.reset()
                if hasattr(agent, 'update_obs_rms'): agent.update_obs_rms(obs_raw)
                if hasattr(agent, 'normalize_obs'):  obs = agent.normalize_obs(obs_raw)
                else: obs = obs_raw
                ep_reward = 0.0

            if total_steps >= TOTAL_STEPS:
                break

        # GAE + update（与V4相同：update后eval）
        with torch.no_grad():
            obs_t  = torch.FloatTensor(obs).unsqueeze(0)
            last_v = agent.critic(obs_t).item()
        agent.total_steps = total_steps
        agent.compute_gae(last_v)
        agent.update()

        # Eval（在update之后！与V4对齐）
        if total_steps - last_eval >= EVAL_FREQ or total_steps >= TOTAL_STEPS:
            er = evaluate_policy(agent, eval_env)
            eval_rewards.append(er)
            eval_steps.append(total_steps)
            last_eval = total_steps
            elapsed = time.time() - t0
            pct = 100 * total_steps / TOTAL_STEPS
            print(f"  {env_name}/{algo_name}/s{seed}  {pct:.0f}%  eval={er:.1f}  ({elapsed:.0f}s)")

    result = {
        "env": env_name, "agent": algo_name, "seed": seed,
        "config": {"hidden_dim": 256, "use_obs_norm": True, "use_adv_norm": True,
                   "use_lr_anneal": True, "eval_mode": "deterministic_mean_post_update"},
        "total_steps": total_steps,
        "final_reward": float(np.mean(eval_rewards[-5:])) if len(eval_rewards) >= 5 else float(np.mean(eval_rewards)),
        "eval_rewards": eval_rewards, "eval_steps": eval_steps,
        "elapsed_s": round(time.time() - t0, 1),
    }
    json.dump(result, open(out_path, 'w'), indent=2)
    print(f"  => Saved {out_path}  final5={result['final_reward']:.1f}")
    return result

# ─── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",  default=None)
    parser.add_argument("--algo", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    envs  = [args.env]  if args.env  else ENVS
    algos = [args.algo] if args.algo else ALGOS
    seeds = [args.seed] if args.seed is not None else SEEDS

    print(f"V4 Baselines: {envs} × {algos} × seeds={seeds} × {TOTAL_STEPS//1000}K steps")
    print("Eval timing: collect → update → EVAL (same as V4FullExperiment)")
    for env_name in envs:
        for algo in algos:
            for s in seeds:
                run_single(env_name, algo, s)

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Baseline Summary (last-5-eval mean ± SEM):")
    print("="*60)
    for env_name in envs:
        print(f"\n{env_name}:")
        ppo_vals = []
        for algo in algos + ['Optimal_HCGAE_v4']:
            vals = []
            for s in seeds:
                p = RESULTS_DIR / env_name / algo / f"{algo}_s{s}.json"
                if p.exists():
                    d = json.load(open(p))
                    er = d.get('eval_rewards', [])
                    if er:
                        vals.append(float(np.mean(er[-5:])))
            if vals:
                m, sem = np.mean(vals), np.std(vals)/max(np.sqrt(len(vals)), 1)
                delta = ""
                if algo != "Optimal_PPO" and ppo_vals:
                    d_pct = (m - np.mean(ppo_vals)) / max(abs(np.mean(ppo_vals)), 1) * 100
                    delta = f"  Δ vs PPO={d_pct:+.1f}%"
                print(f"  {algo:35s}: {m:7.1f} ± {sem:5.1f}  (n={len(vals)}){delta}")
                if algo == "Optimal_PPO":
                    ppo_vals = vals

