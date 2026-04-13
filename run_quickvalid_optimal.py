#!/usr/bin/env python3
"""
HCGAE_Optimal 快速验证实验
===========================
验证三学科统一推导的最完整理论实现 (Optimal) 相对于 v4 和 Optimal_PPO 的性能。

理论改进摘要:
  改进1 (FixSCR 分母修正):
    v4:      SCR = |B| / std(G)                [高估 σ_G]
    Optimal: SCR = |B| / sqrt(Var(G) - Var(V)) [全方差定律修正]
    依据: Var(G) = Var(V*) + E[Var(G|s)], Var(V) ≈ Var(V*) when Critic good
    效果: α_cap 提升 1.5-2× (Walker/Hopper 获益最大)

  改进2 (MC 归一化 per-step sigmoid):
    v4:      z_t = β·(|δ_t| - μ_e) / σ_e  [批内 z-score, 强制 50% 修正]
    Optimal: z_t = β·(|δ_t| / σ_G_ema - 0.5)  [信噪比感知]
    依据: per-step α_t* ∝ SNR_t = |δ_t| / σ_G (卡尔曼局部最优)
    效果: Critic 好时整批 α→0，避免无效修正

对比组: Optimal_PPO, Optimal_HCGAE_v4, Optimal_HCGAE_v4_FixSCR, Optimal_HCGAE_Optimal
结果保存到 results/QuickValid_Optimal/
"""
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from gae_experiments.agents.optimal_ppo import build_optimal_agent

# ─── Config ───────────────────────────────────────────────────────────────────
ENVS = ["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4", "Ant-v4"]
ALGOS = [
    "Optimal_PPO",
    "Optimal_HCGAE_v4",
    "Optimal_HCGAE_v4_FixSCR",
    "Optimal_HCGAE_Optimal",
]
SEEDS = [0, 1, 2, 3, 4]
TOTAL_STEPS = 300_000
EVAL_FREQ = 20_480
N_EVAL_EPS = 10
RESULTS_DIR = Path("results/QuickValid_Optimal")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AGENT_KWARGS = dict(
    hidden_dim=256, lr=3e-4, gamma=0.99, lam=0.95, eps_clip=0.2,
    n_epochs=10, batch_size=64, n_steps=2048, ent_coef=0.0,
    vf_coef=0.5, max_grad_norm=0.5, use_obs_norm=True,
    use_adv_norm=True, use_lr_anneal=True, use_vclip=False, device="cpu",
)


def evaluate_policy(agent, eval_env, n_episodes=N_EVAL_EPS):
    """Evaluate agent deterministically (uses dist.mean for continuous)."""
    rewards = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        obs_n = agent.normalize_obs(obs) if hasattr(agent, 'normalize_obs') else obs
        total_r = 0.0
        done = False
        while not done:
            obs_t = torch.FloatTensor(obs_n).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                dist = agent.actor(obs_t)
                if agent.continuous:
                    action = dist.mean.squeeze(0).cpu().numpy()
                else:
                    action = dist.probs.argmax(dim=-1).squeeze(0).cpu().numpy()
            obs, r, terminated, truncated, _ = eval_env.step(action)
            obs_n = agent.normalize_obs(obs) if hasattr(agent, 'normalize_obs') else obs
            total_r += r
            done = terminated or truncated
        rewards.append(total_r)
    return float(np.mean(rewards)), float(np.std(rewards))


def run_trial(algo_name, env_name, seed):
    """Run a single trial and save results."""
    out_dir = RESULTS_DIR / env_name / algo_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{algo_name}_s{seed}.json"

    if out_file.exists():
        print(f"  [skip] {env_name}/{algo_name} s{seed} already done")
        return

    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 100_000)

    agent = build_optimal_agent(algo_name, env, name=f"{algo_name}_s{seed}", **AGENT_KWARGS)
    agent._total_timesteps = TOTAL_STEPS

    eval_curve = []
    total_steps = 0
    last_eval_step = -EVAL_FREQ
    t0 = time.time()

    obs, _ = env.reset()
    if hasattr(agent, 'update_obs_rms'):
        agent.update_obs_rms(obs)
    obs = agent.normalize_obs(obs) if hasattr(agent, 'normalize_obs') else obs

    while total_steps < TOTAL_STEPS:
        # Rollout
        agent.buffer.reset()
        for _ in range(agent.n_steps):
            if hasattr(agent, 'update_obs_rms'):
                agent.update_obs_rms(obs)
            obs_n = agent.normalize_obs(obs) if hasattr(agent, 'normalize_obs') else obs
            obs_t = torch.FloatTensor(obs_n).unsqueeze(0).to(agent.device)

            with torch.no_grad():
                action, log_prob = agent.actor.get_action_and_logprob(obs_t)
                value = agent.critic(obs_t)

            action_np = action.squeeze(0).cpu().numpy()
            if not agent.continuous:
                action_np = int(action_np)

            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            agent.buffer.add(obs_n, action_np, float(reward), float(terminated),
                             log_prob.item(), value.item())
            obs = next_obs
            total_steps += 1

            if terminated or truncated:
                obs, _ = env.reset()
                if hasattr(agent, 'update_obs_rms'):
                    agent.update_obs_rms(obs)
                obs = agent.normalize_obs(obs) if hasattr(agent, 'normalize_obs') else obs

        # Bootstrap
        obs_n_boot = agent.normalize_obs(obs) if hasattr(agent, 'normalize_obs') else obs
        obs_t = torch.FloatTensor(obs_n_boot).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            last_val = agent.critic(obs_t).item()

        agent.total_steps = total_steps

        # GAE computation
        if hasattr(agent, 'compute_hindsight_gae'):
            agent.compute_hindsight_gae(last_val)
        else:
            agent.compute_gae(last_val)

        # Update
        agent.update()

        # Evaluate
        if total_steps - last_eval_step >= EVAL_FREQ:
            mean_r, std_r = evaluate_policy(agent, eval_env)
            elapsed = time.time() - t0
            print(f"  {env_name}/{algo_name} s{seed} | "
                  f"steps={total_steps:>7d} | "
                  f"reward={mean_r:>8.1f} ± {std_r:>5.1f} | "
                  f"elapsed={elapsed:.1f}s", flush=True)
            eval_curve.append({
                "steps": total_steps,
                "mean_reward": mean_r,
                "std_reward": std_r,
            })
            last_eval_step = total_steps

    # Final evaluation
    final_mean, final_std = evaluate_policy(agent, eval_env, n_episodes=20)
    elapsed = time.time() - t0

    # Diagnostics
    diagnostics = {}
    if hasattr(agent, '_scr_ema'):
        diagnostics['scr_ema_final'] = float(agent._scr_ema)
    if hasattr(agent, '_ev_ema'):
        diagnostics['ev_ema_final'] = float(agent._ev_ema)
    if hasattr(agent, '_g_std_ema'):
        diagnostics['g_std_ema_final'] = float(agent._g_std_ema)

    result = {
        "env":          env_name,
        "agent":        algo_name,
        "seed":         seed,
        "total_steps":  total_steps,
        "elapsed_s":    elapsed,
        "final_reward": final_mean,
        "final_std":    final_std,
        "eval_curve":   eval_curve,
        "diagnostics":  diagnostics,
    }

    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    env.close()
    eval_env.close()
    print(f"  [done] {env_name}/{algo_name} s{seed} | "
          f"final={final_mean:.1f} ± {final_std:.1f} | {elapsed:.0f}s")


def summarize():
    """Print summary table after all trials."""
    import glob

    print()
    print("=" * 85)
    print("HCGAE_Optimal QuickValid Summary (300K steps, 5 seeds)")
    print("Theory: James-Stein / Kalman / Bayes unified → FixSCR + MC-Normalized Sigmoid")
    print("=" * 85)

    ppo_results = {}
    for env_name in ENVS:
        files = glob.glob(str(RESULTS_DIR / env_name / "Optimal_PPO" / "*.json"))
        ppo_results[env_name] = [json.load(open(f))["final_reward"] for f in files]

    env_labels = ["HC", "Hop", "Wal", "Ant"]
    print(f"\n{'Algo':<35} {'HC':>12} {'Hop':>12} {'Wal':>12} {'Ant':>12} {'Avg_Δ':>8}")
    print("-" * 90)

    for algo_name in ALGOS:
        row = []
        deltas = []
        for env_name in ENVS:
            files = glob.glob(str(RESULTS_DIR / env_name / algo_name / "*.json"))
            if files:
                vals = [json.load(open(f))["final_reward"] for f in files]
                row.append(f"{np.mean(vals):.0f}(n={len(vals)})")
                ppo = ppo_results[env_name]
                if ppo and algo_name != "Optimal_PPO":
                    d = (np.mean(vals) - np.mean(ppo)) / abs(np.mean(ppo)) * 100
                    deltas.append(d)
            else:
                row.append("N/A")
        avg_d = np.mean(deltas) if deltas else 0.0
        delta_str = f"{avg_d:>+7.1f}%" if deltas else "  ---"
        print(f"  {algo_name:<33} {row[0]:>12} {row[1]:>12} {row[2]:>12} {row[3]:>12} {delta_str:>8}")

    print()


if __name__ == "__main__":
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", default=ENVS)
    parser.add_argument("--algos", nargs="+", default=ALGOS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    if args.summarize:
        summarize()
        sys.exit(0)

    tasks = [
        (algo, env, seed)
        for env in args.envs
        for algo in args.algos
        for seed in args.seeds
    ]

    print(f"Total trials: {len(tasks)} | Workers: {args.workers}")
    print(f"Algos: {args.algos}")
    print(f"Envs: {args.envs}")
    print()

    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            pool.starmap(run_trial, tasks)
    else:
        for task in tasks:
            run_trial(*task)

    summarize()

