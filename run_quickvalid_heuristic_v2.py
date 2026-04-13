#!/usr/bin/env python3
"""
Optimal_HCGAE_HeuristicV2 快速验证实验
==========================================
验证改进版 Heuristic_HCGAE (HeuristicV2) 相对于 v4 和 Optimal_PPO 的性能。

HeuristicV2 改进:
  1. Noise-Normalized Sigmoid: z_t = β·(|V-G|/σ_G_ema - 0.5)
     → 修正量与 MC 噪声水平挂钩，Critic 好时自动抑制
  2. 继承 v4 的 SCR_cap: α_max ≤ SCR²/(1+SCR²) + 0.05
     → 防止高方差环境（Ant）过度修正

预期：
  HC:      保持 v4 水平（SCR_cap 已有保护）
  Hopper:  略改善（归一化修正，Critic 好时不再浪费修正量）
  Walker2d: 改善（双重约束）
  Ant:     改善（SCR_cap + 归一化）

对比组：Optimal_PPO, Optimal_HCGAE_v4, Optimal_HCGAE_HeuristicV2
结果保存到 results/QuickValid_HeuristicV2/
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
ENVS        = ["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4", "Ant-v4"]
ALGOS       = ["Optimal_PPO", "Optimal_HCGAE_v4", "Optimal_HCGAE_HeuristicV2"]
SEEDS       = [0, 1, 2, 3, 4]
TOTAL_STEPS = 300_000
EVAL_FREQ   = 20_480
N_EVAL_EPS  = 10
RESULTS_DIR = Path("results/QuickValid_HeuristicV2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AGENT_KWARGS = dict(
    hidden_dim=256, lr=3e-4, gamma=0.99, lam=0.95, eps_clip=0.2,
    n_epochs=10, batch_size=64, n_steps=2048, ent_coef=0.0,
    vf_coef=0.5, max_grad_norm=0.5, use_obs_norm=True,
    use_adv_norm=True, use_lr_anneal=True, use_vclip=False, device="cpu",
)


def evaluate_policy(agent, eval_env, n_episodes=N_EVAL_EPS):
    """Evaluate agent deterministically for n_episodes (uses dist.mean)."""
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
        print(f"  [skip] {out_file.name} already exists")
        return

    torch.manual_seed(seed)
    np.random.seed(seed)

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

    ep_reward = 0.0

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
            ep_reward += reward
            agent.buffer.add(obs_n, action_np, float(reward), float(terminated),
                             log_prob.item(), value.item())
            obs = next_obs
            total_steps += 1

            if terminated or truncated:
                ep_reward = 0.0
                obs, _ = env.reset()
                if hasattr(agent, 'update_obs_rms'):
                    agent.update_obs_rms(obs)
                obs = agent.normalize_obs(obs) if hasattr(agent, 'normalize_obs') else obs

        # Bootstrap
        obs_t = torch.FloatTensor(
            agent.normalize_obs(obs) if hasattr(agent, 'normalize_obs') else obs
        ).unsqueeze(0).to(agent.device)
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
        diagnostics['final_scr_ema'] = float(agent._scr_ema)
    if hasattr(agent, '_g_std_ema'):
        diagnostics['final_g_std_ema'] = float(agent._g_std_ema)
    if hasattr(agent, '_ev_ema'):
        diagnostics['final_ev_ema'] = float(agent._ev_ema)

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
    return result


def summarize():
    """Print summary table after all trials."""
    import glob

    print()
    print("=" * 75)
    print("Summary (vs Optimal_PPO)")
    print("=" * 75)

    ppo_results = {}
    for env_name in ENVS:
        files = glob.glob(str(RESULTS_DIR / env_name / "Optimal_PPO" / "*.json"))
        vals = [json.load(open(f))["final_reward"] for f in files]
        ppo_results[env_name] = vals

    print(f"\n{'Algo':<35} {'HC':>12} {'Hop':>12} {'Wal':>12} {'Ant':>12} {'Avg_Δ':>8}")
    print("-" * 85)

    for algo_name in ALGOS:
        deltas = []
        row = []
        for env_name in ENVS:
            files = glob.glob(str(RESULTS_DIR / env_name / algo_name / "*.json"))
            vals = [json.load(open(f))["final_reward"] for f in files]
            if vals:
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
    print("=== Δ vs Optimal_PPO ===")
    print(f"\n{'Algo':<35} {'ΔHC':>10} {'ΔHop':>10} {'ΔWal':>10} {'ΔAnt':>10} {'ΔAvg':>8}")
    print("-" * 80)
    for algo_name in ALGOS:
        if algo_name == "Optimal_PPO":
            continue
        deltas = []
        row = []
        for env_name in ENVS:
            files = glob.glob(str(RESULTS_DIR / env_name / algo_name / "*.json"))
            vals = [json.load(open(f))["final_reward"] for f in files]
            ppo = ppo_results[env_name]
            if vals and ppo:
                d = (np.mean(vals) - np.mean(ppo)) / abs(np.mean(ppo)) * 100
                row.append(f"{d:+.1f}%")
                deltas.append(d)
            else:
                row.append("N/A")
        avg_d = np.mean(deltas) if deltas else float('nan')
        print(f"  {algo_name:<33} {row[0]:>10} {row[1]:>10} {row[2]:>10} {row[3]:>10} {avg_d:>+7.1f}%")


def main():
    print("=" * 68)
    print("Optimal_HCGAE_HeuristicV2 快速验证")
    print(f"  Envs:  {ENVS}")
    print(f"  Algos: {ALGOS}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Steps: {TOTAL_STEPS:,} per trial")
    print(f"  Total trials: {len(ENVS) * len(ALGOS) * len(SEEDS)}")
    print("=" * 68)

    t_total = time.time()

    for env_name in ENVS:
        print(f"\n{'='*60}")
        print(f"Environment: {env_name}")
        print(f"{'='*60}")
        for algo_name in ALGOS:
            print(f"\n  [{algo_name}]")
            for seed in SEEDS:
                run_trial(algo_name, env_name, seed)

    summarize()
    print(f"\nTotal time: {(time.time() - t_total)/3600:.2f} h")


if __name__ == "__main__":
    main()

