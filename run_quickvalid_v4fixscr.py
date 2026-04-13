#!/usr/bin/env python3
"""
Optimal_HCGAE_v4_FixSCR 快速验证实验
=====================================
验证修正 SCR 分母对 v4 性能的影响。

问题根源（全方差定律）：
    v4 原始: SCR = |mean(G-V)| / std(G)
      → std(G) 包含 Var(V*(s)) 结构方差，与 alpha 无关
      → SCR 被过度压低 → alpha_cap 过度保守

修正: v4_FixSCR: SCR = |mean(G-V)| / sqrt(max(Var(G)-Var(V), floor))
    全方差定律: Var(G) = Var(V*) + E[Var(G|s)]
      → E[Var(G|s)] 才是真正的 MC 噪声方差
      → 当 Critic 好时: Var(V) ≈ Var(V*) → 分母更小 → SCR 更大 → alpha_cap 更宽松

数值量化（Critic 质量中等, std(G)=60, std(V)=48, bias=12）：
    v4:       SCR = 12/60 = 0.200 → alpha_cap = 0.04 + 0.05 = 0.09
    v4_Fix:   SCR = 12/36 = 0.333 → alpha_cap = 0.10 + 0.05 = 0.15  (+67%)

预期：
    HalfCheetah (Critic差, EV≈0.15, Var(V)≈0): v4 ≈ v4_Fix（分母相同）
    Ant         (Critic中, EV≈0.40, Var(V)>0):  v4_Fix > v4（更多 MC 修正）
    Hopper      (Critic好, EV≈0.85, Var(V)≈Var(G)): 修正效果最大
    Walker2d    (Critic好, EV≈0.72): 显著改善

结果保存到 results/QuickValid_v4FixSCR/
对比：Optimal_HCGAE_v4 vs Optimal_HCGAE_v4_FixSCR vs Optimal_PPO
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

# ─── Config ───────────────────────────────────────────────────────────────────
ENVS        = ["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4", "Ant-v4"]
ALGOS       = ["Optimal_HCGAE_v4", "Optimal_HCGAE_v4_FixSCR", "Optimal_PPO"]
SEEDS       = [0, 1, 2, 3]
TOTAL_STEPS = 300_000
EVAL_FREQ   = 10_240
N_EVAL_EPS  = 10
RESULTS_DIR = Path("results/QuickValid_v4FixSCR")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AGENT_KWARGS = dict(
    hidden_dim=256, lr=3e-4, gamma=0.99, lam=0.95, eps_clip=0.2,
    n_epochs=10, batch_size=64, n_steps=2048, ent_coef=0.0,
    vf_coef=0.5, max_grad_norm=0.5, use_obs_norm=True,
    use_adv_norm=True, use_lr_anneal=True, use_vclip=False, device="cpu",
)


def evaluate_policy(agent, eval_env, n_eps=N_EVAL_EPS):
    rewards = []
    for _ in range(n_eps):
        obs, _ = eval_env.reset()
        total_r = 0.0
        done = False
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, r, terminated, truncated, _ = eval_env.step(action)
            total_r += r
            done = terminated or truncated
        rewards.append(total_r)
    return float(np.mean(rewards)), float(np.std(rewards))


def run_single(env_name, algo_name, seed):
    out_dir = RESULTS_DIR / env_name / algo_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{algo_name}_s{seed}.json"
    if out_file.exists():
        print(f"  [skip] {env_name}/{algo_name} seed={seed} already exists")
        return

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 1000)

    agent = build_optimal_agent(
        algo_name, env,
        name=f"{algo_name}_s{seed}",
        **AGENT_KWARGS
    )

    eval_curve = []
    steps_done = 0
    next_eval = EVAL_FREQ
    t0 = time.time()

    # Diagnostic tracking for FixSCR analysis
    scr_history = []
    alpha_cap_history = []

    obs, _ = env.reset()
    while steps_done < TOTAL_STEPS:
        # Rollout
        for _ in range(agent.n_steps):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.store_transition(obs, action, reward, terminated, truncated)
            obs = next_obs
            steps_done += 1
            if terminated or truncated:
                obs, _ = env.reset()

        # Update
        agent.update()

        # Collect SCR diagnostics if available
        if hasattr(agent, '_scr_ema'):
            scr_history.append(float(agent._scr_ema))
        if hasattr(agent, '_scr_history') and len(agent._scr_history) > 0:
            pass  # already tracked via _scr_ema

        # Evaluate
        if steps_done >= next_eval:
            mean_r, std_r = evaluate_policy(agent, eval_env)
            elapsed = time.time() - t0
            print(f"  {env_name}/{algo_name} s{seed} | "
                  f"steps={steps_done:>7d} | "
                  f"reward={mean_r:>8.1f} ± {std_r:>6.1f} | "
                  f"elapsed={elapsed:.1f}s", flush=True)
            eval_curve.append({
                "steps": steps_done,
                "mean_reward": mean_r,
                "std_reward": std_r,
            })
            next_eval += EVAL_FREQ

    # Final evaluation
    final_mean, final_std = evaluate_policy(agent, eval_env, n_eps=20)

    result = {
        "env": env_name,
        "agent": algo_name,
        "seed": seed,
        "total_steps": steps_done,
        "eval_curve": eval_curve,
        "final_reward": final_mean,
        "final_std": final_std,
        "scr_history": scr_history[-50:] if scr_history else [],  # 最后50个SCR值
    }

    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    env.close()
    eval_env.close()
    print(f"  [done] {env_name}/{algo_name} s{seed} | final={final_mean:.1f}")
    return result


def print_summary():
    """汇总显示结果，重点对比 v4 vs v4_FixSCR 的 alpha_cap 行为差异。"""
    print("\n" + "="*80)
    print("结果汇总：Optimal_HCGAE_v4 vs v4_FixSCR vs Optimal_PPO")
    print("="*80)
    for env_name in ENVS:
        print(f"\n{env_name}:")
        results = {}
        for algo_name in ALGOS:
            out_dir = RESULTS_DIR / env_name / algo_name
            rewards = []
            scr_vals = []
            for seed in SEEDS:
                out_file = out_dir / f"{algo_name}_s{seed}.json"
                if out_file.exists():
                    with open(out_file) as f:
                        d = json.load(f)
                    rewards.append(d.get("final_reward", float("nan")))
                    scr_vals.extend(d.get("scr_history", []))
            if rewards:
                mean_r = float(np.mean(rewards))
                std_r = float(np.std(rewards))
                scr_mean = float(np.mean(scr_vals)) if scr_vals else float("nan")
                results[algo_name] = (mean_r, std_r, scr_mean)
                print(f"  {algo_name:<35s}: {mean_r:>8.1f} ± {std_r:>6.1f}  (SCR_ema≈{scr_mean:.3f})")
            else:
                print(f"  {algo_name:<35s}: (no data)")

        # 对比分析
        if "Optimal_PPO" in results and "Optimal_HCGAE_v4" in results:
            ppo_r = results["Optimal_PPO"][0]
            v4_r = results["Optimal_HCGAE_v4"][0]
            v4fix_r = results.get("Optimal_HCGAE_v4_FixSCR", (float("nan"),))[0]
            v4_delta = (v4_r - ppo_r) / (abs(ppo_r) + 1e-8) * 100
            v4fix_delta = (v4fix_r - ppo_r) / (abs(ppo_r) + 1e-8) * 100
            print(f"  → v4 vs PPO: {v4_delta:+.1f}%")
            print(f"  → v4_FixSCR vs PPO: {v4fix_delta:+.1f}%")
            print(f"  → v4_FixSCR vs v4: {v4fix_delta - v4_delta:+.1f}%  ← 分母修正效果")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="v4_FixSCR 快速验证（v4 分母修正）")
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--algo", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    if args.summary:
        print_summary()
        sys.exit(0)

    envs  = [args.env]  if args.env  else ENVS
    algos = [args.algo] if args.algo else ALGOS
    seeds = [args.seed] if args.seed is not None else SEEDS

    total = len(envs) * len(algos) * len(seeds)
    done_count = 0

    for env_name in envs:
        for algo_name in algos:
            for seed in seeds:
                done_count += 1
                print(f"\n[{done_count}/{total}] {env_name} / {algo_name} / seed={seed}")
                try:
                    run_single(env_name, algo_name, seed)
                except Exception as e:
                    print(f"  [ERROR] {e}")

    print_summary()

