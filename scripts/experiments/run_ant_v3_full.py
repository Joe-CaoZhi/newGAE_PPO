#!/usr/bin/env python3
"""
Ant-v4 完整对齐实验: HCGAE v3
==============================
与 ICMLExperiment 完全对齐的实验条件：
  - 训练步数: 500K（与其他环境相同）
  - 种子: 0-4（5个种子）
  - 算法:
      * Standard_PPO     (基线)
      * Optimal_PPO      (优化基线)
      * Optimal_HCGAE_v2 (前代，有 Ant 退化问题)
      * Optimal_HCGAE_v3 (v3 全量修复)
  - 评估: 每 10240 步，10 episodes

注意: 仅在 200K 快速验证显示 v3 有效后再运行此脚本。
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

# ═══════════════════════════════════════════
# 实验配置（与 ICMLExperiment 完全对齐）
# ═══════════════════════════════════════════
SEEDS = list(range(5))            # 0-4，共5个种子
TOTAL_TIMESTEPS = 500_000         # 与其他环境相同
EVAL_FREQ = 10_240
N_EVAL_EPISODES = 10
RESULTS_DIR = Path("results/ICMLExperiment/Ant-v4")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 只跑 v3 相关的算法（其他已有 ICMLExperiment 结果）
ALGORITHMS = [
    "Optimal_HCGAE_v3",    # v3 全量修复（新增）
]
# 可选：如果要重新对齐基线，也运行以下
# ALGORITHMS = ["Standard_PPO", "Optimal_PPO", "Optimal_HCGAE_v2", "Optimal_HCGAE_v3"]

STANDARD_PPO_KWARGS = dict(
    hidden_dim=256, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, lam=0.95,  # ★ 256x256 MLP
    eps_clip=0.2, n_epochs=10, batch_size=64, n_steps=2048,
    ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
    use_obs_norm=True, use_adv_norm=True,  # ★ Obs norm + adv norm
    device="cpu",
)
OPTIMAL_PPO_KWARGS = dict(
    hidden_dim=256, lr=3e-4, gamma=0.99, lam=0.95, eps_clip=0.2,  # ★ 256x256 MLP
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
        print(f"  [SKIP] {algo_name} s{seed} — 已存在结果")
        return

    print(f"  [{algo_name} s{seed}] 开始训练 …")
    t0 = time.time()

    env = gym.make("Ant-v4")
    eval_env = gym.make("Ant-v4")
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 50000)

    if algo_name == "Standard_PPO":
        kw = dict(**STANDARD_PPO_KWARGS, save_dir=save_dir)
        agent = build_ppo_baseline("Standard_PPO", env, name=f"{algo_name}_s{seed}", **kw)
    else:
        kw = dict(**OPTIMAL_PPO_KWARGS, save_dir=save_dir)
        agent = build_optimal_agent(algo_name, env, name=f"{algo_name}_s{seed}", **kw)

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
                print(f"    step={total_steps:>7d}  eval_reward={er:.1f}")

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

    er = evaluate_policy(agent, eval_env, N_EVAL_EPISODES)
    eval_rewards.append(er)

    final_mean = float(np.mean(eval_rewards[-5:])) if len(eval_rewards) >= 5 else float(np.mean(eval_rewards))
    elapsed = time.time() - t0

    result = {
        "env": "Ant-v4",
        "agent": algo_name,
        "seed": seed,
        "total_steps": total_steps,
        "final_reward": final_mean,
        "eval_rewards": eval_rewards,
        "eval_steps": eval_steps,
        "episode_rewards": episode_rewards,
        "elapsed_s": elapsed,
    }
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  [{algo_name} s{seed}] 完成 → 最终奖励={final_mean:.1f}  耗时={elapsed:.0f}s")
    env.close()
    eval_env.close()


def print_summary():
    """汇总 ICMLExperiment/Ant-v4 所有算法结果（包含 v3）"""
    all_algos = [
        "Standard_PPO",
        "Optimal_PPO",
        "Optimal_HCGAE",
        "Optimal_HCGAE_SCR",
        "Optimal_HCGAE_v2",
        "Optimal_HCGAE_v3",
    ]
    print("\n" + "=" * 70)
    print("  Ant-v4 完整实验结果汇总 (500K steps, 5 seeds)")
    print("=" * 70)
    print(f"  {'算法':<35} {'均值':>8}  {'标准差':>8}  {'种子数'}")
    print("-" * 70)
    for algo in all_algos:
        algo_dir = RESULTS_DIR / algo
        results = []
        for seed in range(5):
            fp = algo_dir / f"{algo}_s{seed}.json"
            if fp.exists():
                with open(fp) as f:
                    d = json.load(f)
                results.append(d['final_reward'])
        if results:
            mean_r = np.mean(results)
            std_r = np.std(results)
            print(f"  {algo:<35} {mean_r:>8.1f}  {std_r:>8.1f}  [{len(results)}/5]")
        else:
            print(f"  {algo:<35} {'—':>8}  {'—':>8}  [0/5]")
    print("=" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print(f"  Ant-v4 完整实验: {ALGORITHMS} × 5 seeds × 500K 步")
    print("=" * 70)
    for algo in ALGORITHMS:
        print(f"\n── {algo} ──")
        for seed in SEEDS:
            run_single(algo, seed)
    print_summary()
    print("\n完整实验完成！")

