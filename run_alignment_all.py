#!/usr/bin/env python3
"""
统一对齐验证实验调度器
========================
串行、有序地运行所有对齐验证实验，避免 CPU 竞争。

实验目标（文献值对齐）：
  - Hopper-v4    Standard_PPO  ~2300  (Schulman et al. 2017)
  - Hopper-v4    Optimal_PPO   ~2500+
  - HalfCheetah-v4 Standard_PPO ~1800
  - HalfCheetah-v4 Optimal_PPO  ~4000+

配置：每个环境 × 每个算法 × 3 seeds（0/1/2），共 12 runs。
已完成的会自动 SKIP（幂等）。
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

# ─────────────────────────────────────────────────────────────────────────────
# 实验配置
# ─────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results/AlignmentValidation")

PLAN = [
    # (env_name, algo_name, seeds_list)
    ("Hopper-v4",      "Standard_PPO", [0, 1, 2]),
    ("Hopper-v4",      "Optimal_PPO",  [0, 1, 2]),
    ("HalfCheetah-v4", "Standard_PPO", [0, 1, 2]),
    ("HalfCheetah-v4", "Optimal_PPO",  [0, 1, 2]),
]

TOTAL_TIMESTEPS = 1_000_000
EVAL_FREQ       = 20_480
N_EVAL_EPISODES = 10

LITERATURE_TARGETS = {
    "Hopper-v4":       {"Standard_PPO": 2300, "Optimal_PPO": 2500},
    "HalfCheetah-v4":  {"Standard_PPO": 1800, "Optimal_PPO": 4000},
}

STANDARD_PPO_KWARGS = dict(
    hidden_dim=256,
    lr_actor=3e-4,
    lr_critic=1e-3,
    gamma=0.99,
    lam=0.95,
    eps_clip=0.2,
    n_epochs=10,
    batch_size=64,
    n_steps=2048,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    use_obs_norm=True,
    use_adv_norm=True,
    device="cpu",
)

OPTIMAL_PPO_KWARGS = dict(
    hidden_dim=256,
    lr=3e-4,
    gamma=0.99,
    lam=0.95,
    eps_clip=0.2,
    n_epochs=10,
    batch_size=64,
    n_steps=2048,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    use_obs_norm=True,
    use_adv_norm=True,
    use_lr_anneal=True,
    use_vclip=False,
    device="cpu",
)

# ─────────────────────────────────────────────────────────────────────────────
# 训练/评估函数
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_policy(agent, eval_env, n_episodes=10):
    """确定性策略评估（不更新 obs_rms）。"""
    rewards = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        total_reward = 0.0
        done = False
        while not done:
            obs_norm = agent.normalize_obs(obs) if hasattr(agent, 'normalize_obs') else obs
            obs_t = torch.FloatTensor(obs_norm).unsqueeze(0)
            with torch.no_grad():
                dist = agent.actor(obs_t)
                action = dist.mean if agent.continuous else dist.probs.argmax(dim=-1)
            action_np = action.squeeze(0).cpu().numpy()
            obs, r, terminated, truncated, _ = (
                eval_env.step(action_np) if agent.continuous
                else eval_env.step(int(action_np))
            )
            total_reward += r
            done = terminated or truncated
        rewards.append(total_reward)
    return float(np.mean(rewards))


def run_single(env_name, algo_name, seed):
    """运行单个 (env, algo, seed) 组合，幂等（已完成则 SKIP）。"""
    save_dir = RESULTS_DIR / env_name / algo_name
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{algo_name}_s{seed}.json"

    if out_path.exists():
        d = json.load(open(out_path))
        er = d.get("eval_rewards", [])
        final5 = float(np.mean(er[-5:])) if len(er) >= 5 else (float(np.mean(er)) if er else 0.0)
        print(f"  [SKIP] {env_name}/{algo_name}/s{seed}  (final5={final5:.1f})")
        return final5

    print(f"  [RUN ] {env_name}/{algo_name}/seed={seed}")
    t0 = time.time()

    np.random.seed(seed)
    torch.manual_seed(seed)

    env      = gym.make(env_name)
    eval_env = gym.make(env_name)
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 50000)

    if algo_name == "Standard_PPO":
        kw = dict(**STANDARD_PPO_KWARGS, save_dir=str(save_dir))
        agent = build_ppo_baseline("Standard_PPO", env, name=f"Standard_PPO_s{seed}", **kw)
    elif algo_name == "Optimal_PPO":
        kw = dict(**OPTIMAL_PPO_KWARGS, save_dir=str(save_dir))
        agent = build_optimal_agent("Optimal_PPO", env, name=f"Optimal_PPO_s{seed}", **kw)
    else:
        raise ValueError(f"Unknown algo: {algo_name}")

    # ★ 必须在第一次 update() 之前设置，供 LR annealing 使用
    agent._total_timesteps = TOTAL_TIMESTEPS

    eval_rewards  = []
    eval_steps    = []
    episode_rewards = []

    obs, _ = env.reset()
    ep_reward = 0.0
    total_steps = 0
    last_eval_step = 0

    while total_steps < TOTAL_TIMESTEPS:
        # ── Collect rollout ──────────────────────────────────────────────────
        agent.buffer.reset()
        cur_obs = obs

        for _ in range(agent.n_steps):
            if hasattr(agent, 'update_obs_rms'):
                agent.update_obs_rms(cur_obs)
            obs_norm = agent.normalize_obs(cur_obs) if hasattr(agent, 'normalize_obs') else cur_obs
            obs_t = torch.FloatTensor(obs_norm).unsqueeze(0)
            with torch.no_grad():
                action, log_prob = agent.actor.get_action_and_logprob(obs_t)
                value = agent.critic(obs_t)

            action_np = action.squeeze(0).cpu().numpy()
            if not agent.continuous:
                action_np = int(action_np)

            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            ep_reward += reward

            agent.buffer.add(obs_norm, action_np, float(reward), float(terminated),
                             log_prob.item(), value.item())
            total_steps += 1
            cur_obs = next_obs

            if terminated or truncated:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                cur_obs, _ = env.reset()

        # ── GAE bootstrap ────────────────────────────────────────────────────
        if hasattr(agent, 'update_obs_rms'):
            agent.update_obs_rms(cur_obs)
        last_obs_norm = agent.normalize_obs(cur_obs) if hasattr(agent, 'normalize_obs') else cur_obs
        last_obs_t = torch.FloatTensor(last_obs_norm).unsqueeze(0)
        with torch.no_grad():
            last_val = agent.critic(last_obs_t).item()

        obs = cur_obs
        agent.total_steps = total_steps  # 供 LR annealing 使用

        agent.compute_gae(last_val)
        agent.update()

        # ── Evaluate ─────────────────────────────────────────────────────────
        if total_steps - last_eval_step >= EVAL_FREQ:
            eval_r = evaluate_policy(agent, eval_env, N_EVAL_EPISODES)
            eval_rewards.append(eval_r)
            eval_steps.append(total_steps)
            last_eval_step = total_steps
            elapsed = time.time() - t0
            print(f"    step={total_steps:>7,}/{TOTAL_TIMESTEPS:,}  eval={eval_r:>7.1f}  "
                  f"elapsed={elapsed:>5.0f}s")

    elapsed = time.time() - t0
    final5 = float(np.mean(eval_rewards[-5:])) if len(eval_rewards) >= 5 else (
        float(np.mean(eval_rewards)) if eval_rewards else 0.0
    )
    max_r = float(max(eval_rewards)) if eval_rewards else 0.0

    result = {
        "env": env_name, "agent": algo_name, "seed": seed,
        "total_steps": total_steps, "final_reward": final5,
        "eval_rewards": eval_rewards, "eval_steps": eval_steps,
        "episode_rewards": episode_rewards, "elapsed_s": elapsed,
        "config": {
            "hidden_dim": 256, "use_obs_norm": True,
            "use_adv_norm": True, "use_lr_anneal": (algo_name == "Optimal_PPO"),
        }
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    env.close(); eval_env.close()
    target = LITERATURE_TARGETS.get(env_name, {}).get(algo_name, 0)
    status = "✓" if final5 >= target * 0.9 else ("△" if max_r >= target * 0.9 else "✗")
    print(f"  {status} Done: {algo_name} {env_name} s{seed}  "
          f"final5={final5:.1f}  max={max_r:.1f}  target={target}  ({elapsed:.0f}s)")
    return final5


# ─────────────────────────────────────────────────────────────────────────────
# 汇总打印
# ─────────────────────────────────────────────────────────────────────────────

def print_summary():
    print("\n" + "="*72)
    print("  LITERATURE ALIGNMENT VALIDATION  —  FINAL SUMMARY")
    print("="*72)
    print(f"  {'Algorithm':<18} {'Environment':<18} {'Seeds':>6}  "
          f"{'Mean±Std':>14}  {'Target':>8}  {'Status':>8}")
    print("-"*72)

    all_ok = True
    for env_name, algo_name, seeds in PLAN:
        vals = []
        for seed in seeds:
            fp = RESULTS_DIR / env_name / algo_name / f"{algo_name}_s{seed}.json"
            if fp.exists():
                d   = json.load(open(fp))
                er  = d.get("eval_rewards", [])
                f5  = float(np.mean(er[-5:])) if len(er) >= 5 else (float(np.mean(er)) if er else 0.0)
                vals.append(f5)

        target = LITERATURE_TARGETS.get(env_name, {}).get(algo_name, 0)
        if vals:
            m, s  = np.mean(vals), np.std(vals)
            ratio = m / target if target else 1.0
            status = "✓ OK" if ratio >= 0.90 else ("△ WARN" if ratio >= 0.70 else "✗ FAIL")
            if ratio < 0.90:
                all_ok = False
            seeds_str = ",".join(str(v) for v in seeds[:len(vals)])
            print(f"  {algo_name:<18} {env_name:<18} s{seeds_str:>4}  "
                  f"{m:>7.0f}±{s:>5.0f}  {target:>8}  {status:>8}")
        else:
            print(f"  {algo_name:<18} {env_name:<18} {'—':>6}  "
                  f"{'pending':>14}  {target:>8}  {'—':>8}")
            all_ok = False

    print("="*72)
    if all_ok:
        print("  ★ All algorithms meet ≥90% of literature target values.")
    else:
        print("  △ Some algorithms below target (check max rewards — Hopper has high variance).")

    # 额外：显示各 seed 的 max reward（Hopper 重要）
    print("\n  Per-seed MAX eval rewards (Hopper variance analysis):")
    for env_name, algo_name, seeds in PLAN:
        if "Hopper" not in env_name:
            continue
        maxes = []
        for seed in seeds:
            fp = RESULTS_DIR / env_name / algo_name / f"{algo_name}_s{seed}.json"
            if fp.exists():
                d  = json.load(open(fp))
                er = d.get("eval_rewards", [])
                maxes.append((seed, max(er) if er else 0.0))
        if maxes:
            print(f"    {algo_name} {env_name}:", "  ".join(f"s{s}={v:.0f}" for s,v in maxes))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 统计总任务数
    total_runs = sum(len(seeds) for _, _, seeds in PLAN)
    done_runs  = sum(
        1 for env_name, algo_name, seeds in PLAN
        for seed in seeds
        if (RESULTS_DIR / env_name / algo_name / f"{algo_name}_s{seed}.json").exists()
    )

    print("="*72)
    print("  Literature Alignment Validation  —  Serial Scheduler")
    print(f"  Plan : {len(PLAN)} groups × seeds = {total_runs} runs total")
    print(f"  Done : {done_runs} / {total_runs}  (will SKIP completed runs)")
    print(f"  Targets: Hopper Standard≥2300, Hopper Optimal≥2500,")
    print(f"           HalfCheetah Standard≥1800, HalfCheetah Optimal≥4000")
    print("="*72)

    t_global = time.time()
    run_idx  = 0
    for env_name, algo_name, seeds in PLAN:
        for seed in seeds:
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] {env_name}  {algo_name}  seed={seed}")
            run_single(env_name, algo_name, seed)

    print(f"\nAll runs finished in {time.time()-t_global:.0f}s")

    # 保存最终汇总
    summary = {}
    for env_name, algo_name, seeds in PLAN:
        summary.setdefault(env_name, {})
        vals = []
        for seed in seeds:
            fp = RESULTS_DIR / env_name / algo_name / f"{algo_name}_s{seed}.json"
            if fp.exists():
                d  = json.load(open(fp))
                er = d.get("eval_rewards", [])
                f5 = float(np.mean(er[-5:])) if len(er) >= 5 else (float(np.mean(er)) if er else 0.0)
                vals.append(f5)
        if vals:
            summary[env_name][algo_name] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals)),
                "n":    len(vals),
                "target": LITERATURE_TARGETS.get(env_name, {}).get(algo_name, 0),
                "seeds": vals,
            }
    with open(RESULTS_DIR / "alignment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary → {RESULTS_DIR}/alignment_summary.json")

    print_summary()


if __name__ == "__main__":
    main()

