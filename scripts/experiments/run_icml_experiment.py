#!/usr/bin/env python3
"""
ICML 2026 完备实验脚本
======================
5 种算法 × 4 个环境 × 12 个随机种子 × 1M 步

算法（严格对齐超参数，所有算法共享 OptimalPPO 骨架）：
  1. Standard_PPO       ── 原始 PPO（无 obs-norm / adv-norm / lr-anneal）
  2. Optimal_PPO        ── 集成所有最佳实践的基线（obs-norm + adv-norm + lr-anneal）
  3. Heuristic_HCGAE    ── 启发式 MC 融合（OptimalHCGAE_v2）：余弦退火×EV门控×sigmoid
  4. BHVF               ── 本文方法 §2：解析最优增益 α*=SCR²/(SCR²+1)，无任何启发式
  5. BHVF_DCPPO         ── 本文方法 §2+§3：BHVF + EV 线性收缩梯度调制（DCPPO-S）

环境：Hopper-v4 / Walker2d-v4 / HalfCheetah-v4 / Ant-v4

输出目录结构：
  results/FinalExperiment/{env}/{algo}/{algo}_s{seed}.json

每个 JSON 包含：
  - eval_rewards, eval_steps  ── 标准学习曲线（每 20480 步评估一次）
  - final_reward              ── 最后 10 次评估均值（论文 Table 1 数据源）
  - max_reward                ── 全程最高评估分数
  - bhvf_diagnostics          ── BHVF 特有的中间状态（SCR、alpha*、sigma_V 等）
  - config                    ── 完整超参数（可溯源）

可溯源性设计：
  - 每个 JSON 含完整超参数 config，任意结果可独立复现
  - 种子列表固定（0–11），与 paper 报告严格对应
  - 中间 diagnostics 允许离线验证理论预测（如 HalfCheetah SCR ≈ 0，Hopper SCR >> 1）

用法：
  # 快速冒烟测试（50k步，2种子）
  python run_icml_experiment.py --smoke-test

  # 单环境单算法（断点续跑）
  python run_icml_experiment.py --env Hopper-v4 --algo BHVF

  # 完整实验（自动跳过已有结果）
  python run_icml_experiment.py

  # 并行（外部脚本控制进程池）
  python run_icml_experiment.py --env Hopper-v4 --algo BHVF --seeds 0,1,2,3
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

from gae_experiments.agents.optimal_ppo import (
    OptimalPPO,
    OptimalHCGAE_v2,
    OptimalHCGAE_Bayesian,
    build_optimal_agent,
)
from gae_experiments.agents.ppo_baselines import PPOBaseline

# ─────────────────────────────────────────────────────────────────────────────
# 实验设计常量
# ─────────────────────────────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 1_000_000
EVAL_FREQ       = 20_480     # 约 10 次 rollout 评估一次（与 Andrychowicz 2021 对齐）
N_EVAL_EPISODES = 10
N_SEEDS         = 12
SEEDS           = list(range(N_SEEDS))  # 0..11，固定，可溯源

ENVS = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4", "Ant-v4"]

# 5 个算法配置
ALGOS = ["Standard_PPO", "Optimal_PPO", "Heuristic_HCGAE", "BHVF", "BHVF_DCPPO"]

RESULTS_DIR = Path("results/FinalExperiment")

# ─────────────────────────────────────────────────────────────────────────────
# 对齐超参数（所有算法完全相同，是公平对比的基础）
# ─────────────────────────────────────────────────────────────────────────────
SHARED_KWARGS = dict(
    hidden_dim      = 256,
    lr              = 3e-4,
    gamma           = 0.99,
    lam             = 0.95,
    eps_clip        = 0.2,
    n_epochs        = 10,
    batch_size      = 64,
    n_steps         = 2048,
    ent_coef        = 0.0,
    vf_coef         = 0.5,
    max_grad_norm   = 0.5,
    # OptimalPPO 骨架 tricks
    use_obs_norm    = True,
    use_adv_norm    = True,
    use_lr_anneal   = True,
    use_vclip       = False,
)

# Standard_PPO 禁用所有最佳实践（对标 Schulman 2017 原版）
STANDARD_PPO_OVERRIDES = dict(
    use_obs_norm  = False,
    use_adv_norm  = False,
    use_lr_anneal = False,
)

# BHVF 超参数（全环境完全相同，零环境特定调整）
BHVF_KWARGS = dict(
    scr_ema_alpha = 0.1,   # SCR EMA 学习率
    scr_relax     = 0.05,  # alpha* 数值松弛（防趋零）
    clip_c        = 3.0,   # 新息截断系数（3σ = 99.7% Normal）
)

# Heuristic HCGAE (v2) 超参数
HEURISTIC_HCGAE_KWARGS = dict(
    hindsight_beta       = 3.0,
    hindsight_alpha_max  = 0.7,
    hindsight_alpha_min  = 0.1,
    use_scr_adapt        = False,
    use_boundary_correction = True,
    use_ev_rate_gate     = True,
    ev_rate_threshold    = 0.05,
    ev_rate_max          = 0.15,
    ev_gate_min_scale    = 0.1,
)


def build_agent(algo_name: str, env: gym.Env, seed: int, save_dir: str):
    """
    构建对应算法的 agent。
    所有算法共享 SHARED_KWARGS，仅算法特有参数有差异。
    """
    kw = dict(**SHARED_KWARGS, save_dir=save_dir)

    if algo_name == "Standard_PPO":
        kw.update(STANDARD_PPO_OVERRIDES)
        return OptimalPPO(env=env, name=f"Standard_PPO_s{seed}", **kw)

    elif algo_name == "Optimal_PPO":
        return OptimalPPO(env=env, name=f"Optimal_PPO_s{seed}", **kw)

    elif algo_name == "Heuristic_HCGAE":
        hkw = dict(**kw)
        hkw.update(HEURISTIC_HCGAE_KWARGS)
        return OptimalHCGAE_v2(env=env, name=f"Heuristic_HCGAE_s{seed}", **hkw)

    elif algo_name == "BHVF":
        # 纯 BHVF（无 DCPPO-S）：advantage 不乘 EV 权重
        return OptimalHCGAE_Bayesian(
            env=env, name=f"BHVF_s{seed}",
            **{k: v for k, v in {**kw, **BHVF_KWARGS}.items()}
        )

    elif algo_name == "BHVF_DCPPO":
        # BHVF + DCPPO-S：advantage 乘以 clip(EV, 0.1, 1.0)（在 update 中实现）
        agent = OptimalHCGAE_Bayesian(
            env=env, name=f"BHVF_DCPPO_s{seed}",
            **{k: v for k, v in {**kw, **BHVF_KWARGS}.items()}
        )
        agent._use_dcppo_s = True   # 标记启用 DCPPO-S（在 run_single 中处理）
        return agent

    else:
        raise ValueError(f"Unknown algo: {algo_name}")


def evaluate_policy(agent, eval_env: gym.Env, n_episodes: int = 10) -> float:
    """
    评估策略（确定性均值动作，与 Andrychowicz 2021 对齐）。
    禁用 exploration noise，使用 policy mean。
    """
    total_reward = 0.0
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_r = 0.0
        while not done:
            obs_n = agent.normalize_obs(obs) if hasattr(agent, 'normalize_obs') else obs
            obs_t = torch.FloatTensor(obs_n).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                dist = agent.actor(obs_t)
                # 确定性评估：使用分布均值
                if agent.continuous:
                    action = dist.mean
                else:
                    action = dist.probs.argmax(dim=-1)
            a = action.squeeze(0).cpu().numpy()
            next_obs, r, terminated, truncated, _ = eval_env.step(
                a if agent.continuous else int(a)
            )
            ep_r += r
            done = terminated or truncated
            obs = next_obs
        total_reward += ep_r
    return total_reward / n_episodes


def run_single(
    env_name: str,
    algo_name: str,
    seed: int,
    total_timesteps: int = TOTAL_TIMESTEPS,
    results_dir: Path = RESULTS_DIR,
    skip_existing: bool = True,
) -> dict:
    """
    运行单个 (env, algo, seed) 组合。

    返回：包含完整指标的 dict，同时保存到 JSON。
    """
    save_dir = str(results_dir / env_name / algo_name)
    os.makedirs(save_dir, exist_ok=True)
    out_path = Path(save_dir) / f"{algo_name}_s{seed}.json"

    if skip_existing and out_path.exists():
        data = json.load(open(out_path))
        print(f"  [SKIP] {env_name}/{algo_name}/s{seed} "
              f"(final={data.get('final_reward', '?'):.1f})")
        return data

    # ── 种子设置 ──
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── 环境创建 ──
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 100_000)

    # ── Agent 构建 ──
    agent = build_agent(algo_name, env, seed, save_dir)
    use_dcppo_s = getattr(agent, '_use_dcppo_s', False)
    agent._total_timesteps = total_timesteps

    # ── 训练状态 ──
    eval_rewards, eval_steps = [], []
    episode_rewards = []
    bhvf_diag_log = []   # BHVF 诊断记录（每次 rollout 一条）

    obs, _ = env.reset()
    if hasattr(agent, 'update_obs_rms'):
        agent.update_obs_rms(obs)
    obs = agent.normalize_obs(obs) if hasattr(agent, 'normalize_obs') else obs

    ep_reward, ep_length = 0.0, 0
    total_steps = 0
    last_eval_step = -EVAL_FREQ  # 确保第一次立即评估
    t0 = time.time()
    rollout_idx = 0

    print(f"  START {env_name}/{algo_name}/s{seed} "
          f"[DCPPO-S={'ON' if use_dcppo_s else 'OFF'}]")

    while total_steps < total_timesteps:
        # ── Rollout 采集 ──
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
            ep_length += 1

            agent.buffer.add(obs_n, action_np, float(reward), float(terminated),
                             log_prob.item(), value.item())
            obs = next_obs
            total_steps += 1

            if terminated or truncated:
                episode_rewards.append(ep_reward)
                ep_reward, ep_length = 0.0, 0
                obs, _ = env.reset()
                if hasattr(agent, 'update_obs_rms'):
                    agent.update_obs_rms(obs)
                obs = agent.normalize_obs(obs) if hasattr(agent, 'normalize_obs') else obs

        # ── Bootstrap value ──
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            last_val = agent.critic(obs_t).item()

        agent.total_steps = total_steps

        # ── GAE / BHVF 计算 ──
        if hasattr(agent, 'compute_hindsight_gae'):
            agent.compute_hindsight_gae(last_val)
        else:
            agent.compute_gae(last_val)

        # ── DCPPO-S：EV 线性收缩优势（命题 1：仅缩放幅度，不改变方向）──
        if use_dcppo_s:
            ev_weight = float(np.clip(agent._ev_ema, 0.1, 1.0))
            T = agent.buffer.pos
            agent.buffer.advantages[:T] *= ev_weight

        # ── 参数更新 ──
        metrics = agent.update()
        rollout_idx += 1

        # ── 记录 BHVF diagnostics（每次 rollout）──
        if isinstance(agent, OptimalHCGAE_Bayesian):
            bhvf_diag_log.append({
                "step":        total_steps,
                "scr":         agent._diag_scr,
                "sigma_V":     agent._diag_sigma_V,
                "sigma_G":     agent._diag_sigma_G,
                "sigma_e":     agent._diag_sigma_e,
                "alpha_star":  agent._diag_alpha_star,
                "c_mc":        agent._diag_c_mc,
                "ev_now":      agent._diag_ev_now,
                "clip_ratio":  agent._diag_clip_ratio,
            })

        # ── 定期评估 ──
        if total_steps - last_eval_step >= EVAL_FREQ:
            eval_r = evaluate_policy(agent, eval_env, N_EVAL_EPISODES)
            eval_rewards.append(eval_r)
            eval_steps.append(total_steps)
            last_eval_step = total_steps

            # 打印进度
            elapsed = time.time() - t0
            fps = int(total_steps / (elapsed + 1e-8))
            pct = total_steps / total_timesteps * 100
            ev = metrics.get("explained_variance", 0.0)
            print(f"    [{algo_name:<15} s{seed}] "
                  f"{total_steps:7d}/{total_timesteps} ({pct:4.0f}%) "
                  f"| eval={eval_r:7.1f} "
                  f"| EV={ev:+.3f} "
                  f"| {fps:4d}fps {elapsed:5.0f}s",
                  end="")
            if isinstance(agent, OptimalHCGAE_Bayesian) and bhvf_diag_log:
                d = bhvf_diag_log[-1]
                print(f" | SCR={d['scr']:.3f} α*={d['alpha_star']:.3f} "
                      f"clip={d['clip_ratio']:.2f}", end="")
            print()

    # ── 整理最终结果 ──
    n_final = 10
    final_mean = (float(np.mean(eval_rewards[-n_final:])) if len(eval_rewards) >= n_final
                  else float(np.mean(eval_rewards)) if eval_rewards else 0.0)
    max_reward = float(max(eval_rewards)) if eval_rewards else 0.0
    elapsed = time.time() - t0

    # BHVF diagnostics：稀疏化保存（每隔 10 个 rollout 取一次，减小文件大小）
    diag_sparse = bhvf_diag_log[::10] if bhvf_diag_log else []

    result = {
        "env":    env_name,
        "agent":  algo_name,
        "seed":   seed,
        # ── 超参数（完整可溯源）──
        "config": {
            "hidden_dim":     256,
            "lr":             3e-4,
            "gamma":          0.99,
            "lam":            0.95,
            "eps_clip":       0.2,
            "n_epochs":       10,
            "batch_size":     64,
            "n_steps":        2048,
            "ent_coef":       0.0,
            "vf_coef":        0.5,
            "max_grad_norm":  0.5,
            "use_obs_norm":   True if algo_name != "Standard_PPO" else False,
            "use_adv_norm":   True if algo_name != "Standard_PPO" else False,
            "use_lr_anneal":  True if algo_name != "Standard_PPO" else False,
            "total_timesteps": total_timesteps,
            "eval_freq":      EVAL_FREQ,
            "n_eval_episodes": N_EVAL_EPISODES,
            "eval_mode":      "deterministic_mean",
            # BHVF 特有
            "scr_ema_alpha":  BHVF_KWARGS["scr_ema_alpha"] if "BHVF" in algo_name else None,
            "scr_relax":      BHVF_KWARGS["scr_relax"]     if "BHVF" in algo_name else None,
            "clip_c":         BHVF_KWARGS["clip_c"]         if "BHVF" in algo_name else None,
            "use_dcppo_s":    use_dcppo_s,
            "dcppo_w_min":    0.1,
        },
        # ── 主要性能指标 ──
        "total_steps":   total_steps,
        "elapsed_s":     elapsed,
        "final_reward":  final_mean,       # 论文 Table 1 数据源（最后10次评估均值）
        "max_reward":    max_reward,
        "eval_rewards":  eval_rewards,
        "eval_steps":    eval_steps,
        "episode_rewards": episode_rewards[-200:],   # 最近200条 episode
        # ── BHVF 中间状态诊断（用于验证理论预测）──
        "bhvf_diagnostics": diag_sparse,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    env.close()
    eval_env.close()
    print(f"  DONE  {env_name}/{algo_name}/s{seed} "
          f"→ final10={final_mean:.1f} max={max_reward:.1f} ({elapsed:.0f}s)")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 汇总统计
# ─────────────────────────────────────────────────────────────────────────────
def compute_stats(env_name: str, algo_name: str, seeds: list,
                  results_dir: Path, n_final: int = 10) -> dict:
    """
    读取多个种子的结果，计算均值/标准差/中位数。
    """
    finals = []
    for s in seeds:
        fp = results_dir / env_name / algo_name / f"{algo_name}_s{s}.json"
        if fp.exists():
            try:
                d = json.load(open(fp))
                er = d.get("eval_rewards", [])
                if er:
                    v = (float(np.mean(er[-n_final:])) if len(er) >= n_final
                         else float(np.mean(er)))
                    finals.append(v)
            except Exception as e:
                print(f"  [WARN] Failed to load {fp}: {e}")
    if not finals:
        return {"mean": None, "std": None, "median": None, "n": 0, "values": []}
    return {
        "mean":   float(np.mean(finals)),
        "std":    float(np.std(finals)),
        "median": float(np.median(finals)),
        "n":      len(finals),
        "values": finals,
    }


def print_summary_table(results_dir: Path = RESULTS_DIR,
                        seeds: list = SEEDS, n_final: int = 10):
    """
    打印论文 Table 1 风格的汇总表。
    """
    print("\n" + "=" * 90)
    print("  ICML 2026 Table 1 — Final Performance (mean ± std, 12 seeds, last 10 evals)")
    print("=" * 90)
    header = f"  {'Algorithm':<20}" + "".join(f"  {e:<18}" for e in ENVS)
    print(header)
    print("-" * 90)

    for algo in ALGOS:
        row = f"  {algo:<20}"
        for env in ENVS:
            st = compute_stats(env, algo, seeds, results_dir, n_final)
            if st["mean"] is not None:
                row += f"  {st['mean']:7.1f} ± {st['std']:5.1f}  "
            else:
                row += f"  {'[TBD]':>16}  "
        print(row)
    print("=" * 90 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ICML 2026 完备实验")
    parser.add_argument("--env",   type=str, default=None,
                        help="单环境运行（默认：全部4个）")
    parser.add_argument("--algo",  type=str, default=None,
                        help="单算法运行（默认：全部5个）")
    parser.add_argument("--seeds", type=str, default=None,
                        help="逗号分隔的种子列表，如 0,1,2,3（默认：0-11）")
    parser.add_argument("--smoke-test", action="store_true",
                        help="快速冒烟测试（50k步，种子 0,1）")
    parser.add_argument("--no-skip",   action="store_true",
                        help="强制重新运行（覆盖已有结果）")
    parser.add_argument("--summary",   action="store_true",
                        help="仅打印已有结果汇总，不运行实验")
    args = parser.parse_args()

    # ── 参数解析 ──
    envs  = [args.env]  if args.env  else ENVS
    algos = [args.algo] if args.algo else ALGOS
    seeds = ([int(s) for s in args.seeds.split(",")]
             if args.seeds else SEEDS)
    total_ts = 50_000 if args.smoke_test else TOTAL_TIMESTEPS
    skip_existing = not args.no_skip

    if args.smoke_test:
        seeds = [0, 1]
        print(f"\n{'='*60}")
        print(f"  ⚡ SMOKE TEST: 50k steps, seeds {seeds}")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print(f"  ICML 2026 Full Experiment")
        print(f"  Envs:  {envs}")
        print(f"  Algos: {algos}")
        print(f"  Seeds: {seeds} ({len(seeds)} total)")
        print(f"  Steps: {total_ts:,}")
        total_runs = len(envs) * len(algos) * len(seeds)
        print(f"  Total runs: {total_runs}")
        print(f"{'='*60}\n")

    if args.summary:
        print_summary_table(RESULTS_DIR, SEEDS, n_final=10)
        return

    # ── 执行实验 ──
    results_dir = (RESULTS_DIR.parent / "SmokeTest") if args.smoke_test else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    t_global = time.time()

    for env_name in envs:
        for algo_name in algos:
            print(f"\n{'─'*60}")
            print(f"  [{algo_name}] on [{env_name}]")
            print(f"{'─'*60}")
            for seed in seeds:
                try:
                    r = run_single(
                        env_name, algo_name, seed,
                        total_timesteps=total_ts,
                        results_dir=results_dir,
                        skip_existing=skip_existing,
                    )
                    all_results.append(r)
                except Exception as e:
                    import traceback
                    print(f"\n  ✗ FAILED {env_name}/{algo_name}/s{seed}: {e}")
                    traceback.print_exc()
                    continue

    elapsed = time.time() - t_global
    print(f"\n{'='*60}")
    print(f"  All done! Total time: {elapsed/3600:.2f}h ({elapsed:.0f}s)")
    print(f"{'='*60}\n")

    # 打印汇总
    print_summary_table(results_dir, seeds, n_final=5 if args.smoke_test else 10)


if __name__ == "__main__":
    main()

