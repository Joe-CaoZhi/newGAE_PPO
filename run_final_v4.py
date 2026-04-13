#!/usr/bin/env python3
"""
Optimal_HCGAE_v4 完备实验
=========================
目标：与 FinalExperiment 完全对齐，证明 v4 是否真正优于已有算法。

实验设计：
  - 4 个环境: HalfCheetah-v4, Hopper-v4, Walker2d-v4, Ant-v4
  - 12 个随机种子: seeds 0-11 (与 FinalExperiment 完全一致)
  - 1M 步: total_steps = 1_000_000 (与 FinalExperiment TOTAL_TIMESTEPS=1_000_000 对齐)
  - 评估频率: 每 20_480 步评估 10 episode (确定性均值，与 FinalExperiment EVAL_FREQ 对齐)
  - 支持 --env / --seed CLI 参数，便于并行启动

结果保存到: results/FinalExperiment_v4/
  └── {env}/Optimal_HCGAE_v4/{algo}_s{seed}.json

支持断点续跑：已存在结果文件直接 SKIP。

Usage（并行4进程）:
  python3 run_final_v4.py --env HalfCheetah-v4 &
  python3 run_final_v4.py --env Hopper-v4 &
  python3 run_final_v4.py --env Walker2d-v4 &
  python3 run_final_v4.py --env Ant-v4 &
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

# ─── 实验配置（与 FinalExperiment 完全一致）──────────────────────────────────
ENVS        = ["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4", "Ant-v4"]
ALGOS       = ["Optimal_PPO", "Optimal_HCGAE_v4"]
SEEDS       = list(range(12))          # 0-11，与 FinalExperiment 对齐
TOTAL_STEPS = 489 * 2048               # = 1_001_472，与 FinalExperiment 完全对齐（最后一次 eval 在 step=985088）
EVAL_FREQ   = 20_480                   # 与 FinalExperiment 对齐（约10次rollout评估一次）
N_EVAL_EPS  = 10
RESULTS_DIR = Path("results/FinalExperiment_v4")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 超参与 FinalExperiment 完全一致
AGENT_KWARGS = dict(
    hidden_dim=256, lr=3e-4, gamma=0.99, lam=0.95, eps_clip=0.2,
    n_epochs=10, batch_size=64, n_steps=2048, ent_coef=0.0,
    vf_coef=0.5, max_grad_norm=0.5, use_obs_norm=True,
    use_adv_norm=True, use_lr_anneal=True, use_vclip=False, device="cpu",
)


# ─── 确定性评估（严格对齐 run_icml_experiment.py evaluate_policy）─────────────
# 协议：每步开头 normalize obs → 采取确定性动作 → step → obs=next_obs(原始)
#       下步开头再 normalize，避免双重归一化。
def evaluate_policy(agent, eval_env, n_eps=N_EVAL_EPS):
    total_reward = 0.0
    for _ in range(n_eps):
        obs, _ = eval_env.reset()
        done = False
        ep_r = 0.0
        while not done:
            obs_n = agent.normalize_obs(obs) if hasattr(agent, 'normalize_obs') else obs
            obs_t = torch.FloatTensor(obs_n).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                dist = agent.actor(obs_t)
                # 确定性评估：使用分布均值（与 FinalExperiment 对齐）
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
            obs = next_obs   # 保持原始 obs，下步开头再 normalize
        total_reward += ep_r
    return total_reward / n_eps


# ─── 单次运行（含断点续跑）───────────────────────────────────────────────────
def run_single(env_name, algo_name, seed, total_steps=TOTAL_STEPS):
    save_dir = str(RESULTS_DIR / env_name / algo_name)
    os.makedirs(save_dir, exist_ok=True)
    out_path = Path(save_dir) / f"{algo_name}_s{seed}.json"

    # 断点续跑：已存在则跳过
    if out_path.exists():
        data = json.load(open(out_path))
        if data.get("total_steps", 0) >= total_steps:
            print(f"  [SKIP] {env_name}/{algo_name}/s{seed}  "
                  f"(total_steps={data['total_steps']:,})")
            return data
        else:
            print(f"  [RESUME?] {env_name}/{algo_name}/s{seed}  "
                  f"only {data['total_steps']:,} steps, re-running from scratch")

    print(f"\n  [{env_name}/{algo_name}/s{seed}] Starting... ({total_steps:,} steps)")
    np.random.seed(seed)
    torch.manual_seed(seed)
    env      = gym.make(env_name)
    env.reset(seed=seed)
    eval_env = gym.make(env_name)
    eval_env.reset(seed=seed + 100_000)   # 与 FinalExperiment 对齐（+100000）

    kw    = dict(**AGENT_KWARGS, save_dir=save_dir)
    agent = build_optimal_agent(algo_name, env, name=f"{algo_name}_s{seed}", **kw)
    agent._total_timesteps = total_steps

    eval_rewards, eval_steps = [], []
    diag_log = []
    obs, _ = env.reset()
    if hasattr(agent, 'update_obs_rms'):
        agent.update_obs_rms(obs)
    obs = agent.normalize_obs(obs) if hasattr(agent, 'normalize_obs') else obs

    step_count, last_eval = 0, 0
    t0 = time.time()
    ep_reward, ep_length = 0.0, 0

    while step_count < total_steps:
        # ── 收集 rollout ─────────────────────────────────────────────────────
        agent.buffer.reset()
        for _ in range(agent.n_steps):
            # obs_rms 在每步开头更新（与 FinalExperiment 对齐：先 update 再 normalize）
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

            nobs, r, term, trunc, _ = env.step(action_np)
            ep_reward += r
            ep_length += 1

            # buffer 存入已归一化的 obs_n（与 FinalExperiment 对齐）
            agent.buffer.add(obs_n, action_np, float(r), float(term),
                             log_prob.item(), value.item())
            obs = nobs
            step_count += 1

            if term or trunc:
                ep_reward, ep_length = 0.0, 0
                obs, _ = env.reset()
                if hasattr(agent, 'update_obs_rms'):
                    agent.update_obs_rms(obs)
                obs = agent.normalize_obs(obs) if hasattr(agent, 'normalize_obs') else obs

            if step_count >= total_steps:
                break

        # ── Bootstrap value ──────────────────────────────────────────────────
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            last_v = agent.critic(obs_t).item()
        agent.total_steps = step_count

        # ── GAE / BHVF 计算（优先使用 compute_hindsight_gae）────────────────
        if hasattr(agent, 'compute_hindsight_gae'):
            agent.compute_hindsight_gae(last_v)
        else:
            agent.compute_gae(last_v)
        agent.update()

        # ── v4 诊断日志 ──────────────────────────────────────────────────────
        if algo_name == "Optimal_HCGAE_v4" and hasattr(agent, '_scr_ema'):
            scr_ema = float(agent._scr_ema)
            alpha_cap = scr_ema**2 / (1 + scr_ema**2) + agent.scr_relax
            diag_log.append({
                "step":      step_count,
                "scr_ema":   round(scr_ema, 4),
                "alpha_cap": round(min(alpha_cap, 1.0), 4),
                "ev_ema":    round(float(agent._ev_ema), 4),
            })

        # ── 评估 ─────────────────────────────────────────────────────────────
        if step_count - last_eval >= EVAL_FREQ or step_count >= total_steps:
            er = evaluate_policy(agent, eval_env)
            eval_rewards.append(er)
            eval_steps.append(step_count)
            last_eval = step_count

            elapsed = time.time() - t0
            pct     = 100 * step_count / total_steps

            diag_str = ""
            if diag_log and algo_name == "Optimal_HCGAE_v4":
                d = diag_log[-1]
                diag_str = (f"  SCR={d['scr_ema']:.3f}"
                            f"  α_cap={d['alpha_cap']:.3f}"
                            f"  EV={d['ev_ema']:.3f}")
            print(f"  {env_name}/{algo_name}/s{seed}  "
                  f"{pct:.0f}%  eval={er:.1f}{diag_str}  ({elapsed:.0f}s)")

    # ── 保存结果 ─────────────────────────────────────────────────────────────
    top10 = (float(np.mean(sorted(eval_rewards)[-10:]))
             if len(eval_rewards) >= 10 else float(np.mean(eval_rewards)))
    last10 = (float(np.mean(eval_rewards[-10:]))
              if len(eval_rewards) >= 10 else float(np.mean(eval_rewards)))

    result = {
        "env":          env_name,
        "agent":        algo_name,
        "seed":         seed,
        "config": {
            "hidden_dim":   256,
            "use_obs_norm": True,
            "use_adv_norm": True,
            "use_lr_anneal": True,
            "eval_mode":    "deterministic_mean",
            # v4-specific
            "scr_ema_alpha": 0.1,
            "scr_relax":     0.05,
        },
        "total_steps":  step_count,
        "final_reward": last10,
        "top10_reward": top10,
        "eval_rewards": eval_rewards,
        "eval_steps":   eval_steps,
        "diag_log":     diag_log[-50:] if diag_log else [],  # 保留最后50条
        "elapsed_s":    round(time.time() - t0, 1),
    }
    json.dump(result, open(out_path, 'w'), indent=2)
    elapsed_total = time.time() - t0
    print(f"  => Saved {out_path}  "
          f"last10={last10:.1f}  top10={top10:.1f}  ({elapsed_total:.0f}s)")
    env.close()
    eval_env.close()
    return result


# ─── 汇总打印 ────────────────────────────────────────────────────────────────
def print_summary(envs, algos, seeds):
    """打印 Top-10 奖励均值 ± SEM，并与 FinalExperiment 对比"""
    final_dir = Path("results/FinalExperiment")
    print("\n" + "="*80)
    print("完备实验汇总 (Top-10 eval reward mean ± SEM)")
    print("="*80)

    for env_name in envs:
        print(f"\n  {env_name}:")
        print(f"  {'算法':35s} | {'本实验':>10} | {'FinalExp(@1M)':>14} | {'差值':>10}")
        print(f"  {'-'*35}-+-{'-'*10}-+-{'-'*14}-+-{'-'*10}")

        for algo in algos:
            # 本实验
            tops = []
            for s in seeds:
                p = RESULTS_DIR / env_name / algo / f"{algo}_s{s}.json"
                if p.exists():
                    d = json.load(open(p))
                    er = d.get('eval_rewards', [])
                    if er:
                        tops.append(float(np.mean(sorted(er)[-10:])))

            # FinalExperiment（只查 Optimal_PPO 作为基线）
            final_tops = []
            final_algo = "Optimal_PPO" if algo == "Optimal_HCGAE_v4" else algo
            fp = final_dir / env_name / final_algo
            if fp.exists():
                for j in sorted(fp.glob("*.json")):
                    d = json.load(open(j))
                    er = d.get('eval_rewards', [])
                    if er:
                        final_tops.append(float(np.mean(sorted(er)[-10:])))

            if tops:
                m   = np.mean(tops)
                sem = np.std(tops) / max(np.sqrt(len(tops)), 1)
                local_str = f"{m:8.1f} ± {sem:4.1f} (n={len(tops)})"
            else:
                local_str = "        N/A"

            if final_tops:
                fm   = np.mean(final_tops)
                fsem = np.std(final_tops) / max(np.sqrt(len(final_tops)), 1)
                final_str = f"{fm:8.1f} ± {fsem:4.1f}"
                delta_str = f"{np.mean(tops)-fm:+8.1f}" if tops else "     N/A"
            else:
                final_str = "            N/A"
                delta_str = "       N/A"

            print(f"  {algo:35s} | {local_str:>10} | {final_str:>14} | {delta_str:>10}")


# ─── 入口 ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Optimal_HCGAE_v4 完备实验 (4env × 12seeds × 1M steps)"
    )
    parser.add_argument("--env",   default=None,
                        help="只跑指定环境 (e.g. HalfCheetah-v4)")
    parser.add_argument("--algo",  default=None,
                        help="只跑指定算法 (Optimal_PPO 或 Optimal_HCGAE_v4)")
    parser.add_argument("--seed",  type=int, default=None,
                        help="只跑指定种子")
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS,
                        help=f"总步数 (default: {TOTAL_STEPS})")
    parser.add_argument("--summary", action="store_true",
                        help="只打印汇总，不运行实验")
    args = parser.parse_args()

    envs   = [args.env]  if args.env  else ENVS
    algos  = [args.algo] if args.algo else ALGOS
    seeds  = [args.seed] if args.seed is not None else SEEDS
    tsteps = args.steps

    if args.summary:
        print_summary(ENVS, ALGOS, SEEDS)
        sys.exit(0)

    print(f"\n{'='*70}")
    print(f"Optimal_HCGAE_v4 完备实验")
    print(f"  算法: {algos}")
    print(f"  环境: {envs}")
    print(f"  种子: {seeds}  ({len(seeds)} 个)")
    print(f"  步数: {tsteps:,}  (≈ {tsteps/1e6:.2f}M)")
    print(f"  结果目录: {RESULTS_DIR}")
    print(f"{'='*70}\n")

    for env_name in envs:
        for algo in algos:
            for s in seeds:
                run_single(env_name, algo, s, tsteps)

    # 打印汇总
    print_summary(envs, algos, seeds)
    print("\nDone!")

