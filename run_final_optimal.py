#!/usr/bin/env python3
"""
HCGAE_Optimal vs Optimal_PPO 正式对齐实验
==========================================
条件严格对齐：
  - 1M 步训练
  - 20 seeds (0-19)
  - 4 个 MuJoCo 环境 (HC / Hopper / Walker2d / Ant)
  - 相同超参数骨架 (OptimalPPO SHARED_KWARGS)
  - 确定性均值动作评估
  - 断点续跑支持

算法对比:
  1. Optimal_PPO          ── 基线
  2. Optimal_HCGAE_Optimal ── FixSCR + MC-Normalized Sigmoid
     理论依据: α* = (B²+σ²_V)/(B²+σ²_V+σ²_G) [James-Stein/Kalman/Bayes 统一]

输出: results/FinalOptimal/{env}/{algo}/{algo}_s{seed}.json
      每个 JSON 含 eval_curve, final_reward (末10次均值), peak_reward (峰值窗口10)
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

# ─────────────────────────────────────────────────────────────────────────────
# 实验常量
# ─────────────────────────────────────────────────────────────────────────────
TOTAL_TIMESTEPS  = 1_000_000
EVAL_FREQ        = 20_480        # ≈ 10 rollouts，与 ICML 实验对齐
N_EVAL_EPISODES  = 10
N_PEAK           = 10            # peak 窗口大小
SEEDS            = list(range(20))  # 0-19, 固定，可溯源

ENVS  = ["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4", "Ant-v4"]
ALGOS = ["Optimal_PPO", "Optimal_HCGAE_Optimal"]

RESULTS_DIR = Path("results/FinalOptimal")

# 所有算法完全共享的超参数（与 run_icml_experiment.py 的 SHARED_KWARGS 一致）
SHARED_KWARGS = dict(
    hidden_dim     = 256,
    lr             = 3e-4,
    gamma          = 0.99,
    lam            = 0.95,
    eps_clip       = 0.2,
    n_epochs       = 10,
    batch_size     = 64,
    n_steps        = 2048,
    ent_coef       = 0.0,
    vf_coef        = 0.5,
    max_grad_norm  = 0.5,
    use_obs_norm   = True,
    use_adv_norm   = True,
    use_lr_anneal  = True,
    use_vclip      = False,
    device         = "cpu",
)


# ─────────────────────────────────────────────────────────────────────────────
# 评估
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_policy(agent, eval_env, n_episodes=N_EVAL_EPISODES):
    """确定性均值动作评估（与 run_icml_experiment.py 完全一致）。"""
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


# ─────────────────────────────────────────────────────────────────────────────
# 单次 trial
# ─────────────────────────────────────────────────────────────────────────────
def run_single(env_name, algo_name, seed,
               total_timesteps=TOTAL_TIMESTEPS,
               results_dir=RESULTS_DIR,
               skip_existing=True):
    save_dir = results_dir / env_name / algo_name
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{algo_name}_s{seed}.json"

    if skip_existing and out_path.exists():
        d = json.load(open(out_path))
        print(f"  [SKIP] {env_name}/{algo_name}/s{seed} "
              f"(final={d.get('final_reward', '?'):.1f})")
        return d

    np.random.seed(seed)
    torch.manual_seed(seed)

    env      = gym.make(env_name)
    eval_env = gym.make(env_name)
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 100_000)

    agent = build_optimal_agent(
        algo_name, env,
        name=f"{algo_name}_s{seed}",
        **SHARED_KWARGS
    )
    agent._total_timesteps = total_timesteps

    eval_rewards, eval_steps = [], []
    episode_rewards = []
    diag_log = []      # diagnostics (scr_ema, ev_ema, g_std_ema)
    peak10_running = 0.0

    obs, _ = env.reset()
    if hasattr(agent, 'update_obs_rms'):
        agent.update_obs_rms(obs)
    obs = agent.normalize_obs(obs) if hasattr(agent, 'normalize_obs') else obs

    ep_reward = 0.0
    total_steps = 0
    last_eval_step = -EVAL_FREQ
    t0 = time.time()

    print(f"  START {env_name}/{algo_name}/s{seed}")

    while total_steps < total_timesteps:
        # ── Rollout ──
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
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                obs, _ = env.reset()
                if hasattr(agent, 'update_obs_rms'):
                    agent.update_obs_rms(obs)
                obs = agent.normalize_obs(obs) if hasattr(agent, 'normalize_obs') else obs

        # ── Bootstrap ──
        obs_n_boot = agent.normalize_obs(obs) if hasattr(agent, 'normalize_obs') else obs
        obs_t = torch.FloatTensor(obs_n_boot).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            last_val = agent.critic(obs_t).item()

        agent.total_steps = total_steps

        # ── GAE ──
        if hasattr(agent, 'compute_hindsight_gae'):
            agent.compute_hindsight_gae(last_val)
        else:
            agent.compute_gae(last_val)

        # ── Update ──
        metrics = agent.update()

        # ── Diagnostics ──
        diag_entry = {"step": total_steps}
        if hasattr(agent, '_scr_ema'):    diag_entry['scr_ema']    = float(agent._scr_ema)
        if hasattr(agent, '_ev_ema'):     diag_entry['ev_ema']     = float(agent._ev_ema)
        if hasattr(agent, '_g_std_ema'):  diag_entry['g_std_ema']  = float(agent._g_std_ema)
        diag_log.append(diag_entry)

        # ── 评估 ──
        if total_steps - last_eval_step >= EVAL_FREQ:
            eval_r = evaluate_policy(agent, eval_env)
            eval_rewards.append(eval_r)
            eval_steps.append(total_steps)
            last_eval_step = total_steps

            # 实时 peak10
            peak_idx = int(np.argmax(eval_rewards))
            half = N_PEAK // 2
            lo = max(0, peak_idx - half)
            hi = min(len(eval_rewards), lo + N_PEAK)
            lo = max(0, hi - N_PEAK)
            peak10_running = float(np.mean(eval_rewards[lo:hi]))

            elapsed = time.time() - t0
            ev = metrics.get("explained_variance", 0.0)
            pct = total_steps / total_timesteps * 100
            fps = int(total_steps / (elapsed + 1e-8))
            final_so_far = float(np.mean(eval_rewards[-min(N_PEAK, len(eval_rewards)):]))
            scr_str = f" SCR={agent._scr_ema:.3f}" if hasattr(agent, '_scr_ema') else ""
            ev_str  = f" EV={ev:+.3f}"
            print(f"    [{algo_name:<26} s{seed}] "
                  f"{total_steps:7d}/{total_timesteps} ({pct:4.0f}%) "
                  f"| eval={eval_r:7.1f} peak10={peak10_running:7.1f}"
                  f"{ev_str}{scr_str} | {fps}fps {elapsed:.0f}s")

    # ── 整理结果 ──
    final_mean = (float(np.mean(eval_rewards[-N_PEAK:])) if len(eval_rewards) >= N_PEAK
                  else float(np.mean(eval_rewards)) if eval_rewards else 0.0)

    result = {
        "env":            env_name,
        "agent":          algo_name,
        "seed":           seed,
        "config": {
            **SHARED_KWARGS,
            "total_timesteps":  total_timesteps,
            "eval_freq":        EVAL_FREQ,
            "n_eval_episodes":  N_EVAL_EPISODES,
            "eval_mode":        "deterministic_mean",
        },
        "total_steps":    total_steps,
        "elapsed_s":      time.time() - t0,
        "final_reward":   final_mean,
        "peak_reward":    peak10_running,
        "max_reward":     float(max(eval_rewards)) if eval_rewards else 0.0,
        "eval_rewards":   eval_rewards,
        "eval_steps":     eval_steps,
        "episode_rewards": episode_rewards[-200:],
        "diagnostics":    diag_log[::10],   # 稀疏保存，每10步一条
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    env.close()
    eval_env.close()
    print(f"  DONE  {env_name}/{algo_name}/s{seed} "
          f"→ final10={final_mean:.1f} peak10={peak10_running:.1f} "
          f"max={result['max_reward']:.1f} ({result['elapsed_s']:.0f}s)")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 汇总
# ─────────────────────────────────────────────────────────────────────────────
def _load_stats(results_dir, algo, env, seeds):
    finals, peaks, maxs = [], [], []
    for s in seeds:
        fp = results_dir / env / algo / f"{algo}_s{s}.json"
        if fp.exists():
            try:
                d = json.load(open(fp))
                er = d.get("eval_rewards", [])
                if er:
                    finals.append(float(np.mean(er[-N_PEAK:])) if len(er) >= N_PEAK
                                  else float(np.mean(er)))
                    peaks.append(d.get("peak_reward", float(np.max(er))))
                    maxs.append(d.get("max_reward", float(np.max(er))))
            except Exception as e:
                print(f"  [WARN] {fp}: {e}")
    return finals, peaks, maxs


def print_summary(results_dir=RESULTS_DIR, seeds=SEEDS):
    from scipy import stats as scipy_stats

    env_labels = {"HalfCheetah-v4": "HC", "Hopper-v4": "Hop",
                  "Walker2d-v4": "Wal", "Ant-v4": "Ant"}

    print("\n" + "=" * 100)
    print(f"  FinalOptimal Summary  (1M steps, {len(seeds)} seeds)")
    print("  James-Stein/Kalman/Bayes unified → HCGAE_Optimal = FixSCR + MC-Normalized Sigmoid")
    print("=" * 100)

    # 收集数据
    data = {}
    for algo in ALGOS:
        data[algo] = {}
        for env in ENVS:
            f, p, m = _load_stats(results_dir, algo, env, seeds)
            data[algo][env] = dict(finals=f, peaks=p, maxs=m)

    ppo_key = "Optimal_PPO"

    for metric_name, key in [("Final (last-10 mean)", "finals"),
                               ("Peak  (peak-10 window)", "peaks")]:
        print(f"\n── {metric_name} ──")
        print(f"  {'Algo':<35}", end="")
        for env in ENVS:
            print(f"  {env_labels[env]:>16}", end="")
        print(f"  {'Avg Δ':>8}  {'n':>4}")
        print("  " + "-" * 95)

        for algo in ALGOS:
            print(f"  {algo:<35}", end="")
            deltas = []
            n_seeds = 0
            for env in ENVS:
                vals = data[algo][env][key]
                ppo_vals = data[ppo_key][env][key]
                if vals:
                    m, s = np.mean(vals), np.std(vals)
                    n_seeds = max(n_seeds, len(vals))
                    cell = f"{m:.0f}±{s:.0f}(n={len(vals)})"
                    if algo != ppo_key and ppo_vals:
                        d_pct = (m - np.mean(ppo_vals)) / abs(np.mean(ppo_vals)) * 100
                        deltas.append(d_pct)
                        cell += f"({d_pct:+.1f}%)"
                else:
                    cell = "N/A"
                print(f"  {cell:>16}", end="")
            avg_d = f"{np.mean(deltas):+.1f}%" if deltas else "---"
            print(f"  {avg_d:>8}  {n_seeds:>4}")

    # t-test 显著性
    print(f"\n── t-test vs Optimal_PPO (p-value, 双侧) ──")
    print(f"  {'Algo':<35}", end="")
    for env in ENVS:
        print(f"  {env_labels[env]:>14}", end="")
    print()
    print("  " + "-" * 90)
    for algo in [a for a in ALGOS if a != ppo_key]:
        print(f"  {algo:<35}", end="")
        for env in ENVS:
            vals = data[algo][env]["finals"]
            ppo_vals = data[ppo_key][env]["finals"]
            if vals and ppo_vals and len(vals) >= 3 and len(ppo_vals) >= 3:
                _, p = scipy_stats.ttest_ind(vals, ppo_vals)
                d = (np.mean(vals) - np.mean(ppo_vals)) / abs(np.mean(ppo_vals)) * 100
                sig = "**" if p < 0.05 else ("*" if p < 0.10 else "  ")
                cell = f"{d:+.1f}%{sig}(p={p:.3f})"
            else:
                cell = "N/A"
            print(f"  {cell:>14}", end="")
        print()

    print()


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import multiprocessing as mp
    import subprocess

    parser = argparse.ArgumentParser(description="HCGAE_Optimal vs Optimal_PPO 正式对齐实验")
    parser.add_argument("--envs",    nargs="+", default=ENVS,  help="环境列表")
    parser.add_argument("--algos",   nargs="+", default=ALGOS, help="算法列表")
    parser.add_argument("--seeds",   nargs="+", type=int, default=SEEDS, help="种子列表")
    parser.add_argument("--steps",   type=int, default=TOTAL_TIMESTEPS, help="训练步数")
    parser.add_argument("--workers", type=int, default=None, help="并行 worker 数（默认 CPU 数）")
    parser.add_argument("--summarize", action="store_true", help="只打印汇总表")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR), help="结果目录")
    # 单 trial 模式（由并行调度器内部调用）
    parser.add_argument("--single-env",   default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-algo",  default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-seed",  type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── 单 trial 模式：由子进程调用 ──────────────────────────────────────────
    if args.single_env is not None:
        run_single(args.single_env, args.single_algo, args.single_seed,
                   args.steps, results_dir, skip_existing=True)
        sys.exit(0)

    # ── 汇总模式 ─────────────────────────────────────────────────────────────
    if args.summarize:
        print_summary(results_dir, args.seeds)
        sys.exit(0)

    # ── 主调度：用 subprocess 启动独立子进程，避免 macOS spawn 初始化问题 ────
    tasks = [
        (env, algo, seed)
        for env in args.envs
        for algo in args.algos
        for seed in args.seeds
    ]
    n_workers = args.workers or mp.cpu_count()
    script = os.path.abspath(__file__)

    print(f"Total trials : {len(tasks)}")
    print(f"  Envs       : {args.envs}")
    print(f"  Algos      : {args.algos}")
    print(f"  Seeds      : {len(args.seeds)}  ({args.seeds[0]}..{args.seeds[-1]})")
    print(f"  Steps      : {args.steps:,}")
    print(f"  Workers    : {n_workers}")
    print(f"  Output     : {results_dir}\n")

    pending = list(tasks)
    running: dict = {}   # pid → (Popen, env, algo, seed)
    done_count = 0

    while pending or running:
        # 补充空位
        while len(running) < n_workers and pending:
            env, algo, seed = pending.pop(0)
            cmd = [
                sys.executable, script,
                "--single-env",  env,
                "--single-algo", algo,
                "--single-seed", str(seed),
                "--steps",       str(args.steps),
                "--results-dir", str(results_dir),
            ]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(Path(script).parent),
            )
            running[proc.pid] = (proc, env, algo, seed)

        # 轮询已完成的进程
        for pid in list(running.keys()):
            proc, env, algo, seed = running[pid]
            ret = proc.poll()
            if ret is not None:
                out, _ = proc.communicate()
                for line in out.strip().splitlines():
                    print(f"    {line}", flush=True)
                done_count += 1
                print(f"  [{done_count}/{len(tasks)}] DONE {env}/{algo}/s{seed} "
                      f"(exit={ret})", flush=True)
                del running[pid]

        if running:
            time.sleep(3)

    print(f"\nAll {len(tasks)} trials finished.")
    print_summary(results_dir, args.seeds)

