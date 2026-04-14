#!/usr/bin/env python3
"""
HCGAE_Optimal 消融实验
=======================

目的：验证 OptimalHCGAE_Optimal 三个核心设计组件各自的贡献：

  组件 1 - FixSCR 分母修正：
    Full:    SCR = |B| / sqrt(Var(G) - Var(V))   [条件MC噪声估计]
    -FixSCR: SCR = |B| / std(G)                  [退回 v4 原始，高估分母]

  组件 2 - MC 归一化 per-step sigmoid：
    Full:     z_t = β*(|δ_t|/σ_G - θ)  [MC噪声归一化，SNR驱动]
    -Sigmoid: z_t → +∞, sigmoid≈1 → 全批均匀 α_max_k

  组件 3 - EV 自适应门控：
    Full:    alpha_max 随 EV 和 cosine 动态调整
    -EVGate: alpha_max 固定（无EV/cosine缩放）

实验协议：
  - 4 个 MuJoCo 环境 (HalfCheetah-v4 / Hopper-v4 / Walker2d-v4 / Ant-v4)
  - 每个算法 8 seeds (0-7)
  - 1M steps（与 FinalOptimal 对齐）
  - 断点续跑支持（skip_existing=True）

输出: results/Ablation/{env}/{algo}/{algo}_s{seed}.json

用法:
  python3 run_ablation.py                    # 默认所有环境，8 seeds，16 workers
  python3 run_ablation.py --workers 4        # 限制并行数
  python3 run_ablation.py --envs HalfCheetah-v4 --seeds 0 1 2  # 指定环境+seed
  python3 run_ablation.py --summarize        # 只看汇总结果
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
TOTAL_TIMESTEPS  = 1_000_000     # 与 FinalOptimal 对齐
EVAL_FREQ        = 20_480
N_EVAL_EPISODES  = 10
N_PEAK           = 10
SEEDS            = list(range(8))   # 0-7（消融实验 8 seeds 即可）

ENVS  = ["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4", "Ant-v4"]

# 消融算法列表（+完整版作为对照）
ALGOS = [
    "Optimal_HCGAE_Optimal",     # Full（对照）
    "Optimal_HCGAE_NoFixSCR",    # -FixSCR（去掉分母修正）
    "Optimal_HCGAE_NoMCSigmoid", # -MCSigmoid（去掉 per-step sigmoid）
    "Optimal_HCGAE_NoEVGate",    # -EVGate（去掉 EV 自适应）
]

RESULTS_DIR = Path("results/Ablation")

# 超参数与 FinalOptimal 完全一致
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
    """确定性均值动作评估。"""
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
        if d.get("total_steps", 0) >= total_timesteps * 0.95:
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
    diag_log = []
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
            scr_str = f" SCR={agent._scr_ema:.3f}" if hasattr(agent, '_scr_ema') else ""
            ev_str  = f" EV={ev:+.3f}"
            print(f"    [{algo_name:<30} s{seed}] "
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
        "diagnostics":    diag_log[::10],
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

    print("\n" + "=" * 110)
    print(f"  Ablation Study Summary  (1M steps, {len(seeds)} seeds per algo)")
    print("  HCGAE_Optimal = FixSCR + MC-Normalized-Sigmoid + EV-Gate")
    print("=" * 110)

    data = {}
    for algo in ALGOS:
        data[algo] = {}
        for env in ENVS:
            f, p, m = _load_stats(results_dir, algo, env, seeds)
            data[algo][env] = dict(finals=f, peaks=p, maxs=m)

    full_key = "Optimal_HCGAE_Optimal"

    ablation_labels = {
        "Optimal_HCGAE_Optimal":    "Full (对照)",
        "Optimal_HCGAE_NoFixSCR":   "- FixSCR      (-分母修正)",
        "Optimal_HCGAE_NoMCSigmoid":"- MCSigmoid   (-per-step sigmoid)",
        "Optimal_HCGAE_NoEVGate":   "- EVGate      (-EV自适应门控)",
    }

    for metric_name, key in [("Final (last-10 mean)", "finals")]:
        print(f"\n── {metric_name} ──")
        header = f"  {'算法变体':<40}"
        for env in ENVS:
            header += f"  {env_labels[env]:>18}"
        header += f"  {'Avg Δ vs Full':>14}  {'n':>4}"
        print(header)
        print("  " + "-" * 105)

        for algo in ALGOS:
            lbl = ablation_labels.get(algo, algo)
            row = f"  {lbl:<40}"
            deltas = []
            n_seeds = 0
            for env in ENVS:
                vals = data[algo][env][key]
                full_vals = data[full_key][env][key]
                if vals:
                    m, s = np.mean(vals), np.std(vals)
                    n_seeds = max(n_seeds, len(vals))
                    cell = f"{m:.0f}±{s:.0f}"
                    if algo != full_key and full_vals:
                        d_pct = (m - np.mean(full_vals)) / abs(np.mean(full_vals) + 1e-8) * 100
                        deltas.append(d_pct)
                        cell += f"({d_pct:+.1f}%)"
                else:
                    cell = "N/A"
                row += f"  {cell:>18}"
            avg_d = f"{np.mean(deltas):+.1f}%" if deltas else "FULL"
            row += f"  {avg_d:>14}  {n_seeds:>4}"
            print(row)

    # Welch t-test 与 Full 对比
    print(f"\n── 显著性检验 (Welch t-test vs Full, p-value) ──")
    header2 = f"  {'算法变体':<40}"
    for env in ENVS:
        header2 += f"  {env_labels[env]:>18}"
    print(header2)
    print("  " + "-" * 100)
    for algo in [a for a in ALGOS if a != full_key]:
        lbl = ablation_labels.get(algo, algo)
        row = f"  {lbl:<40}"
        for env in ENVS:
            vals = data[algo][env]["finals"]
            full_vals = data[full_key][env]["finals"]
            if vals and full_vals and len(vals) >= 3 and len(full_vals) >= 3:
                _, p = scipy_stats.ttest_ind(vals, full_vals)
                d = (np.mean(vals) - np.mean(full_vals)) / abs(np.mean(full_vals) + 1e-8) * 100
                sig = "**" if p < 0.05 else ("*" if p < 0.10 else "  ")
                cell = f"{d:+.1f}%{sig}(p={p:.3f})"
            else:
                n = len(vals) if vals else 0
                cell = f"N/A(n={n})"
            row += f"  {cell:>18}"
        print(row)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import multiprocessing as mp
    import subprocess

    parser = argparse.ArgumentParser(description="HCGAE_Optimal 消融实验")
    parser.add_argument("--envs",    nargs="+", default=ENVS,  help="环境列表")
    parser.add_argument("--algos",   nargs="+", default=ALGOS, help="算法列表")
    parser.add_argument("--seeds",   nargs="+", type=int, default=SEEDS, help="种子列表")
    parser.add_argument("--steps",   type=int, default=TOTAL_TIMESTEPS, help="训练步数")
    parser.add_argument("--workers", type=int, default=None, help="并行 worker 数（默认 CPU 数）")
    parser.add_argument("--summarize", action="store_true", help="只打印汇总表")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR), help="结果目录")
    # 单 trial 模式
    parser.add_argument("--single-env",   default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-algo",  default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-seed",  type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.single_env is not None:
        run_single(args.single_env, args.single_algo, args.single_seed,
                   args.steps, results_dir, skip_existing=True)
        sys.exit(0)

    if args.summarize:
        print_summary(results_dir, args.seeds)
        sys.exit(0)

    tasks = [
        (env, algo, seed)
        for env in args.envs
        for algo in args.algos
        for seed in args.seeds
    ]
    n_workers = args.workers or mp.cpu_count()
    script = os.path.abspath(__file__)

    print(f"消融实验启动")
    print(f"  Total trials : {len(tasks)}")
    print(f"  Envs         : {args.envs}")
    print(f"  Algos        : {args.algos}")
    print(f"  Seeds        : {len(args.seeds)}  ({args.seeds[0]}..{args.seeds[-1]})")
    print(f"  Steps        : {args.steps:,}")
    print(f"  Workers      : {n_workers}")
    print(f"  Output       : {results_dir}\n")

    pending = list(tasks)
    running: dict = {}
    done_count = 0

    while pending or running:
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

    print(f"\nAll {len(tasks)} ablation trials finished.")
    print_summary(results_dir, args.seeds)

