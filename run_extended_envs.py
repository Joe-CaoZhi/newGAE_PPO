#!/usr/bin/env python3
"""
ICML 扩展环境实验
=================
在 4 个核心 MuJoCo 环境 (HC/Hopper/Walker/Ant) 基础上，
进一步测试 ICML 常用的 3 个附加环境：
  - Swimmer-v4           (低维连续控制，简单)
  - Humanoid-v4          (高维连续控制，极难)
  - HumanoidStandup-v4   (奖励密集，高维)

与 run_final_optimal.py 共享相同的 SHARED_KWARGS 和实验框架。
Swimmer/InvertedDP 步数设为 3M（动力学简单但需更多步数），
Humanoid 系列依然 1M（极高维，较慢）。

用法:
  python run_extended_envs.py                 # 全量 (6 envs × 2 algos × 20 seeds)
  python run_extended_envs.py --envs Swimmer-v4 Humanoid-v4
  python run_extended_envs.py --summarize
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
SEEDS   = list(range(20))
ALGOS   = ["Optimal_PPO", "Optimal_HCGAE_Optimal"]
N_EVAL_EPISODES = 10
N_PEAK  = 10
EVAL_FREQ = 20_480

# 每个环境的训练步数（根据环境复杂度调整）
ENV_STEPS = {
    "Swimmer-v4":             1_000_000,   # 简单但需 1M
    "Humanoid-v4":            1_000_000,   # 高维，1M
    "HumanoidStandup-v4":     1_000_000,   # 高维，1M
    "InvertedDoublePendulum-v4": 1_000_000,
    "InvertedPendulum-v4":    1_000_000,
}

EXTENDED_ENVS = list(ENV_STEPS.keys())
RESULTS_DIR = Path("results/ExtendedEnvs")

# 与 run_final_optimal.py 完全一致的超参数
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
# 评估（与 run_final_optimal.py 完全一致）
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_policy(agent, eval_env, n_episodes=N_EVAL_EPISODES):
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
                action = dist.mean if agent.continuous else dist.probs.argmax(dim=-1)
            a = action.squeeze(0).cpu().numpy()
            next_obs, r, terminated, truncated, _ = eval_env.step(
                a if agent.continuous else int(a))
            ep_r += r
            done = terminated or truncated
            obs = next_obs
        total_reward += ep_r
    return total_reward / n_episodes


# ─────────────────────────────────────────────────────────────────────────────
# 单次 trial
# ─────────────────────────────────────────────────────────────────────────────
def run_single(env_name, algo_name, seed,
               total_timesteps=None,
               results_dir=RESULTS_DIR,
               skip_existing=True):
    if total_timesteps is None:
        total_timesteps = ENV_STEPS.get(env_name, 1_000_000)

    save_dir = results_dir / env_name / algo_name
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{algo_name}_s{seed}.json"

    if skip_existing and out_path.exists():
        d = json.load(open(out_path))
        print(f"  [SKIP] {env_name}/{algo_name}/s{seed} (final={d.get('final_reward','?'):.1f})")
        return d

    np.random.seed(seed)
    torch.manual_seed(seed)

    env      = gym.make(env_name)
    eval_env = gym.make(env_name)
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 100_000)

    agent = build_optimal_agent(algo_name, env, name=f"{algo_name}_s{seed}", **SHARED_KWARGS)
    agent._total_timesteps = total_timesteps

    eval_rewards, eval_steps, episode_rewards, diag_log = [], [], [], []
    peak10_running = 0.0
    obs, _ = env.reset()
    if hasattr(agent, 'update_obs_rms'):
        agent.update_obs_rms(obs)
    obs = agent.normalize_obs(obs) if hasattr(agent, 'normalize_obs') else obs

    ep_reward, total_steps, last_eval_step = 0.0, 0, -EVAL_FREQ
    t0 = time.time()
    print(f"  START {env_name}/{algo_name}/s{seed}  ({total_timesteps:,} steps)")

    while total_steps < total_timesteps:
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

        obs_n_b = agent.normalize_obs(obs) if hasattr(agent, 'normalize_obs') else obs
        obs_t = torch.FloatTensor(obs_n_b).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            last_val = agent.critic(obs_t).item()

        agent.total_steps = total_steps
        if hasattr(agent, 'compute_hindsight_gae'):
            agent.compute_hindsight_gae(last_val)
        else:
            agent.compute_gae(last_val)
        metrics = agent.update()

        diag_entry = {"step": total_steps}
        if hasattr(agent, '_scr_ema'):   diag_entry['scr_ema']   = float(agent._scr_ema)
        if hasattr(agent, '_ev_ema'):    diag_entry['ev_ema']    = float(agent._ev_ema)
        if hasattr(agent, '_g_std_ema'): diag_entry['g_std_ema'] = float(agent._g_std_ema)
        diag_log.append(diag_entry)

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
            print(f"    [{algo_name:<26} s{seed}] "
                  f"{total_steps:7d}/{total_timesteps} ({pct:4.0f}%) "
                  f"| eval={eval_r:7.1f} peak10={peak10_running:7.1f} "
                  f"EV={ev:+.3f}{scr_str} | {fps}fps {elapsed:.0f}s")

    final_mean = (float(np.mean(eval_rewards[-N_PEAK:])) if len(eval_rewards) >= N_PEAK
                  else float(np.mean(eval_rewards)) if eval_rewards else 0.0)

    result = {
        "env": env_name, "agent": algo_name, "seed": seed,
        "config": {**SHARED_KWARGS, "total_timesteps": total_timesteps,
                   "eval_freq": EVAL_FREQ, "n_eval_episodes": N_EVAL_EPISODES,
                   "eval_mode": "deterministic_mean"},
        "total_steps": total_steps, "elapsed_s": time.time() - t0,
        "final_reward": final_mean, "peak_reward": peak10_running,
        "max_reward": float(max(eval_rewards)) if eval_rewards else 0.0,
        "eval_rewards": eval_rewards, "eval_steps": eval_steps,
        "episode_rewards": episode_rewards[-200:], "diagnostics": diag_log[::10],
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    env.close(); eval_env.close()
    print(f"  DONE  {env_name}/{algo_name}/s{seed} "
          f"→ final10={final_mean:.1f} peak10={peak10_running:.1f} ({result['elapsed_s']:.0f}s)")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 汇总（与核心环境结果合并展示）
# ─────────────────────────────────────────────────────────────────────────────
def _load_stats(results_dir, algo, env, seeds):
    finals, peaks = [], []
    for s in seeds:
        fp = results_dir / env / algo / f"{algo}_s{s}.json"
        if fp.exists():
            try:
                d = json.load(open(fp))
                er = d.get("eval_rewards", [])
                if er:
                    finals.append(float(np.mean(er[-N_PEAK:])) if len(er) >= N_PEAK else float(np.mean(er)))
                    peaks.append(d.get("peak_reward", float(np.max(er))))
            except Exception as e:
                print(f"  [WARN] {fp}: {e}")
    return finals, peaks


def print_summary(results_dir=RESULTS_DIR, seeds=SEEDS, envs=None):
    from scipy import stats as scipy_stats
    if envs is None:
        envs = [e for e in EXTENDED_ENVS
                if any((results_dir / e / a).exists() for a in ALGOS)]

    print("\n" + "=" * 95)
    print(f"  Extended Envs Summary  (1M steps, {len(seeds)} seeds)")
    print("=" * 95)
    ppo_key = "Optimal_PPO"
    for env in envs:
        ppo_f, ppo_p = _load_stats(results_dir, ppo_key, env, seeds)
        opt_f, opt_p = _load_stats(results_dir, "Optimal_HCGAE_Optimal", env, seeds)
        if not ppo_f and not opt_f:
            continue
        print(f"\n  {env}:")
        for algo, fs, ps in [(ppo_key, ppo_f, ppo_p),
                              ("Optimal_HCGAE_Optimal", opt_f, opt_p)]:
            if fs:
                arr = np.array(fs)
                q1, med, q3 = np.percentile(arr, [25, 50, 75])
                print(f"    {algo:<35}: n={len(arr):2d}  mean={np.mean(arr):8.1f}  "
                      f"std={np.std(arr):6.1f}  median={med:8.1f}  Q1={q1:8.1f}  Q3={q3:8.1f}")
        if ppo_f and opt_f:
            _, pv = scipy_stats.ttest_ind(opt_f, ppo_f)
            dm = (np.mean(opt_f) - np.mean(ppo_f)) / abs(np.mean(ppo_f)) * 100
            wins = sum(b > a for a, b in zip(ppo_f, opt_f))
            sig = "** p<0.05" if pv < 0.05 else ("* p<0.10" if pv < 0.10 else f"ns p={pv:.3f}")
            print(f"    → Δ={dm:+.1f}%  {sig}  胜率={wins}/{len(ppo_f)}={wins/len(ppo_f)*100:.0f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import multiprocessing as mp
    import subprocess

    parser = argparse.ArgumentParser()
    parser.add_argument("--envs",    nargs="+", default=EXTENDED_ENVS)
    parser.add_argument("--algos",   nargs="+", default=ALGOS)
    parser.add_argument("--seeds",   nargs="+", type=int, default=SEEDS)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--single-env",  default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-algo", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-seed", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-steps", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.single_env is not None:
        steps = args.single_steps or ENV_STEPS.get(args.single_env, 1_000_000)
        run_single(args.single_env, args.single_algo, args.single_seed,
                   steps, results_dir, skip_existing=True)
        sys.exit(0)

    if args.summarize:
        print_summary(results_dir, args.seeds, args.envs)
        sys.exit(0)

    tasks = [
        (env, algo, seed, ENV_STEPS.get(env, 1_000_000))
        for env in args.envs
        for algo in args.algos
        for seed in args.seeds
    ]
    n_workers = args.workers or mp.cpu_count()
    script = os.path.abspath(__file__)

    print(f"Extended Envs Experiment")
    print(f"  Envs    : {args.envs}")
    print(f"  Algos   : {args.algos}")
    print(f"  Seeds   : {len(args.seeds)}")
    print(f"  Workers : {n_workers}")
    print(f"  Output  : {results_dir}\n")

    pending = [(e, a, s, st) for e, a, s, st in tasks]
    running: dict = {}
    done_count = 0

    while pending or running:
        while len(running) < n_workers and pending:
            env, algo, seed, steps = pending.pop(0)
            cmd = [
                sys.executable, script,
                "--single-env",   env,
                "--single-algo",  algo,
                "--single-seed",  str(seed),
                "--single-steps", str(steps),
                "--results-dir",  str(results_dir),
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    cwd=str(Path(script).parent))
            running[proc.pid] = (proc, env, algo, seed)

        for pid in list(running.keys()):
            proc, env, algo, seed = running[pid]
            ret = proc.poll()
            if ret is not None:
                out, _ = proc.communicate()
                for line in out.strip().splitlines():
                    print(f"    {line}", flush=True)
                done_count += 1
                print(f"  [{done_count}/{len(tasks)}] DONE {env}/{algo}/s{seed} (exit={ret})",
                      flush=True)
                del running[pid]

        if running:
            time.sleep(3)

    print(f"\nAll {len(tasks)} trials finished.")
    print_summary(results_dir, args.seeds)

