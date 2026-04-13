#!/usr/bin/env python3
"""
GRPO 系列完备实验
=================

算法对比（4个）：
  1. Standard_GRPO_NoTrick  ── 纯 GRPO 基线，无任何 PPO trick
                               (无 obs_norm / adv_norm / lr_anneal / Critic baseline)
  2. HCGAE_Standard_GRPO   ── 纯 GRPO + HCGAE FixSCR + SNR 加权 (无其他 trick)
                               验证 HCGAE 修正在公平基线上的净增益
  3. Optimal_GRPO          ── GRPO + 全套 PPO tricks (obs_norm + adv_norm + lr_anneal)
                               + Critic baseline（原 Standard_GRPO 改名）
  4. HCGAE_Optimal_GRPO    ── GRPO + 全套 PPO tricks + HCGAE FixSCR + SNR 加权
                               （原 HCGAE_GRPO 改名）

理论依据 (James-Stein / Kalman / Bayes 统一):
  标准 GRPO 用 std(G_t) 做分母，高估了真实 MC 噪声 σ²_G = E[Var(G|s)]。
  由全方差定律: Var(G) = Var(V*(s)) + E[Var(G|s)]
  FixSCR 修正: σ²_G_corrected = max(Var(G) - Var(V), floor × Var(G))
  SNR 加权:    w_t = σ(β·(|G_t-V_t|/σ_G - θ))  [局部 Kalman 最优]

实验协议：
  - 1.5M 步训练
  - 15 seeds (0-14)
  - 4 个 MuJoCo 环境 (HC / Hopper / Walker2d / Ant)
  - 完全相同的超参数骨架 (SHARED_KWARGS)
  - 确定性均值动作评估
  - subprocess 并行，断点续跑支持

快速实验模式 (--quick):
  - 3 seeds (0-2), 500K 步, 2 环境 (HC + Hopper)

输出: results/GRPO/{env}/{algo}/{algo}_s{seed}.json
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
TOTAL_TIMESTEPS  = 1_500_000
EVAL_FREQ        = 20_480
N_EVAL_EPISODES  = 10
N_PEAK           = 10
SEEDS            = list(range(15))   # 0-14

ENVS  = ["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4", "Ant-v4"]
ALGOS = [
    "Standard_GRPO_NoTrick",   # 纯 GRPO，无任何 trick
    "HCGAE_Standard_GRPO",     # 纯 GRPO + HCGAE（无其他 trick）
    "Optimal_GRPO",            # GRPO + 全套 PPO tricks（原 Standard_GRPO）
    "HCGAE_Optimal_GRPO",      # GRPO + 全套 tricks + HCGAE（原 HCGAE_GRPO）
]

RESULTS_DIR = Path("results/GRPO")

# 与 run_final_optimal.py 完全对齐的超参数
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
    """确定性均值动作评估（与 run_final_optimal.py 完全一致）。"""
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
# Checkpoint 工具函数
# ─────────────────────────────────────────────────────────────────────────────
def _ckpt_path(save_dir, algo_name, seed):
    """返回 checkpoint 文件路径（.pt）"""
    return save_dir / f"{algo_name}_s{seed}.ckpt.pt"


def _save_checkpoint(agent, path, total_steps, ep_reward, last_eval_step,
                     eval_rewards, eval_steps, episode_rewards, diag_log, peak10_running):
    """保存训练状态：模型权重 + obs_rms + 训练进度"""
    state = {
        "actor":          agent.actor.state_dict(),
        "critic":         agent.critic.state_dict(),
        "total_steps":    total_steps,
        "ep_reward":      ep_reward,
        "last_eval_step": last_eval_step,
        "eval_rewards":   eval_rewards,
        "eval_steps":     eval_steps,
        "episode_rewards": episode_rewards,
        "diag_log":       diag_log,
        "peak10_running": peak10_running,
    }
    # optimizer 状态（OptimalPPO 用 self.optimizer，兼容两种命名）
    if hasattr(agent, 'optimizer'):
        state["optimizer"] = agent.optimizer.state_dict()
    elif hasattr(agent, 'actor_optimizer'):
        state["actor_opt"] = agent.actor_optimizer.state_dict()
        state["critic_opt"] = agent.critic_optimizer.state_dict()
    # obs_rms 统计量（若有）
    if hasattr(agent, 'obs_rms') and agent.obs_rms is not None:
        state["obs_rms_mean"] = agent.obs_rms.mean.tolist()
        state["obs_rms_var"]  = agent.obs_rms.var.tolist()
        state["obs_rms_count"] = float(agent.obs_rms.count)
    # HCGAE 在线状态
    if hasattr(agent, '_g_std_ema'):
        state["g_std_ema"] = float(agent._g_std_ema)
    if hasattr(agent, '_ev_ema'):
        state["ev_ema"] = float(agent._ev_ema)
    torch.save(state, path)


def _load_checkpoint(agent, path):
    """加载 checkpoint，恢复模型权重和训练进度，返回恢复的状态字典"""
    state = torch.load(path, map_location=agent.device)
    agent.actor.load_state_dict(state["actor"])
    agent.critic.load_state_dict(state["critic"])
    # OptimalPPO 用单一 optimizer；兼容两种命名
    if "optimizer" in state and hasattr(agent, 'optimizer'):
        agent.optimizer.load_state_dict(state["optimizer"])
    elif "actor_opt" in state and hasattr(agent, 'actor_optimizer'):
        agent.actor_optimizer.load_state_dict(state["actor_opt"])
        agent.critic_optimizer.load_state_dict(state["critic_opt"])
    # obs_rms 恢复
    if "obs_rms_mean" in state and hasattr(agent, 'obs_rms') and agent.obs_rms is not None:
        agent.obs_rms.mean  = np.array(state["obs_rms_mean"], dtype=np.float64)
        agent.obs_rms.var   = np.array(state["obs_rms_var"],  dtype=np.float64)
        agent.obs_rms.count = float(state["obs_rms_count"])
    # HCGAE 在线状态
    if "g_std_ema" in state and hasattr(agent, '_g_std_ema'):
        agent._g_std_ema = state["g_std_ema"]
    if "ev_ema" in state and hasattr(agent, '_ev_ema'):
        agent._ev_ema = state["ev_ema"]
    return state


# ─────────────────────────────────────────────────────────────────────────────
# 单次 trial
# ─────────────────────────────────────────────────────────────────────────────
CKPT_FREQ = EVAL_FREQ * 5   # 每 5 次评估保存一次 checkpoint（约每 100K 步）


def run_single(env_name, algo_name, seed,
               total_timesteps=TOTAL_TIMESTEPS,
               results_dir=RESULTS_DIR,
               skip_existing=True):
    save_dir = results_dir / env_name / algo_name
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path  = save_dir / f"{algo_name}_s{seed}.json"
    ckpt_path = _ckpt_path(save_dir, algo_name, seed)

    # ── 检查是否已完成 ──────────────────────────────────────────────────────
    if skip_existing and out_path.exists():
        d = json.load(open(out_path))
        ts = d.get('total_steps', 0)
        if ts >= total_timesteps * 0.95:
            print(f"  [SKIP] {env_name}/{algo_name}/s{seed} "
                  f"(steps={ts:,}, final={d.get('final_reward', 0):.1f})")
            return d
        else:
            # 旧数据步数不足，但有 checkpoint → 续训
            if ckpt_path.exists():
                print(f"  [RESUME] {env_name}/{algo_name}/s{seed} "
                      f"from {ts:,} steps (target {total_timesteps:,})")
            else:
                # 无 checkpoint：从头重跑（保留旧 JSON 历史作为参考，直接覆盖）
                print(f"  [RESTART] {env_name}/{algo_name}/s{seed} "
                      f"(old steps={ts:,}, no ckpt, restarting)")

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
    ep_reward   = 0.0
    total_steps = 0
    last_eval_step = -EVAL_FREQ

    # ── 尝试从 checkpoint 恢复 ───────────────────────────────────────────────
    resumed = False
    if ckpt_path.exists() and out_path.exists():
        try:
            state = _load_checkpoint(agent, ckpt_path)
            total_steps     = state["total_steps"]
            ep_reward       = state["ep_reward"]
            last_eval_step  = state["last_eval_step"]
            eval_rewards    = state["eval_rewards"]
            eval_steps      = state["eval_steps"]
            episode_rewards = state["episode_rewards"]
            diag_log        = state["diag_log"]
            peak10_running  = state["peak10_running"]
            agent.total_steps = total_steps
            resumed = True
            print(f"  [RESUMED] {env_name}/{algo_name}/s{seed} "
                  f"at step {total_steps:,}")
        except Exception as e:
            print(f"  [WARN] checkpoint load failed ({e}), restarting from scratch")
            total_steps = 0
            ep_reward = 0.0
            last_eval_step = -EVAL_FREQ
            eval_rewards, eval_steps = [], []
            episode_rewards, diag_log = [], []
            peak10_running = 0.0

    obs, _ = env.reset()
    if hasattr(agent, 'update_obs_rms') and not resumed:
        agent.update_obs_rms(obs)
    obs = agent.normalize_obs(obs) if hasattr(agent, 'normalize_obs') else obs

    t0 = time.time()
    eval_count_since_ckpt = 0

    if not resumed:
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

        # ── GAE / GRPO advantage ──
        agent.compute_gae(last_val)

        # ── Update ──
        metrics = agent.update()

        # ── Diagnostics ──
        diag_entry = {"step": total_steps}
        if hasattr(agent, '_ev_ema'):      diag_entry['ev_ema']      = float(agent._ev_ema)
        if hasattr(agent, '_g_std_ema'):   diag_entry['g_std_ema']   = float(agent._g_std_ema)
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
            g_str = (f" σ_G={agent._g_std_ema:.1f}"
                     if hasattr(agent, '_g_std_ema') else "")
            ev_str = f" EV={ev:+.3f}"
            print(f"    [{algo_name:<26} s{seed}] "
                  f"{total_steps:7d}/{total_timesteps} ({pct:4.0f}%) "
                  f"| eval={eval_r:7.1f} peak10={peak10_running:7.1f}"
                  f"{ev_str}{g_str} | {fps}fps {elapsed:.0f}s")

            # ── 周期性保存 checkpoint ────────────────────────────────────────
            eval_count_since_ckpt += 1
            if eval_count_since_ckpt >= 5:   # 每 5 次 eval（≈100K 步）保存一次
                _save_checkpoint(
                    agent, ckpt_path, total_steps, ep_reward, last_eval_step,
                    eval_rewards, eval_steps, episode_rewards, diag_log, peak10_running
                )
                eval_count_since_ckpt = 0
                # 同步写入中间 JSON（方便进度查询，不影响最终结果）
                interim = {
                    "env": env_name, "agent": algo_name, "seed": seed,
                    "total_steps": total_steps,
                    "eval_rewards": eval_rewards,
                    "eval_steps": eval_steps,
                    "peak_reward": peak10_running,
                }
                with open(out_path, "w") as _f:
                    json.dump(interim, _f)

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

    # 训练完成后删除 checkpoint（不再需要）
    if ckpt_path.exists():
        ckpt_path.unlink()

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


def print_summary(results_dir=RESULTS_DIR, seeds=SEEDS, envs=ENVS, algos=ALGOS):
    try:
        from scipy import stats as scipy_stats
        has_scipy = True
    except ImportError:
        has_scipy = False

    env_labels = {"HalfCheetah-v4": "HC", "Hopper-v4": "Hop",
                  "Walker2d-v4": "Wal", "Ant-v4": "Ant"}

    print("\n" + "=" * 105)
    print(f"  GRPO Experiment Summary  ({len(seeds)} seeds)")
    print("  Standard_GRPO vs HCGAE_GRPO (FixSCR + SNR-Aware)")
    print("  理论: 全方差定律分解 Var(G)=Var(V*)+E[Var(G|s)] → FixSCR 修正分母")
    print("=" * 105)

    data = {}
    for algo in algos:
        data[algo] = {}
        for env in envs:
            f, p, m = _load_stats(results_dir, algo, env, seeds)
            data[algo][env] = dict(finals=f, peaks=p, maxs=m)

    base_key = algos[0]  # Standard_GRPO

    for metric_name, key in [("Final (last-10 mean)", "finals"),
                               ("Peak  (peak-10 window)", "peaks")]:
        print(f"\n── {metric_name} ──")
        print(f"  {'Algo':<30}", end="")
        for env in envs:
            print(f"  {env_labels.get(env, env):>18}", end="")
        print(f"  {'Avg Δ':>8}  {'n':>4}")
        print("  " + "-" * 100)

        for algo in algos:
            print(f"  {algo:<30}", end="")
            deltas = []
            n_seeds = 0
            for env in envs:
                vals = data[algo][env][key]
                base_vals = data[base_key][env][key]
                if vals:
                    m, s = np.mean(vals), np.std(vals)
                    n_seeds = max(n_seeds, len(vals))
                    cell = f"{m:.0f}±{s:.0f}(n={len(vals)})"
                    if algo != base_key and base_vals:
                        d_pct = (m - np.mean(base_vals)) / (abs(np.mean(base_vals)) + 1e-8) * 100
                        deltas.append(d_pct)
                        cell += f"({d_pct:+.1f}%)"
                else:
                    cell = "N/A"
                print(f"  {cell:>18}", end="")
            avg_d = f"{np.mean(deltas):+.1f}%" if deltas else "---"
            print(f"  {avg_d:>8}  {n_seeds:>4}")

    # t-test 显著性
    if has_scipy:
        from scipy import stats as scipy_stats
        print(f"\n── t-test vs Standard_GRPO (p-value, 双侧) ──")
        print(f"  {'Algo':<30}", end="")
        for env in envs:
            print(f"  {env_labels.get(env, env):>16}", end="")
        print()
        print("  " + "-" * 95)
        for algo in [a for a in algos if a != base_key]:
            print(f"  {algo:<30}", end="")
            for env in envs:
                vals = data[algo][env]["finals"]
                base_vals = data[base_key][env]["finals"]
                if vals and base_vals and len(vals) >= 3 and len(base_vals) >= 3:
                    _, p = scipy_stats.ttest_ind(vals, base_vals)
                    d = (np.mean(vals) - np.mean(base_vals)) / (abs(np.mean(base_vals)) + 1e-8) * 100
                    sig = "**" if p < 0.05 else ("*" if p < 0.10 else "  ")
                    cell = f"{d:+.1f}%{sig}(p={p:.3f})"
                else:
                    cell = "N/A"
                print(f"  {cell:>16}", end="")
            print()

    print(f"\n  Results dir: {results_dir}")
    print("=" * 105)


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import multiprocessing as mp
    import subprocess

    parser = argparse.ArgumentParser(description="GRPO vs HCGAE_GRPO 实验")
    parser.add_argument("--envs",    nargs="+", default=ENVS,  help="环境列表")
    parser.add_argument("--algos",   nargs="+", default=ALGOS, help="算法列表")
    parser.add_argument("--seeds",   nargs="+", type=int, default=SEEDS, help="种子列表")
    parser.add_argument("--steps",   type=int, default=TOTAL_TIMESTEPS, help="训练步数")
    parser.add_argument("--workers", type=int, default=None, help="并行 worker 数")
    parser.add_argument("--summarize", action="store_true", help="只打印汇总表")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR), help="结果目录")
    parser.add_argument("--quick",   action="store_true",
                        help="快速实验: 3seeds×2envs×500K步 (验证用)")
    # 单 trial 模式（内部调用）
    parser.add_argument("--single-env",   default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-algo",  default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-seed",  type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # 快速实验配置
    if args.quick:
        quick_envs  = ["HalfCheetah-v4", "Hopper-v4"]
        quick_seeds = list(range(3))
        quick_steps = 500_000
        args.envs   = args.envs if args.envs != ENVS else quick_envs
        args.seeds  = args.seeds if args.seeds != SEEDS else quick_seeds
        args.steps  = args.steps if args.steps != TOTAL_TIMESTEPS else quick_steps
        if args.results_dir == str(RESULTS_DIR):
            args.results_dir = "results/GRPO_Quick"

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── 单 trial 模式 ─────────────────────────────────────────────────────────
    if args.single_env is not None:
        run_single(args.single_env, args.single_algo, args.single_seed,
                   args.steps, results_dir, skip_existing=True)
        sys.exit(0)

    # ── 汇总模式 ──────────────────────────────────────────────────────────────
    if args.summarize:
        print_summary(results_dir, args.seeds, args.envs, args.algos)
        sys.exit(0)

    # ── 主调度：subprocess 并行 ───────────────────────────────────────────────
    tasks = [
        (env, algo, seed)
        for env in args.envs
        for algo in args.algos
        for seed in args.seeds
    ]
    n_workers = args.workers or mp.cpu_count()
    script = os.path.abspath(__file__)

    print("=" * 70)
    print(f"GRPO 系列完备实验")
    print(f"  Total trials : {len(tasks)}")
    print(f"  Envs         : {args.envs}")
    print(f"  Algos        : {args.algos}")
    print(f"  Seeds        : {len(args.seeds)}  ({args.seeds[0]}..{args.seeds[-1]})")
    print(f"  Steps        : {args.steps:,}")
    print(f"  Workers      : {n_workers}")
    print(f"  Output       : {results_dir}")
    print("=" * 70)
    print()

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

    print(f"\nAll {len(tasks)} trials finished.")
    print_summary(results_dir, args.seeds, args.envs, args.algos)


# ─────────────────────────────────────────────────────────────────────────────
# 版本历史
# ─────────────────────────────────────────────────────────────────────────────
# v1 (2025-04): 初始版本 Standard_GRPO vs HCGAE_GRPO，1M 步，20 seeds
# v2 (2026-04): 重命名 Standard_GRPO→Optimal_GRPO, HCGAE_GRPO→HCGAE_Optimal_GRPO
#               新增 Standard_GRPO_NoTrick（纯 GRPO 无 trick）
#               新增 HCGAE_Standard_GRPO（HCGAE + 无 trick 基线）
#               步长升级至 1.5M，seeds 调整为 15 个

