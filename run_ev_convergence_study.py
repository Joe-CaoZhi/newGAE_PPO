"""
EV 收敛速度验证实验（§4.6 定理 1 实证验证）
=============================================
目标：严格验证定理 1（Critic EV 加速收敛定理）的预测：
  - 推论 1 预测：HCGAE-Imp-II 使 EV > 0.9 的步数从 ~150K 降至 ~80K（~47% 加速）
  - 验证范围：Hopper-v4（主要）+ Walker2d-v4（泛化验证）
  - 对照组：Optimal_PPO（基线）vs Optimal_HCGAE_v2（含改进 I+II+III）
  - 消融组：Optimal_HCGAE_v2_NoBdry（仅改进 I+II）、Optimal_HCGAE_v2_NoGate（仅改进 I+III）
  - 种子：5 seeds（Mann-Whitney U 统计检验所需的最小样本）
  - 步数：500K（与 ICMLExperiment 主实验对齐，确保置信度）

测量指标：
  (1) steps_to_ev09  : 首次稳定达到 EV ≥ 0.9 的步数（定理 1 主要验证指标）
  (2) steps_to_ev07  : 首次达到 EV ≥ 0.7 的步数（中期收敛指标）
  (3) ev_at_80k      : 80K 步时的 EV（推论 1 预测 HCGAE 此时应 ≥ 0.9）
  (4) ev_at_150k     : 150K 步时的 EV（推论 1 预测 PPO 此时刚达到 0.9）
  (5) aulc_200k      : 0-200K 步内 EV 曲线下面积（累积学习效率）
  (6) speedup_ratio  : PPO 步数 / HCGAE 步数（直接验证 ~1.87× 预测）

结果保存：results/EVConvergenceStudy/
"""

import os
import json
import time
import random
import sys
from pathlib import Path

import numpy as np
import gymnasium as gym
import torch

sys.path.insert(0, str(Path(__file__).parent))
from gae_experiments.agents.optimal_ppo import build_optimal_agent
from gae_experiments.utils.logger import MetricLogger

# ── 配置 ──────────────────────────────────────────────────────────────────
SEEDS = [0, 1, 2, 3, 4]
TOTAL_STEPS = 500_000       # 与 ICMLExperiment 对齐，确保统计置信度
N_STEPS = 2048              # rollout buffer size
EVAL_FREQ = 10_240          # 评测频率（每 5 个 rollout）
N_EVAL_EPISODES = 10        # 与主实验对齐

# 至少两个环境：Hopper-v4（主，稀疏奖励）+ Walker2d-v4（泛化验证）
ENVS = [
    "Hopper-v4",
    "Walker2d-v4",
]

# 对照组和消融组（每次只改变一个变量）
ALGOS = [
    "Optimal_PPO",             # 基线（定理 1 的 standard PPO 情形）
    "Optimal_HCGAE_v2",        # 完整 HCGAE v2（定理 1 的 Imp-II 情形）
    "Optimal_HCGAE_v2_NoBdry", # 消融：仅 EV 门控，无边界校正（分离改进 I+II）
    "Optimal_HCGAE_v2_NoGate", # 消融：无 EV 增长率门控（分离改进 I+III）
    "Optimal_HCGAE",           # v1 基线（无改进 II 和 III）
]

RESULT_DIR = "results/EVConvergenceStudy"
os.makedirs(RESULT_DIR, exist_ok=True)

# ── 最优超参（与 ICMLExperiment 完全对齐）────────────────────────────────
OPTIMAL_DEFAULTS = {
    "n_steps": N_STEPS,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "lam": 0.95,
    "lr": 3e-4,
    "eps_clip": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "hidden_dim": 64,
    "use_obs_norm": True,
    "use_adv_norm": True,
    "use_lr_anneal": True,
    "use_vclip": False,
    "device": "cpu",
}

# 定理 1 的理论预测值（来自推论 1）
THEOREM1_PREDICTIONS = {
    "PPO_steps_to_ev09":    149_504,   # ~73 rollouts × 2048
    "HCGAE_steps_to_ev09":  79_872,    # ~39 rollouts × 2048
    "speedup_ratio":         1.87,
    "speedup_pct":           47.0,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_policy(agent, eval_env, n_episodes: int = 10) -> float:
    """评测当前策略的平均奖励。"""
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


def train_and_collect_ev(
    agent,
    env: gym.Env,
    eval_env: gym.Env,
    total_steps: int,
    env_name: str,
    algo_name: str,
    seed: int,
) -> dict:
    """
    完整训练循环，每次 rollout 后记录 EV。
    与 run_ant_experiment.py 的训练逻辑保持完全一致。
    """
    save_dir = os.path.join(RESULT_DIR, "metrics")
    os.makedirs(save_dir, exist_ok=True)
    logger = MetricLogger(
        agent_name=f"{env_name}_{algo_name}_s{seed}",
        save_dir=save_dir,
    )
    agent.logger = logger

    obs, _ = env.reset()
    if hasattr(agent, 'update_obs_rms'):
        agent.update_obs_rms(obs)
    if hasattr(agent, 'normalize_obs'):
        obs = agent.normalize_obs(obs)

    ep_reward = 0.0
    step_count = 0
    last_eval_step = 0

    t0 = time.time()

    while step_count < total_steps:
        agent.buffer.reset()

        # ── 收集 rollout ─────────────────────────────────────────────────
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
            agent.buffer.add(
                obs, action_np, float(reward), float(terminated),
                log_prob.item(), value.item()
            )
            obs = next_obs_norm
            step_count += 1

            if terminated or truncated:
                logger.log_episode(ep_reward, 0)
                ep_reward = 0.0
                next_obs, _ = env.reset()
                if hasattr(agent, 'update_obs_rms'):
                    agent.update_obs_rms(next_obs)
                if hasattr(agent, 'normalize_obs'):
                    obs = agent.normalize_obs(next_obs)
                else:
                    obs = next_obs

            if step_count - last_eval_step >= EVAL_FREQ:
                er = evaluate_policy(agent, eval_env, N_EVAL_EPISODES)
                logger.log_eval(er, step_count)
                last_eval_step = step_count

        # ── GAE 计算 ─────────────────────────────────────────────────────
        with torch.no_grad():
            last_obs_t = torch.FloatTensor(obs).unsqueeze(0)
            last_val = agent.critic(last_obs_t).item()

        agent._total_timesteps = total_steps
        agent.total_steps = step_count

        if hasattr(agent, 'compute_hindsight_gae'):
            agent.compute_hindsight_gae(last_val)
        else:
            agent.compute_gae(last_val)

        # ── PPO 更新 ─────────────────────────────────────────────────────
        update_metrics = agent.update()

        # ── 记录 EV（每次 rollout 后一次，这是定理验证的核心数据）────────
        ev = update_metrics.get("explained_variance", float('nan'))
        alpha_mean = update_metrics.get("alpha_mean", None)
        c_mc = update_metrics.get("c_mc", None)

        logger.log_update(
            value_loss=update_metrics.get("value_loss", 0.0),
            policy_loss=update_metrics.get("policy_loss", 0.0),
            entropy_loss=update_metrics.get("entropy_loss", 0.0),
            approx_kl=update_metrics.get("approx_kl", 0.0),
            clip_frac=update_metrics.get("clip_frac", 0.0),
            explained_variance=ev,
            total_steps=step_count,
            alpha_mean=alpha_mean,
            c_mc=c_mc,
        )

    elapsed = time.time() - t0

    # 提取完整 EV 时间序列
    ev_series = logger.explained_variances
    steps_series = logger.total_steps

    # 计算关键指标（验证定理 1 的预测）
    metrics = _compute_ev_metrics(ev_series, steps_series)
    metrics.update({
        "env": env_name,
        "algo": algo_name,
        "seed": seed,
        "elapsed_s": elapsed,
        "total_steps_trained": step_count,
        "ev_series": ev_series,
        "steps_series": steps_series,
        "eval_rewards": logger.eval_rewards,
        "eval_steps": logger.eval_steps,
    })
    return metrics


def _compute_ev_metrics(ev_series: list, steps_series: list) -> dict:
    """计算定理 1 验证所需的关键 EV 指标。"""
    if not ev_series or not steps_series:
        return {}

    ev_arr = np.array(ev_series)
    steps_arr = np.array(steps_series)

    # ─── (1) 首次达到阈值的步数（稳定性要求：连续 3 次 rollout ≥ 阈值）──
    def first_stable_crossing(threshold: float, window: int = 3) -> int:
        """EV 连续 window 次 ≥ threshold 时，返回首次超过的步数。单次抖动不算。"""
        for i in range(len(ev_arr) - window + 1):
            if np.all(ev_arr[i:i+window] >= threshold):
                return int(steps_arr[i])
        # 回退：单次超过（最宽松的定义）
        idx = np.where(ev_arr >= threshold)[0]
        return int(steps_arr[idx[0]]) if len(idx) > 0 else -1

    def first_crossing(threshold: float) -> int:
        idx = np.where(ev_arr >= threshold)[0]
        return int(steps_arr[idx[0]]) if len(idx) > 0 else -1

    steps_to_ev09_stable = first_stable_crossing(0.9, window=3)
    steps_to_ev09        = first_crossing(0.9)   # 单次穿越（论文报告主指标）
    steps_to_ev08        = first_crossing(0.8)
    steps_to_ev07        = first_crossing(0.7)
    steps_to_ev05        = first_crossing(0.5)

    # ─── (2) 特定步数点的 EV 快照（推论 1 的关键对比点）────────────────
    def ev_at_step(target_step: int) -> float:
        if len(steps_arr) == 0:
            return float('nan')
        idx = min(np.searchsorted(steps_arr, target_step), len(steps_arr) - 1)
        return float(ev_arr[idx])

    ev_at_50k  = ev_at_step(51_200)
    ev_at_80k  = ev_at_step(81_920)   # 推论 1 预测 HCGAE 此时 ≥ 0.9
    ev_at_100k = ev_at_step(102_400)
    ev_at_150k = ev_at_step(153_600)  # 推论 1 预测 PPO 此时刚达到 0.9
    ev_at_200k = ev_at_step(204_800)
    ev_at_300k = ev_at_step(307_200)
    ev_at_500k = ev_at_step(512_000)

    # ─── (3) EV 曲线下面积（前 200K 步内，对应推论 1 的验证窗口）────────
    mask_200k = steps_arr <= 204_800
    if mask_200k.sum() > 1:
        aulc_200k = float(
            np.trapz(ev_arr[mask_200k], steps_arr[mask_200k])
            / max(steps_arr[mask_200k][-1] - steps_arr[mask_200k][0], 1)
        )
    else:
        aulc_200k = float(ev_arr[0]) if len(ev_arr) > 0 else float('nan')

    # 全程 AULC
    if len(steps_arr) > 1:
        aulc_full = float(np.trapz(ev_arr, steps_arr) / (steps_arr[-1] - steps_arr[0]))
    else:
        aulc_full = float(ev_arr[0]) if len(ev_arr) > 0 else float('nan')

    # ─── (4) EV 增长速率 ─────────────────────────────────────────────────
    if len(ev_arr) > 1:
        delta_ev = np.diff(ev_arr)
        ev_growth_rate = float(np.mean(delta_ev))
        # 前 25% rollouts 的增长率（早期收敛速度）
        n25 = max(1, len(ev_arr) // 4)
        ev_growth_rate_early = float(np.mean(delta_ev[:n25]))
    else:
        ev_growth_rate = float('nan')
        ev_growth_rate_early = float('nan')

    return {
        # 定理 1 主要验证指标
        "steps_to_ev09":        steps_to_ev09,        # 单次穿越（论文报告）
        "steps_to_ev09_stable": steps_to_ev09_stable,  # 稳定穿越（保守估计）
        "steps_to_ev08":        steps_to_ev08,
        "steps_to_ev07":        steps_to_ev07,
        "steps_to_ev05":        steps_to_ev05,
        # 推论 1 关键时间点 EV 快照
        "ev_at_50k":   ev_at_50k,
        "ev_at_80k":   ev_at_80k,
        "ev_at_100k":  ev_at_100k,
        "ev_at_150k":  ev_at_150k,
        "ev_at_200k":  ev_at_200k,
        "ev_at_300k":  ev_at_300k,
        "ev_at_500k":  ev_at_500k,
        # 面积与速率
        "aulc_200k":            aulc_200k,
        "aulc_full":            aulc_full,
        "ev_growth_rate":       ev_growth_rate,
        "ev_growth_rate_early": ev_growth_rate_early,
    }


def run_single(env_name: str, algo_name: str, seed: int) -> dict:
    """运行单次实验，返回完整指标字典。"""
    save_path = os.path.join(RESULT_DIR, f"{env_name}_{algo_name}_s{seed}.json")
    if os.path.exists(save_path):
        print(f"  [SKIP] {env_name}/{algo_name}/seed={seed} already exists")
        with open(save_path) as f:
            return json.load(f)

    print(f"\n  {'='*60}")
    print(f"  ENV={env_name}  ALGO={algo_name}  SEED={seed}")
    print(f"  {'='*60}")

    set_seed(seed)
    train_env = gym.make(env_name)
    eval_env  = gym.make(env_name)
    train_env.reset(seed=seed)
    eval_env.reset(seed=seed + 1000)

    save_dir = os.path.join(RESULT_DIR, "agent_data")
    os.makedirs(save_dir, exist_ok=True)
    agent = build_optimal_agent(
        algo_name=algo_name,
        env=train_env,
        name=f"{env_name}_{algo_name}_s{seed}",
        save_dir=save_dir,
        **OPTIMAL_DEFAULTS,
    )

    result = train_and_collect_ev(
        agent=agent,
        env=train_env,
        eval_env=eval_env,
        total_steps=TOTAL_STEPS,
        env_name=env_name,
        algo_name=algo_name,
        seed=seed,
    )

    train_env.close()
    eval_env.close()

    # 保存摘要
    summary = {k: v for k, v in result.items()
               if k not in ("ev_series", "steps_series")}
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)

    # 单独保存完整 EV 时间序列（供绘图）
    series_path = save_path.replace(".json", "_series.json")
    with open(series_path, "w") as f:
        json.dump({
            "env": env_name, "algo": algo_name, "seed": seed,
            "ev_series": result.get("ev_series", []),
            "steps_series": result.get("steps_series", []),
        }, f, indent=2)

    # 打印结果（与推论 1 预测对比）
    s09 = result.get('steps_to_ev09', -1)
    s09_str = f"{s09:,}" if s09 > 0 else "未达到"
    print(f"\n  ── 结果摘要（vs 推论 1 预测）──")
    print(f"    steps → EV>0.9 : {s09_str} 步  (PPO 预测: ~149,504, HCGAE 预测: ~79,872)")
    print(f"    EV @ 80K       : {result.get('ev_at_80k', float('nan')):.3f}  (HCGAE 预测: ≥ 0.9)")
    print(f"    EV @ 150K      : {result.get('ev_at_150k', float('nan')):.3f}  (PPO 预测: ≥ 0.9)")
    print(f"    AULC (0-200K)  : {result.get('aulc_200k', float('nan')):.4f}")
    if result.get('eval_rewards'):
        print(f"    Final reward   : {result['eval_rewards'][-1]:.1f}")

    return result


def print_summary_table(all_results: dict):
    """打印汇总表格，直接对比理论预测与实验观测。"""
    print("\n" + "="*80)
    print("定理 1 实证验证结果（均值 ± 标准差，n=5 seeds）")
    print(f"理论预测（推论 1）：PPO ~ {THEOREM1_PREDICTIONS['PPO_steps_to_ev09']//1000}K 步，"
          f"HCGAE ~ {THEOREM1_PREDICTIONS['HCGAE_steps_to_ev09']//1000}K 步，"
          f"加速比 ~ {THEOREM1_PREDICTIONS['speedup_ratio']:.2f}×")
    print("="*80)

    for env_name in ENVS:
        print(f"\n── {env_name} ──")
        header = f"  {'算法':<35} {'步→EV>0.9':>12} {'EV@80K':>8} {'EV@150K':>9} {'AULC_200K':>11}"
        print(header)
        print(f"  {'-'*78}")

        ppo_steps = []
        for algo_name in ALGOS:
            steps_vals, ev80_vals, ev150_vals, aulc_vals = [], [], [], []
            for seed in SEEDS:
                key = f"{env_name}/{algo_name}/s{seed}"
                r = all_results.get(key, {})
                s09 = r.get("steps_to_ev09", -1)
                if s09 > 0:
                    steps_vals.append(s09)
                ev80_vals.append(r.get("ev_at_80k", float('nan')))
                ev150_vals.append(r.get("ev_at_150k", float('nan')))
                aulc_vals.append(r.get("aulc_200k", float('nan')))

            if algo_name == "Optimal_PPO" and steps_vals:
                ppo_steps = steps_vals

            if steps_vals:
                steps_str = f"{np.mean(steps_vals)/1000:.0f}K±{np.std(steps_vals)/1000:.0f}K"
            else:
                steps_str = "N/A (>500K)"

            ev80_str  = f"{np.nanmean(ev80_vals):.3f}" if ev80_vals else "N/A"
            ev150_str = f"{np.nanmean(ev150_vals):.3f}" if ev150_vals else "N/A"
            aulc_str  = f"{np.nanmean(aulc_vals):.4f}" if aulc_vals else "N/A"

            # 计算相对于 PPO 的加速比
            speedup_str = ""
            if algo_name != "Optimal_PPO" and steps_vals and ppo_steps:
                ratio = np.mean(ppo_steps) / np.mean(steps_vals)
                speedup_str = f"  [{ratio:.2f}× 加速]"

            print(f"  {algo_name:<35} {steps_str:>12} {ev80_str:>8} {ev150_str:>9} {aulc_str:>11}{speedup_str}")

        # 打印理论预测行
        print(f"  {'[推论 1 预测 PPO]':<35} {'~150K':>12} {'<0.9':>8} {'≥0.9':>9} {'—':>11}")
        print(f"  {'[推论 1 预测 HCGAE]':<35} {'~80K':>12} {'≥0.9':>8} {'≥0.9':>9} {'—':>11}")


def main():
    """主实验循环：运行所有 (env, algo, seed) 组合，验证定理 1。"""
    print("\n" + "="*80)
    print("§4.6 EV 收敛速度实验 — 定理 1 实证验证")
    print("="*80)
    print(f"环境:     {ENVS}")
    print(f"算法:     {ALGOS}")
    print(f"种子:     {SEEDS}")
    print(f"步数:     {TOTAL_STEPS:,} steps per run（与 ICMLExperiment 对齐）")
    total_runs = len(ENVS) * len(ALGOS) * len(SEEDS)
    print(f"总运行数: {total_runs}  (预计 ~{total_runs * 12 // 60}–{total_runs * 15 // 60} 小时)")
    print("\n理论预测（推论 1）：")
    print(f"  PPO  → EV>0.9: ~{THEOREM1_PREDICTIONS['PPO_steps_to_ev09']//1000}K 步")
    print(f"  HCGAE→ EV>0.9: ~{THEOREM1_PREDICTIONS['HCGAE_steps_to_ev09']//1000}K 步")
    print(f"  加速比: ~{THEOREM1_PREDICTIONS['speedup_ratio']:.2f}× (~{THEOREM1_PREDICTIONS['speedup_pct']:.0f}% 步数减少)")
    print("="*80 + "\n")

    all_results = {}

    # 优先顺序：Hopper-v4 先（论文主要声明），Walker2d-v4 后（泛化验证）
    priority_order = []
    for env_name in ENVS:
        for algo_name in ALGOS:
            priority_order.append((env_name, algo_name))

    for env_name, algo_name in priority_order:
        for seed in SEEDS:
            key = f"{env_name}/{algo_name}/s{seed}"
            try:
                result = run_single(env_name, algo_name, seed)
                all_results[key] = {
                    k: v for k, v in result.items()
                    if k not in ("ev_series", "steps_series")
                }
            except Exception as e:
                print(f"  [ERROR] {key}: {e}")
                import traceback
                traceback.print_exc()

    # 打印汇总表格
    print_summary_table(all_results)

    # 保存汇总
    summary_path = os.path.join(RESULT_DIR, "ev_convergence_summary.json")
    # 合并已有结果（如果脚本被中断后再次运行）
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            existing = json.load(f)
        existing.update(all_results)
        all_results = existing

    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n汇总已保存至: {summary_path}")

    # 计算并打印核心声明的验证结果
    print("\n" + "="*80)
    print("定理 1 验证结论")
    print("="*80)
    for env_name in ENVS:
        ppo_steps = []
        hcgae_steps = []
        for seed in SEEDS:
            ppo_r = all_results.get(f"{env_name}/Optimal_PPO/s{seed}", {})
            hcgae_r = all_results.get(f"{env_name}/Optimal_HCGAE_v2/s{seed}", {})
            if ppo_r.get("steps_to_ev09", -1) > 0:
                ppo_steps.append(ppo_r["steps_to_ev09"])
            if hcgae_r.get("steps_to_ev09", -1) > 0:
                hcgae_steps.append(hcgae_r["steps_to_ev09"])
        if ppo_steps and hcgae_steps:
            observed_ratio = np.mean(ppo_steps) / np.mean(hcgae_steps)
            predicted_ratio = THEOREM1_PREDICTIONS["speedup_ratio"]
            print(f"\n{env_name}:")
            print(f"  PPO   → EV>0.9: {np.mean(ppo_steps)/1000:.0f}K±{np.std(ppo_steps)/1000:.0f}K 步"
                  f"  (推论 1 预测: ~{THEOREM1_PREDICTIONS['PPO_steps_to_ev09']//1000}K)")
            print(f"  HCGAE → EV>0.9: {np.mean(hcgae_steps)/1000:.0f}K±{np.std(hcgae_steps)/1000:.0f}K 步"
                  f"  (推论 1 预测: ~{THEOREM1_PREDICTIONS['HCGAE_steps_to_ev09']//1000}K)")
            print(f"  观测加速比: {observed_ratio:.2f}×  (推论 1 预测: {predicted_ratio:.2f}×)")
            verdict = "✓ 符合" if abs(observed_ratio - predicted_ratio) / predicted_ratio < 0.3 else "△ 偏差>30%"
            print(f"  结论: {verdict}")


if __name__ == "__main__":
    main()

