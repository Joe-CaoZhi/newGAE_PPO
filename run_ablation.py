#!/usr/bin/env python3
"""
HCGAE 消融实验主脚本
====================
在 Hopper-v4 上依次运行 10 个变体，逐一分析各改进的贡献。
运行约需 60-90 分钟（单 CPU）。

结果保存至 results/Hopper-v4-Ablation/
"""
import json
import os
import random
import sys
import time

import gymnasium as gym
import numpy as np
import torch

# ── 路径设置 ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gae_experiments.agents.hindsight_ablation import (
    build_ablation_agent
)

# ════════════════════════════════════════════════════════════════════
# 实验配置
# ════════════════════════════════════════════════════════════════════
ENV_NAME        = "Hopper-v4"
TOTAL_TIMESTEPS = 300_000      # 30 万步（比完整实验短，消融不需要完全收敛）
N_STEPS         = 2048
BATCH_SIZE      = 64
N_EPOCHS        = 10
GAMMA           = 0.99
LAM             = 0.95
LR_ACTOR        = 3e-4
LR_CRITIC       = 1e-3
EPS_CLIP        = 0.2
HIDDEN_DIM      = 64
EVAL_FREQ       = 10_000
N_EVAL_EPS      = 10
SEED            = 42
SAVE_DIR        = "results/Hopper-v4-Ablation"
PRINT_INTERVAL  = 4            # 每 4 次 update 打印一次

# 本次消融运行的变体（按逻辑顺序：从最少改进到最多）
VARIANTS = [
    "HCGAE_Base",    # v1 基线
    "HCGAE_Imp1",    # 仅 ① 批内中心化归一化
    "HCGAE_Imp2",    # 仅 ② EV 驱动混合
    "HCGAE_Imp3",    # 仅 ③ 末端 Bootstrap 修正
    "HCGAE_Imp4",    # 仅 ④ 冻结统计量
    "HCGAE_Imp12",   # ①+②
    "HCGAE_Imp14",   # ①+④
    "HCGAE_Imp24",   # ②+④
    "HCGAE_Imp124",  # ①+②+④（不含末端）
    "HCGAE_Full",    # 全量 v2
]


# ════════════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════════════
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(env_name: str, seed: int) -> gym.Env:
    env = gym.make(env_name)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def print_section(title: str, width: int = 78):
    print(f"\n{'═'*width}")
    print(f"  {title}")
    print(f"{'═'*width}")


def run_variant(variant_name: str, exp_idx: int, n_total: int) -> dict:
    """运行单个消融变体，返回汇总结果字典"""
    print_section(f"[{exp_idx}/{n_total}] 变体: {variant_name}")

    set_seed(SEED)
    train_env = make_env(ENV_NAME, SEED)
    eval_env  = make_env(ENV_NAME, SEED + 1000)

    agent = build_ablation_agent(
        variant_name,
        train_env,
        hidden_dim      = HIDDEN_DIM,
        lr_actor        = LR_ACTOR,
        lr_critic       = LR_CRITIC,
        gamma           = GAMMA,
        lam             = LAM,
        eps_clip        = EPS_CLIP,
        n_epochs        = N_EPOCHS,
        batch_size      = BATCH_SIZE,
        n_steps         = N_STEPS,
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        device          = "cpu",
        save_dir        = SAVE_DIR,
    )

    # 打印实验标头
    print(f"  改进开关: ①{agent.use_imp1} ②{agent.use_imp2} "
          f"③{agent.use_imp3} ④{agent.use_imp4}")
    print(f"  环境: {ENV_NAME}  步数: {TOTAL_TIMESTEPS:,}  种子: {SEED}")
    print(f"  {'─'*74}")

    t0 = time.time()
    logger = agent.train(
        total_timesteps  = TOTAL_TIMESTEPS,
        eval_env         = eval_env,
        eval_freq        = EVAL_FREQ,
        n_eval_episodes  = N_EVAL_EPS,
        verbose          = True,
        print_interval   = PRINT_INTERVAL,
    )
    elapsed = time.time() - t0

    train_env.close()
    eval_env.close()

    # ── 汇总统计 ─────────────────────────────────────────────────
    eval_rewards = logger.eval_rewards if logger.eval_rewards else [0.0]
    final_reward  = float(np.mean(eval_rewards[-5:])) if len(eval_rewards) >= 5 else float(np.mean(eval_rewards))
    best_reward   = float(max(eval_rewards))
    peak_step     = int(logger.eval_steps[eval_rewards.index(max(eval_rewards))]) if eval_rewards else 0

    # 收敛步数：首次达到 best * 0.9 的步数
    threshold = best_reward * 0.9
    conv_step = TOTAL_TIMESTEPS
    for s, r in zip(logger.eval_steps, eval_rewards):
        if r >= threshold:
            conv_step = int(s)
            break

    # 训练稳定性：后半段评估奖励的标准差
    half = len(eval_rewards) // 2
    stability = float(np.std(eval_rewards[half:])) if half > 0 else 0.0

    # EV 统计
    ev_list = [m.get("explained_variance", 0) for m in logger.update_metrics] if hasattr(logger, "update_metrics") and logger.update_metrics else [0]
    # 从 logger 数据提取
    ev_data = logger.value_losses  # 用 value_loss 代理，再查 ev
    # 实际从 eval_rewards 中提取收集的 update 数据
    final_ev = float(agent._ev_ema)

    result = {
        "variant"       : variant_name,
        "use_imp1"      : agent.use_imp1,
        "use_imp2"      : agent.use_imp2,
        "use_imp3"      : agent.use_imp3,
        "use_imp4"      : agent.use_imp4,
        "final_reward"  : final_reward,
        "best_reward"   : best_reward,
        "peak_step"     : peak_step,
        "conv_step_90"  : conv_step,    # 达到最高奖励 90% 的步数
        "stability_std" : stability,    # 后半程评估标准差（越小越稳）
        "final_ev_ema"  : final_ev,
        "elapsed_s"     : round(elapsed, 1),
        "n_episodes"    : len(logger.episode_rewards),
        "eval_steps"    : list(logger.eval_steps),
        "eval_rewards"  : [float(r) for r in eval_rewards],
    }

    # 保存单个变体的详细结果
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(os.path.join(SAVE_DIR, f"{variant_name}_summary.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  {'─'*74}")
    print(f"  ✓ {variant_name} | 最终={final_reward:.1f} | 最高={best_reward:.1f} "
          f"| 耗时={elapsed:.0f}s | 收敛步={conv_step:,}")

    return result


# ════════════════════════════════════════════════════════════════════
# 数学分析打印
# ════════════════════════════════════════════════════════════════════
def print_math_analysis(results: list):
    """对消融结果进行数学分析，打印各改进的 Shapley 近似贡献"""
    print_section("数学分析：各改进的贡献量估计")

    # 构建查找字典
    r = {x["variant"]: x["final_reward"] for x in results}

    base = r.get("HCGAE_Base", 0.0)
    full = r.get("HCGAE_Full", 0.0)
    total_gain = full - base

    print(f"\n  基线（HCGAE_Base）最终奖励 : {base:.1f}")
    print(f"  全量（HCGAE_Full）最终奖励  : {full:.1f}")
    print(f"  总增益 Δ_total              : {total_gain:+.1f}")

    print(f"\n  {'─'*74}")
    print(f"  单一改进的增量贡献（相对于基线）")
    print(f"  {'─'*74}")
    for k, vname in [("①", "HCGAE_Imp1"), ("②", "HCGAE_Imp2"),
                      ("③", "HCGAE_Imp3"), ("④", "HCGAE_Imp4")]:
        v = r.get(vname, base)
        delta = v - base
        pct_of_total = (delta / (total_gain + 1e-8)) * 100
        print(f"  改进{k} ({vname:<14}): {v:7.1f}  Δ={delta:+7.1f}  "
              f"({pct_of_total:+6.1f}% of total gain)")

    print(f"\n  {'─'*74}")
    print(f"  交互效应分析（组合 vs 单独之和）")
    print(f"  {'─'*74}")

    # ①+② 组合
    if "HCGAE_Imp12" in r:
        imp12    = r["HCGAE_Imp12"]
        imp1_d   = r.get("HCGAE_Imp1", base) - base
        imp2_d   = r.get("HCGAE_Imp2", base) - base
        actual   = imp12 - base
        additive = imp1_d + imp2_d
        interact = actual - additive
        print(f"  ①+② 实际增益={actual:+.1f}  加性估计={additive:+.1f}  "
              f"交互效应={interact:+.1f}  "
              f"({'协同' if interact > 0 else '拮抗'})")

    # ①+② vs ①+②+④
    if "HCGAE_Imp124" in r:
        imp124   = r["HCGAE_Imp124"]
        imp12    = r.get("HCGAE_Imp12", base)
        imp4_d   = r.get("HCGAE_Imp4", base) - base
        actual   = imp124 - imp12
        print(f"  在①②基础上加④: Δ={actual:+.1f}  (④单独={imp4_d:+.1f})")

    # ③ 的边际贡献（在全量中）
    if "HCGAE_Imp124" in r and "HCGAE_Full" in r:
        delta_3 = r["HCGAE_Full"] - r["HCGAE_Imp124"]
        print(f"  在①②④基础上加③: Δ={delta_3:+.1f}  "
              f"(③单独={r.get('HCGAE_Imp3', base)-base:+.1f})")

    print(f"\n  {'─'*74}")
    print(f"  偏差-方差视角分析")
    print(f"  {'─'*74}")
    print(f"  改进① 主要作用：消除 EMA 滞后 → 修正 Critic 偏差更及时")
    print(f"    预期：提升 EV 收敛速度，减小 err_ema 与 err_batch_mean 的偏离")
    print(f"  改进② 主要作用：降低 Critic 目标方差（高 EV 时减少 MC 噪声）")
    print(f"    预期：降低 value_loss 方差，提升 EV 上限")
    print(f"  改进③ 主要作用：修正 rollout 边界 Critic 估计不一致性")
    print(f"    预期：减小 δ 的自相关（边界步的 δ 异常减少）")
    print(f"  改进④ 主要作用：消除 mini-batch 归一化漂移 → 稳定策略梯度")
    print(f"    预期：降低 approx_kl 方差，减少 clip_frac 峰值")

    print(f"\n  {'─'*74}")
    print(f"  Shapley 值近似（等权重 marginalization over subsets）")
    print(f"  {'─'*74}")
    # Shapley ≈ (贡献于各子集平均增益)
    # 简化：用已知的子集组合估计
    shapley = {}
    for i, (imp, vname_single) in enumerate(
        [("①", "HCGAE_Imp1"), ("②", "HCGAE_Imp2"),
         ("③", "HCGAE_Imp3"), ("④", "HCGAE_Imp4")]
    ):
        v_alone  = r.get(vname_single, base) - base          # {i} \ {}
        v_full   = r.get("HCGAE_Full", base) - base          # {all} \ {all\i} ≈ marginal
        shapley[imp] = (v_alone + v_full) / 2.0
    total_shapley = sum(shapley.values()) + 1e-8
    for imp, sv in shapley.items():
        print(f"  φ({imp}) ≈ {sv:+7.1f}  ({100*sv/total_shapley:.1f}% of total)")

    print(f"\n  注：Shapley 值使用了简化的两点估计（需完整 2^n 子集才能精确计算）")


# ════════════════════════════════════════════════════════════════════
# 汇总表打印
# ════════════════════════════════════════════════════════════════════
def print_summary_table(results: list):
    print_section("消融实验汇总表")
    header = (
        f"  {'变体':<16} {'①':^3} {'②':^3} {'③':^3} {'④':^3} "
        f"{'最终':>8} {'最高':>8} {'Δ vs Base':>10} "
        f"{'收敛步':>8} {'稳定性σ':>8} {'时间(s)':>8}"
    )
    print(header)
    print(f"  {'─'*16} {'─'*3} {'─'*3} {'─'*3} {'─'*3} "
          f"{'─'*8} {'─'*8} {'─'*10} {'─'*8} {'─'*8} {'─'*8}")

    base_reward = next((r["final_reward"] for r in results if r["variant"] == "HCGAE_Base"), 0.0)

    for res in results:
        delta = res["final_reward"] - base_reward
        flag  = "↑" if delta > 50 else ("↓" if delta < -50 else "~")
        print(
            f"  {res['variant']:<16} "
            f"{'✓' if res['use_imp1'] else '✗':^3} "
            f"{'✓' if res['use_imp2'] else '✗':^3} "
            f"{'✓' if res['use_imp3'] else '✗':^3} "
            f"{'✓' if res['use_imp4'] else '✗':^3} "
            f"{res['final_reward']:8.1f} "
            f"{res['best_reward']:8.1f} "
            f"{delta:+9.1f}{flag} "
            f"{res['conv_step_90']:8,} "
            f"{res['stability_std']:8.1f} "
            f"{res['elapsed_s']:8.1f}"
        )


# ════════════════════════════════════════════════════════════════════
# 可视化
# ════════════════════════════════════════════════════════════════════
def plot_ablation_results(results: list):
    """生成 4 张消融分析图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── 颜色方案 ─────────────────────────────────────────────────
    COLOR_MAP = {
        "HCGAE_Base"  : "#9E9E9E",
        "HCGAE_Imp1"  : "#2196F3",
        "HCGAE_Imp2"  : "#FF9800",
        "HCGAE_Imp3"  : "#9C27B0",
        "HCGAE_Imp4"  : "#00BCD4",
        "HCGAE_Imp12" : "#4CAF50",
        "HCGAE_Imp14" : "#8BC34A",
        "HCGAE_Imp24" : "#FFC107",
        "HCGAE_Imp124": "#FF5722",
        "HCGAE_Full"  : "#F44336",
    }
    MARKERS = {
        "HCGAE_Base"  : "o",
        "HCGAE_Imp1"  : "s",
        "HCGAE_Imp2"  : "^",
        "HCGAE_Imp3"  : "D",
        "HCGAE_Imp4"  : "p",
        "HCGAE_Imp12" : "h",
        "HCGAE_Imp14" : "8",
        "HCGAE_Imp24" : "*",
        "HCGAE_Imp124": "X",
        "HCGAE_Full"  : "P",
    }

    # ── 图 1：学习曲线对比 ────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(14, 7))
    for res in results:
        name   = res["variant"]
        steps  = res["eval_steps"]
        rewards = res["eval_rewards"]
        if not steps:
            continue
        lw = 2.5 if name in ("HCGAE_Base", "HCGAE_Full") else 1.5
        ls = "--" if name == "HCGAE_Base" else "-"
        ax1.plot(steps, rewards,
                 color=COLOR_MAP.get(name, "#607D8B"),
                 linewidth=lw, linestyle=ls,
                 marker=MARKERS.get(name, "o"), markevery=max(1, len(steps)//8),
                 markersize=6, label=name)

    ax1.set_xlabel("Environment Steps", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Evaluation Reward", fontsize=12, fontweight="bold")
    ax1.set_title("HCGAE Ablation Study: Learning Curves (Hopper-v4)", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9, ncol=2, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle="--")
    fig1.tight_layout()
    fig1.savefig(os.path.join(SAVE_DIR, "ablation_learning_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  📊 学习曲线已保存")

    # ── 图 2：最终性能柱状图 ──────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))

    names  = [r["variant"] for r in results]
    finals = [r["final_reward"] for r in results]
    bests  = [r["best_reward"] for r in results]
    stabs  = [r["stability_std"] for r in results]
    colors = [COLOR_MAP.get(n, "#607D8B") for n in names]
    base_r = next((r["final_reward"] for r in results if r["variant"] == "HCGAE_Base"), 0.0)

    # 子图 a：最终奖励
    bars = axes2[0].bar(range(len(names)), finals, color=colors, edgecolor="white", linewidth=0.5)
    axes2[0].axhline(base_r, color="#9E9E9E", linestyle="--", linewidth=1.5, label="Base")
    axes2[0].set_xticks(range(len(names)))
    axes2[0].set_xticklabels([n.replace("HCGAE_", "") for n in names], rotation=35, ha="right", fontsize=9)
    axes2[0].set_ylabel("Final Reward (last 5 evals)", fontsize=11)
    axes2[0].set_title("Final Performance", fontweight="bold")
    axes2[0].grid(axis="y", alpha=0.3, linestyle="--")
    for bar, val in zip(bars, finals):
        axes2[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                      f"{val:.0f}", ha="center", va="bottom", fontsize=7)

    # 子图 b：增量贡献（Δ vs 基线）
    deltas = [f - base_r for f in finals]
    bar_colors2 = ["#4CAF50" if d >= 0 else "#F44336" for d in deltas]
    axes2[1].bar(range(len(names)), deltas, color=bar_colors2, edgecolor="white", linewidth=0.5)
    axes2[1].axhline(0, color="black", linewidth=1.0)
    axes2[1].set_xticks(range(len(names)))
    axes2[1].set_xticklabels([n.replace("HCGAE_", "") for n in names], rotation=35, ha="right", fontsize=9)
    axes2[1].set_ylabel("Δ Final Reward vs Base", fontsize=11)
    axes2[1].set_title("Marginal Gain vs Baseline", fontweight="bold")
    axes2[1].grid(axis="y", alpha=0.3, linestyle="--")

    # 子图 c：稳定性（后半程 eval σ）
    axes2[2].bar(range(len(names)), stabs, color=colors, edgecolor="white", linewidth=0.5)
    axes2[2].set_xticks(range(len(names)))
    axes2[2].set_xticklabels([n.replace("HCGAE_", "") for n in names], rotation=35, ha="right", fontsize=9)
    axes2[2].set_ylabel("Eval Reward Std (2nd half)", fontsize=11)
    axes2[2].set_title("Training Stability (lower = more stable)", fontweight="bold")
    axes2[2].grid(axis="y", alpha=0.3, linestyle="--")

    fig2.suptitle("HCGAE Ablation Study: Performance Breakdown (Hopper-v4)", fontsize=13, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(os.path.join(SAVE_DIR, "ablation_bar_charts.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  📊 柱状图已保存")

    # ── 图 3：改进贡献热力图（4×4 二进制矩阵着色）──────────────
    fig3, ax3 = plt.subplots(figsize=(10, 7))

    # 将变体编码为二进制向量
    imp_matrix = np.array([
        [int(r["use_imp1"]), int(r["use_imp2"]),
         int(r["use_imp3"]), int(r["use_imp4"])]
        for r in results
    ], dtype=float)

    # reward 着色
    reward_arr = np.array(finals)
    reward_norm = (reward_arr - reward_arr.min()) / (reward_arr.max() - reward_arr.min() + 1e-8)

    ax3.set_xlim(-0.5, 3.5)
    ax3.set_ylim(-0.5, len(results) - 0.5)

    # 绘制矩阵单元格
    import matplotlib.cm as cm
    cmap = cm.RdYlGn
    for i, res in enumerate(results):
        row_reward = (res["final_reward"] - reward_arr.min()) / (reward_arr.max() - reward_arr.min() + 1e-8)
        for j in range(4):
            active = imp_matrix[i, j]
            cell_color = cmap(row_reward) if active else "#F5F5F5"
            edge_color = "gray" if active else "#E0E0E0"
            rect = plt.Rectangle([j - 0.45, i - 0.45], 0.9, 0.9,
                                  color=cell_color, ec=edge_color, lw=1.0)
            ax3.add_patch(rect)
            ax3.text(j, i, "✓" if active else "✗",
                     ha="center", va="center", fontsize=14,
                     color="white" if active else "#BDBDBD", fontweight="bold")
        # 右侧标注奖励值
        ax3.text(4.1, i, f"{res['final_reward']:.0f}", ha="left", va="center",
                 fontsize=9, color=cmap(row_reward))

    ax3.set_xticks([0, 1, 2, 3])
    ax3.set_xticklabels(["① 批内归一化", "② EV混合", "③ 末端修正", "④ 冻结统计量"],
                         fontsize=10, fontweight="bold")
    ax3.set_yticks(range(len(results)))
    ax3.set_yticklabels([r["variant"] for r in results], fontsize=9)
    ax3.set_title("HCGAE Ablation Matrix\n(green=high reward, red=low reward)", fontsize=12, fontweight="bold")

    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=reward_arr.min(), vmax=reward_arr.max()))
    sm.set_array([])
    plt.colorbar(sm, ax=ax3, label="Final Reward", shrink=0.6)

    fig3.tight_layout()
    fig3.savefig(os.path.join(SAVE_DIR, "ablation_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"  📊 改进矩阵热力图已保存")

    # ── 图 4：Shapley 贡献估计图 ─────────────────────────────────
    r_dict = {x["variant"]: x["final_reward"] for x in results}
    base_v = r_dict.get("HCGAE_Base", 0.0)

    shapley_labels = ["①批内归一化", "②EV混合", "③末端修正", "④冻结统计量"]
    single_deltas  = [
        r_dict.get("HCGAE_Imp1", base_v) - base_v,
        r_dict.get("HCGAE_Imp2", base_v) - base_v,
        r_dict.get("HCGAE_Imp3", base_v) - base_v,
        r_dict.get("HCGAE_Imp4", base_v) - base_v,
    ]
    full_marginals = [
        r_dict.get("HCGAE_Full", base_v) - r_dict.get("HCGAE_Imp124",  base_v),  # ③ 的边际
        r_dict.get("HCGAE_Full", base_v) - r_dict.get("HCGAE_Imp124",  base_v),  # ③ 边际（proxy）
        r_dict.get("HCGAE_Full", base_v) - r_dict.get("HCGAE_Imp124",  base_v),  # ③ 边际
        r_dict.get("HCGAE_Imp124",  base_v) - r_dict.get("HCGAE_Imp12",  base_v),  # ④ 的边际
    ]
    # 简化 Shapley：(单独贡献 + 全量边际贡献) / 2
    shapley_vals = [(s + f) / 2.0 for s, f in zip(single_deltas, full_marginals)]
    # 修正③和④
    shapley_vals[2] = (single_deltas[2] + (r_dict.get("HCGAE_Full", base_v) - r_dict.get("HCGAE_Imp124", base_v))) / 2.0
    shapley_vals[3] = (single_deltas[3] + (r_dict.get("HCGAE_Imp124", base_v) - r_dict.get("HCGAE_Imp12", base_v))) / 2.0

    fig4, ax4 = plt.subplots(figsize=(9, 5))
    bar_c = ["#4CAF50" if v >= 0 else "#F44336" for v in shapley_vals]
    bars4 = ax4.barh(shapley_labels, shapley_vals, color=bar_c, edgecolor="white", height=0.6)
    ax4.axvline(0, color="black", linewidth=1.0)
    for bar, val in zip(bars4, shapley_vals):
        ax4.text(val + (5 if val >= 0 else -5), bar.get_y() + bar.get_height()/2,
                 f"{val:+.0f}", ha="left" if val >= 0 else "right",
                 va="center", fontsize=10, fontweight="bold")
    ax4.set_xlabel("Approximate Shapley Value (reward contribution)", fontsize=11)
    ax4.set_title("HCGAE v2: Approximate Shapley Values of Each Improvement\n(Hopper-v4, simplified 2-point estimation)",
                  fontsize=11, fontweight="bold")
    ax4.grid(axis="x", alpha=0.3, linestyle="--")
    fig4.tight_layout()
    fig4.savefig(os.path.join(SAVE_DIR, "ablation_shapley.png"), dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print(f"  📊 Shapley 贡献图已保存")


# ════════════════════════════════════════════════════════════════════
# 主程序
# ════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"\n{'#'*78}")
    print(f"  HCGAE 消融实验  |  环境: {ENV_NAME}  |  步数: {TOTAL_TIMESTEPS:,}/变体")
    print(f"  共 {len(VARIANTS)} 个变体  |  结果保存至: {SAVE_DIR}")
    print(f"{'#'*78}")

    all_results = []
    total_start = time.time()

    for idx, variant in enumerate(VARIANTS, start=1):
        result = run_variant(variant, idx, len(VARIANTS))
        all_results.append(result)

        # 每个变体完成后打印当前排行
        sorted_r = sorted(all_results, key=lambda x: x["final_reward"], reverse=True)
        print(f"\n  当前排行（已完成 {idx}/{len(VARIANTS)}）：")
        for rank, res in enumerate(sorted_r, 1):
            base_r = next((r["final_reward"] for r in all_results if r["variant"] == "HCGAE_Base"), 0.0)
            delta = res["final_reward"] - base_r
            print(f"    #{rank:2d} {res['variant']:<18} {res['final_reward']:7.1f}  (Δ={delta:+7.1f})")

    # ── 汇总 ────────────────────────────────────────────────────
    print_summary_table(all_results)
    print_math_analysis(all_results)

    # ── 可视化 ──────────────────────────────────────────────────
    print_section("生成可视化图表")
    plot_ablation_results(all_results)

    # ── 保存汇总 JSON ────────────────────────────────────────────
    summary_path = os.path.join(SAVE_DIR, "ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  完整汇总已保存至: {summary_path}")

    total_elapsed = time.time() - total_start
    print(f"\n{'#'*78}")
    print(f"  消融实验全部完成！总耗时: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print(f"{'#'*78}\n")


if __name__ == "__main__":
    main()

