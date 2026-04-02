"""
可视化模块：对比多个 GAE 改进方案的实验结果
"""
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Dict
from scipy.signal import savgol_filter

from .logger import MetricLogger


# ─────────────────────────────────────────────
# 颜色方案和标签
# ─────────────────────────────────────────────
AGENT_COLORS = {
    "Standard_GAE":                "#2196F3",   # 蓝色 - 基线
    "Conservative_Bootstrap_GAE":  "#F44336",   # 红色
    "Adaptive_Lambda_GAE":         "#4CAF50",   # 绿色
    "Confidence_Weighted_GAE":     "#FF9800",   # 橙色
    "Combined_GAE":                "#9C27B0",   # 紫色
    # 革新方法
    "Hindsight_GAE":               "#00BCD4",   # 青色 - HCGAE
    "MultiScale_GAE":              "#FF5722",   # 深橙 - MSGAE
    "CausalAttn_GAE":              "#E91E63",   # 粉红 - CAGAE（最革新）
}

AGENT_LABELS = {
    "Standard_GAE":                "Baseline (Standard GAE)",
    "Conservative_Bootstrap_GAE":  "Conservative Bootstrap GAE",
    "Adaptive_Lambda_GAE":         "Adaptive λ GAE",
    "Confidence_Weighted_GAE":     "Confidence-Weighted GAE",
    "Combined_GAE":                "Combined GAE",
    # 革新方法
    "Hindsight_GAE":               "★ Hindsight-Corrected GAE (Novel)",
    "MultiScale_GAE":              "★ Multi-Scale GAE (Novel)",
    "CausalAttn_GAE":              "★ Causal Attention GAE (Novel)",
}

AGENT_LINESTYLES = {
    "Standard_GAE":                "-",
    "Conservative_Bootstrap_GAE":  "--",
    "Adaptive_Lambda_GAE":         "-.",
    "Confidence_Weighted_GAE":     ":",
    "Combined_GAE":                "-",
    # 革新方法
    "Hindsight_GAE":               "-",
    "MultiScale_GAE":              "-",
    "CausalAttn_GAE":              "-",
}


def smooth(values: List[float], window: int = 15) -> np.ndarray:
    """滑动平均平滑"""
    arr = np.array(values, dtype=np.float64)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def smooth_savgol(values: List[float], window: int = 21, polyorder: int = 3) -> np.ndarray:
    """Savitzky-Golay 滤波平滑（保留峰值特征）"""
    arr = np.array(values, dtype=np.float64)
    if len(arr) < window:
        return arr
    window = min(window, len(arr))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return arr
    return savgol_filter(arr, window_length=window, polyorder=min(polyorder, window - 1))


def load_all_loggers(results_dir: str) -> Dict[str, MetricLogger]:
    """从 results 目录加载所有 agent 的日志"""
    loggers = {}
    for fname in os.listdir(results_dir):
        if fname.endswith("_metrics.json"):
            path = os.path.join(results_dir, fname)
            try:
                logger = MetricLogger.load(path)
                loggers[logger.agent_name] = logger
            except Exception as e:
                print(f"⚠ 加载 {fname} 失败: {e}")
    return loggers


def plot_all_results(
    loggers: Dict[str, MetricLogger],
    env_name: str,
    save_dir: str = "results",
    show: bool = False,
):
    """
    主可视化函数：生成多子图对比图
    - 子图1: 评估奖励对比（主图）
    - 子图2: 训练 Episode 奖励（平滑）
    - 子图3: 价值损失
    - 子图4: 策略损失
    - 子图5: Explained Variance
    - 子图6: Approx KL 散度
    - 子图7: 自适应 λ 均值（仅支持的 agent）
    - 子图8: 性能汇总柱状图
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f9fa",
        "axes.grid": True,
        "grid.alpha": 0.4,
        "grid.linestyle": "--",
    })

    fig = plt.figure(figsize=(22, 18))
    fig.suptitle(
        f"GAE Improvement Methods Comparison\nEnvironment: {env_name}",
        fontsize=16, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_eval    = fig.add_subplot(gs[0, :2])   # 评估奖励（大图）
    ax_ep      = fig.add_subplot(gs[0, 2])    # Episode 奖励
    ax_vloss   = fig.add_subplot(gs[1, 0])    # 价值损失
    ax_ploss   = fig.add_subplot(gs[1, 1])    # 策略损失
    ax_ev      = fig.add_subplot(gs[1, 2])    # Explained Variance
    ax_kl      = fig.add_subplot(gs[2, 0])    # Approx KL
    ax_lambda  = fig.add_subplot(gs[2, 1])    # 自适应 λ
    ax_bar     = fig.add_subplot(gs[2, 2])    # 性能汇总

    sorted_names = sorted(loggers.keys(), key=lambda n: (n != "Standard_GAE", n))
    legend_handles = []

    for name in sorted_names:
        logger = loggers[name]
        color = AGENT_COLORS.get(name, "#607D8B")
        label = AGENT_LABELS.get(name, name)
        ls = AGENT_LINESTYLES.get(name, "-")
        lw = 2.5 if name == "Combined_GAE" else 1.8

        # ── 评估奖励 ──
        if logger.eval_rewards:
            steps = np.array(logger.eval_steps)
            rewards = np.array(logger.eval_rewards)
            smooth_r = smooth(rewards, window=min(5, len(rewards)))
            line, = ax_eval.plot(
                steps, smooth_r, color=color, linestyle=ls, linewidth=lw, label=label
            )
            ax_eval.fill_between(steps, rewards - 0, rewards + 0, color=color, alpha=0.1)
            legend_handles.append(line)

        # ── Episode 奖励 ──
        if logger.episode_rewards:
            ep_r = np.array(logger.episode_rewards)
            w = min(30, len(ep_r))
            smooth_ep = smooth(ep_r, window=w)
            ax_ep.plot(smooth_ep, color=color, linestyle=ls, linewidth=lw, label=label)

        # ── 训练指标（按 update 步数对齐）──
        steps_train = np.array(logger.total_steps)

        if logger.value_losses:
            sv = smooth(logger.value_losses, window=5)
            ax_vloss.plot(steps_train[:len(sv)], sv, color=color, linestyle=ls, linewidth=lw)

        if logger.policy_losses:
            sp = smooth(logger.policy_losses, window=5)
            ax_ploss.plot(steps_train[:len(sp)], sp, color=color, linestyle=ls, linewidth=lw)

        if logger.explained_variances:
            sev = smooth(logger.explained_variances, window=5)
            ax_ev.plot(steps_train[:len(sev)], sev, color=color, linestyle=ls, linewidth=lw)

        if logger.approx_kls:
            skl = smooth(logger.approx_kls, window=5)
            ax_kl.plot(steps_train[:len(skl)], skl, color=color, linestyle=ls, linewidth=lw)

        # ── 自适应 λ ──
        if logger.mean_lambda_values:
            sl = smooth(logger.mean_lambda_values, window=5)
            ax_lambda.plot(
                steps_train[:len(sl)], sl,
                color=color, linestyle=ls, linewidth=lw, label=label
            )

    # ── 柱状图：最终评估奖励 ──
    bar_names = []
    bar_values = []
    bar_colors = []
    for name in sorted_names:
        logger = loggers[name]
        if logger.eval_rewards:
            last_n = min(5, len(logger.eval_rewards))
            final_reward = np.mean(logger.eval_rewards[-last_n:])
            bar_names.append(AGENT_LABELS.get(name, name).replace(" ", "\n"))
            bar_values.append(final_reward)
            bar_colors.append(AGENT_COLORS.get(name, "#607D8B"))

    if bar_values:
        bars = ax_bar.bar(range(len(bar_names)), bar_values, color=bar_colors, edgecolor="white", linewidth=1.5)
        ax_bar.set_xticks(range(len(bar_names)))
        ax_bar.set_xticklabels(bar_names, fontsize=8)
        ax_bar.set_title("Final Eval Reward (last 5 evals)")
        ax_bar.set_ylabel("Mean Reward")
        # 添加数值标签
        for bar, val in zip(bars, bar_values):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + abs(bar.get_height()) * 0.02,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold"
            )

    # 设置坐标轴标签和标题
    ax_eval.set_title("Evaluation Reward over Training Steps", fontweight="bold")
    ax_eval.set_xlabel("Environment Steps")
    ax_eval.set_ylabel("Mean Eval Reward")

    ax_ep.set_title("Episode Reward (smoothed)")
    ax_ep.set_xlabel("Episode")
    ax_ep.set_ylabel("Reward")

    ax_vloss.set_title("Value Loss")
    ax_vloss.set_xlabel("Steps")
    ax_vloss.set_ylabel("Loss")

    ax_ploss.set_title("Policy Loss")
    ax_ploss.set_xlabel("Steps")
    ax_ploss.set_ylabel("Loss")

    ax_ev.set_title("Explained Variance")
    ax_ev.set_xlabel("Steps")
    ax_ev.set_ylabel("EV (↑ better)")

    ax_kl.set_title("Approx KL Divergence")
    ax_kl.set_xlabel("Steps")
    ax_kl.set_ylabel("KL")

    ax_lambda.set_title("Mean λ Value (Adaptive agents)")
    ax_lambda.set_xlabel("Steps")
    ax_lambda.set_ylabel("λ")
    ax_lambda.set_ylim(0, 1)

    # 主图图例
    if legend_handles:
        ax_eval.legend(handles=legend_handles, loc="lower right", framealpha=0.9)

    if ax_lambda.lines:
        ax_lambda.legend(fontsize=8, loc="best")

    save_path = os.path.join(save_dir, f"comparison_{env_name.replace('/', '_')}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n📊 对比图已保存至: {save_path}")

    if show:
        plt.show()
    plt.close(fig)
    return save_path


def plot_learning_curves_single(
    loggers: Dict[str, MetricLogger],
    env_name: str,
    save_dir: str = "results",
):
    """单独绘制学习曲线（论文级别图表）"""
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "figure.facecolor": "white",
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    sorted_names = sorted(loggers.keys(), key=lambda n: (n != "Standard_GAE", n))

    for name in sorted_names:
        logger = loggers[name]
        if not logger.eval_rewards:
            continue
        color = AGENT_COLORS.get(name, "#607D8B")
        label = AGENT_LABELS.get(name, name)
        ls = AGENT_LINESTYLES.get(name, "-")
        lw = 2.5 if name in ["Combined_GAE", "Standard_GAE"] else 1.8

        steps = np.array(logger.eval_steps)
        rewards = np.array(logger.eval_rewards)
        smooth_r = smooth(rewards, window=min(5, len(rewards)))

        ax.plot(steps, smooth_r, color=color, linestyle=ls, linewidth=lw, label=label, zorder=3)
        ax.fill_between(steps, rewards * 0.97, rewards * 1.03, color=color, alpha=0.08, zorder=2)

    ax.set_xlabel("Environment Steps", fontweight="bold")
    ax.set_ylabel("Mean Evaluation Reward", fontweight="bold")
    ax.set_title(f"GAE Methods Comparison — {env_name}", fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.95, edgecolor="gray")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_facecolor("#fafafa")

    save_path = os.path.join(save_dir, f"learning_curves_{env_name.replace('/', '_')}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"📈 学习曲线已保存至: {save_path}")
    plt.close(fig)
    return save_path


def print_summary_table(loggers: Dict[str, MetricLogger]):
    """打印文字格式的结果汇总表"""
    print("\n" + "═" * 80)
    print("  GAE 改进方案实验结果汇总")
    print("═" * 80)
    print(f"  {'方法':<30}  {'最终评估奖励':>12}  {'最高评估奖励':>12}  {'评估次数':>8}")
    print("─" * 80)

    sorted_names = sorted(loggers.keys(), key=lambda n: (n != "Standard_GAE", n))

    results = {}
    for name in sorted_names:
        logger = loggers[name]
        if logger.eval_rewards:
            last_n = min(5, len(logger.eval_rewards))
            final = np.mean(logger.eval_rewards[-last_n:])
            best = np.max(logger.eval_rewards)
            n_evals = len(logger.eval_rewards)
        else:
            final, best, n_evals = 0, 0, 0
        results[name] = final

        label = AGENT_LABELS.get(name, name)
        marker = "★" if name == "Combined_GAE" else " "
        print(f"  {marker} {label:<28}  {final:>12.2f}  {best:>12.2f}  {n_evals:>8}")

    print("═" * 80)

    # 计算相对提升
    if "Standard_GAE" in results and results["Standard_GAE"] != 0:
        baseline = results["Standard_GAE"]
        print("\n  相对基线提升:")
        for name, val in results.items():
            if name != "Standard_GAE":
                improvement = (val - baseline) / abs(baseline) * 100
                label = AGENT_LABELS.get(name, name)
                arrow = "↑" if improvement >= 0 else "↓"
                print(f"  {label:<30} {arrow} {improvement:+.1f}%")
    print()


def plot_comprehensive_analysis(
    loggers: Dict[str, MetricLogger],
    env_name: str,
    save_dir: str = "results",
    show: bool = False,
) -> str:
    """
    综合分析图（论文级别）：
    - 图1: 评估奖励对比（主图，含置信区间）
    - 图2: 收敛速度雷达图
    - 图3: 最终性能对比柱状图（含改进率）
    - 图4: 训练稳定性分析（EV 曲线）
    - 图5: 价值损失对比
    - 图6: 各方法收敛到不同阈值的步数热力图
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f9fa",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    })

    fig = plt.figure(figsize=(24, 20))
    fig.suptitle(
        f"Comprehensive GAE Methods Analysis — {env_name}\n"
        f"(Hindsight GAE / Multi-Scale GAE / Causal Attention GAE vs. Baselines)",
        fontsize=15, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)

    ax_main   = fig.add_subplot(gs[0, :2])   # 主曲线图（大）
    ax_bar    = fig.add_subplot(gs[0, 2])    # 最终性能柱状图
    ax_ev     = fig.add_subplot(gs[1, 0])    # Explained Variance
    ax_vloss  = fig.add_subplot(gs[1, 1])    # Value Loss
    ax_kl     = fig.add_subplot(gs[1, 2])    # Approx KL
    ax_heat   = fig.add_subplot(gs[2, :2])   # 收敛热力图
    ax_box    = fig.add_subplot(gs[2, 2])    # 后20%奖励分布图

    sorted_names = sorted(loggers.keys(), key=lambda n: (n != "Standard_GAE", n))
    novel_methods = {"Hindsight_GAE", "MultiScale_GAE", "CausalAttn_GAE"}

    # ── 主评估曲线 ──
    final_rewards = {}
    for name in sorted_names:
        logger = loggers[name]
        if not logger.eval_rewards:
            continue
        color = AGENT_COLORS.get(name, "#607D8B")
        label = AGENT_LABELS.get(name, name)
        ls = AGENT_LINESTYLES.get(name, "-")
        lw = 2.8 if name in novel_methods else (2.2 if name == "Standard_GAE" else 1.5)
        alpha = 1.0 if name in novel_methods or name == "Standard_GAE" else 0.7

        steps = np.array(logger.eval_steps)
        rewards = np.array(logger.eval_rewards)
        smooth_r = smooth(rewards, window=min(7, len(rewards)))

        ax_main.plot(steps, smooth_r, color=color, linestyle=ls,
                     linewidth=lw, label=label, alpha=alpha, zorder=3)
        if name in novel_methods:
            # 添加原始奖励的半透明阴影
            ax_main.fill_between(steps,
                                  np.minimum(smooth_r, rewards) - np.abs(smooth_r - rewards) * 0.5,
                                  np.maximum(smooth_r, rewards) + np.abs(smooth_r - rewards) * 0.5,
                                  color=color, alpha=0.12, zorder=2)

        last_n = min(5, len(rewards))
        final_rewards[name] = float(np.mean(rewards[-last_n:]))

    ax_main.set_xlabel("Environment Steps", fontweight="bold")
    ax_main.set_ylabel("Mean Evaluation Reward", fontweight="bold")
    ax_main.set_title(f"Evaluation Reward — {env_name}", fontweight="bold")
    ax_main.legend(loc="lower right", framealpha=0.95, edgecolor="gray", fontsize=8)

    # ── 最终性能柱状图（含改进率标注）──
    names_list = [n for n in sorted_names if n in final_rewards]
    vals = [final_rewards[n] for n in names_list]
    colors_bar = [AGENT_COLORS.get(n, "#607D8B") for n in names_list]
    baseline_val = final_rewards.get("Standard_GAE", 1.0)

    bars = ax_bar.barh(range(len(names_list)), vals, color=colors_bar, alpha=0.82, edgecolor="white")
    ax_bar.set_yticks(range(len(names_list)))
    short_labels = [AGENT_LABELS.get(n, n).replace("★ ", "").replace(" (Novel)", "")[:22]
                    for n in names_list]
    ax_bar.set_yticklabels(short_labels, fontsize=8)
    ax_bar.set_xlabel("Final Reward", fontweight="bold")
    ax_bar.set_title("Final Performance", fontweight="bold")
    # 添加改进率标注
    for i, (n, v) in enumerate(zip(names_list, vals)):
        if n != "Standard_GAE" and abs(baseline_val) > 1e-3:
            imp = (v - baseline_val) / abs(baseline_val) * 100
            color_txt = "#2e7d32" if imp >= 0 else "#c62828"
            ax_bar.text(max(v, 0) + abs(max(vals) - min(vals)) * 0.02,
                        i, f"{imp:+.0f}%", va="center", ha="left",
                        fontsize=8, color=color_txt, fontweight="bold")
    # 标注革新方法
    for i, n in enumerate(names_list):
        if n in novel_methods:
            bars[i].set_edgecolor("#333")
            bars[i].set_linewidth(1.5)

    # ── Explained Variance ──
    for name in sorted_names:
        logger = loggers[name]
        evs = logger.explained_variances
        steps_train = logger.total_steps
        if not evs or not steps_train:
            continue
        n = min(len(evs), len(steps_train))
        color = AGENT_COLORS.get(name, "#607D8B")
        ls = AGENT_LINESTYLES.get(name, "-")
        lw = 2.2 if name in novel_methods else 1.4
        alpha_val = 1.0 if name in novel_methods or name == "Standard_GAE" else 0.6
        smooth_ev = smooth(evs[:n], window=min(15, n))
        ax_ev.plot(steps_train[:n], smooth_ev, color=color, linestyle=ls,
                   linewidth=lw, alpha=alpha_val, label=AGENT_LABELS.get(name, name))
    ax_ev.set_xlabel("Environment Steps")
    ax_ev.set_ylabel("Explained Variance")
    ax_ev.set_title("Critic Accuracy (EV)", fontweight="bold")
    all_evs = [v for nm in sorted_names for v in loggers[nm].explained_variances]
    if all_evs:
        lo, hi = min(all_evs), max(all_evs)
        ax_ev.set_ylim(min(-0.05, lo - 0.05), max(1.05, hi + 0.05))
    ax_ev.axhline(0, color="black", alpha=0.3, linewidth=0.8)
    ax_ev.legend(fontsize=7, loc="lower right")

    # ── Value Loss ──
    for name in sorted_names:
        logger = loggers[name]
        vlosses = logger.value_losses
        steps_train = logger.total_steps
        if not vlosses or not steps_train:
            continue
        n = min(len(vlosses), len(steps_train))
        color = AGENT_COLORS.get(name, "#607D8B")
        ls = AGENT_LINESTYLES.get(name, "-")
        lw = 2.2 if name in novel_methods else 1.4
        alpha_val = 1.0 if name in novel_methods or name == "Standard_GAE" else 0.6
        smooth_vl = smooth(vlosses[:n], window=min(15, n))
        ax_vloss.plot(steps_train[:n], smooth_vl, color=color, linestyle=ls,
                      linewidth=lw, alpha=alpha_val, label=AGENT_LABELS.get(name, name))
    ax_vloss.set_xlabel("Environment Steps")
    ax_vloss.set_ylabel("Value Loss")
    ax_vloss.set_title("Value Loss", fontweight="bold")
    ax_vloss.legend(fontsize=7)

    # ── Approx KL ──
    for name in sorted_names:
        logger = loggers[name]
        kls = logger.approx_kls
        steps_train = logger.total_steps
        if not kls or not steps_train:
            continue
        n = min(len(kls), len(steps_train))
        color = AGENT_COLORS.get(name, "#607D8B")
        ls = AGENT_LINESTYLES.get(name, "-")
        lw = 2.2 if name in novel_methods else 1.4
        alpha_val = 1.0 if name in novel_methods or name == "Standard_GAE" else 0.6
        smooth_kl = smooth(kls[:n], window=min(15, n))
        ax_kl.plot(steps_train[:n], smooth_kl, color=color, linestyle=ls,
                   linewidth=lw, alpha=alpha_val, label=AGENT_LABELS.get(name, name))
    ax_kl.set_xlabel("Environment Steps")
    ax_kl.set_ylabel("Approx KL")
    ax_kl.set_title("Policy Update KL", fontweight="bold")
    ax_kl.legend(fontsize=7)

    # ── 收敛热力图 ──
    # 计算各方法首次达到不同阈值的步数
    if final_rewards:
        max_reward = max(final_rewards.values())
        min_reward = min(final_rewards.values())
        reward_range = max_reward - min_reward + 1e-3
        # 用百分位数设置阈值（50%, 70%, 85%, 95% of max performance）
        thresholds = [0.50, 0.65, 0.80, 0.90, 0.95]
        heat_data = np.full((len(names_list), len(thresholds)), np.nan)
        max_steps_any = 1

        for i, name in enumerate(names_list):
            logger = loggers[name]
            if not logger.eval_rewards:
                continue
            steps = np.array(logger.eval_steps, dtype=float)
            rewards_arr = np.array(logger.eval_rewards, dtype=float)
            max_steps_any = max(max_steps_any, float(steps[-1]) if len(steps) > 0 else 1)
            # 计算平滑奖励
            smooth_r = smooth(rewards_arr, window=min(3, len(rewards_arr)))
            for j, pct in enumerate(thresholds):
                threshold_val = min_reward + pct * reward_range
                # 找到第一次超过阈值的步数
                crossed = np.where(smooth_r >= threshold_val)[0]
                if len(crossed) > 0:
                    heat_data[i, j] = steps[crossed[0]]
                else:
                    heat_data[i, j] = float(steps[-1]) if len(steps) > 0 else np.nan

        # 归一化到 [0, 1]（步数越少越好 → 颜色越深）
        heat_norm = heat_data / max_steps_any
        im = ax_heat.imshow(heat_norm.T, aspect="auto", cmap="RdYlGn_r",
                             vmin=0, vmax=1)
        ax_heat.set_xticks(range(len(names_list)))
        ax_heat.set_xticklabels(
            [AGENT_LABELS.get(n, n).replace("★ ", "").replace(" (Novel)", "")[:18]
             for n in names_list],
            rotation=25, ha="right", fontsize=8,
        )
        ax_heat.set_yticks(range(len(thresholds)))
        ax_heat.set_yticklabels([f"{int(p*100)}% of max" for p in thresholds], fontsize=8)
        ax_heat.set_title("Steps to Reach Threshold\n(Green=Faster, Red=Slower)", fontweight="bold")
        plt.colorbar(im, ax=ax_heat, shrink=0.8, label="Relative Steps (lower=faster)")

        # 在格子内显示步数
        for i in range(len(names_list)):
            for j in range(len(thresholds)):
                val = heat_data[i, j]
                if not np.isnan(val):
                    txt = f"{int(val/1000)}k"
                    ax_heat.text(i, j, txt, ha="center", va="center",
                                 fontsize=7, color="black", fontweight="bold")

    # ── 后20%奖励分布箱线图 ──
    box_data = []
    box_labels = []
    box_colors = []
    for name in sorted_names:
        logger = loggers[name]
        if not logger.eval_rewards:
            continue
        rewards_arr = np.array(logger.eval_rewards)
        last_n = max(1, int(len(rewards_arr) * 0.20))
        tail_rewards = rewards_arr[-last_n:]
        box_data.append(tail_rewards)
        box_labels.append(AGENT_LABELS.get(name, name).replace("★ ", "").replace(" (Novel)", "")[:18])
        box_colors.append(AGENT_COLORS.get(name, "#607D8B"))

    if box_data:
        bp = ax_box.boxplot(box_data, patch_artist=True, notch=False,
                             medianprops=dict(color="black", linewidth=2),
                             whiskerprops=dict(linewidth=1.2),
                             capprops=dict(linewidth=1.2))
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
        ax_box.set_xticks(range(1, len(box_labels) + 1))
        ax_box.set_xticklabels(box_labels, rotation=30, ha="right", fontsize=7)
        ax_box.set_ylabel("Reward (last 20%)", fontweight="bold")
        ax_box.set_title("Final Performance Distribution\n(last 20% of training)", fontweight="bold")

    save_path = os.path.join(save_dir, f"analysis_{env_name.replace('/', '_')}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n📊 综合分析图已保存至: {save_path}")
    if show:
        plt.show()
    plt.close(fig)
    return save_path

