"""
综合可视化分析脚本（等待实验完成后运行）
适用于 ADVANCE-PPO 消融实验和多环境对比实验
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

ADVANCE_DIR = "results/Advance-Ablation"
MULTI_ENV_DIR = "results/MultiEnv"
SAVE_DIR = "results/Final-Analysis"
os.makedirs(SAVE_DIR, exist_ok=True)

ENVS = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]
ADVANCE_VARIANTS = [
    "ADVANCE_Base", "ADVANCE_ImpA", "ADVANCE_ImpB", "ADVANCE_ImpC",
    "ADVANCE_ImpAB", "ADVANCE_ImpAC", "ADVANCE_ImpBC", "ADVANCE_Full"
]
MULTI_ALGORITHMS = ["Standard_PPO", "GAE_Lambda1", "HCGAE_Base", "HCGAE_Imp12"]

# 颜色方案
ADVANCE_COLORS = {
    "ADVANCE_Base":  "#7f8c8d",  # 灰色
    "ADVANCE_ImpA":  "#3498db",  # 蓝
    "ADVANCE_ImpB":  "#2ecc71",  # 绿
    "ADVANCE_ImpC":  "#e67e22",  # 橙
    "ADVANCE_ImpAB": "#9b59b6",  # 紫
    "ADVANCE_ImpAC": "#1abc9c",  # 青
    "ADVANCE_ImpBC": "#e74c3c",  # 红
    "ADVANCE_Full":  "#f39c12",  # 金色
}
ALG_COLORS = {
    "Standard_PPO": "#95a5a6",
    "GAE_Lambda1":  "#3498db",
    "HCGAE_Base":   "#e67e22",
    "HCGAE_Imp12":  "#2ecc71",
}
ALG_LABELS = {
    "Standard_PPO": "PPO (baseline)",
    "GAE_Lambda1":  r"GAE($\lambda$=1)",
    "HCGAE_Base":   "HCGAE-Base",
    "HCGAE_Imp12":  "HCGAE (ours)",
}
ADVANCE_LABELS = {
    "ADVANCE_Base":  "ADVANCE-Base",
    "ADVANCE_ImpA":  "+ Adaptive ε (A)",
    "ADVANCE_ImpB":  "+ Epoch-Decay IS (B)",
    "ADVANCE_ImpC":  "+ EV-Gated Critic (C)",
    "ADVANCE_ImpAB": "+ A + B",
    "ADVANCE_ImpAC": "+ A + C",
    "ADVANCE_ImpBC": "+ B + C",
    "ADVANCE_Full":  "ADVANCE-Full (A+B+C)",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def load_advance_summary(env_id):
    path = os.path.join(ADVANCE_DIR, env_id, "summary.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def load_multi_summary(env_id):
    path = os.path.join(MULTI_ENV_DIR, env_id, "summary.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def load_learning_curves(base_dir, env_id, variant, seeds):
    """加载多个 seed 的学习曲线"""
    all_curves = []
    all_steps = None
    for seed in seeds:
        path = os.path.join(base_dir, env_id, f"{variant}_s{seed}.json")
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        rewards = d.get("all_eval_rewards", [])
        steps = d.get("eval_steps", list(range(0, len(rewards) * 10240, 10240)))
        if rewards:
            all_curves.append(rewards)
            if all_steps is None:
                all_steps = steps
    return all_steps, all_curves


# ─────────────────────────────────────────────
# 图 1：多环境对比（5 seeds 均值 ± 标准差）
# ─────────────────────────────────────────────
def plot_multi_env_comparison():
    data_available = all(
        os.path.exists(os.path.join(MULTI_ENV_DIR, e, "summary.json"))
        for e in ENVS
    )
    if not data_available:
        print("  [skip] 多环境对比图：数据未完成")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, env_id in zip(axes, ENVS):
        summary = load_multi_summary(env_id)
        if summary is None:
            continue

        algs = list(ALG_COLORS.keys())
        means = [summary.get(a, {}).get("mean", 0) for a in algs]
        stds  = [summary.get(a, {}).get("std", 0) for a in algs]
        colors = [ALG_COLORS[a] for a in algs]
        labels = [ALG_LABELS[a] for a in algs]

        x = np.arange(len(algs))
        bars = ax.bar(x, means, yerr=stds, color=colors, capsize=5, alpha=0.85, width=0.6)

        # 标注数值
        for bar, mean, std in zip(bars, means, stds):
            if mean > 0:
                ax.text(bar.get_x() + bar.get_width()/2, mean + std + 5,
                        f"{mean:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(f"{env_id}", fontweight="bold")
        ax.set_ylabel("Average Return" if env_id == ENVS[0] else "")

    fig.suptitle("Multi-Environment Comparison (5 seeds, 300k steps)", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "fig1_multi_env_comparison.png"), bbox_inches="tight")
    plt.close()
    print(f"  [ok] 图1 已保存: fig1_multi_env_comparison.png")


# ─────────────────────────────────────────────
# 图 2：ADVANCE-PPO 消融柱状图
# ─────────────────────────────────────────────
def plot_advance_ablation_bars():
    data_available = all(
        os.path.exists(os.path.join(ADVANCE_DIR, e, "summary.json"))
        for e in ENVS
    )
    if not data_available:
        print("  [skip] ADVANCE 消融柱状图：数据未完成")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, env_id in zip(axes, ENVS):
        summary = load_advance_summary(env_id)
        if summary is None:
            continue

        variants = [v for v in ADVANCE_VARIANTS if v in summary]
        means = [summary[v]["mean"] for v in variants]
        stds  = [summary[v]["std"] for v in variants]
        colors = [ADVANCE_COLORS[v] for v in variants]

        x = np.arange(len(variants))
        bars = ax.bar(x, means, yerr=stds, color=colors, capsize=4, alpha=0.85, width=0.65)

        # 标注相对增益
        base_mean = summary.get("ADVANCE_Base", {}).get("mean", 1.0)
        for bar, mean, v in zip(bars, means, variants):
            gain = (mean / (base_mean + 1e-8) - 1) * 100
            color = "green" if gain > 2 else ("red" if gain < -2 else "gray")
            ax.text(bar.get_x() + bar.get_width()/2, mean + 5,
                    f"{gain:+.0f}%", ha="center", va="bottom", fontsize=7, color=color, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([ADVANCE_LABELS.get(v, v) for v in variants], rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{env_id}", fontweight="bold")
        ax.set_ylabel("Average Return" if env_id == ENVS[0] else "")

    fig.suptitle("ADVANCE-PPO Ablation Study (3 seeds)", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "fig2_advance_ablation.png"), bbox_inches="tight")
    plt.close()
    print(f"  [ok] 图2 已保存: fig2_advance_ablation.png")


# ─────────────────────────────────────────────
# 图 3：学习曲线对比（均值 ± 标准差区间）
# ─────────────────────────────────────────────
def plot_learning_curves_comparison():
    env_id = "Hopper-v4"
    multi_seeds = [42, 123, 456, 789, 1234]
    advance_seeds = [42, 123, 456]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：标准对比（Standard_PPO vs GAE_Lambda1 vs HCGAE_Base vs HCGAE_Imp12）
    ax = axes[0]
    ax.set_title("HCGAE vs Baselines (Hopper-v4)", fontweight="bold")
    for alg in MULTI_ALGORITHMS:
        steps, curves = load_learning_curves(MULTI_ENV_DIR, env_id, alg, multi_seeds)
        if not curves:
            continue
        min_len = min(len(c) for c in curves)
        mat = np.array([c[:min_len] for c in curves])
        mean = mat.mean(axis=0)
        sem  = mat.std(axis=0) / np.sqrt(len(curves))
        if steps is not None:
            x = np.array(steps[:min_len]) / 1e3
        else:
            x = np.arange(min_len) * 10.24
        ax.plot(x, mean, color=ALG_COLORS[alg], label=ALG_LABELS[alg], linewidth=2)
        ax.fill_between(x, mean - sem, mean + sem, color=ALG_COLORS[alg], alpha=0.2)
    ax.set_xlabel("Timesteps (k)")
    ax.set_ylabel("Evaluation Return")
    ax.legend(loc="upper left")

    # 右图：ADVANCE-PPO 消融学习曲线（仅关键变体）
    ax = axes[1]
    ax.set_title("ADVANCE-PPO Ablation Curves (Hopper-v4)", fontweight="bold")
    key_variants = ["ADVANCE_Base", "ADVANCE_ImpA", "ADVANCE_ImpB", "ADVANCE_ImpC", "ADVANCE_Full"]
    for v in key_variants:
        steps, curves = load_learning_curves(ADVANCE_DIR, env_id, v, advance_seeds)
        if not curves:
            continue
        min_len = min(len(c) for c in curves)
        mat = np.array([c[:min_len] for c in curves])
        mean = mat.mean(axis=0)
        sem  = mat.std(axis=0) / np.sqrt(len(curves))
        if steps is not None:
            x = np.array(steps[:min_len]) / 1e3
        else:
            x = np.arange(min_len) * 10.24
        lw = 2.5 if v == "ADVANCE_Full" else 1.5
        ls = "-" if v in ["ADVANCE_Base", "ADVANCE_Full"] else "--"
        ax.plot(x, mean, color=ADVANCE_COLORS[v], label=ADVANCE_LABELS.get(v, v), linewidth=lw, linestyle=ls)
        ax.fill_between(x, mean - sem, mean + sem, color=ADVANCE_COLORS[v], alpha=0.15)
    ax.set_xlabel("Timesteps (k)")
    ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "fig3_learning_curves.png"), bbox_inches="tight")
    plt.close()
    print(f"  [ok] 图3 已保存: fig3_learning_curves.png")


# ─────────────────────────────────────────────
# 图 4：热力图（环境 × 算法性能矩阵）
# ─────────────────────────────────────────────
def plot_heatmap():
    # ADVANCE 消融热力图
    all_summaries = {}
    for env_id in ENVS:
        s = load_advance_summary(env_id)
        if s:
            all_summaries[env_id] = s

    if not all_summaries:
        print("  [skip] 热力图：数据未完成")
        return

    variants = ADVANCE_VARIANTS
    n_variants = len(variants)
    n_envs = len(ENVS)

    matrix = np.zeros((n_envs, n_variants))
    for i, env_id in enumerate(ENVS):
        s = all_summaries.get(env_id, {})
        base_mean = s.get("ADVANCE_Base", {}).get("mean", 1.0)
        for j, v in enumerate(variants):
            mean = s.get(v, {}).get("mean", 0)
            matrix[i, j] = (mean / (base_mean + 1e-8) - 1) * 100

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-20, vmax=30)

    ax.set_xticks(range(n_variants))
    ax.set_xticklabels([ADVANCE_LABELS.get(v, v) for v in variants], rotation=25, ha="right")
    ax.set_yticks(range(n_envs))
    ax.set_yticklabels(ENVS)

    for i in range(n_envs):
        for j in range(n_variants):
            val = matrix[i, j]
            color = "white" if abs(val) > 15 else "black"
            ax.text(j, i, f"{val:+.1f}%", ha="center", va="center", fontsize=9, color=color)

    plt.colorbar(im, ax=ax, label="Relative improvement over ADVANCE_Base (%)")
    ax.set_title("ADVANCE-PPO Ablation Heatmap (% gain over base)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "fig4_heatmap.png"), bbox_inches="tight")
    plt.close()
    print(f"  [ok] 图4 已保存: fig4_heatmap.png")


# ─────────────────────────────────────────────
# 图 5：综合对比（统计显著性检验）
# ─────────────────────────────────────────────
def plot_statistical_significance():
    env_id = "Hopper-v4"
    s = load_advance_summary(env_id)
    if s is None:
        print("  [skip] 显著性检验图：数据未完成")
        return

    methods = [v for v in ADVANCE_VARIANTS if v in s]
    base_seeds = s.get("ADVANCE_Base", {}).get("seeds", [])
    full_seeds = s.get("ADVANCE_Full", {}).get("seeds", [])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：各变体 vs Base 的 t-test 结果
    ax = axes[0]
    p_values = []
    gains = []
    for v in methods:
        v_seeds = s.get(v, {}).get("seeds", [])
        if len(v_seeds) < 2 or len(base_seeds) < 2:
            p_values.append(1.0)
            gains.append(0.0)
            continue
        try:
            t_stat, p_val = stats.ttest_ind(v_seeds, base_seeds, alternative="greater")
            p_values.append(min(float(p_val), 1.0))
        except Exception:
            p_values.append(1.0)
        base_m = np.mean(base_seeds)
        gains.append((np.mean(v_seeds) / (base_m + 1e-8) - 1) * 100)

    colors = ["#2ecc71" if p < 0.05 else "#e74c3c" for p in p_values]
    x = np.arange(len(methods))
    bars = ax.bar(x, gains, color=colors, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([ADVANCE_LABELS.get(v, v) for v in methods], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Relative gain over ADVANCE_Base (%)")
    ax.set_title(f"Statistical Significance (Hopper-v4)\n(Green = p<0.05, Red = not sig.)", fontweight="bold")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#2ecc71", label="p < 0.05"),
        Patch(color="#e74c3c", label="p ≥ 0.05")
    ])

    # 右图：ADVANCE_Full 学习曲线分布
    ax2 = axes[1]
    advance_seeds_ids = [42, 123, 456]
    steps, curves = load_learning_curves(ADVANCE_DIR, env_id, "ADVANCE_Full", advance_seeds_ids)
    base_steps, base_curves = load_learning_curves(ADVANCE_DIR, env_id, "ADVANCE_Base", advance_seeds_ids)

    if curves and base_curves:
        min_len = min(min(len(c) for c in curves), min(len(c) for c in base_curves))
        c_mat = np.array([c[:min_len] for c in curves])
        b_mat = np.array([c[:min_len] for c in base_curves])
        x = np.arange(min_len) * 10.24

        c_mean = c_mat.mean(0); c_std = c_mat.std(0)
        b_mean = b_mat.mean(0); b_std = b_mat.std(0)

        ax2.plot(x, c_mean, color=ADVANCE_COLORS["ADVANCE_Full"], label="ADVANCE_Full", lw=2)
        ax2.fill_between(x, c_mean - c_std, c_mean + c_std, color=ADVANCE_COLORS["ADVANCE_Full"], alpha=0.2)
        ax2.plot(x, b_mean, color=ADVANCE_COLORS["ADVANCE_Base"], label="ADVANCE_Base", lw=2, linestyle="--")
        ax2.fill_between(x, b_mean - b_std, b_mean + b_std, color=ADVANCE_COLORS["ADVANCE_Base"], alpha=0.2)
        ax2.set_xlabel("Timesteps (k)")
        ax2.set_ylabel("Evaluation Return")
        ax2.set_title("ADVANCE_Full vs ADVANCE_Base (mean ± std)", fontweight="bold")
        ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "fig5_significance.png"), bbox_inches="tight")
    plt.close()
    print(f"  [ok] 图5 已保存: fig5_significance.png")


def main():
    print(f"\n{'='*60}")
    print(f"  综合可视化分析")
    print(f"{'='*60}")

    plot_multi_env_comparison()
    plot_advance_ablation_bars()
    plot_learning_curves_comparison()
    plot_heatmap()
    plot_statistical_significance()

    print(f"\n  所有图表已保存至: {SAVE_DIR}/")


if __name__ == "__main__":
    main()

