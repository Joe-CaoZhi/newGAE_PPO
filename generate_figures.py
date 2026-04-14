#!/usr/bin/env python3
"""
HCGAE Paper Figure Generator
==============================
生成所有论文图表，使用最新实验数据（FinalOptimal + GRPO + Ablation）。

生成的图表（results/paper_figures/）:
  fig1_learning_curves.png  — PPO/GRPO 学习曲线 (4环境 × 3算法)
  fig3_summary_bars.png     — 最终性能柱状图 (PPO vs HCGAE-PPO vs GRPO vs HCGAE-GRPO)
  figA8_grpo_advantage.png  — GRPO 提升分析（HCGAE vs Standard GRPO，4环境）

用法:
  python3 generate_figures.py          # 生成所有图
  python3 generate_figures.py --fig 1  # 只生成 fig1
  python3 generate_figures.py --fig 3
  python3 generate_figures.py --fig A8
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.ndimage import uniform_filter1d

# ─────────────────────────────────────────────────────────────────────────────
# 全局配置
# ─────────────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent
OUT     = ROOT / "results" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

ENVS    = ["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4", "Ant-v4"]
ENV_LABELS = {
    "HalfCheetah-v4": "HalfCheetah-v4",
    "Hopper-v4":      "Hopper-v4",
    "Walker2d-v4":    "Walker2d-v4",
    "Ant-v4":         "Ant-v4",
}

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        10,
    'axes.titlesize':   11,
    'axes.labelsize':   10,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'legend.fontsize':  8.5,
    'figure.dpi':       150,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.25,
    'grid.linestyle':   '--',
    'lines.linewidth':  1.8,
})

# ─────────────────────────────────────────────────────────────────────────────
# 颜色 / 样式
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    'Optimal_PPO':          '#4472C4',   # 蓝
    'Optimal_HCGAE_Optimal':'#E74C3C',   # 红
    'Optimal_GRPO':         '#2ECC71',   # 绿
    'HCGAE_Optimal_GRPO':   '#E67E22',   # 橙
    # Ablation（暂留）
    'Optimal_HCGAE_NoFixSCR':    '#9B59B6',
    'Optimal_HCGAE_NoMCSigmoid': '#1ABC9C',
    'Optimal_HCGAE_NoEVGate':    '#F39C12',
}

LABELS = {
    'Optimal_PPO':           'Standard PPO',
    'Optimal_HCGAE_Optimal': 'HCGAE-PPO (Ours)',
    'Optimal_GRPO':          'Standard GRPO',
    'HCGAE_Optimal_GRPO':    'HCGAE-GRPO (Ours)',
    'Optimal_HCGAE_NoFixSCR':    r'HCGAE-PPO $-$FixSCR',
    'Optimal_HCGAE_NoMCSigmoid': r'HCGAE-PPO $-$MCSigmoid',
    'Optimal_HCGAE_NoEVGate':    r'HCGAE-PPO $-$EVGate',
}

LINESTYLES = {
    'Optimal_PPO':           (0, (5, 2)),
    'Optimal_HCGAE_Optimal': 'solid',
    'Optimal_GRPO':          (0, (3, 1, 1, 1)),
    'HCGAE_Optimal_GRPO':    'solid',
}

# ─────────────────────────────────────────────────────────────────────────────
# 数据加载工具
# ─────────────────────────────────────────────────────────────────────────────
def load_seed_data(base_dir: Path, env: str, algo: str, min_steps: int = 0):
    """加载某环境某算法的全部 seed 数据，返回 list[dict]"""
    d = base_dir / env / algo
    if not d.exists():
        return []
    result = []
    for fp in sorted(d.glob("*.json")):
        try:
            data = json.load(open(fp))
            if data.get('total_steps', 0) >= min_steps:
                result.append(data)
        except Exception:
            pass
    return result


def get_curves(seeds_data, smooth_w=5, n_common=60):
    """
    从 seed 数据提取学习曲线，插值到公共步数网格。
    返回 (steps_M, mean, sem)，步数单位为 M（百万）。
    """
    all_steps = []
    all_vals  = []
    for d in seeds_data:
        steps = d.get('eval_steps', [])
        vals  = d.get('eval_rewards', [])
        if steps and vals and len(steps) == len(vals):
            all_steps.append(np.array(steps, dtype=float))
            all_vals.append(np.array(vals,  dtype=float))
    if not all_steps:
        return None, None, None

    max_step = min(s[-1] for s in all_steps)   # 取所有 seed 共有的步数范围
    min_step = max(s[0]  for s in all_steps)
    grid = np.linspace(min_step, max_step, n_common)

    interped = []
    for s, v in zip(all_steps, all_vals):
        interped.append(np.interp(grid, s, v))

    mat  = np.array(interped)               # (n_seeds, n_points)
    mean = np.mean(mat, axis=0)
    sem  = np.std(mat, axis=0) / np.sqrt(len(mat))

    if smooth_w > 1:
        mean = uniform_filter1d(mean, size=smooth_w, mode='nearest')
        sem  = uniform_filter1d(sem,  size=smooth_w, mode='nearest')

    return grid / 1e6, mean, sem


def get_final_score(seeds_data, last_n=10):
    """返回 (mean, std) 最终 eval reward"""
    scores = []
    for d in seeds_data:
        ev = d.get('eval_rewards', [])
        if ev:
            scores.append(float(np.mean(ev[-last_n:])))
    if not scores:
        return None, None
    return float(np.mean(scores)), float(np.std(scores))


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — 学习曲线（4 环境 × 4 算法）
# ─────────────────────────────────────────────────────────────────────────────
def make_fig1():
    print("  [Fig 1] Learning curves ...")
    PPO_DIR  = ROOT / "results" / "FinalOptimal"
    GRPO_DIR = ROOT / "results" / "GRPO"

    # 4 算法
    algo_configs = [
        ('Optimal_PPO',           PPO_DIR,  700_000),
        ('Optimal_HCGAE_Optimal', PPO_DIR,  700_000),
        ('Optimal_GRPO',          GRPO_DIR, 1_000_000),
        ('HCGAE_Optimal_GRPO',    GRPO_DIR, 1_000_000),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    for ax_idx, env in enumerate(ENVS):
        ax = axes[ax_idx]
        for algo, base_dir, min_steps in algo_configs:
            seeds_data = load_seed_data(base_dir, env, algo, min_steps=min_steps)
            if not seeds_data:
                continue
            steps_M, mean, sem = get_curves(seeds_data, smooth_w=5)
            if steps_M is None:
                continue
            c  = COLORS[algo]
            ls = LINESTYLES.get(algo, 'solid')
            lb = LABELS[algo]
            n  = len(seeds_data)
            ax.plot(steps_M, mean, color=c, linestyle=ls, label=f'{lb} (n={n})', linewidth=1.8)
            ax.fill_between(steps_M, mean - sem, mean + sem, alpha=0.15, color=c)

        ax.set_title(ENV_LABELS[env], fontweight='bold', pad=4)
        ax.set_xlabel("Environment Steps (M)")
        ax.set_ylabel("Episode Return")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))

    # 共享图例放在底部
    handles, labels_ = axes[0].get_legend_handles_labels()
    # 收集全部图例
    all_h, all_l = [], []
    seen = set()
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            key = li.split(' (n=')[0]
            if key not in seen:
                all_h.append(hi); all_l.append(li); seen.add(key)

    fig.legend(all_h, all_l, loc='lower center', ncol=2,
               frameon=True, fancybox=False, edgecolor='#ccc',
               bbox_to_anchor=(0.5, 0.0), fontsize=9)
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    out_path = str(OUT / "fig1_learning_curves.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    ✓ fig1_learning_curves.png")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — 性能对比柱状图（PPO vs HCGAE-PPO vs GRPO vs HCGAE-GRPO）
# ─────────────────────────────────────────────────────────────────────────────
def make_fig3():
    print("  [Fig 3] Summary bars ...")
    PPO_DIR  = ROOT / "results" / "FinalOptimal"
    GRPO_DIR = ROOT / "results" / "GRPO"

    algo_configs = [
        ('Optimal_PPO',           PPO_DIR,  700_000),
        ('Optimal_HCGAE_Optimal', PPO_DIR,  700_000),
        ('Optimal_GRPO',          GRPO_DIR, 1_000_000),
        ('HCGAE_Optimal_GRPO',    GRPO_DIR, 1_000_000),
    ]

    # 收集数据
    data = {}
    for algo, base_dir, min_steps in algo_configs:
        data[algo] = {}
        for env in ENVS:
            seeds_data = load_seed_data(base_dir, env, algo, min_steps=min_steps)
            m, s = get_final_score(seeds_data, last_n=10)
            data[algo][env] = (m, s, len(seeds_data))

    fig, axes = plt.subplots(1, 4, figsize=(14, 5))

    bar_algos = ['Optimal_PPO', 'Optimal_HCGAE_Optimal', 'Optimal_GRPO', 'HCGAE_Optimal_GRPO']
    x = np.arange(len(bar_algos))
    width = 0.55

    for ax_idx, env in enumerate(ENVS):
        ax = axes[ax_idx]
        means = []
        stds  = []
        ns    = []
        for algo in bar_algos:
            m, s, n = data[algo].get(env, (None, None, 0))
            means.append(m if m is not None else 0)
            stds.append(s  if s  is not None else 0)
            ns.append(n)

        colors = [COLORS[a] for a in bar_algos]
        bars = ax.bar(x, means, width=width, color=colors, alpha=0.85,
                      edgecolor='white', linewidth=0.8, zorder=3)

        # 误差线
        ax.errorbar(x, means, yerr=stds, fmt='none', color='black',
                    capsize=4, capthick=1.2, linewidth=1.2, zorder=4)

        # 标注 n 数
        for i, (bar, n) in enumerate(zip(bars, ns)):
            if means[i] > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[i] * 0.1,
                        f'n={n}', ha='center', va='bottom', fontsize=7.5, color='#555')

        # 标注 HCGAE 提升百分比（vs PPO baseline 和 GRPO baseline）
        ppo_m   = data['Optimal_PPO'].get(env, (None, None, 0))[0]
        hcgae_m = data['Optimal_HCGAE_Optimal'].get(env, (None, None, 0))[0]
        grpo_m  = data['Optimal_GRPO'].get(env, (None, None, 0))[0]
        hcgae_grpo_m = data['HCGAE_Optimal_GRPO'].get(env, (None, None, 0))[0]

        if ppo_m and hcgae_m and ppo_m != 0:
            imp_ppo = (hcgae_m - ppo_m) / abs(ppo_m) * 100
            y_ppo   = max(means[0], means[1]) + max(stds[0], stds[1]) + abs(max(means)) * 0.04
            ax.annotate(f'{imp_ppo:+.0f}%',
                        xy=(0.5, y_ppo), xycoords=('axes fraction', 'data'),
                        ha='center', fontsize=8, color='#C0392B', fontweight='bold')

        if grpo_m and hcgae_grpo_m and grpo_m != 0:
            imp_grpo = (hcgae_grpo_m - grpo_m) / abs(grpo_m) * 100
            y_grpo   = max(means[2], means[3]) + max(stds[2], stds[3]) + abs(max(means)) * 0.04
            ax.annotate(f'{imp_grpo:+.0f}%',
                        xy=(0.5, y_grpo), xycoords=('axes fraction', 'data'),
                        ha='center', fontsize=8, color='#E67E22', fontweight='bold')

        ax.set_title(ENV_LABELS[env], fontweight='bold', pad=4)
        ax.set_xticks(x)
        short_labels = ['PPO', 'HCGAE-PPO', 'GRPO', 'HCGAE-GRPO']
        ax.set_xticklabels(short_labels, rotation=25, ha='right', fontsize=8.5)
        ax.set_ylabel("Final Episode Return")
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        # 去掉 x 轴网格
        ax.xaxis.grid(False)

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS[a], label=LABELS[a]) for a in bar_algos]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               frameon=True, fancybox=False, edgecolor='#ccc',
               bbox_to_anchor=(0.5, -0.02), fontsize=9)

    fig.suptitle("Final Performance: PPO vs HCGAE-PPO vs GRPO vs HCGAE-GRPO",
                 fontsize=12, fontweight='bold', y=1.01)
    fig.tight_layout(rect=[0, 0.09, 1, 1])

    out_path = str(OUT / "fig3_summary_bars.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    ✓ fig3_summary_bars.png")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Fig A8 — GRPO 提升分析（4 环境学习曲线对比 + 提升百分比）
# ─────────────────────────────────────────────────────────────────────────────
def make_figA8():
    print("  [FigA8] GRPO Advantage analysis ...")
    GRPO_DIR = ROOT / "results" / "GRPO"

    grpo_algos = [
        ('Optimal_GRPO',       1_000_000),
        ('HCGAE_Optimal_GRPO', 1_000_000),
    ]
    grpo_colors = {
        'Optimal_GRPO':       '#4472C4',
        'HCGAE_Optimal_GRPO': '#E74C3C',
    }
    grpo_labels = {
        'Optimal_GRPO':       'Standard GRPO',
        'HCGAE_Optimal_GRPO': 'HCGAE-GRPO (Ours)',
    }
    grpo_ls = {
        'Optimal_GRPO':       (0, (4, 2)),
        'HCGAE_Optimal_GRPO': 'solid',
    }

    fig = plt.figure(figsize=(14, 9))
    outer = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.32)

    for env_idx, env in enumerate(ENVS):
        row, col = divmod(env_idx, 2)
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[row, col],
                                                  height_ratios=[3, 1.2], hspace=0.08)
        ax_main = fig.add_subplot(inner[0])
        ax_diff = fig.add_subplot(inner[1], sharex=ax_main)

        curves = {}
        for algo, min_steps in grpo_algos:
            seeds_data = load_seed_data(GRPO_DIR, env, algo, min_steps=min_steps)
            if not seeds_data:
                continue
            steps_M, mean, sem = get_curves(seeds_data, smooth_w=7, n_common=80)
            if steps_M is None:
                continue
            curves[algo] = (steps_M, mean, sem, len(seeds_data))
            c  = grpo_colors[algo]
            ls = grpo_ls[algo]
            n  = len(seeds_data)
            ax_main.plot(steps_M, mean, color=c, linestyle=ls,
                         label=f'{grpo_labels[algo]} (n={n})', linewidth=1.8)
            ax_main.fill_between(steps_M, mean - sem, mean + sem, alpha=0.15, color=c)

        ax_main.set_title(ENV_LABELS[env], fontweight='bold', pad=4)
        ax_main.set_ylabel("Episode Return")
        ax_main.legend(loc='upper left', fontsize=7.5)
        plt.setp(ax_main.get_xticklabels(), visible=False)

        # 差值图（HCGAE-GRPO 相对 Standard GRPO 的绝对提升）
        if 'Optimal_GRPO' in curves and 'HCGAE_Optimal_GRPO' in curves:
            s1, m1, se1, _ = curves['Optimal_GRPO']
            s2, m2, se2, _ = curves['HCGAE_Optimal_GRPO']
            # 对齐到相同 steps（取交集）
            m1_ali = np.interp(s2, s1, m1)
            diff   = m2 - m1_ali
            se_c   = np.sqrt(se1**2 + se2**2)
            se_ali = np.interp(s2, s1, se_c)

            ax_diff.axhline(0, color='#999', linewidth=0.8, linestyle='--')
            ax_diff.plot(s2, diff, color='#E67E22', linewidth=1.5, label='HCGAE - GRPO')
            ax_diff.fill_between(s2, diff - se_ali, diff + se_ali, alpha=0.2, color='#E67E22')

            # 标注最终提升百分比
            final_grpo  = float(np.mean(m1_ali[-10:]))
            final_hcgae = float(np.mean(m2[-10:]))
            if final_grpo != 0:
                pct = (final_hcgae - final_grpo) / abs(final_grpo) * 100
                ax_diff.text(0.97, 0.82, f'{pct:+.1f}%',
                             transform=ax_diff.transAxes, ha='right', va='top',
                             fontsize=9, fontweight='bold', color='#C0392B')

        ax_diff.set_xlabel("Environment Steps (M)")
        ax_diff.set_ylabel("Δ Return", fontsize=8)
        ax_diff.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        ax_diff.tick_params(labelsize=8)

    fig.suptitle("HCGAE-GRPO vs Standard GRPO: Learning Curves & Absolute Improvement",
                 fontsize=12, fontweight='bold', y=1.01)

    out_path = str(OUT / "figA8_grpo_advantage.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    ✓ figA8_grpo_advantage.png")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fig', nargs='+', default=['1', '3', 'A8'],
                        help='要生成的图：1  3  A8  (默认全部)')
    args = parser.parse_args()
    figs = set(args.fig)

    print(f"\n  Output: {OUT}")
    print(f"  Figures: {', '.join(sorted(figs))}\n")

    if '1'  in figs: make_fig1()
    if '3'  in figs: make_fig3()
    if 'A8' in figs: make_figA8()

    print("\n  All done.")

