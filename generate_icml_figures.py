"""
Generate ICML-quality Figures
==============================

Generates all figures for the HCGAE/DCPPO-S paper submission.
Reads from: results/BaselineComparison/**/*_metrics.json

Figures:
  Fig 1: Learning curves (Hopper + Walker2d, 5-seed shaded)
  Fig 2: Final performance bar chart (all 3 envs, SEM bars)
  Fig 3: Ablation study (Hopper, bar + p-values)
  Fig 4: HCGAE diagnostic plots (EV trajectory, alpha trajectory, c_mc)
  Fig 5: Statistical significance table (heatmap)

Style: ICML 2025 compatible — single-column or double-column, 300 DPI PDF+PNG
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASELINE_DIR = Path("results/BaselineComparison")
OUT_DIR = Path("results/paper_figures_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456, 789, 1234]
ENVS  = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]

# ─────────────────────────────────────────────────────────────────────────────
# ICML Style
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "Standard_PPO":      "#4C72B0",  # blue
    "PPO_KLPEN":         "#55A868",  # green
    "PPO_Anneal":        "#C44E52",  # red
    "PPO_EntDecay":      "#8172B2",  # purple
    "PPO_VClip":         "#CCB974",  # gold
    "PPO_Full_Baseline": "#64B5CD",  # light blue
    "HCGAE_Imp12":       "#DD4444",  # vivid red (ours)
    "DCPPO_Full":        "#2CA02C",  # dark green
    "DCPPO_ImpS":        "#17BECF",  # cyan
    "DCPPO_Base":        "#9467BD",  # purple
}

LABELS = {
    "Standard_PPO":      "PPO",
    "PPO_KLPEN":         "PPO-KLPEN",
    "PPO_Anneal":        "PPO-Anneal",
    "PPO_EntDecay":      "PPO-EntDecay",
    "PPO_VClip":         "PPO-VClip",
    "PPO_Full_Baseline": "PPO-Full",
    "HCGAE_Imp12":       "HCGAE (Ours)",
    "DCPPO_Full":        "DCPPO (Ours)",
    "DCPPO_ImpS":        "DCPPO-S",
    "DCPPO_Base":        "DCPPO-Base",
}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.labelsize":    10,
    "axes.titlesize":    10,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "lines.linewidth":   1.5,
})


# ─────────────────────────────────────────────────────────────────────────────
# Data loading utilities
# ─────────────────────────────────────────────────────────────────────────────
def load_metrics(env_name, algo_name, seed):
    p = BASELINE_DIR / env_name / algo_name / f"{algo_name}_s{seed}_metrics.json"
    if not p.exists():
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def load_dcppo_metrics(env_name, algo_name, seed):
    p = Path("results/MultiEnv_DCPPO") / env_name / algo_name / f"{algo_name}_s{seed}_metrics.json"
    if not p.exists():
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def get_eval_curve(env_name, algo_name, seed, source="baseline"):
    """Returns (eval_steps, eval_rewards) or (None, None)."""
    if source == "dcppo":
        d = load_dcppo_metrics(env_name, algo_name, seed)
    else:
        d = load_metrics(env_name, algo_name, seed)
    if d is None:
        return None, None
    evr = d.get("eval_rewards", [])
    evs = d.get("eval_steps", list(range(0, len(evr) * 10_240, 10_240)))
    if not evr:
        return None, None
    return np.array(evs), np.array(evr, dtype=float)


def get_final_rewards(env_name, algo_name, n_tail=5):
    """Returns list of final rewards (last n_tail evals) per seed."""
    rewards = []
    for seed in SEEDS:
        d = load_metrics(env_name, algo_name, seed)
        if d is None:
            continue
        evr = d.get("eval_rewards", [])
        if len(evr) >= n_tail:
            rewards.append(float(np.mean(evr[-n_tail:])))
        elif evr:
            rewards.append(float(np.mean(evr)))
    return rewards


def interpolate_to_common_steps(curves, target_n=30):
    """Interpolate all curves to same step grid."""
    if not curves:
        return None, None
    all_steps = [s for s, _ in curves if s is not None]
    if not all_steps:
        return None, None
    # Common x-axis: from min_start to max_end
    max_len = max(len(s) for s in all_steps)
    # Use the median max step
    all_maxsteps = [s[-1] for s in all_steps]
    common_end = int(np.median(all_maxsteps))
    x_common = np.linspace(0, common_end, target_n)
    interp_y = []
    for steps, rewards in curves:
        if steps is None or rewards is None or len(steps) < 2:
            continue
        y_interp = np.interp(x_common, steps, rewards, left=rewards[0], right=rewards[-1])
        interp_y.append(y_interp)
    if not interp_y:
        return None, None
    return x_common, np.array(interp_y)


def sem(vals):
    if len(vals) <= 1:
        return 0.0
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def mann_whitney_test(a, b):
    """Returns (u_stat, p_value). Uses two-sided test."""
    if len(a) < 2 or len(b) < 2:
        return None, None
    try:
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        return float(u), float(p)
    except Exception:
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Learning Curves (Hopper + Walker2d)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig1_learning_curves():
    """5-seed shaded learning curves for Hopper-v4 and Walker2d-v4."""
    ENVS_SHOW = ["Hopper-v4", "Walker2d-v4"]
    ALGOS_SHOW = ["Standard_PPO", "PPO_Full_Baseline", "HCGAE_Imp12"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    fig.subplots_adjust(wspace=0.35)

    for ax, env_name in zip(axes, ENVS_SHOW):
        for algo in ALGOS_SHOW:
            curves = []
            for seed in SEEDS:
                steps, rewards = get_eval_curve(env_name, algo, seed)
                if steps is not None:
                    curves.append((steps, rewards))
            if not curves:
                continue

            x_common, y_matrix = interpolate_to_common_steps(curves, target_n=50)
            if x_common is None:
                continue

            y_mean = y_matrix.mean(axis=0)
            y_std  = y_matrix.std(axis=0)
            y_lo   = y_mean - y_std
            y_hi   = y_mean + y_std

            color = COLORS.get(algo, "#888888")
            lw    = 2.0 if algo == "HCGAE_Imp12" else 1.5
            zorder = 5 if algo == "HCGAE_Imp12" else 3

            ax.plot(x_common / 1e6, y_mean, color=color, lw=lw,
                    label=LABELS.get(algo, algo), zorder=zorder)
            ax.fill_between(x_common / 1e6, y_lo, y_hi,
                            color=color, alpha=0.15, zorder=zorder - 1)

        env_short = env_name.split("-")[0]
        ax.set_title(env_name, fontweight="bold")
        ax.set_xlabel("Environment Steps (×10⁶)")
        if ax == axes[0]:
            ax.set_ylabel("Mean Episode Reward")
        ax.legend(loc="upper left", framealpha=0.85)

    fig.suptitle("Learning Curves (mean ± 1 std, 5 seeds)",
                 fontweight="bold", y=1.01)
    _save(fig, "fig1_learning_curves")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Final Performance Bar Chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig2_bar_comparison():
    """Bar chart: final performance across environments, with SEM error bars."""
    ALGOS = [
        "Standard_PPO", "PPO_KLPEN", "PPO_Anneal",
        "PPO_EntDecay", "PPO_VClip", "PPO_Full_Baseline", "HCGAE_Imp12",
    ]

    # Determine which envs have data
    available_envs = []
    for env_name in ENVS:
        has_any = any(get_final_rewards(env_name, a) for a in ALGOS)
        if has_any:
            available_envs.append(env_name)

    n_envs = len(available_envs)
    if n_envs == 0:
        print("  [Fig 2] No data available")
        return None

    fig, axes = plt.subplots(1, n_envs, figsize=(4.5 * n_envs, 4.2))
    if n_envs == 1:
        axes = [axes]

    for ax, env_name in zip(axes, available_envs):
        means, sems, colors_list, xlabels = [], [], [], []
        for algo in ALGOS:
            vals = get_final_rewards(env_name, algo)
            if vals:
                means.append(float(np.mean(vals)))
                sems.append(sem(vals))
            else:
                means.append(0.0)
                sems.append(0.0)
            colors_list.append(COLORS.get(algo, "#888888"))
            xlabels.append(LABELS.get(algo, algo))

        x = np.arange(len(ALGOS))
        bars = ax.bar(x, means, yerr=sems, capsize=4,
                      color=colors_list, alpha=0.82,
                      edgecolor="none",
                      error_kw=dict(elinewidth=1.5, capthick=1.5, ecolor="gray"))

        # Highlight HCGAE
        hcgae_idx = ALGOS.index("HCGAE_Imp12")
        bars[hcgae_idx].set_edgecolor("#CC2222")
        bars[hcgae_idx].set_linewidth(2.0)
        bars[hcgae_idx].set_alpha(1.0)

        # Add n= annotation on bars
        for i, (bar, m, s) in enumerate(zip(bars, means, sems)):
            vals = get_final_rewards(env_name, ALGOS[i])
            n = len(vals)
            if m > 0:
                ax.text(bar.get_x() + bar.get_width()/2, m + s + max(means)*0.02,
                        f"n={n}", ha="center", va="bottom", fontsize=6.5, color="gray")

        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=38, ha="right", fontsize=7.5)
        ax.set_title(env_name, fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("Mean Episode Reward (5-seed, last 5 evals)")
        ax.set_xlim(-0.6, len(ALGOS) - 0.4)

    fig.suptitle("Final Performance Comparison: HCGAE vs PPO Baselines",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, "fig2_bar_comparison")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Statistical Significance (p-value heatmap)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig3_significance():
    """Heatmap of Mann-Whitney p-values: HCGAE vs each baseline."""
    from matplotlib.colors import LinearSegmentedColormap

    BASELINES = [
        "Standard_PPO", "PPO_KLPEN", "PPO_Anneal",
        "PPO_EntDecay", "PPO_VClip", "PPO_Full_Baseline",
    ]

    # Collect p-values and effect sizes
    available_envs = [e for e in ENVS
                      if get_final_rewards(e, "HCGAE_Imp12")]

    if not available_envs:
        print("  [Fig 3] No HCGAE data for significance test")
        return None

    p_matrix = np.ones((len(BASELINES), len(available_envs)))
    d_matrix = np.zeros((len(BASELINES), len(available_envs)))  # Cohen's d

    for j, env_name in enumerate(available_envs):
        hcgae_vals = get_final_rewards(env_name, "HCGAE_Imp12")
        for i, bl in enumerate(BASELINES):
            bl_vals = get_final_rewards(env_name, bl)
            if hcgae_vals and bl_vals:
                _, p = mann_whitney_test(hcgae_vals, bl_vals)
                if p is not None:
                    p_matrix[i, j] = p
                # Cohen's d (effect size)
                n1, n2 = len(hcgae_vals), len(bl_vals)
                if n1 >= 2 and n2 >= 2:
                    pooled_std = np.sqrt((np.std(hcgae_vals, ddof=1)**2 +
                                          np.std(bl_vals, ddof=1)**2) / 2)
                    if pooled_std > 0:
                        d_matrix[i, j] = (np.mean(hcgae_vals) - np.mean(bl_vals)) / pooled_std

    # Custom colormap: red (p<0.05) → yellow → white (p>0.5)
    cmap = LinearSegmentedColormap.from_list(
        "sig", ["#D32F2F", "#FF8C00", "#FFEB3B", "#FFFFFF"], N=256)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    # Left: p-value heatmap
    im1 = ax1.imshow(p_matrix, cmap=cmap, vmin=0, vmax=0.5, aspect="auto")
    ax1.set_xticks(range(len(available_envs)))
    ax1.set_xticklabels([e.split("-")[0] for e in available_envs], fontsize=9)
    ax1.set_yticks(range(len(BASELINES)))
    ax1.set_yticklabels([LABELS.get(b, b) for b in BASELINES], fontsize=8)
    ax1.set_title("HCGAE vs Baselines\nMann-Whitney p-value", fontweight="bold")

    for i in range(len(BASELINES)):
        for j in range(len(available_envs)):
            p = p_matrix[i, j]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            color = "white" if p < 0.15 else "black"
            ax1.text(j, i, f"{p:.3f}\n{sig}", ha="center", va="center",
                     fontsize=7, color=color, fontweight="bold")

    plt.colorbar(im1, ax=ax1, label="p-value", shrink=0.8)

    # Right: Cohen's d effect size
    cmap2 = LinearSegmentedColormap.from_list(
        "effect", ["#FFFFFF", "#FFF9C4", "#81C784", "#2E7D32"], N=256)
    im2 = ax2.imshow(d_matrix, cmap=cmap2, vmin=-0.5, vmax=2.0, aspect="auto")
    ax2.set_xticks(range(len(available_envs)))
    ax2.set_xticklabels([e.split("-")[0] for e in available_envs], fontsize=9)
    ax2.set_yticks(range(len(BASELINES)))
    ax2.set_yticklabels([LABELS.get(b, b) for b in BASELINES], fontsize=8)
    ax2.set_title("Cohen's d Effect Size\n(HCGAE − Baseline)", fontweight="bold")

    for i in range(len(BASELINES)):
        for j in range(len(available_envs)):
            d = d_matrix[i, j]
            interp = "large" if d > 0.8 else "med" if d > 0.5 else "small" if d > 0.2 else "neg"
            ax2.text(j, i, f"{d:.2f}\n({interp})", ha="center", va="center",
                     fontsize=7, color="black")

    plt.colorbar(im2, ax=ax2, label="Cohen's d", shrink=0.8)

    plt.tight_layout()
    _save(fig, "fig3_significance_heatmap")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: HCGAE Diagnostics (EV, alpha, c_mc)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig4_hcgae_diagnostics():
    """EV trajectory, alpha mean, c_mc across seeds and envs."""
    ENVS_SHOW = ["Hopper-v4", "Walker2d-v4"]

    fig, axes = plt.subplots(3, 2, figsize=(9, 8))
    metric_names = ["ev_ema_history", "alpha_mean_history", "c_mc_history"]
    ylabels = ["EV (Explained Variance)", "α (Correction Strength)", "c_mc (MC Weight)"]
    ylims = [(None, None), (0, 0.5), (0, 1.0)]

    for col, env_name in enumerate(ENVS_SHOW):
        for row, (metric, ylabel, ylim) in enumerate(zip(metric_names, ylabels, ylims)):
            ax = axes[row][col]
            all_curves = []
            for seed in SEEDS:
                d = load_metrics(env_name, "HCGAE_Imp12", seed)
                if d is None:
                    continue
                hist = d.get(metric, [])
                if hist:
                    all_curves.append(np.array(hist, dtype=float))

            if all_curves:
                min_len = min(len(c) for c in all_curves)
                mat = np.array([c[:min_len] for c in all_curves])
                x   = np.arange(min_len)
                y_mean = mat.mean(axis=0)
                y_std  = mat.std(axis=0)

                ax.plot(x, y_mean, color=COLORS["HCGAE_Imp12"], lw=1.8)
                ax.fill_between(x, y_mean - y_std, y_mean + y_std,
                                color=COLORS["HCGAE_Imp12"], alpha=0.2)
                ax.axhline(y=y_mean[-len(y_mean)//4:].mean(), color="gray",
                           lw=1, ls="--", alpha=0.6, label=f"late avg={y_mean[-len(y_mean)//4:].mean():.3f}")
                ax.legend(fontsize=7)

            ax.set_ylabel(ylabel if col == 0 else "", fontsize=8)
            if ylim[0] is not None:
                ax.set_ylim(*ylim)
            ax.set_xlabel("Update Steps" if row == 2 else "")
            if row == 0:
                ax.set_title(env_name, fontweight="bold")

    fig.suptitle("HCGAE Internal Diagnostics (5-seed mean ± 1 std)",
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "fig4_hcgae_diagnostics")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Performance Table (Latex-style heatmap)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig5_perf_table():
    """Publication-style performance table as heatmap."""
    ALGOS = [
        "Standard_PPO", "PPO_KLPEN", "PPO_Anneal",
        "PPO_EntDecay", "PPO_VClip", "PPO_Full_Baseline", "HCGAE_Imp12",
    ]

    available_envs = [e for e in ENVS
                      if any(get_final_rewards(e, a) for a in ALGOS)]
    if not available_envs:
        print("  [Fig 5] No data")
        return None

    # Build matrix
    mean_matrix = np.zeros((len(ALGOS), len(available_envs)))
    std_matrix  = np.zeros((len(ALGOS), len(available_envs)))
    n_matrix    = np.zeros((len(ALGOS), len(available_envs)), dtype=int)

    for i, algo in enumerate(ALGOS):
        for j, env_name in enumerate(available_envs):
            vals = get_final_rewards(env_name, algo)
            if vals:
                mean_matrix[i, j] = np.mean(vals)
                std_matrix[i, j]  = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                n_matrix[i, j]    = len(vals)

    # Normalize per column for color
    norm_matrix = mean_matrix.copy()
    for j in range(len(available_envs)):
        col = mean_matrix[:, j]
        col_max = col.max() if col.max() > 0 else 1.0
        col_min = col[col > 0].min() if (col > 0).any() else 0.0
        if col_max > col_min:
            norm_matrix[:, j] = (col - col_min) / (col_max - col_min)
        else:
            norm_matrix[:, j] = 0.5

    cmap = plt.cm.RdYlGn
    fig, ax = plt.subplots(figsize=(4 * len(available_envs), 0.55 * len(ALGOS) + 1.5))

    im = ax.imshow(norm_matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Annotate
    for i, algo in enumerate(ALGOS):
        for j, env_name in enumerate(available_envs):
            m = mean_matrix[i, j]
            s = std_matrix[i, j]
            n = n_matrix[i, j]
            if m > 0:
                is_best = (m == mean_matrix[:, j].max())
                weight = "bold" if is_best else "normal"
                color  = "black"
                # Bold HCGAE
                if algo == "HCGAE_Imp12":
                    weight = "bold"
                ax.text(j, i, f"{m:.0f}±{s:.0f}\n(n={n})",
                        ha="center", va="center", fontsize=8,
                        fontweight=weight, color=color)
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="gray")

    ax.set_xticks(range(len(available_envs)))
    ax.set_xticklabels(available_envs, fontsize=9)
    ax.set_yticks(range(len(ALGOS)))
    yticklabels = [LABELS.get(a, a) for a in ALGOS]
    yticklabels[-1] = f"★ {yticklabels[-1]}"  # Highlight ours
    ax.set_yticklabels(yticklabels, fontsize=9)

    # Color bar
    plt.colorbar(im, ax=ax, label="Normalized Performance (per env)", shrink=0.7, pad=0.02)

    ax.set_title(
        "Final Performance: Mean ± Std (5 seeds, last 5 evals)\n"
        "Color: relative rank within environment (green=best, red=worst)",
        fontweight="bold", pad=10)

    plt.tight_layout()
    _save(fig, "fig5_performance_table")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: HCGAE mechanism illustration (MC vs TD tradeoff)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig6_mechanism():
    """Visualize HCGAE's bias-correction mechanism conceptually."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # ── Left: alpha distribution over time ───────────────────────────
    ax = axes[0]
    all_alpha_early, all_alpha_late = [], []

    for env_name in ["Hopper-v4", "Walker2d-v4"]:
        for seed in SEEDS:
            d = load_metrics(env_name, "HCGAE_Imp12", seed)
            if d is None:
                continue
            ah = d.get("alpha_mean_history", [])
            if len(ah) >= 10:
                n = len(ah)
                all_alpha_early.extend(ah[:n//3])
                all_alpha_late.extend(ah[-n//3:])

    if all_alpha_early:
        ax.hist(all_alpha_early, bins=25, alpha=0.65, color="#E07B39",
                label="Early training (0–33%)", density=True)
        ax.hist(all_alpha_late, bins=25, alpha=0.65, color="#4878D0",
                label="Late training (67–100%)", density=True)
        ax.axvline(np.mean(all_alpha_early), color="#E07B39", lw=2, ls="--",
                   alpha=0.8, label=f"Early mean={np.mean(all_alpha_early):.3f}")
        ax.axvline(np.mean(all_alpha_late), color="#4878D0", lw=2, ls="--",
                   alpha=0.8, label=f"Late mean={np.mean(all_alpha_late):.3f}")
        ax.set_xlabel("Correction Strength α")
        ax.set_ylabel("Density")
        ax.set_title("α Distribution: Early vs Late Training\n(Hopper + Walker2d)", fontweight="bold")
        ax.legend(fontsize=7.5)

    # ── Right: EV vs alpha scatter ────────────────────────────────────
    ax = axes[1]
    all_ev_pts, all_alpha_pts = [], []

    for env_name in ["Hopper-v4", "Walker2d-v4"]:
        for seed in SEEDS:
            d = load_metrics(env_name, "HCGAE_Imp12", seed)
            if d is None:
                continue
            evh  = d.get("ev_ema_history", [])
            ah   = d.get("alpha_mean_history", [])
            n = min(len(evh), len(ah))
            if n >= 5:
                all_ev_pts.extend(evh[:n])
                all_alpha_pts.extend(ah[:n])

    if all_ev_pts:
        ax.scatter(all_ev_pts, all_alpha_pts, alpha=0.06, s=4,
                   color=COLORS["HCGAE_Imp12"], rasterized=True)
        # Add trend line
        ev_arr  = np.array(all_ev_pts)
        al_arr  = np.array(all_alpha_pts)
        mask = np.isfinite(ev_arr) & np.isfinite(al_arr)
        if mask.sum() > 10:
            z = np.polyfit(ev_arr[mask], al_arr[mask], 1)
            p = np.poly1d(z)
            xs = np.linspace(ev_arr[mask].min(), ev_arr[mask].max(), 100)
            r, pval = stats.pearsonr(ev_arr[mask], al_arr[mask])
            ax.plot(xs, p(xs), color="black", lw=2, ls="--",
                    label=f"Pearson r={r:.3f} (p={pval:.2e})")
            ax.legend(fontsize=7.5)

        ax.set_xlabel("EV EMA (Critic Quality)")
        ax.set_ylabel("Mean α (Correction Strength)")
        ax.set_title("EV vs α: Higher Critic Quality → Smaller Correction\n(Design Verification)", fontweight="bold")

    plt.tight_layout()
    _save(fig, "fig6_hcgae_mechanism")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: Sensitivity analysis (placeholder if data available)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig7_sensitivity():
    """If sensitivity data is available, plot it."""
    sens_dir = Path("results/Sensitivity")
    if not sens_dir.exists():
        print("  [Fig 7] Sensitivity data not found, skipping")
        return None

    # Look for any sensitivity JSON files
    sens_files = list(sens_dir.glob("*.json"))
    if not sens_files:
        print("  [Fig 7] No sensitivity JSON found")
        return None

    # Generic sensitivity plot
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.set_title("Hyperparameter Sensitivity", fontweight="bold")
    ax.text(0.5, 0.5, "Sensitivity data available\nSee results/Sensitivity/",
            ha="center", va="center", transform=ax.transAxes, fontsize=10)
    plt.tight_layout()
    _save(fig, "fig7_sensitivity")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Save helper
# ─────────────────────────────────────────────────────────────────────────────
def _save(fig, name):
    pdf_path = OUT_DIR / f"{name}.pdf"
    png_path = OUT_DIR / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved: {png_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Print LaTeX table
# ─────────────────────────────────────────────────────────────────────────────
def print_latex_table():
    """Print LaTeX-ready performance table."""
    ALGOS = [
        "Standard_PPO", "PPO_KLPEN", "PPO_Anneal",
        "PPO_EntDecay", "PPO_VClip", "PPO_Full_Baseline", "HCGAE_Imp12",
    ]
    available_envs = [e for e in ENVS if any(get_final_rewards(e, a) for a in ALGOS)]

    print("\n% ────────────────────────────────────────────────────────────")
    print("% LaTeX Performance Table (copy-paste to paper)")
    print("% ────────────────────────────────────────────────────────────")
    header_envs = " & ".join([f"\\textbf{{{e}}}" for e in available_envs])
    print(f"\\textbf{{Method}} & {header_envs} \\\\")
    print("\\hline")

    for algo in ALGOS:
        row_parts = [LABELS.get(algo, algo)]
        if algo == "HCGAE_Imp12":
            row_parts = [f"\\textbf{{{LABELS.get(algo, algo)}}}"]

        for env_name in available_envs:
            vals = get_final_rewards(env_name, algo)
            if vals:
                m = np.mean(vals)
                s = sem(vals)
                n = len(vals)
                # Statistical significance vs Standard_PPO
                ppo_vals = get_final_rewards(env_name, "Standard_PPO")
                if algo != "Standard_PPO" and ppo_vals:
                    _, p = mann_whitney_test(vals, ppo_vals)
                    sig = "$^{***}$" if p and p < 0.001 else \
                          "$^{**}$"  if p and p < 0.01  else \
                          "$^{*}$"   if p and p < 0.05  else ""
                else:
                    sig = ""
                if algo == "HCGAE_Imp12":
                    row_parts.append(f"\\textbf{{{m:.0f}}}$\\pm${s:.0f}{sig}")
                else:
                    row_parts.append(f"{m:.0f}$\\pm${s:.0f}{sig}")
            else:
                row_parts.append("—")
        print(" & ".join(row_parts) + " \\\\")
    print("\\hline")
    print("\\multicolumn{4}{l}{\\footnotesize $^{*}$p<0.05, $^{**}$p<0.01, $^{***}$p<0.001 (Mann-Whitney U test, two-sided)}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*65}")
    print("  Generating ICML Paper Figures")
    print(f"  Output: {OUT_DIR}")
    print(f"{'='*65}\n")

    print("  [Fig 1] Learning curves...")
    plot_fig1_learning_curves()

    print("  [Fig 2] Bar comparison...")
    plot_fig2_bar_comparison()

    print("  [Fig 3] Statistical significance...")
    plot_fig3_significance()

    print("  [Fig 4] HCGAE diagnostics...")
    plot_fig4_hcgae_diagnostics()

    print("  [Fig 5] Performance table...")
    plot_fig5_perf_table()

    print("  [Fig 6] Mechanism illustration...")
    plot_fig6_mechanism()

    print("  [Fig 7] Sensitivity...")
    plot_fig7_sensitivity()

    print_latex_table()

    print(f"\n  ✓ All figures saved to: {OUT_DIR}")
    print(f"  Files: {sorted([f.name for f in OUT_DIR.iterdir() if f.suffix in ('.pdf','.png')])}")


if __name__ == "__main__":
    main()

