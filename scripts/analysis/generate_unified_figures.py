"""
Generate ICML-quality figures from unified comparison results.

Requires: results/UnifiedComparison/ with data from run_unified_comparison.py

Figures produced (all in results/paper_figures_final/):
  fig1_learning_curves.{pdf,png}   – Learning curves (4-panel) with SEM bands
  fig2_final_performance.{pdf,png} – Final reward bar chart (3 envs x N algos)
  fig3_ablation.{pdf,png}          – HCGAE ablation (5-seed, Hopper-v4)
  fig4_sensitivity.{pdf,png}       – Hyperparameter sensitivity (existing data)
  fig5_dcppo_multienv.{pdf,png}    – DCPPO-S multienv (existing global_summary)
  fig6_overhead.{pdf,png}          – Computational overhead (existing)

Nature/ICML palette:
  Standard PPO  : #7F7F7F (gray)
  PPO-KLPEN     : #FF7F0E (orange)
  PPO-Anneal    : #2CA02C (green)
  PPO-EntDecay  : #D62728 (red)
  PPO-VClip     : #9467BD (purple)
  HCGAE_Imp12   : #1F77B4 (blue)  <-- ours
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
UNIFIED_DIR = Path("results/UnifiedComparison")
ABLATION_DIR = Path("results/Hopper-v4-Ablation-MultiSeed")
DCPPO_SUMMARY = Path("results/MultiEnv_DCPPO/global_summary.json")
SENSITIVITY_DIR = Path("results/Sensitivity")
OVERHEAD_FILE = Path("results/overhead_measurement.json")
OUT_DIR = Path("results/paper_figures_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENVS = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]
SEEDS = [42, 123, 456, 789, 1234]

ALGORITHMS = [
    "Standard_PPO",
    "PPO_KLPEN",
    "PPO_Anneal",
    "PPO_EntDecay",
    "PPO_VClip",
    "HCGAE_Imp12",
]

ALG_LABELS = {
    "Standard_PPO": "Standard PPO",
    "PPO_KLPEN":    "PPO-KLPEN",
    "PPO_Anneal":   "PPO-Anneal",
    "PPO_EntDecay": "PPO-EntDecay",
    "PPO_VClip":    "PPO-VClip",
    "HCGAE_Imp12":  "HCGAE (Ours)",
}

COLORS = {
    "Standard_PPO": "#7F7F7F",
    "PPO_KLPEN":    "#FF7F0E",
    "PPO_Anneal":   "#2CA02C",
    "PPO_EntDecay": "#D62728",
    "PPO_VClip":    "#9467BD",
    "HCGAE_Imp12":  "#1F77B4",
}

LINESTYLES = {
    "Standard_PPO": "--",
    "PPO_KLPEN":    ":",
    "PPO_Anneal":   "-.",
    "PPO_EntDecay": (0, (3, 1, 1, 1)),
    "PPO_VClip":    (0, (5, 2)),
    "HCGAE_Imp12":  "-",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_seed_data(env, algo, seeds=SEEDS):
    """Load all seed eval_rewards for one (env, algo). Returns list of arrays."""
    curves = []
    for seed in seeds:
        fp = UNIFIED_DIR / env / algo / f"{algo}_s{seed}.json"
        if fp.exists():
            d = json.load(open(fp))
            er = d.get("eval_rewards", [])
            es = d.get("eval_steps", [])
            if er:
                curves.append((np.array(es), np.array(er)))
    return curves


def interpolate_curves(curves, n_points=100):
    """Interpolate all curves to common x grid."""
    if not curves:
        return None, None, None
    # Find common x range
    max_steps = min(c[0][-1] for c in curves if len(c[0]) > 0)
    x_grid = np.linspace(0, max_steps, n_points)
    interp_curves = []
    for steps, rewards in curves:
        if len(steps) < 2:
            continue
        y = np.interp(x_grid, steps, rewards)
        interp_curves.append(y)
    if not interp_curves:
        return None, None, None
    mat = np.array(interp_curves)
    mean = mat.mean(axis=0)
    sem = mat.std(axis=0) / np.sqrt(len(interp_curves))
    return x_grid, mean, sem


def get_final_means(env, algos=ALGORITHMS, seeds=SEEDS):
    """Get final 5-eval mean for each algo in one env."""
    results = {}
    for algo in algos:
        curves = load_seed_data(env, algo, seeds)
        if not curves:
            continue
        seed_finals = []
        for _, rewards in curves:
            if len(rewards) >= 5:
                seed_finals.append(float(np.mean(rewards[-5:])))
            elif len(rewards) > 0:
                seed_finals.append(float(np.mean(rewards)))
        if seed_finals:
            results[algo] = {
                "mean": float(np.mean(seed_finals)),
                "std": float(np.std(seed_finals)),
                "n": len(seed_finals),
                "values": seed_finals,
            }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Learning curves (3 envs)
# ─────────────────────────────────────────────────────────────────────────────
def fig1_learning_curves():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Learning Curves: HCGAE vs PPO Baselines\n(5 seeds, 1M steps, mean +/- SEM)",
                 fontsize=12, fontweight="bold", y=1.02)

    for ax, env_name in zip(axes, ENVS):
        for algo in ALGORITHMS:
            curves = load_seed_data(env_name, algo)
            x, mean, sem = interpolate_curves(curves, n_points=100)
            if x is None:
                continue
            label = ALG_LABELS[algo]
            lw = 2.5 if algo == "HCGAE_Imp12" else 1.5
            zorder = 10 if algo == "HCGAE_Imp12" else 5
            ax.plot(x / 1e6, mean, color=COLORS[algo], label=label,
                    linewidth=lw, linestyle=LINESTYLES[algo], zorder=zorder)
            ax.fill_between(x / 1e6, mean - sem, mean + sem,
                            color=COLORS[algo], alpha=0.15, zorder=zorder - 1)

        ax.set_title(env_name.replace("-v4", ""), fontweight="bold")
        ax.set_xlabel("Environment Steps (M)")
        ax.set_ylabel("Eval Reward")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Legend outside
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=6,
                   bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=9)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(OUT_DIR / f"fig1_learning_curves.{ext}")
    plt.close()
    print("  Saved fig1_learning_curves")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Final performance bar chart
# ─────────────────────────────────────────────────────────────────────────────
def fig2_final_performance():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    fig.suptitle("Final Performance: HCGAE vs PPO Baselines\n(5 seeds, 1M steps, mean +/- std)",
                 fontsize=12, fontweight="bold", y=1.02)

    for ax, env_name in zip(axes, ENVS):
        results = get_final_means(env_name)
        algos_available = [a for a in ALGORITHMS if a in results]
        means = [results[a]["mean"] for a in algos_available]
        stds = [results[a]["std"] for a in algos_available]
        colors = [COLORS[a] for a in algos_available]
        labels = [ALG_LABELS[a] for a in algos_available]

        x = np.arange(len(algos_available))
        bars = ax.bar(x, means, yerr=stds, capsize=4,
                      color=colors, edgecolor="white", linewidth=0.5,
                      error_kw={"elinewidth": 1.5, "capthick": 1.5})

        # Highlight HCGAE
        for i, algo in enumerate(algos_available):
            if algo == "HCGAE_Imp12":
                bars[i].set_edgecolor("#1F77B4")
                bars[i].set_linewidth(2.5)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(env_name.replace("-v4", ""), fontweight="bold")
        ax.set_ylabel("Mean Final Reward")
        ax.grid(True, alpha=0.3, axis="y", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Annotate n
        for i, (m, s, algo) in enumerate(zip(means, stds, algos_available)):
            n = results[algo]["n"]
            ax.text(i, m + s + max(means) * 0.01, f"n={n}",
                    ha="center", va="bottom", fontsize=7, color="#555")

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(OUT_DIR / f"fig2_final_performance.{ext}")
    plt.close()
    print("  Saved fig2_final_performance")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Relative improvement over Standard PPO
# ─────────────────────────────────────────────────────────────────────────────
def fig3_relative_improvement():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Relative Improvement vs. Standard PPO\n(5 seeds, 1M steps)",
                 fontsize=12, fontweight="bold", y=1.02)

    for ax, env_name in zip(axes, ENVS):
        results = get_final_means(env_name)
        if "Standard_PPO" not in results:
            ax.set_title(f"{env_name} (pending)")
            continue
        baseline = results["Standard_PPO"]["mean"]
        if abs(baseline) < 1e-6:
            continue

        algos_to_plot = [a for a in ALGORITHMS if a != "Standard_PPO" and a in results]
        pcts = [(results[a]["mean"] - baseline) / abs(baseline) * 100 for a in algos_to_plot]
        colors = [COLORS[a] if pcts[i] > 0 else "#AAAAAA" for i, a in enumerate(algos_to_plot)]
        labels = [ALG_LABELS[a] for a in algos_to_plot]

        x = np.arange(len(algos_to_plot))
        bars = ax.bar(x, pcts, color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=1.0, linestyle="-")

        for i, (algo, pct) in enumerate(zip(algos_to_plot, pcts)):
            if algo == "HCGAE_Imp12":
                bars[i].set_edgecolor("#1F77B4")
                bars[i].set_linewidth(2.5)
            va = "bottom" if pct >= 0 else "top"
            offset = max(abs(max(pcts, default=0)), 5) * 0.02
            ax.text(i, pct + (offset if pct >= 0 else -offset),
                    f"{pct:+.0f}%", ha="center", va=va, fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(env_name.replace("-v4", ""), fontweight="bold")
        ax.set_ylabel("Improvement vs Std PPO (%)")
        ax.grid(True, alpha=0.3, axis="y", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(OUT_DIR / f"fig3_relative_improvement.{ext}")
    plt.close()
    print("  Saved fig3_relative_improvement")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: HCGAE ablation (from Hopper-v4-Ablation-MultiSeed)
# ─────────────────────────────────────────────────────────────────────────────
def fig4_ablation():
    variants = ["HCGAE_Base", "HCGAE_Imp1", "HCGAE_Imp2", "HCGAE_Imp12"]
    labels =   ["HCGAE Base\n(no I, no II)", "+I only", "+II only", "HCGAE Full\n(+I+II) [Ours]*"]
    colors_abl = ["#AAAAAA", "#FF7F0E", "#2CA02C", "#1F77B4"]

    abl_means = []
    abl_stds = []
    abl_ns = []

    for variant in variants:
        seed_vals = []
        for fname in os.listdir(ABLATION_DIR) if ABLATION_DIR.exists() else []:
            if fname.startswith(f"{variant}_s") and fname.endswith(".json"):
                d = json.load(open(ABLATION_DIR / fname))
                er = d.get("all_eval_rewards", [])
                if er:
                    seed_vals.append(float(np.mean(er[-5:])) if len(er) >= 5 else float(np.mean(er)))
        abl_means.append(np.mean(seed_vals) if seed_vals else 0)
        abl_stds.append(np.std(seed_vals) if seed_vals else 0)
        abl_ns.append(len(seed_vals))

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    x = np.arange(len(variants))
    bars = ax.bar(x, abl_means, yerr=abl_stds, capsize=5,
                  color=colors_abl, edgecolor="white", linewidth=0.5,
                  error_kw={"elinewidth": 1.8, "capthick": 1.8})
    bars[-1].set_edgecolor("#1F77B4")
    bars[-1].set_linewidth(2.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean Final Reward (5 seeds)")
    ax.set_title("HCGAE Ablation: Synergistic Interaction of Imp-I + Imp-II\n"
                 "(Hopper-v4, 5 Seeds x 300K Steps)", fontweight="bold")

    # Annotate synergy
    if all(m > 0 for m in abl_means[:3]):
        additive = abl_means[1] - abl_means[0] + (abl_means[2] - abl_means[0]) + abl_means[0]
        synergy = abl_means[3] - additive
        ax.annotate(f"Synergy: +{synergy:.0f} pts\n(vs. additive prediction)",
                    xy=(3, abl_means[3]), xytext=(2.2, abl_means[3] + abl_stds[3] + 200),
                    fontsize=9, color="#1F77B4", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#1F77B4"))

    for i, (m, s, n) in enumerate(zip(abl_means, abl_stds, abl_ns)):
        if m > 0:
            ax.text(i, m + s + 50, f"{m:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax.text(i, -250, f"n={n}", ha="center", va="top", fontsize=8, color="#777")

    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(OUT_DIR / f"fig4_ablation.{ext}")
    plt.close()
    print("  Saved fig4_ablation")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: DCPPO-S multienv (from global_summary.json)
# ─────────────────────────────────────────────────────────────────────────────
def fig5_dcppo_multienv():
    if not DCPPO_SUMMARY.exists():
        print("  Skipping fig5: no global_summary.json")
        return

    with open(DCPPO_SUMMARY) as f:
        data = json.load(f)
    results = data.get("results", {})

    envs_all = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4", "Ant-v4"]
    algos_dcppo = ["Standard_PPO", "DCPPO_S"]
    labels_dcppo = {"Standard_PPO": "Standard PPO", "DCPPO_S": "DCPPO-S (Ours)"}
    colors_dcppo = {"Standard_PPO": "#7F7F7F", "DCPPO_S": "#E64B35"}

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
    fig.suptitle("DCPPO-S vs Standard PPO: Multi-Environment (5 seeds x 300K steps)",
                 fontsize=12, fontweight="bold")

    for ax, env_name in zip(axes, envs_all):
        env_data = results.get(env_name, {})
        means_d = []
        stds_d = []
        cols_d = []
        labs_d = []
        for algo in algos_dcppo:
            if algo in env_data:
                means_d.append(env_data[algo]["mean"])
                stds_d.append(env_data[algo]["std"])
                cols_d.append(colors_dcppo[algo])
                labs_d.append(labels_dcppo[algo])

        x = np.arange(len(means_d))
        ax.bar(x, means_d, yerr=stds_d, capsize=5, color=cols_d,
               edgecolor="white", error_kw={"elinewidth": 1.8})
        ax.set_xticks(x)
        ax.set_xticklabels(labs_d, rotation=20, ha="right", fontsize=9)
        ax.set_title(env_name.replace("-v4", ""), fontweight="bold")
        ax.set_ylabel("Mean Final Reward")
        ax.grid(True, alpha=0.3, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if len(means_d) == 2 and means_d[0] > 0:
            pct = (means_d[1] - means_d[0]) / abs(means_d[0]) * 100
            color_ann = "#E64B35" if pct > 0 else "#7F7F7F"
            # Use axes-fraction coordinates (0-1 range) to avoid giant pixel issues
            ax.text(0.5, 0.95, f"{pct:+.0f}%", ha="center", va="top", fontsize=10,
                    fontweight="bold", color=color_ann, transform=ax.transAxes)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(OUT_DIR / f"fig5_dcppo_multienv.{ext}", dpi=150)
    plt.close()
    print("  Saved fig5_dcppo_multienv")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Overhead measurement
# ─────────────────────────────────────────────────────────────────────────────
def fig6_overhead():
    if not OVERHEAD_FILE.exists():
        print("  Skipping fig6: no overhead_measurement.json")
        return

    with open(OVERHEAD_FILE) as f:
        data = json.load(f)

    methods = list(data.keys())
    gae_times = [data[m].get("gae_time_ms", 0) for m in methods]
    update_times = [data[m].get("update_time_ms", 0) for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Computational Overhead Measurement\n(Hopper-v4, CPU, 20 runs avg)",
                 fontsize=12, fontweight="bold", y=1.02)

    x = np.arange(len(methods))
    colors_ov = ["#7F7F7F", "#1F77B4", "#E64B35"][:len(methods)]

    ax1.barh(x, gae_times, color=colors_ov)
    ax1.set_yticks(x)
    ax1.set_yticklabels(methods, fontsize=9)
    ax1.set_xlabel("GAE Time (ms)")
    ax1.set_title("GAE Computation Time", fontweight="bold")
    for i, v in enumerate(gae_times):
        ax1.text(v + 0.1, i, f"{v:.1f}ms", va="center", fontsize=9)

    ax2.barh(x, update_times, color=colors_ov)
    ax2.set_yticks(x)
    ax2.set_yticklabels(methods, fontsize=9)
    ax2.set_xlabel("Update Time (ms)")
    ax2.set_title("PPO Update Time", fontweight="bold")
    for i, v in enumerate(update_times):
        ax2.text(v + 1, i, f"{v:.1f}ms", va="center", fontsize=9)

    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(OUT_DIR / f"fig6_overhead.{ext}")
    plt.close()
    print("  Saved fig6_overhead")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: Summary table heatmap
# ─────────────────────────────────────────────────────────────────────────────
def fig7_summary_heatmap():
    # Collect all available data from UnifiedComparison
    table = {}
    for env in ENVS:
        table[env] = {}
        for algo in ALGORITHMS:
            curves = load_seed_data(env, algo)
            if curves:
                seed_finals = []
                for _, rewards in curves:
                    if len(rewards) >= 5:
                        seed_finals.append(float(np.mean(rewards[-5:])))
                    elif rewards:
                        seed_finals.append(float(np.mean(rewards)))
                if seed_finals:
                    table[env][algo] = {
                        "mean": float(np.mean(seed_finals)),
                        "std": float(np.std(seed_finals)),
                        "n": len(seed_finals),
                    }

    algos_with_data = [a for a in ALGORITHMS if any(a in table[e] for e in ENVS)]
    if not algos_with_data:
        print("  Skipping fig7: no data yet")
        return

    means_mat = np.full((len(algos_with_data), len(ENVS)), np.nan)
    for i, algo in enumerate(algos_with_data):
        for j, env in enumerate(ENVS):
            if algo in table[env]:
                means_mat[i, j] = table[env][algo]["mean"]

    fig, ax = plt.subplots(figsize=(9, 5))

    # Normalize per-column (env) for coloring
    norm_mat = np.full_like(means_mat, np.nan)
    for j in range(means_mat.shape[1]):
        col = means_mat[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) > 0:
            mn, mx = valid.min(), valid.max()
            if mx > mn:
                norm_mat[:, j] = (col - mn) / (mx - mn)
            else:
                norm_mat[:, j] = 0.5

    im = ax.imshow(norm_mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(ENVS)))
    ax.set_xticklabels([e.replace("-v4", "") for e in ENVS], fontsize=10)
    ax.set_yticks(range(len(algos_with_data)))
    ax.set_yticklabels([ALG_LABELS[a] for a in algos_with_data], fontsize=9)

    for i in range(len(algos_with_data)):
        for j in range(len(ENVS)):
            m = means_mat[i, j]
            algo = algos_with_data[i]
            n = table[ENVS[j]].get(algo, {}).get("n", 0)
            if not np.isnan(m):
                txt = f"{m:.0f}\n(n={n})"
                nv = norm_mat[i, j]
                color = "black" if (nv > 0.3 and nv < 0.7) else ("white" if nv < 0.3 else "black")
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color=color, fontweight="bold" if algo == "HCGAE_Imp12" else "normal")
            else:
                ax.text(j, i, "pending", ha="center", va="center", fontsize=8, color="#666")

    plt.colorbar(im, ax=ax, label="Normalized Performance (per-env)")
    ax.set_title("Performance Summary Heatmap\n(5 seeds, 1M steps, mean final reward)",
                 fontweight="bold")
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(OUT_DIR / f"fig7_summary_heatmap.{ext}")
    plt.close()
    print("  Saved fig7_summary_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# Save unified summary JSON
# ─────────────────────────────────────────────────────────────────────────────
def save_summary():
    summary = {}
    for env in ENVS:
        summary[env] = {}
        for algo in ALGORITHMS:
            curves = load_seed_data(env, algo)
            if curves:
                seed_finals = []
                for _, rewards in curves:
                    if len(rewards) >= 5:
                        seed_finals.append(float(np.mean(rewards[-5:])))
                    elif rewards:
                        seed_finals.append(float(np.mean(rewards)))
                if seed_finals:
                    summary[env][algo] = {
                        "mean": float(np.mean(seed_finals)),
                        "std": float(np.std(seed_finals)),
                        "n": len(seed_finals),
                        "seeds": seed_finals,
                    }

    out_path = UNIFIED_DIR / "unified_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n  Results available so far:")
    print(f"  {'Algorithm':<25} | {'Hopper-v4':>15} | {'Walker2d-v4':>15} | {'HalfCheetah-v4':>16}")
    print(f"  {'-'*25} | {'-'*15} | {'-'*15} | {'-'*16}")
    for algo in ALGORITHMS:
        row = f"  {ALG_LABELS[algo]:<25}"
        for env in ENVS:
            info = summary.get(env, {}).get(algo, {})
            if info:
                m, s, n = info["mean"], info["std"], info["n"]
                row += f" | {m:>7.0f}+/-{s:>4.0f}(n={n})"
            else:
                row += f" | {'pending':>15}"
        marker = " <-- OURS" if algo == "HCGAE_Imp12" else ""
        print(row + marker)
    print(f"  Saved: {out_path}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--figs", nargs="+", default=["all"],
                        help="Which figures to generate: 1 2 3 4 5 6 7 or all")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  Generating ICML Figures")
    print(f"{'='*60}")

    summary = save_summary()
    if args.summary_only:
        return

    figs = args.figs
    if "all" in figs or "1" in figs:
        fig1_learning_curves()
    if "all" in figs or "2" in figs:
        fig2_final_performance()
    if "all" in figs or "3" in figs:
        fig3_relative_improvement()
    if "all" in figs or "4" in figs:
        fig4_ablation()
    if "all" in figs or "5" in figs:
        fig5_dcppo_multienv()
    if "all" in figs or "6" in figs:
        fig6_overhead()
    if "all" in figs or "7" in figs:
        fig7_summary_heatmap()

    print(f"\n  All figures saved to: {OUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

