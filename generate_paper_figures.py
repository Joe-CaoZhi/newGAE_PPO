#!/usr/bin/env python3
"""
Generate all paper figures from REAL experimental data.
=========================================================
Uses only actual JSON result files. No synthetic data.

Figures produced:
  fig1_learning_curves.png    - Multi-env learning curves (5 seeds, SEM)
  fig2_hcgae_ablation.png     - HCGAE ablation multi-seed bar + synergy
  fig3_dcppo_multienv.png     - DCPPO-S vs baseline multi-env bars
  fig4_sensitivity.png        - Hyperparameter sensitivity curves
  fig5_ev_snr_trajectory.png  - EV/SNR diagnostic over training
  fig6_overhead.png           - Computational overhead
  fig7_improvement_heatmap.png - Relative improvement heatmap

All text is English. No garbled characters.
"""
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "figure.dpi": 150,
})
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS_DIR = "results"
SAVE_DIR    = "results/paper_figures_v2"
os.makedirs(SAVE_DIR, exist_ok=True)

ENVS  = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4", "Ant-v4"]
SEEDS = [42, 123, 456, 789, 1234]

COLORS = {
    "Standard_PPO":  "#555555",
    "GAE_Lambda1":   "#E08000",
    "HCGAE_Base":    "#1E88E5",
    "HCGAE_Imp12":   "#00ACC1",
    "DCPPO_S":       "#43A047",
}
LABELS = {
    "Standard_PPO":  "Standard PPO",
    "GAE_Lambda1":   "GAE (λ=1, MC)",
    "HCGAE_Base":    "HCGAE-Base",
    "HCGAE_Imp12":   "HCGAE-Imp12",
    "DCPPO_S":       "DCPPO-S (ours)",
}
LINESTYLES = {
    "Standard_PPO": ":",
    "GAE_Lambda1":  "--",
    "HCGAE_Base":   "-.",
    "HCGAE_Imp12":  "-",
    "DCPPO_S":      "-",
}

# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────
def smooth(x, window=5):
    if len(x) < window:
        return np.array(x, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def load_multienv(algo, env, seeds=None):
    """Load multi-seed results from MultiEnv directory."""
    seeds = seeds or SEEDS
    rews, stps = [], []
    base = os.path.join(RESULTS_DIR, "MultiEnv", env)
    for seed in seeds:
        path = os.path.join(base, f"{algo}_s{seed}.json")
        if os.path.exists(path):
            d = json.load(open(path))
            rews.append(d.get("all_eval_rewards", []))
            stps.append(d.get("eval_steps", []))
    return rews, stps


def load_multienv_dcppo(algo, env, seeds=None):
    """Load DCPPO multi-seed results."""
    seeds = seeds or SEEDS
    rews, stps = [], []
    base = os.path.join(RESULTS_DIR, "MultiEnv_DCPPO", env)
    for seed in seeds:
        path = os.path.join(base, f"{algo}_s{seed}.json")
        if os.path.exists(path):
            d = json.load(open(path))
            rews.append(d.get("all_eval_rewards", []))
            stps.append(d.get("eval_steps", []))
    return rews, stps


def interpolate_to_grid(rews_list, stps_list, grid=None):
    """Interpolate all runs to a common step grid."""
    if not rews_list:
        return None, None
    if grid is None:
        max_len = max(len(s) for s in stps_list)
        min_steps = min(min(s) for s in stps_list if s)
        max_steps = max(max(s) for s in stps_list if s)
        grid = np.linspace(min_steps, max_steps, max_len)
    interp = []
    for r, s in zip(rews_list, stps_list):
        if len(r) < 2 or len(s) < 2:
            continue
        r_interp = np.interp(grid, s, r)
        interp.append(r_interp)
    if not interp:
        return None, None
    interp = np.array(interp)
    return interp, grid


def get_final_reward(algo, env, source="MultiEnv"):
    """Get mean±std final reward over seeds."""
    vals = []
    base = os.path.join(RESULTS_DIR, source, env)
    for seed in SEEDS:
        path = os.path.join(base, f"{algo}_s{seed}.json")
        if os.path.exists(path):
            d = json.load(open(path))
            vals.append(d.get("final_reward", 0.0))
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


# ─────────────────────────────────────────────────────────────────────────────
# FIG 1: Multi-environment learning curves
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig1_learning_curves():
    algos = ["Standard_PPO", "GAE_Lambda1", "HCGAE_Base", "HCGAE_Imp12"]
    fig, axes = plt.subplots(1, len(ENVS), figsize=(16, 4), sharey=False)

    for ax, env in zip(axes, ENVS):
        has_data = False
        for algo in algos:
            rews, stps = load_multienv(algo, env)
            if not rews:
                continue
            has_data = True
            interp, grid = interpolate_to_grid(rews, stps)
            if interp is None:
                continue
            mean = np.mean(interp, axis=0)
            sem  = np.std(interp, axis=0) / np.sqrt(len(interp))
            grid_k = grid / 1000
            ax.plot(grid_k, smooth(mean), color=COLORS[algo],
                    ls=LINESTYLES.get(algo, "-"), lw=1.8,
                    label=LABELS[algo])
            ax.fill_between(grid_k, smooth(mean - sem), smooth(mean + sem),
                            alpha=0.15, color=COLORS[algo])
        if not has_data:
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_title(env.replace("-v4", "").replace("-v3", ""), fontsize=12)
        ax.set_xlabel("Steps (K)")
        if ax == axes[0]:
            ax.set_ylabel("Eval Reward")
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        ax.grid(alpha=0.3, linestyle=":")
        ax.tick_params(labelsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4,
                   bbox_to_anchor=(0.5, 1.03), fontsize=10,
                   framealpha=0.9)
    plt.suptitle("Multi-Environment Learning Curves (5 Seeds, mean ± SEM)",
                 y=1.08, fontsize=13)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "fig1_learning_curves.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2: HCGAE Ablation multi-seed (bar + synergy annotation)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig2_hcgae_ablation():
    """Load ablation data from per-seed JSON files in any available directory."""
    candidates_dirs = [
        os.path.join(RESULTS_DIR, "Hopper-v4-Ablation-MultiSeed"),
        os.path.join(RESULTS_DIR, "Hopper-v4-Ablation"),
        os.path.join(RESULTS_DIR, "Advance-Ablation", "Hopper-v4"),
    ]
    ablation_dir = None
    for d in candidates_dirs:
        if os.path.isdir(d):
            ablation_dir = d
            break
    if ablation_dir is None:
        print("  [FIG2] No ablation directory found, skipping.")
        return None

    variants  = ["HCGAE_Base", "HCGAE_Imp1", "HCGAE_Imp2", "HCGAE_Imp12"]
    v_labels  = ["Base", "+Imp1\n(Norm)", "+Imp2\n(EV-mix)", "+Imp1+2\n(ours)"]

    # Load per-seed files
    def collect_variant_finals(vname):
        vals = []
        for seed in SEEDS:
            p = os.path.join(ablation_dir, f"{vname}_s{seed}.json")
            if os.path.exists(p):
                d = json.load(open(p))
                vals.append(d.get("final_reward", 0.0))
        # Also try without seed suffix (single-seed)
        if not vals:
            p = os.path.join(ablation_dir, f"{vname}.json")
            if os.path.exists(p):
                d = json.load(open(p))
                vals.append(d.get("final_reward", 0.0))
        return vals

    means, stds = [], []
    for v in variants:
        vals = collect_variant_finals(v)
        if vals:
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
        else:
            means.append(0.0); stds.append(0.0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors_abl = ["#90CAF9", "#64B5F6", "#42A5F5", "#1565C0"]
    bars = ax.bar(v_labels, means, color=colors_abl,
                  yerr=stds, capsize=5, width=0.55, zorder=3)
    ax.set_ylabel("Final Eval Reward (Hopper-v4)")
    ax.set_title("HCGAE Ablation Study (5 Seeds)")
    ax.grid(axis="y", alpha=0.4, zorder=0)
    ax.set_ylim(0, max(means) * 1.25 if means else 1)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{m:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Synergy annotation
    if all(m > 0 for m in means[:4]):
        d1  = means[1] - means[0]
        d2  = means[2] - means[0]
        d12 = means[3] - means[0]
        syn = d12 - d1 - d2
        synergy_text = (f"Synergy: {syn:+.0f}\n"
                        f"(Imp1+Imp2={d1+d2:+.0f}, actual={d12:+.0f})")
        ax.annotate(synergy_text,
                    xy=(3, means[3]), xytext=(2, means[3] * 1.05),
                    arrowprops=dict(arrowstyle="->", color="#B71C1C"),
                    fontsize=8, color="#B71C1C",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#FFCDD2", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "fig2_hcgae_ablation.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3: DCPPO-S multi-environment bar chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig3_dcppo_multienv():
    summary_path = os.path.join(RESULTS_DIR, "MultiEnv_DCPPO", "global_summary.json")
    # Also need Standard_PPO baseline from MultiEnv
    multi_summary = os.path.join(RESULTS_DIR, "MultiEnv", "global_summary.json")

    # Collect per-env results
    env_results = {}
    for env in ENVS:
        env_results[env] = {}
        # Standard PPO from MultiEnv
        sp_mean, sp_std = get_final_reward("Standard_PPO", env, "MultiEnv")
        env_results[env]["Standard_PPO"] = (sp_mean, sp_std)
        # HCGAE_Imp12 from MultiEnv
        hc_mean, hc_std = get_final_reward("HCGAE_Imp12", env, "MultiEnv")
        env_results[env]["HCGAE_Imp12"] = (hc_mean, hc_std)
        # DCPPO-S from MultiEnv_DCPPO
        ds_mean, ds_std = get_final_reward("DCPPO_S", env, "MultiEnv_DCPPO")
        env_results[env]["DCPPO_S"] = (ds_mean, ds_std)

    algos   = ["Standard_PPO", "HCGAE_Imp12", "DCPPO_S"]
    colors_  = [COLORS["Standard_PPO"], COLORS["HCGAE_Imp12"], COLORS["DCPPO_S"]]
    labels_  = [LABELS["Standard_PPO"], LABELS["HCGAE_Imp12"], LABELS["DCPPO_S"]]

    x = np.arange(len(ENVS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (algo, color, label) in enumerate(zip(algos, colors_, labels_)):
        vals = [env_results[env][algo][0] or 0 for env in ENVS]
        errs = [env_results[env][algo][1] or 0 for env in ENVS]
        rects = ax.bar(x + i * width - width, vals, width,
                       label=label, color=color, alpha=0.85,
                       yerr=errs, capsize=4, zorder=3)
        for r, v in zip(rects, vals):
            if v and abs(v) > 10:
                ax.text(r.get_x() + r.get_width() / 2,
                        r.get_height() + max(errs) * 0.5 + 20,
                        f"{v:.0f}", ha="center", va="bottom",
                        fontsize=7, rotation=45)

    env_names = [e.replace("-v4", "") for e in ENVS]
    ax.set_xticks(x)
    ax.set_xticklabels(env_names, fontsize=11)
    ax.set_ylabel("Final Eval Reward (mean ± std over 5 seeds)")
    ax.set_title("DCPPO-S vs Baselines: Multi-Environment Comparison")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.35, zorder=0)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "fig3_dcppo_multienv.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4: Hyperparameter sensitivity
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig4_sensitivity():
    sens_path = os.path.join(RESULTS_DIR, "Sensitivity", "sensitivity_summary.json")
    if not os.path.exists(sens_path):
        print("  [FIG4] Sensitivity results not yet available, skipping.")
        return None

    data = json.load(open(sens_path))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # ─ beta sensitivity ─
    ax = axes[0]
    betas = [r["beta"] for r in data["beta_sensitivity"]]
    b_rews = [r["final_reward"] for r in data["beta_sensitivity"]]
    ax.plot(betas, b_rews, "o-", color="#1565C0", lw=2, markersize=7)
    ax.axvline(x=3.0, color="red", ls="--", lw=1.2, label="Default β=3")
    ax.set_xlabel("HCGAE Sigmoid Steepness β")
    ax.set_ylabel("Final Eval Reward")
    ax.set_title("Sensitivity to β (alpha_max=0.7)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.35)
    ax.set_xticks(betas)

    # ─ alpha_max sensitivity ─
    ax = axes[1]
    amaxs = [r["alpha_max0"] for r in data["amax_sensitivity"]]
    a_rews = [r["final_reward"] for r in data["amax_sensitivity"]]
    ax.plot(amaxs, a_rews, "s-", color="#00838F", lw=2, markersize=7)
    ax.axvline(x=0.7, color="red", ls="--", lw=1.2, label="Default α_max=0.7")
    ax.set_xlabel("Max MC Blend (alpha_max)")
    ax.set_title("Sensitivity to alpha_max (β=3)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.35)
    ax.set_xticks(amaxs)

    # ─ SNR* sensitivity ─
    ax = axes[2]
    snrs = [r["snr_target"] for r in data["snr_sensitivity"]]
    s_rews = [r["final_reward"] for r in data["snr_sensitivity"]]
    ax.plot(snrs, s_rews, "^-", color="#2E7D32", lw=2, markersize=7)
    ax.axvline(x=0.3, color="red", ls="--", lw=1.2, label="Default SNR*=0.3")
    ax.set_xlabel("SNR Target (SNR*)")
    ax.set_title("DCPPO-S Sensitivity to SNR*")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.35)
    ax.set_xticks(snrs)

    plt.suptitle("Hyperparameter Sensitivity Analysis (Hopper-v4, seed=42)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "fig4_sensitivity.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# FIG 5: EV / SNR diagnostic trajectory (from metrics JSON)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig5_ev_snr():
    # Use DCPPO-S metrics from any available run
    candidates = [
        os.path.join(RESULTS_DIR, "MultiEnv_DCPPO", "Hopper-v4", "DCPPO_S_metrics.json"),
        os.path.join(RESULTS_DIR, "Hopper-v4-DCPPO", "DCPPO_ImpS_metrics.json"),
    ]
    # Also baseline metrics
    baseline_candidates = [
        os.path.join(RESULTS_DIR, "MultiEnv", "Hopper-v4", "Standard_GAE_metrics.json"),
        os.path.join(RESULTS_DIR, "MultiEnv_DCPPO", "Hopper-v4", "Standard_GAE_metrics.json"),
    ]

    dcppo_data = None
    for p in candidates:
        if os.path.exists(p):
            dcppo_data = json.load(open(p))
            break

    baseline_data = None
    for p in baseline_candidates:
        if os.path.exists(p):
            baseline_data = json.load(open(p))
            break

    if dcppo_data is None:
        print("  [FIG5] No DCPPO metrics found, skipping.")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    def to_arr(d, key, default_len=None):
        v = d.get(key, [])
        if not v and default_len:
            return np.zeros(default_len)
        return np.array(v, dtype=float)

    n = len(dcppo_data.get("total_steps", dcppo_data.get("explained_variances", [])))
    steps = to_arr(dcppo_data, "total_steps")
    if len(steps) == 0:
        steps = np.arange(n) * 2048

    # ─ EV over training ─
    ax = axes[0]
    ev = to_arr(dcppo_data, "ev_ema_history")
    if len(ev) == 0:
        ev = to_arr(dcppo_data, "explained_variances")
    if len(ev) > 0:
        ax.plot(steps[:len(ev)] / 1000, smooth(ev, 5),
                color=COLORS["DCPPO_S"], lw=2, label="DCPPO-S")
    if baseline_data:
        ev_b = to_arr(baseline_data, "ev_ema_history")
        if len(ev_b) == 0:
            ev_b = to_arr(baseline_data, "explained_variances")
        steps_b = to_arr(baseline_data, "total_steps")
        if len(steps_b) == 0:
            steps_b = np.arange(len(ev_b)) * 2048
        if len(ev_b) > 0:
            ax.plot(steps_b[:len(ev_b)] / 1000, smooth(ev_b, 5),
                    color=COLORS["Standard_PPO"], lw=2, ls=":", label="Standard PPO")
    ax.set_xlabel("Steps (K)")
    ax.set_ylabel("Explained Variance")
    ax.set_title("Critic EV over Training")
    ax.set_ylim(-0.5, 1.05)
    ax.axhline(1.0, color="green", ls="--", lw=0.8, alpha=0.6)
    ax.axhline(0.0, color="gray",  ls=":",  lw=0.8, alpha=0.5)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ─ SNR over training ─
    ax = axes[1]
    snr = to_arr(dcppo_data, "snr_history")
    if len(snr) > 0:
        ax.plot(steps[:len(snr)] / 1000, smooth(snr, 5),
                color="#E65100", lw=2, label="SNR (batch)")
    snr_w = to_arr(dcppo_data, "snr_weight_history")
    if len(snr_w) > 0:
        ax2 = ax.twinx()
        ax2.plot(steps[:len(snr_w)] / 1000, smooth(snr_w, 5),
                 color="#2E7D32", lw=1.8, ls="--", label="SNR weight w")
        ax2.set_ylabel("SNR Weight w", color="#2E7D32")
        ax2.set_ylim(0, 1.15)
        ax2.tick_params(axis="y", labelcolor="#2E7D32")
        ax2.legend(loc="lower right", fontsize=9)
    ax.set_xlabel("Steps (K)")
    ax.set_ylabel("Advantage SNR")
    ax.set_title("SNR & Gradient Scale Weight")
    ax.axhline(0.3, color="red", ls="--", lw=0.8, alpha=0.7, label="SNR*=0.3")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)

    # ─ MC blend c_mc over training ─
    ax = axes[2]
    c_mc = to_arr(dcppo_data, "c_mc_history")
    alpha_m = to_arr(dcppo_data, "alpha_mean_history")
    if len(c_mc) > 0:
        ax.plot(steps[:len(c_mc)] / 1000, smooth(c_mc, 5),
                color="#7B1FA2", lw=2, label="c_mc (EV-driven)")
    if len(alpha_m) > 0:
        ax.plot(steps[:len(alpha_m)] / 1000, smooth(alpha_m, 5),
                color="#00838F", lw=1.8, ls="--", label="mean α (HCGAE)")
    ax.set_xlabel("Steps (K)")
    ax.set_ylabel("Blend Coefficient")
    ax.set_title("EV-Driven Blend Coefficients")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.suptitle("EV / SNR / Blend Diagnostic Trajectories (Hopper-v4)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "fig5_ev_snr_trajectory.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# FIG 6: Computational overhead
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig6_overhead():
    overhead_path = os.path.join(RESULTS_DIR, "overhead_measurement.json")
    if not os.path.exists(overhead_path):
        print("  [FIG6] No overhead measurement found, skipping.")
        return None

    data = json.load(open(overhead_path))
    methods = list(data.get("methods", {}).keys())
    if not methods:
        # Try older structure
        methods = [k for k in data.keys() if k not in ("timestamp", "env")]
    if not methods:
        print("  [FIG6] Empty overhead data, skipping.")
        return None

    fps_vals  = [data.get("methods", data).get(m, {}).get("fps_mean", 0) for m in methods]
    time_vals = [data.get("methods", data).get(m, {}).get("time_per_update_ms", 0) for m in methods]

    # Clean method names for display
    clean_names = []
    for m in methods:
        n = m.replace("_", " ").replace("Standard GAE", "Std PPO")
        n = n.replace("HCGAE Imp12", "HCGAE").replace("DCPPO ImpS", "DCPPO-S")
        clean_names.append(n)

    x = np.arange(len(methods))
    colors_ov = ["#9E9E9E", "#FFA726", "#42A5F5", "#00ACC1", "#43A047"][:len(methods)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    bars = ax.bar(x, fps_vals, color=colors_ov, alpha=0.85, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(clean_names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Throughput (steps/sec)")
    ax.set_title("Computational Throughput")
    ax.grid(axis="y", alpha=0.35, zorder=0)
    for b, v in zip(bars, fps_vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 5,
                f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    bars2 = ax.bar(x, time_vals, color=colors_ov, alpha=0.85, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(clean_names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Time per Update (ms)")
    ax.set_title("Per-Update Computation Time")
    ax.grid(axis="y", alpha=0.35, zorder=0)
    for b, v in zip(bars2, time_vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02 + 0.01,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Computational Overhead Comparison (Hopper-v4)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "fig6_overhead.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# FIG 7: Improvement heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig7_heatmap():
    # Collect final rewards: Standard PPO (baseline) vs HCGAE_Imp12 vs DCPPO_S
    rows = ["HCGAE_Imp12", "DCPPO_S"]
    row_labels = ["HCGAE-Imp12", "DCPPO-S"]
    env_short = [e.replace("-v4", "") for e in ENVS]

    pct_improvement = np.zeros((2, len(ENVS)))
    valid_mask = np.zeros((2, len(ENVS)), dtype=bool)

    for j, env in enumerate(ENVS):
        sp_mean, _ = get_final_reward("Standard_PPO", env, "MultiEnv")
        if sp_mean is None:
            continue

        hc_mean, _ = get_final_reward("HCGAE_Imp12", env, "MultiEnv")
        if hc_mean is not None and abs(sp_mean) > 1:
            pct_improvement[0, j] = 100 * (hc_mean - sp_mean) / abs(sp_mean)
            valid_mask[0, j] = True

        ds_mean, _ = get_final_reward("DCPPO_S", env, "MultiEnv_DCPPO")
        if ds_mean is not None and abs(sp_mean) > 1:
            pct_improvement[1, j] = 100 * (ds_mean - sp_mean) / abs(sp_mean)
            valid_mask[1, j] = True

    if not valid_mask.any():
        print("  [FIG7] Insufficient data for heatmap, skipping.")
        return None

    # Mask invalid cells
    pct_plot = np.where(valid_mask, pct_improvement, np.nan)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    vmax = max(200, np.nanmax(np.abs(pct_plot)))
    im = ax.imshow(pct_plot, cmap="RdYlGn", aspect="auto",
                   vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="% Improvement vs Standard PPO")

    ax.set_xticks(range(len(ENVS)))
    ax.set_xticklabels(env_short, fontsize=11)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.set_title("Relative Improvement over Standard PPO (%)", fontsize=13)

    for i in range(2):
        for j in range(len(ENVS)):
            if valid_mask[i, j]:
                val = pct_improvement[i, j]
                ax.text(j, i, f"{val:+.0f}%",
                        ha="center", va="center", fontsize=11,
                        fontweight="bold",
                        color="black" if abs(val) < vmax * 0.5 else "white")
            else:
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=9, color="gray")

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "fig7_improvement_heatmap.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# FIG 8: DCPPO-S learning curves (Hopper-v4, multi-seed)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig8_dcppo_curves():
    algos  = ["Standard_PPO", "HCGAE_Imp12", "DCPPO_S"]
    fig, ax = plt.subplots(figsize=(7, 5))
    env = "Hopper-v4"
    has_data = False
    for algo in algos:
        # Try DCPPO dir for DCPPO_S, MultiEnv for rest
        if algo == "DCPPO_S":
            rews, stps = load_multienv_dcppo(algo, env)
        else:
            rews, stps = load_multienv(algo, env)
        if not rews:
            continue
        has_data = True
        interp, grid = interpolate_to_grid(rews, stps)
        if interp is None:
            continue
        mean = np.mean(interp, axis=0)
        sem  = np.std(interp, axis=0) / np.sqrt(len(interp))
        grid_k = grid / 1000
        ax.plot(grid_k, smooth(mean), color=COLORS[algo],
                ls=LINESTYLES.get(algo, "-"), lw=2,
                label=f"{LABELS[algo]} (n={len(interp)})")
        ax.fill_between(grid_k, smooth(mean - sem), smooth(mean + sem),
                        alpha=0.18, color=COLORS[algo])

    if not has_data:
        ax.text(0.5, 0.5, "No DCPPO-S data yet", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")

    ax.set_xlabel("Steps (K)")
    ax.set_ylabel("Eval Reward")
    ax.set_title("DCPPO-S vs Baselines: Hopper-v4 (5 Seeds, mean ± SEM)")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(alpha=0.35, linestyle=":")
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "fig8_dcppo_curves.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Generating paper figures from REAL experimental data")
    print(f"  Output: {SAVE_DIR}")
    print("=" * 60)

    generated = {}
    fns = [
        ("fig1_learning_curves",    plot_fig1_learning_curves),
        ("fig2_hcgae_ablation",     plot_fig2_hcgae_ablation),
        ("fig3_dcppo_multienv",     plot_fig3_dcppo_multienv),
        ("fig4_sensitivity",        plot_fig4_sensitivity),
        ("fig5_ev_snr_trajectory",  plot_fig5_ev_snr),
        ("fig6_overhead",           plot_fig6_overhead),
        ("fig7_improvement_heatmap",plot_fig7_heatmap),
        ("fig8_dcppo_curves",       plot_fig8_dcppo_curves),
    ]

    for name, fn in fns:
        print(f"\n  [{name}]")
        try:
            path = fn()
            if path:
                generated[name] = path
        except Exception as e:
            import traceback
            print(f"  [ERROR] {e}")
            traceback.print_exc()

    # Save index
    index = {"figures": generated, "count": len(generated)}
    ipath = os.path.join(SAVE_DIR, "figure_index.json")
    with open(ipath, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Generated {len(generated)}/{len(fns)} figures")
    for k, v in generated.items():
        print(f"    {k}: {v}")
    print(f"  Index: {ipath}")


if __name__ == "__main__":
    main()

