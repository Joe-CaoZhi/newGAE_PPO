#!/usr/bin/env python3
"""
Generate HighPower Experiment Figure (n=30 seeds, Hopper-v4, 300K steps)
Shows the critical finding: Standard PPO outperforms Optimal PPO at short horizons.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configuration
RESULTS_DIR = Path("results/HighPowerExperiment/Hopper-v4")
OUT_DIR = Path("results/paper_figures_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 200,
})


def load_highpower_data():
    """Load all HighPower Hopper-v4 results."""
    data = {}
    for algo in ["Standard_PPO", "Optimal_PPO", "Optimal_HCGAE_v2"]:
        algo_path = RESULTS_DIR / algo
        if not algo_path.exists():
            continue
        finals = []
        for f in sorted(algo_path.glob("*.json")):
            try:
                d = json.load(open(f))
                er = d.get("eval_rewards", [])
                if er:
                    finals.append(float(np.mean(er[-5:])))
            except Exception:
                pass
        if finals:
            data[algo] = np.array(finals)
    return data


def compute_stats(data):
    """Compute statistics."""
    results = {}
    for algo, scores in data.items():
        n = len(scores)
        results[algo] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores, ddof=1)),
            "sem": float(np.std(scores, ddof=1) / np.sqrt(n)),
            "n": n,
        }

    # Statistical comparisons vs Standard PPO
    baseline = data.get("Standard_PPO", np.array([]))
    for algo, scores in data.items():
        if algo == "Standard_PPO":
            continue
        if len(baseline) >= 2 and len(scores) >= 2:
            u, p = stats.mannwhitneyu(scores, baseline, alternative='two-sided')
            pooled_std = np.sqrt((np.std(scores, ddof=1)**2 + np.std(baseline, ddof=1)**2) / 2)
            d = (np.mean(scores) - np.mean(baseline)) / (pooled_std + 1e-8)
            results[algo]["p_value"] = float(p)
            results[algo]["cohens_d"] = float(d)
            results[algo]["pct_diff"] = float((np.mean(scores) - np.mean(baseline)) / np.mean(baseline) * 100)

    return results


def plot_highpower_figure(data, stats_dict):
    """Create the main figure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Color scheme
    colors = {
        "Standard_PPO": "#3B82F6",      # Blue - best at 300K
        "Optimal_PPO": "#F59E0B",        # Orange
        "Optimal_HCGAE_v2": "#10B981",   # Green
    }
    labels = {
        "Standard_PPO": "Standard PPO",
        "Optimal_PPO": "Optimal PPO",
        "Optimal_HCGAE_v2": "HCGAE v2 (Ours)",
    }

    # ── Left: Bar comparison ────────────────────────────────────────────────
    ax = axes[0]
    algos = ["Standard_PPO", "Optimal_PPO", "Optimal_HCGAE_v2"]
    x = np.arange(len(algos))
    means = [stats_dict[a]["mean"] for a in algos]
    sems = [stats_dict[a]["sem"] for a in algos]
    bar_colors = [colors[a] for a in algos]
    bar_labels = [labels[a] for a in algos]

    bars = ax.bar(x, means, 0.6, yerr=sems, capsize=6,
                  color=bar_colors, alpha=0.85, edgecolor='white', linewidth=1.5,
                  error_kw={'ecolor': '#374151', 'elinewidth': 1.5})

    # Highlight best
    best_idx = int(np.argmax(means))
    bars[best_idx].set_edgecolor('#1F2937')
    bars[best_idx].set_linewidth(3)

    # Add significance annotations
    for i, algo in enumerate(algos):
        if algo == "Standard_PPO":
            continue
        if "p_value" in stats_dict[algo]:
            p = stats_dict[algo]["p_value"]
            d = stats_dict[algo]["cohens_d"]
            pct = stats_dict[algo]["pct_diff"]
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            # Draw bracket
            ymax = max(means[i] + sems[i], means[0] + sems[0])
            ax.annotate('', xy=(i, ymax * 1.05), xytext=(0, ymax * 1.05),
                        arrowprops=dict(arrowstyle='-', lw=1.5, color='#374151'))
            text = f"{pct:+.1f}%\np={p:.3f}{sig}\nd={d:.2f}"
            ax.text((i + 0) / 2 + 0.3, ymax * 1.08, text,
                    ha='center', va='bottom', fontsize=9,
                    color='#DC2626' if p < 0.05 else '#374151')

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=11)
    ax.set_ylabel("Mean Final Return ± SEM")
    ax.set_title("(a) Performance at 300K Steps (n=30 seeds)", fontweight='bold')

    # Add value labels
    for bar, mean, sem in zip(bars, means, sems):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + sem + max(means) * 0.015,
                f'{mean:.0f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    # ── Right: Distribution violin/box plot ─────────────────────────────────
    ax = axes[1]

    positions = [0, 1, 2]
    all_data = [data[a] for a in algos]

    # Violin plot
    parts = ax.violinplot(all_data, positions=positions, showmeans=True, showmedians=False)

    for i, (pc, algo) in enumerate(zip(parts['bodies'], algos)):
        pc.set_facecolor(colors[algo])
        pc.set_alpha(0.6)

    for partname in ['cbars', 'cmins', 'cmaxes', 'cmeans']:
        parts[partname].set_edgecolor('#374151')
        parts[partname].set_linewidth(1.2)

    # Overlay scatter points
    for i, (algo, d) in enumerate(zip(algos, all_data)):
        x_jitter = np.random.normal(i, 0.06, size=len(d))
        ax.scatter(x_jitter, d, alpha=0.4, s=25, color=colors[algo], edgecolor='white', linewidth=0.5)

    ax.set_xticks(positions)
    ax.set_xticklabels([labels[a] for a in algos], fontsize=11)
    ax.set_ylabel("Final Return (per seed)")
    ax.set_title("(b) Distribution of Individual Seeds", fontweight='bold')

    # Add n=30 annotation
    ax.text(0.98, 0.02, f"n = {stats_dict[algos[0]]['n']} seeds per algorithm",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
            style='italic', color='#6B7280')

    fig.suptitle(
        "High-Power Experiment: Observation Normalization Warm-up Cost\n"
        "Hopper-v4, 300K steps — Standard PPO significantly outperforms Optimal PPO",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()

    # Save
    for fmt in ['png', 'pdf']:
        path = OUT_DIR / f"fig_highpower_300k.{fmt}"
        fig.savefig(path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"[✓] HighPower figure saved to {OUT_DIR}")


def main():
    print("=" * 60)
    print("Generating HighPower Experiment Figure (n=30, Hopper-v4)")
    print("=" * 60)

    data = load_highpower_data()
    if not data:
        print("[!] No data found!")
        return

    stats_dict = compute_stats(data)

    # Print summary
    print("\n📊 Results Summary:")
    print("-" * 50)
    for algo, s in stats_dict.items():
        print(f"  {algo:25s}: {s['mean']:.1f} ± {s['sem']:.1f} (n={s['n']})")
        if "p_value" in s:
            sig = "***" if s["p_value"] < 0.001 else ""
            print(f"    → vs. Standard PPO: {s['pct_diff']:+.1f}%, p={s['p_value']:.4f}{sig}, d={s['cohens_d']:.2f}")

    plot_highpower_figure(data, stats_dict)

    # Save stats
    out_json = OUT_DIR / "highpower_300k_stats.json"
    with open(out_json, 'w') as f:
        json.dump(stats_dict, f, indent=2)
    print(f"[✓] Stats saved to {out_json}")


if __name__ == "__main__":
    main()

