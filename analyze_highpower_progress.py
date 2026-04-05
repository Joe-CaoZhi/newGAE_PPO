"""
Analyze HighPower experiment progress and interim results.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS_DIR = Path("results/HighPowerExperiment")
ENVS = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]
ALGOS = ["Standard_PPO", "Optimal_PPO", "Optimal_HCGAE_v2"]
N_TARGET_SEEDS = 30


def load_results():
    """Load all available results."""
    data = {}
    for env_name in ENVS:
        data[env_name] = {}
        for algo in ALGOS:
            algo_path = RESULTS_DIR / env_name / algo
            if algo_path.exists():
                seeds = []
                for f in algo_path.glob("*.json"):
                    try:
                        d = json.load(open(f))
                        er = d.get("eval_rewards", [])
                        if er:
                            final = float(np.mean(er[-5:])) if len(er) >= 5 else float(np.mean(er))
                            seeds.append({"seed": d.get("seed", 0), "final": final})
                    except Exception:
                        pass
                if seeds:
                    seeds.sort(key=lambda x: x["seed"])
                    data[env_name][algo] = seeds
    return data


def compute_statistics(scores_a, scores_b, name_a, name_b):
    """Compute statistical comparison."""
    if len(scores_a) < 2 or len(scores_b) < 2:
        return {}

    n_a, n_b = len(scores_a), len(scores_b)
    mean_a, mean_b = float(np.mean(scores_a)), float(np.mean(scores_b))
    std_a = float(np.std(scores_a, ddof=1))
    std_b = float(np.std(scores_b, ddof=1))

    u, p = stats.mannwhitneyu(scores_a, scores_b, alternative='two-sided')
    pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
    d = (mean_a - mean_b) / (pooled_std + 1e-8)

    # Bootstrap CI
    rng = np.random.default_rng(42)
    diffs = [float(np.mean(rng.choice(scores_a, n_a, True)) - np.mean(rng.choice(scores_b, n_b, True)))
             for _ in range(2000)]
    ci_lo, ci_hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

    # Power estimate
    lam = abs(d) * np.sqrt(n_a / 2.0)
    power = float(1 - stats.norm.cdf(1.96 - lam) + stats.norm.cdf(-1.96 - lam))

    return {
        f"mean_{name_a}": mean_a, f"mean_{name_b}": mean_b,
        f"std_{name_a}": std_a, f"std_{name_b}": std_b,
        f"sem_{name_a}": std_a / max(n_a**0.5, 1),
        f"sem_{name_b}": std_b / max(n_b**0.5, 1),
        "n_a": n_a, "n_b": n_b,
        "mann_whitney_u": float(u), "p_value": float(p),
        "cohens_d": float(d),
        "ci_low": ci_lo, "ci_high": ci_hi,
        "power_estimate": power,
        "pct_improvement": float((mean_a - mean_b) / (abs(mean_b) + 1e-8) * 100),
        "significant_p05": bool(p < 0.05),
    }


def main():
    print("=" * 70)
    print("  High Statistical Power Experiment - Progress Report")
    print("=" * 70)

    data = load_results()

    # Progress summary
    print("\n📊 Progress Summary:")
    print("-" * 70)
    total_done = 0
    total_target = len(ENVS) * len(ALGOS) * N_TARGET_SEEDS
    for env_name in ENVS:
        print(f"\n  {env_name}:")
        for algo in ALGOS:
            n = len(data.get(env_name, {}).get(algo, []))
            total_done += n
            bar = "█" * (n // 2) + "░" * ((N_TARGET_SEEDS - n) // 2)
            print(f"    {algo:<30} [{bar}] {n}/{N_TARGET_SEEDS}")

    print(f"\n  Total: {total_done}/{total_target} ({total_done/total_target*100:.1f}%)")

    # Interim results (only for algorithms with enough data)
    print("\n📈 Interim Results (where n >= 5):")
    print("-" * 70)

    for env_name in ENVS:
        env_data = data.get(env_name, {})
        if not env_data:
            continue

        print(f"\n  {env_name}:")
        for algo in ALGOS:
            if algo in env_data and len(env_data[algo]) >= 5:
                scores = [s["final"] for s in env_data[algo]]
                mean = np.mean(scores)
                std = np.std(scores, ddof=1)
                sem = std / len(scores)**0.5
                n = len(scores)
                print(f"    {algo:<30} {mean:8.1f} ± {sem:5.1f} SEM (n={n})")

        # Comparisons if both groups have enough data
        if "Standard_PPO" in env_data and len(env_data["Standard_PPO"]) >= 5:
            baseline = [s["final"] for s in env_data["Standard_PPO"]]
            baseline_mean = np.mean(baseline)

            for algo in ["Optimal_PPO", "Optimal_HCGAE_v2"]:
                if algo in env_data and len(env_data[algo]) >= 5:
                    scores = [s["final"] for s in env_data[algo]]
                    stat = compute_statistics(scores, baseline, algo, "Standard_PPO")
                    if stat:
                        pct = stat["pct_improvement"]
                        p = stat["p_value"]
                        d = stat["cohens_d"]
                        pwr = stat.get("power_estimate", float("nan"))
                        sig = "* p<0.05" if p < 0.05 else ("~ p<0.10" if p < 0.10 else "ns")
                        print(f"    {algo} vs Standard_PPO: {pct:+.1f}% "
                              f"(p={p:.3f} {sig}, d={d:+.2f}, power={pwr:.2f})")

    # Save interim summary
    summary = {
        "progress": {"total_done": total_done, "total_target": total_target},
        "results": {},
        "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S")
    }

    for env_name in ENVS:
        env_data = data.get(env_name, {})
        if env_data:
            summary["results"][env_name] = {}
            for algo in ALGOS:
                if algo in env_data and len(env_data[algo]) >= 5:
                    scores = [s["final"] for s in env_data[algo]]
                    summary["results"][env_name][algo] = {
                        "mean": float(np.mean(scores)),
                        "std": float(np.std(scores, ddof=1)),
                        "sem": float(np.std(scores, ddof=1) / len(scores)**0.5),
                        "n": len(scores),
                    }

    out_path = RESULTS_DIR / "highpower_interim.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Interim summary saved: {out_path}")


if __name__ == "__main__":
    main()

