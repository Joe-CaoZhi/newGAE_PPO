"""
Analyze completed AutoSCR experiment results.
"""
import json
import numpy as np
import os

from scipy import stats


def analyze_env(env_path, env_name):
    """Analyze a single environment."""
    results = {}
    for algo in sorted(os.listdir(env_path)):
        algo_path = os.path.join(env_path, algo)
        if os.path.isdir(algo_path):
            seeds = []
            for f in os.listdir(algo_path):
                if f.endswith('.json'):
                    try:
                        d = json.load(open(os.path.join(algo_path, f)))
                        if 'eval_rewards' in d:
                            final = d['eval_rewards'][-5:]
                        elif 'eval_returns' in d:
                            final = d['eval_returns'][-5:]
                        else:
                            final = [d.get('final_reward', 0)]
                        seeds.append(sum(final)/len(final))
                    except Exception as e:
                        pass
            if seeds:
                results[algo] = seeds

    if not results:
        return None

    print(f"\n{'='*60}")
    print(f"Environment: {env_name}")
    print(f"{'='*60}")

    # Find baseline
    baseline = results.get('Optimal_PPO', None)
    baseline_mean = np.mean(baseline) if baseline else 0

    for algo, seeds in sorted(results.items()):
        mean = np.mean(seeds)
        std = np.std(seeds, ddof=1)
        n = len(seeds)

        # Compute vs baseline
        if algo != 'Optimal_PPO' and baseline is not None:
            delta_pct = (mean - baseline_mean) / baseline_mean * 100
            # Cohen's d
            pooled_std = np.sqrt((std**2 + np.std(baseline, ddof=1)**2) / 2)
            d = (mean - baseline_mean) / pooled_std if pooled_std > 0 else 0
            # Mann-Whitney U
            if n >= 3 and len(baseline) >= 3:
                try:
                    u_stat, p_val = stats.mannwhitneyu(seeds, baseline, alternative='two-sided')
                except:
                    p_val = 1.0
            else:
                p_val = 1.0

            print(f"{algo}:")
            print(f"  Mean: {mean:.1f} ± {std:.1f} (n={n})")
            print(f"  vs Optimal PPO: {delta_pct:+.1f}%")
            print(f"  Cohen's d: {d:.2f}")
            print(f"  p-value: {p_val:.4f}")
        else:
            print(f"{algo}:")
            print(f"  Mean: {mean:.1f} ± {std:.1f} (n={n}) [BASELINE]")

    return results

# AutoSCR Experiment
print("="*60)
print("AutoSCR Experiment Results")
print("="*60)

base = '/Users/joe-caozhi/newGAE_ppo/results/AutoSCRExperiment'
all_results = {}
for env in sorted(os.listdir(base)):
    env_path = os.path.join(base, env)
    if os.path.isdir(env_path) and env not in ['test']:
        r = analyze_env(env_path, env)
        if r:
            all_results[env] = r

# Summary
print("\n" + "="*60)
print("SUMMARY: AutoSCR vs HCGAE v2")
print("="*60)
for env, results in all_results.items():
    if 'Optimal_HCGAE_v2_AutoSCR' in results and 'Optimal_HCGAE_v2' in results:
        v2 = results['Optimal_HCGAE_v2']
        autoscr = results['Optimal_HCGAE_v2_AutoSCR']
        baseline = results.get('Optimal_PPO', [0])

        v2_mean = np.mean(v2)
        autoscr_mean = np.mean(autoscr)
        baseline_mean = np.mean(baseline)

        print(f"\n{env}:")
        print(f"  HCGAE v2:     {v2_mean:.1f} ({(v2_mean-baseline_mean)/baseline_mean*100:+.1f}% vs baseline)")
        print(f"  AutoSCR:      {autoscr_mean:.1f} ({(autoscr_mean-baseline_mean)/baseline_mean*100:+.1f}% vs baseline)")
        print(f"  AutoSCR vs v2: {(autoscr_mean-v2_mean)/v2_mean*100:+.1f}%")

