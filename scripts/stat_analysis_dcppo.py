import json
import numpy as np
from scipy import stats

# Load DCPPO data
with open('results/MultiEnv_DCPPO/dcppo_multiseed_summary.json', 'r') as f:
    dcppo_data = json.load(f)

# Load baseline data
with open('results/BaselineComparison/baseline_comparison_summary.json', 'r') as f:
    baseline_data = json.load(f)

print("=" * 60)
print("Statistical Analysis: DCPPO_ImpS vs Standard PPO")
print("=" * 60)

# Hopper-v4 analysis
print("\n### Hopper-v4 (500K steps)")
print("-" * 40)

dcppo_imps_seeds = dcppo_data['Hopper-v4']['DCPPO_ImpS']['seeds']
standard_ppo_seeds = baseline_data['Hopper-v4']['Standard PPO']['seeds']

print(f"DCPPO_ImpS seeds: {[f'{x:.1f}' for x in dcppo_imps_seeds]}")
print(f"Standard PPO seeds: {[f'{x:.1f}' for x in standard_ppo_seeds]}")

# Mann-Whitney U test
u_stat, p_value = stats.mannwhitneyu(dcppo_imps_seeds, standard_ppo_seeds, alternative='two-sided')
print(f"\nMann-Whitney U test:")
print(f"  U statistic: {u_stat}")
print(f"  p-value: {p_value:.4f}")

# Cohen's d
mean_diff = np.mean(dcppo_imps_seeds) - np.mean(standard_ppo_seeds)
pooled_std = np.sqrt((np.var(dcppo_imps_seeds) + np.var(standard_ppo_seeds)) / 2)
cohens_d = mean_diff / pooled_std
print(f"  Cohen's d: {cohens_d:.2f}")

# Percentage improvement
improvement = (np.mean(dcppo_imps_seeds) / np.mean(standard_ppo_seeds) - 1) * 100
print(f"  Improvement: {improvement:.1f}%")

# Significance level
if p_value < 0.01:
    sig = "**"
elif p_value < 0.05:
    sig = "*"
elif p_value < 0.1:
    sig = "."
else:
    sig = "n.s."
print(f"  Significance: {sig}")

# Walker2d-v4 analysis
print("\n### Walker2d-v4 (500K steps)")
print("-" * 40)
print("Note: DCPPO_ImpS seeds are identical to DCPPO_Base seeds")
print("      (SNR scaling may not have been applied)")

dcppo_walker_seeds = dcppo_data['Walker2d-v4']['DCPPO_ImpS']['seeds']
standard_walker_seeds = baseline_data['Walker2d-v4']['Standard PPO']['seeds']

print(f"\nDCPPO_ImpS seeds: {[f'{x:.1f}' for x in dcppo_walker_seeds]}")
print(f"Standard PPO seeds: {[f'{x:.1f}' for x in standard_walker_seeds]}")

u_stat, p_value = stats.mannwhitneyu(dcppo_walker_seeds, standard_walker_seeds, alternative='two-sided')
print(f"\nMann-Whitney U test:")
print(f"  U statistic: {u_stat}")
print(f"  p-value: {p_value:.4f}")

mean_diff = np.mean(dcppo_walker_seeds) - np.mean(standard_walker_seeds)
pooled_std = np.sqrt((np.var(dcppo_walker_seeds) + np.var(standard_walker_seeds)) / 2)
cohens_d = mean_diff / pooled_std
print(f"  Cohen's d: {cohens_d:.2f}")

improvement = (np.mean(dcppo_walker_seeds) / np.mean(standard_walker_seeds) - 1) * 100
print(f"  Improvement: {improvement:.1f}%")

# DCPPO_Full analysis
print("\n" + "=" * 60)
print("Statistical Analysis: DCPPO_Full vs DCPPO_ImpS")
print("=" * 60)

print("\n### Hopper-v4")
full_seeds = dcppo_data['Hopper-v4']['DCPPO_Full']['seeds']
imps_seeds = dcppo_data['Hopper-v4']['DCPPO_ImpS']['seeds']

u_stat, p_value = stats.mannwhitneyu(full_seeds, imps_seeds, alternative='two-sided')
print(f"Mann-Whitney U test:")
print(f"  U statistic: {u_stat}")
print(f"  p-value: {p_value:.4f}")

mean_diff = np.mean(full_seeds) - np.mean(imps_seeds)
pooled_std = np.sqrt((np.var(full_seeds) + np.var(imps_seeds)) / 2)
cohens_d = mean_diff / pooled_std
print(f"  Cohen's d: {cohens_d:.2f}")
print(f"  DCPPO_Full mean: {np.mean(full_seeds):.1f}")
print(f"  DCPPO_ImpS mean: {np.mean(imps_seeds):.1f}")
print(f"  Degradation: {(np.mean(full_seeds)/np.mean(imps_seeds)-1)*100:.1f}%")

print("\n" + "=" * 60)
print("Summary Table")
print("=" * 60)
print("\n| Comparison | Env | U | p-value | Cohen's d | Sig |")
print("|------------|-----|---|---------|-----------|-----|")

# Recalculate for summary
u1, p1 = stats.mannwhitneyu(dcppo_data['Hopper-v4']['DCPPO_ImpS']['seeds'],
                            baseline_data['Hopper-v4']['Standard PPO']['seeds'])
d1 = (np.mean(dcppo_data['Hopper-v4']['DCPPO_ImpS']['seeds']) - np.mean(baseline_data['Hopper-v4']['Standard PPO']['seeds'])) / \
     np.sqrt((np.var(dcppo_data['Hopper-v4']['DCPPO_ImpS']['seeds']) + np.var(baseline_data['Hopper-v4']['Standard PPO']['seeds']))/2)
sig1 = "**" if p1 < 0.01 else "*" if p1 < 0.05 else "." if p1 < 0.1 else "n.s."

u2, p2 = stats.mannwhitneyu(dcppo_data['Walker2d-v4']['DCPPO_ImpS']['seeds'],
                            baseline_data['Hopper-v4']['Standard PPO']['seeds'] if False else baseline_data['Walker2d-v4']['Standard PPO']['seeds'])
d2 = (np.mean(dcppo_data['Walker2d-v4']['DCPPO_ImpS']['seeds']) - np.mean(baseline_data['Walker2d-v4']['Standard PPO']['seeds'])) / \
     np.sqrt((np.var(dcppo_data['Walker2d-v4']['DCPPO_ImpS']['seeds']) + np.var(baseline_data['Walker2d-v4']['Standard PPO']['seeds']))/2)
sig2 = "**" if p2 < 0.01 else "*" if p2 < 0.05 else "." if p2 < 0.1 else "n.s."

print(f"| DCPPO_ImpS vs Std PPO | Hopper | {u1:.0f} | {p1:.4f} | {d1:+.2f} | {sig1} |")
print(f"| DCPPO_ImpS vs Std PPO | Walker | {u2:.0f} | {p2:.4f} | {d2:+.2f} | {sig2} |")

# Additional: DCPPO_Base vs Standard PPO
print("\n| DCPPO_Base vs Std PPO |")
u3, p3 = stats.mannwhitneyu(dcppo_data['Hopper-v4']['DCPPO_Base']['seeds'],
                            baseline_data['Hopper-v4']['Standard PPO']['seeds'])
d3 = (np.mean(dcppo_data['Hopper-v4']['DCPPO_Base']['seeds']) - np.mean(baseline_data['Hopper-v4']['Standard PPO']['seeds'])) / \
     np.sqrt((np.var(dcppo_data['Hopper-v4']['DCPPO_Base']['seeds']) + np.var(baseline_data['Hopper-v4']['Standard PPO']['seeds']))/2)
sig3 = "**" if p3 < 0.01 else "*" if p3 < 0.05 else "." if p3 < 0.1 else "n.s."
print(f"| DCPPO_Base vs Std PPO | Hopper | {u3:.0f} | {p3:.4f} | {d3:+.2f} | {sig3} |")

