#!/usr/bin/env python3
"""计算Bonferroni校正后的p值"""
alpha = 0.05
n_tests = 9  # 3 envs x 3 comparison pairs
bonferroni_alpha = alpha / n_tests
print(f"Bonferroni corrected alpha = {bonferroni_alpha:.5f} (= 0.05/{n_tests})")
print()

comparisons = [
    ('HCGAE vs Opt_PPO', 'Hopper',       0.222, '+9.6%',   '+1.28 (large)'),
    ('HCGAE vs Opt_PPO', 'Walker2d',     0.841, '+17.3%',  '+0.57 (medium)'),
    ('HCGAE vs Opt_PPO', 'HalfCheetah', 0.008, '-16.0%',  '-4.14 (large)'),
    ('HCGAE vs Std_PPO', 'Hopper',       0.421, '-2.9%',   '-0.69 (medium)'),
    ('HCGAE vs Std_PPO', 'Walker2d',     0.151, '+31.4%',  '+1.07 (large)'),
    ('HCGAE vs Std_PPO', 'HalfCheetah', 0.008, '+18.9%',  '+1.95 (large)'),
    ('Opt_PPO vs Std_PPO', 'Hopper',     0.032, '-11.4%',  '-1.78 (large)'),
    ('Opt_PPO vs Std_PPO', 'Walker2d',   0.548, '+12.0%',  '+0.51 (medium)'),
    ('Opt_PPO vs Std_PPO', 'HalfCheetah',0.008, '+41.5%', '+4.18 (large)'),
]

print("| Comparison | Env | Δ% | p (raw) | p×9 (Bonf-adj) | d | Sig (Bonf α=0.006) |")
print("|---|---|---|---|---|---|---|")
for cmp, env, p, delta, d in comparisons:
    adj_p = min(p * n_tests, 1.0)
    sig = '**' if p < bonferroni_alpha else 'ns'
    print(f"| {cmp} | {env} | {delta} | {p:.3f} | {adj_p:.3f} | {d} | {sig} |")

