import os

RESULTS_DIR = "results/ICMLExperiment"
envs = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4", "Ant-v4", "Swimmer-v4"]
algos = ["Optimal_PPO", "Optimal_HCGAE_Bayesian"]
seeds = [1, 2, 3, 4, 5]

total = len(envs) * len(algos) * len(seeds)
completed = 0

for env in envs:
    for algo in algos:
        for seed in seeds:
            path = f"{RESULTS_DIR}/{env}/{algo}/{algo}_s{seed}.json"
            if os.path.exists(path):
                completed += 1

print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")

