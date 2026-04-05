#!/usr/bin/env python3
"""
启动缺失的实验：
1. AutoSCR: Walker2d-v4 和 HalfCheetah-v4 的 Optimal_HCGAE_v2 和 Optimal_HCGAE_v2_AutoSCR
2. HighPower: 继续运行 Hopper-v4 的 Optimal_PPO 和 Optimal_HCGAE_v2

使用 Optimal PPO 作为基准。
"""
import os
import subprocess
import sys

# 切换到项目目录
os.chdir("/Users/joe-caozhi/newGAE_ppo")

# 1. 先运行 Walker2d-v4 的 AutoSCR 实验
print("=" * 60)
print("Running AutoSCR Walker2d-v4...")
print("=" * 60)
result = subprocess.run([
    sys.executable, "run_autoscr_experiment.py",
    "--envs", "Walker2d-v4",
    "--algos", "Optimal_PPO", "Optimal_HCGAE_v2", "Optimal_HCGAE_v2_AutoSCR",
    "--seeds", "5"
], check=False)
print(f"Walker2d-v4 AutoSCR completed with code {result.returncode}")

# 2. 运行 HalfCheetah-v4 的 AutoSCR 实验
print("\n" + "=" * 60)
print("Running AutoSCR HalfCheetah-v4...")
print("=" * 60)
result = subprocess.run([
    sys.executable, "run_autoscr_experiment.py",
    "--envs", "HalfCheetah-v4",
    "--algos", "Optimal_PPO", "Optimal_HCGAE_v2", "Optimal_HCGAE_v2_AutoSCR",
    "--seeds", "5"
], check=False)
print(f"HalfCheetah-v4 AutoSCR completed with code {result.returncode}")

# 3. 运行 HighPower Hopper-v4 的 Optimal_PPO
print("\n" + "=" * 60)
print("Running HighPower Hopper-v4 Optimal_PPO (n=30)...")
print("=" * 60)
result = subprocess.run([
    sys.executable, "run_highpower_experiment.py",
    "--envs", "Hopper-v4",
    "--algos", "Optimal_PPO",
    "--n_seeds", "30"
], check=False)
print(f"HighPower Hopper-v4 Optimal_PPO completed with code {result.returncode}")

# 4. 运行 HighPower Hopper-v4 的 Optimal_HCGAE_v2
print("\n" + "=" * 60)
print("Running HighPower Hopper-v4 Optimal_HCGAE_v2 (n=30)...")
print("=" * 60)
result = subprocess.run([
    sys.executable, "run_highpower_experiment.py",
    "--envs", "Hopper-v4",
    "--algos", "Optimal_HCGAE_v2",
    "--n_seeds", "30"
], check=False)
print(f"HighPower Hopper-v4 Optimal_HCGAE_v2 completed with code {result.returncode}")

print("\n" + "=" * 60)
print("All experiments completed!")
print("=" * 60)

