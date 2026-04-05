"""测试 EV 收敛实验的 agent 构建是否正常"""
import sys
sys.path.insert(0, '.')
from gae_experiments.agents.optimal_ppo import build_optimal_agent
import gymnasium as gym

OPTIMAL_DEFAULTS = {
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "lam": 0.95,
    "lr": 3e-4,
    "eps_clip": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "hidden_dim": 64,
    "use_obs_norm": True,
    "use_adv_norm": True,
    "use_lr_anneal": True,
    "use_vclip": False,
}

algos = [
    "Optimal_PPO",
    "Optimal_HCGAE_v2",
    "Optimal_HCGAE_v2_NoBdry",
    "Optimal_HCGAE_v2_NoGate",
    "Optimal_HCGAE",
]

env = gym.make("Hopper-v4")
try:
    for algo in algos:
        agent = build_optimal_agent(algo_name=algo, env=env, **OPTIMAL_DEFAULTS)
        print(f"OK: {algo:35s} -> {type(agent).__name__}")
    print("\nAll agents built successfully!")
finally:
    env.close()

