"""EV 记录冒烟测试：确认 EV 在真实训练循环中被正确记录"""
import sys, os
sys.path.insert(0, '.')
import gymnasium as gym
import numpy as np
import torch
from gae_experiments.agents.optimal_ppo import build_optimal_agent
from gae_experiments.utils.logger import MetricLogger

os.makedirs("/tmp/ev_smoke_test", exist_ok=True)

DEFAULTS = {
    "n_steps": 512,   # 小 buffer，快速测试
    "batch_size": 64,
    "n_epochs": 2,
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
    "device": "cpu",
    "save_dir": "/tmp/ev_smoke_test",
}

TOTAL_STEPS = 3072  # ~6 rollouts of 512 steps

for algo in ["Optimal_PPO", "Optimal_HCGAE_v2"]:
    print(f"\n── Testing {algo} ──")
    env = gym.make("Hopper-v4")
    eval_env = gym.make("Hopper-v4")
    env.reset(seed=42)
    eval_env.reset(seed=9999)

    agent = build_optimal_agent(algo_name=algo, env=env, name=f"{algo}_smoke", **DEFAULTS)
    logger = MetricLogger(agent_name=f"{algo}_smoke", save_dir="/tmp/ev_smoke_test")
    agent.logger = logger

    # 训练循环（与 run_ev_convergence_study.py 保持一致）
    obs, _ = env.reset()
    if hasattr(agent, 'normalize_obs'):
        obs = agent.normalize_obs(obs)
    step_count = 0

    while step_count < TOTAL_STEPS:
        agent.buffer.reset()
        for _ in range(agent.n_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, log_prob = agent.actor.get_action_and_logprob(obs_t)
                value = agent.critic(obs_t)
            act_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, _ = env.step(act_np)
            if hasattr(agent, 'update_obs_rms'):
                agent.update_obs_rms(next_obs)
            if hasattr(agent, 'normalize_obs'):
                next_obs = agent.normalize_obs(next_obs)
            agent.buffer.add(obs, act_np, float(reward), float(terminated), log_prob.item(), value.item())
            obs = next_obs
            step_count += 1
            if terminated or truncated:
                obs, _ = env.reset()
                if hasattr(agent, 'normalize_obs'):
                    obs = agent.normalize_obs(obs)

        with torch.no_grad():
            last_obs_t = torch.FloatTensor(obs).unsqueeze(0)
            last_val = agent.critic(last_obs_t).item()

        agent._total_timesteps = TOTAL_STEPS
        agent.total_steps = step_count

        if hasattr(agent, 'compute_hindsight_gae'):
            agent.compute_hindsight_gae(last_val)
        else:
            agent.compute_gae(last_val)

        metrics = agent.update()
        ev = metrics.get("explained_variance", float('nan'))
        alpha_mean = metrics.get("alpha_mean", None)

        logger.log_update(
            value_loss=metrics.get("value_loss", 0.0),
            policy_loss=metrics.get("policy_loss", 0.0),
            entropy_loss=metrics.get("entropy_loss", 0.0),
            approx_kl=metrics.get("approx_kl", 0.0),
            clip_frac=metrics.get("clip_frac", 0.0),
            explained_variance=ev,
            total_steps=step_count,
            alpha_mean=alpha_mean,
        )

    print(f"  explained_variances : {len(logger.explained_variances)} entries")
    print(f"  total_steps         : {logger.total_steps}")
    print(f"  EV values           : {[f'{v:.3f}' for v in logger.explained_variances]}")
    print(f"  alpha_mean_history  : {len(logger.alpha_mean_history)} entries")
    if logger.alpha_mean_history:
        print(f"    alpha values      : {[f'{v:.3f}' for v in logger.alpha_mean_history]}")

    # 检查 update() 返回值中有哪些键
    print(f"  update() keys       : {list(metrics.keys())}")

    env.close()
    eval_env.close()

print("\n✓ Smoke test PASSED!")

