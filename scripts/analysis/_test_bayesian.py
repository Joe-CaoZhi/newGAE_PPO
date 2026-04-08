import gym
import numpy as np
import torch

from gae_experiments.agents.optimal_ppo import build_optimal_agent


def test_bayesian_agent(env_name="HalfCheetah-v4", steps=10000):
    print(f"--- Testing Bayesian Agent on {env_name} ---")
    env = gym.make(env_name)
    agent = build_optimal_agent("Optimal_HCGAE_Bayesian", env)

    obs, _ = env.reset()
    total_steps = 0

    while total_steps < steps:
        agent.buffer.reset()
        for _ in range(agent.n_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, log_prob = agent.actor.get_action_and_logprob(obs_t)
                value = agent.critic(obs_t)

            action_np = action.squeeze(0).cpu().numpy()
            if not agent.continuous:
                action_np = int(action_np)

            next_obs, reward, terminated, truncated, _ = env.step(action_np)

            if hasattr(agent, 'update_obs_rms'):
                agent.update_obs_rms(next_obs)
            next_obs_norm = agent.normalize_obs(next_obs) if hasattr(agent, 'normalize_obs') else next_obs

            agent.buffer.add(obs, action_np, float(reward), float(terminated), log_prob.item(), value.item())
            obs = next_obs_norm
            total_steps += 1

            if terminated or truncated:
                obs, _ = env.reset()
                if hasattr(agent, 'update_obs_rms'):
                    agent.update_obs_rms(obs)
                obs = agent.normalize_obs(obs) if hasattr(agent, 'normalize_obs') else obs

        # Compute GAE
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            last_val = agent.critic(obs_t).item()

        # We want to print the internal states during compute_hindsight_gae
        # Let's monkey patch it to print
        original_compute = agent.compute_hindsight_gae
        def compute_with_logs(last_value):
            T = agent.buffer.pos
            rewards = agent.buffer.rewards[:T]
            terminated = agent.buffer.terminated[:T]
            values = agent.buffer.values[:T]

            returns_mc = np.zeros(T, dtype=np.float32)
            running_return = last_value
            for t in reversed(range(T)):
                if terminated[t]:
                    running_return = 0.0
                running_return = rewards[t] + agent.gamma * running_return
                returns_mc[t] = running_return

            delta = returns_mc - values
            B_batch = float(np.abs(np.mean(delta)))
            sigma_G = float(np.std(returns_mc)) + 1e-8
            scr_now = B_batch / sigma_G

            sigma_e_now = float(np.std(delta)) + 1e-8

            print(f"Step {total_steps}:")
            print(f"  Bias (B_batch): {B_batch:.2f}, Noise (sigma_G): {sigma_G:.2f}")
            print(f"  SCR_now: {scr_now:.4f}, SCR_ema: {agent._scr_ema:.4f}")

            alpha_star = (agent._scr_ema ** 2) / (agent._scr_ema ** 2 + 1.0) + agent.scr_relax
            alpha_star = float(np.clip(alpha_star, 0.0, 1.0))
            print(f"  alpha_star: {alpha_star:.4f}")
            print(f"  sigma_e_now: {sigma_e_now:.2f}, sigma_e_ema: {agent._sigma_e_ema:.2f}")

            original_compute(last_value)

            print(f"  EV_ema: {agent._ev_ema:.4f}")
            print("-" * 40)

        agent.compute_hindsight_gae = compute_with_logs
        agent.compute_gae(last_val)
        agent.compute_hindsight_gae = original_compute # restore

        agent.update()

if __name__ == "__main__":
    test_bayesian_agent("HalfCheetah-v4", 10000)
    test_bayesian_agent("Hopper-v4", 10000)

