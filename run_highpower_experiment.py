"""
High Statistical Power Experiment (n=30 seeds)
===============================================

Purpose: Address Future Work 2 from the paper — increase statistical power
to reliably detect medium-sized effects (Cohen's d ≈ 0.27-0.61) on episodic tasks.

Design:
- 3 environments: Hopper-v4, Walker2d-v4, HalfCheetah-v4
- 3 algorithms:
  * Standard_PPO        (baseline, no obs norm)
  * Optimal_PPO         (strong baseline, with obs norm)
  * Optimal_HCGAE_v2   (paper's method)
- 30 seeds (seeds 0..29)
- 300,000 steps (fast runs, ~5 min each on CPU)
- Eval every 10,240 steps, 10 deterministic episodes

Statistical analysis:
- Mann-Whitney U test (non-parametric, no normality assumption)
- Cohen's d effect size
- Bootstrap 95% CI (2000 iterations)
- Post-hoc power analysis

Power analysis (pre-registered):
- For d=0.27 (Hopper HCGAE vs StandardPPO): n=30 → power ≈ 21% (still low but informative)
- For d=0.61 (Hopper HCGAE_SCR vs StandardPPO): n=30 → power ≈ 73%
- For d=1.17 (HalfCheetah, large effect): n=30 → power > 99%

Results saved to: results/HighPowerExperiment/
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from gae_experiments.agents.optimal_ppo import build_optimal_agent
from gae_experiments.agents.ppo_baselines import build_ppo_baseline

# ──────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────
ENVS = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]
ALGOS = ["Standard_PPO", "Optimal_PPO", "Optimal_HCGAE_v2"]
TOTAL_TIMESTEPS = 300_000  # 300K steps → ~5 min per run on CPU
EVAL_FREQ = 10_240
N_EVAL_EPISODES = 10
N_SEEDS = 30

RESULTS_DIR = Path("results/HighPowerExperiment")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
STANDARD_PPO_KWARGS = dict(
    hidden_dim=64, lr_actor=3e-4, lr_critic=1e-3,
    gamma=0.99, lam=0.95, eps_clip=0.2,
    n_epochs=10, batch_size=64, n_steps=2048,
    ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
    device="cpu",
)
OPTIMAL_PPO_KWARGS = dict(
    hidden_dim=64, lr=3e-4,
    gamma=0.99, lam=0.95, eps_clip=0.2,
    n_epochs=10, batch_size=64, n_steps=2048,
    ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
    use_obs_norm=True, use_adv_norm=True,
    use_lr_anneal=True, use_vclip=False,
    device="cpu",
)


# ──────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────
def evaluate_policy(agent, eval_env, n_episodes=10):
    rewards = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        if hasattr(agent, 'normalize_obs'):
            obs = agent.normalize_obs(obs)
        total_reward = 0.0
        done = False
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, _ = agent.actor.get_action_and_logprob(obs_t)
                action_np = action.squeeze(0).cpu().numpy()
                if not agent.continuous:
                    action_np = int(action_np)
            obs, r, terminated, truncated, _ = eval_env.step(action_np)
            if hasattr(agent, 'normalize_obs'):
                obs = agent.normalize_obs(obs)
            total_reward += r
            done = terminated or truncated
        rewards.append(total_reward)
    return float(np.mean(rewards))


# ──────────────────────────────────────────────────────────────────────────
# Single run
# ──────────────────────────────────────────────────────────────────────────
def run_single(env_name: str, algo_name: str, seed: int) -> dict:
    """Run one (env, algo, seed) combination."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    save_dir = str(RESULTS_DIR / env_name / algo_name)
    os.makedirs(save_dir, exist_ok=True)
    out_path = Path(save_dir) / f"{algo_name}_s{seed}.json"

    if out_path.exists():
        print(f"    [SKIP] {env_name}/{algo_name}/s{seed} already done")
        return json.load(open(out_path))

    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 50000)

    if algo_name == "Standard_PPO":
        kw = dict(**STANDARD_PPO_KWARGS, save_dir=save_dir)
        agent = build_ppo_baseline("Standard_PPO", env, name=f"{algo_name}_s{seed}", **kw)
    else:
        kw = dict(**OPTIMAL_PPO_KWARGS, save_dir=save_dir)
        agent = build_optimal_agent(algo_name, env, name=f"{algo_name}_s{seed}", **kw)

    eval_rewards = []
    eval_steps = []
    obs, _ = env.reset()
    if hasattr(agent, 'update_obs_rms'):
        agent.update_obs_rms(obs)
    if hasattr(agent, 'normalize_obs'):
        obs = agent.normalize_obs(obs)

    ep_reward = 0.0
    total_steps = 0
    last_eval_step = 0
    t0 = time.time()

    while total_steps < TOTAL_TIMESTEPS:
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

            ep_reward += reward
            agent.buffer.add(obs, action_np, float(reward), float(terminated),
                             log_prob.item(), value.item())
            obs = next_obs_norm
            total_steps += 1

            if terminated or truncated:
                ep_reward = 0.0
                next_obs, _ = env.reset()
                if hasattr(agent, 'update_obs_rms'):
                    agent.update_obs_rms(next_obs)
                obs = agent.normalize_obs(next_obs) if hasattr(agent, 'normalize_obs') else next_obs

            if total_steps - last_eval_step >= EVAL_FREQ:
                eval_r = evaluate_policy(agent, eval_env, N_EVAL_EPISODES)
                eval_rewards.append(eval_r)
                eval_steps.append(total_steps)
                last_eval_step = total_steps

        with torch.no_grad():
            last_val = agent.critic(torch.FloatTensor(obs).unsqueeze(0)).item()
        agent._total_timesteps = TOTAL_TIMESTEPS
        agent.total_steps = total_steps
        if hasattr(agent, 'compute_hindsight_gae'):
            agent.compute_hindsight_gae(last_val)
        else:
            agent.compute_gae(last_val)
        agent.update()

    elapsed = time.time() - t0
    final_mean = float(np.mean(eval_rewards[-5:])) if len(eval_rewards) >= 5 else (
        float(np.mean(eval_rewards)) if eval_rewards else 0.0)

    result = {
        "env": env_name, "agent": algo_name, "seed": seed,
        "total_steps": total_steps, "final_reward": final_mean,
        "eval_rewards": eval_rewards, "eval_steps": eval_steps,
        "elapsed_s": elapsed,
        "ev_final": float(getattr(agent, '_ev_ema', 0.0)),
    }
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    env.close()
    eval_env.close()
    print(f"    Done: {algo_name} {env_name} s{seed} -> {final_mean:.1f} ({elapsed:.0f}s)")
    return result


# ──────────────────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────────────────
def compute_statistics(scores_a, scores_b, name_a, name_b):
    try:
        from scipy import stats
    except ImportError:
        return {}
    if len(scores_a) < 2 or len(scores_b) < 2:
        return {}

    n_a, n_b = len(scores_a), len(scores_b)
    mean_a, mean_b = float(np.mean(scores_a)), float(np.mean(scores_b))
    std_a = float(np.std(scores_a, ddof=1))
    std_b = float(np.std(scores_b, ddof=1))
    u, p = stats.mannwhitneyu(scores_a, scores_b, alternative='two-sided')
    pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
    d = (mean_a - mean_b) / (pooled_std + 1e-8)
    rng = np.random.default_rng(42)
    diffs = [float(np.mean(rng.choice(scores_a, n_a, True)) - np.mean(rng.choice(scores_b, n_b, True)))
             for _ in range(2000)]
    ci_lo, ci_hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

    # Post-hoc power estimate (t-test approximation)
    try:
        import importlib
        sm = importlib.util.find_spec("statsmodels")
        if sm:
            from statsmodels.stats.power import TTestIndPower
            power = TTestIndPower().solve_power(
                effect_size=abs(d), nobs1=n_a, ratio=n_b/n_a, alpha=0.05)
        else:
            from scipy.stats import norm
            lam = abs(d) * np.sqrt(n_a / 2.0)
            power = float(1 - norm.cdf(1.96 - lam) + norm.cdf(-1.96 - lam))
    except Exception:
        power = float("nan")

    return {
        f"mean_{name_a}": mean_a, f"mean_{name_b}": mean_b,
        f"std_{name_a}": std_a, f"std_{name_b}": std_b,
        f"sem_{name_a}": std_a / max(n_a**0.5, 1), f"sem_{name_b}": std_b / max(n_b**0.5, 1),
        "n_a": n_a, "n_b": n_b,
        "mann_whitney_u": float(u), "p_value": float(p),
        "cohens_d": float(d),
        "ci_low": ci_lo, "ci_high": ci_hi,
        "power_estimate": float(power),
        "pct_improvement": float((mean_a - mean_b) / (abs(mean_b) + 1e-8) * 100),
        "significant_p05": bool(p < 0.05),
        "significant_p10": bool(p < 0.10),
    }


def load_results(results_dir, envs, algos, seeds):
    data = {}
    for env_name in envs:
        data[env_name] = {}
        for algo in algos:
            seed_means = []
            for seed in seeds:
                fp = results_dir / env_name / algo / f"{algo}_s{seed}.json"
                if fp.exists():
                    try:
                        d = json.load(open(fp))
                        er = d.get("eval_rewards", [])
                        if er:
                            seed_means.append(float(np.mean(er[-5:])) if len(er) >= 5 else float(np.mean(er)))
                    except Exception:
                        pass
            if seed_means:
                data[env_name][algo] = {
                    "mean": float(np.mean(seed_means)), "std": float(np.std(seed_means)),
                    "sem": float(np.std(seed_means) / max(len(seed_means)**0.5, 1)),
                    "n": len(seed_means), "seeds": seed_means,
                }
    return data


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", default=None)
    parser.add_argument("--algos", nargs="+", default=None)
    parser.add_argument("--n_seeds", type=int, default=30)
    parser.add_argument("--start_seed", type=int, default=0)
    args = parser.parse_args()

    envs = args.envs if args.envs else ENVS
    algos = args.algos if args.algos else ALGOS
    seeds = list(range(args.start_seed, args.start_seed + args.n_seeds))
    n_seeds = len(seeds)

    total_runs = len(envs) * len(algos) * n_seeds
    t0 = time.time()
    run_idx = 0

    print(f"\n{'='*70}")
    print(f"  High Statistical Power Experiment (Future Work 2)")
    print(f"  Envs: {envs}")
    print(f"  Algorithms: {algos}")
    print(f"  Seeds: {seeds[0]}..{seeds[-1]} (n={n_seeds})")
    print(f"  Timesteps: {TOTAL_TIMESTEPS:,} per run")
    print(f"  Total runs: {total_runs}")
    est_min = total_runs * 5 / 60  # ~5 min per 300K run
    print(f"  Estimated time: ~{est_min:.0f} min ({est_min/60:.1f} hours)")
    print(f"{'='*70}\n")

    for env_name in envs:
        for algo_name in algos:
            for seed in seeds:
                run_idx += 1
                print(f"\n[{run_idx}/{total_runs}] {env_name} | {algo_name} | seed={seed}")
                run_single(env_name, algo_name, seed)

    # Load all results and compute statistics
    summary = load_results(RESULTS_DIR, envs, algos, seeds)

    print(f"\n{'='*70}")
    print(f"  Statistical Summary (n={n_seeds} seeds each)")
    print(f"{'='*70}")

    comparisons = []
    for env_name in envs:
        print(f"\n  {env_name}:")
        env_data = summary.get(env_name, {})
        for algo in algos:
            d = env_data.get(algo, {})
            if d:
                print(f"    {algo:<40} {d['mean']:8.1f} ± {d['sem']:5.1f} SEM  (n={d['n']})")

        # Primary comparisons
        for algo_a, algo_b in [
            ("Optimal_HCGAE_v2", "Optimal_PPO"),
            ("Optimal_HCGAE_v2", "Standard_PPO"),
            ("Optimal_PPO", "Standard_PPO"),
        ]:
            if algo_a not in env_data or algo_b not in env_data:
                continue
            sa = env_data[algo_a]["seeds"]
            sb = env_data[algo_b]["seeds"]
            stat = compute_statistics(sa, sb, algo_a, algo_b)
            comparisons.append({"env": env_name, "name_a": algo_a, "name_b": algo_b, "stats": stat})
            pct = stat.get("pct_improvement", 0)
            p = stat.get("p_value", 1)
            d = stat.get("cohens_d", 0)
            pwr = stat.get("power_estimate", float("nan"))
            sig = "* p<0.05" if p < 0.05 else ("~ p<0.10" if p < 0.10 else "ns")
            print(f"    {algo_a:<35} vs {algo_b:<20}: {pct:+.1f}% "
                  f"(p={p:.3f} {sig}, d={d:+.2f}, power={pwr:.2f})")

    # Save
    out = {"summary": summary, "comparisons": comparisons,
           "meta": {"envs": envs, "algos": algos, "n_seeds": n_seeds, "seeds": seeds,
                    "total_timesteps": TOTAL_TIMESTEPS,
                    "total_time_s": time.time()-t0,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}}
    out_path = RESULTS_DIR / f"highpower_summary_n{n_seeds}.json"
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results: {out_path}")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")
    return out


if __name__ == "__main__":
    main()

