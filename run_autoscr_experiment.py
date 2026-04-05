"""
AutoSCR Experiment: Optimal HCGAE v2 + Automatic SCR-Based Mode Detection
=========================================================================

This experiment compares:
1. Optimal_PPO                  - Strong baseline (observation normalization etc.)
2. Optimal_HCGAE_v2             - Our paper's main result (EV rate gate + boundary correction)
3. Optimal_HCGAE_v2_AutoSCR    - Future Work 1: automatic SCR mode detector added on top

Purpose: Demonstrate that the AutoSCR estimator provides additional robustness
across episodic (Hopper, Walker2d) and dense-reward (HalfCheetah) environments.

Protocol (identical to ICMLExperiment):
- 3 environments: Hopper-v4, Walker2d-v4, HalfCheetah-v4
- 5 seeds per algorithm (seeds 0..4, consistent with ICMLExperiment)
- 500,000 training steps
- Optimal PPO baseline (obs normalization, adv normalization, LR annealing)
- Eval every 10,240 steps, 10 deterministic episodes

Results saved to: results/AutoSCRExperiment/{env}/{algo}/{algo}_s{seed}.json
Summary:          results/AutoSCRExperiment/autoscr_summary.json
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

# ──────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────
ENVS = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]
ALGOS = ["Optimal_PPO", "Optimal_HCGAE_v2", "Optimal_HCGAE_v2_AutoSCR"]
TOTAL_TIMESTEPS = 500_000
EVAL_FREQ = 10_240
N_EVAL_EPISODES = 10
N_SEEDS = 5
SEEDS = list(range(N_SEEDS))  # 0..4

RESULTS_DIR = Path("results/AutoSCRExperiment")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Optimal PPO hyperparameters (identical to ICMLExperiment)
OPTIMAL_PPO_KWARGS = dict(
    hidden_dim=64,
    lr=3e-4,
    gamma=0.99,
    lam=0.95,
    eps_clip=0.2,
    n_epochs=10,
    batch_size=64,
    n_steps=2048,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    use_obs_norm=True,
    use_adv_norm=True,
    use_lr_anneal=True,
    use_vclip=False,
    device="cpu",
)


# ──────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────
def evaluate_policy(agent, eval_env, n_episodes=10):
    """Deterministic policy evaluation."""
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
                if agent.continuous:
                    action_np = action.squeeze(0).cpu().numpy()
                else:
                    action_np = int(action.squeeze(0).cpu().numpy())
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
    """Run one (env, algo, seed) combination (same training loop as run_icml_experiment.py)."""
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

    kw = dict(**OPTIMAL_PPO_KWARGS, save_dir=save_dir)
    agent = build_optimal_agent(algo_name, env, name=f"{algo_name}_s{seed}", **kw)

    # Training loop (manual, consistent with run_icml_experiment.py)
    eval_rewards = []
    eval_steps = []
    episode_rewards = []

    obs, _ = env.reset()
    if hasattr(agent, 'update_obs_rms'):
        agent.update_obs_rms(obs)
    if hasattr(agent, 'normalize_obs'):
        obs = agent.normalize_obs(obs)

    ep_reward = 0.0
    ep_length = 0
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
            if hasattr(agent, 'normalize_obs'):
                next_obs_norm = agent.normalize_obs(next_obs)
            else:
                next_obs_norm = next_obs

            ep_reward += reward
            ep_length += 1

            agent.buffer.add(
                obs, action_np, float(reward), float(terminated),
                log_prob.item(), value.item()
            )
            obs = next_obs_norm
            total_steps += 1

            if terminated or truncated:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                ep_length = 0
                next_obs, _ = env.reset()
                if hasattr(agent, 'update_obs_rms'):
                    agent.update_obs_rms(next_obs)
                if hasattr(agent, 'normalize_obs'):
                    obs = agent.normalize_obs(next_obs)
                else:
                    obs = next_obs

            if total_steps - last_eval_step >= EVAL_FREQ:
                eval_r = evaluate_policy(agent, eval_env, N_EVAL_EPISODES)
                eval_rewards.append(eval_r)
                eval_steps.append(total_steps)
                last_eval_step = total_steps

        with torch.no_grad():
            last_obs_t = torch.FloatTensor(obs).unsqueeze(0)
            last_val = agent.critic(last_obs_t).item()

        agent._total_timesteps = TOTAL_TIMESTEPS
        agent.total_steps = total_steps

        if hasattr(agent, 'compute_hindsight_gae'):
            agent.compute_hindsight_gae(last_val)
        else:
            agent.compute_gae(last_val)

        agent.update()

    elapsed = time.time() - t0
    final_mean = float(np.mean(eval_rewards[-5:])) if len(eval_rewards) >= 5 else (
        float(np.mean(eval_rewards)) if eval_rewards else 0.0
    )

    # Collect AutoSCR diagnostics
    scr_info = {}
    if hasattr(agent, '_scr_history') and agent._scr_history:
        scr_info = {
            "scr_ema_final": float(getattr(agent, '_scr_ema', 0.0)),
            "scr_history_mean": float(np.mean(agent._scr_history)),
            "scr_history_std": float(np.std(agent._scr_history)),
            "scr_scale_mean": float(np.mean(agent._scr_scale_history)),
        }

    result = {
        "env": env_name,
        "agent": algo_name,
        "seed": seed,
        "total_steps": total_steps,
        "final_reward": final_mean,
        "eval_rewards": eval_rewards,
        "eval_steps": eval_steps,
        "episode_rewards": episode_rewards,
        "elapsed_s": elapsed,
        "ev_final": float(getattr(agent, '_ev_ema', 0.0)),
        **scr_info,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    env.close()
    eval_env.close()
    print(f"    Done: {algo_name} {env_name} s{seed} -> final={final_mean:.1f} ({elapsed:.0f}s)")
    return result


# ──────────────────────────────────────────────────────────────────────────
# Summary & Statistics
# ──────────────────────────────────────────────────────────────────────────
def compute_summary(results_dir=RESULTS_DIR, envs=ENVS, algos=ALGOS, seeds=SEEDS):
    summary = {}
    for env_name in envs:
        summary[env_name] = {}
        for algo in algos:
            seed_means = []
            for seed in seeds:
                fp = results_dir / env_name / algo / f"{algo}_s{seed}.json"
                if fp.exists():
                    try:
                        d = json.load(open(fp))
                        er = d.get("eval_rewards", [])
                        if er:
                            val = float(np.mean(er[-5:])) if len(er) >= 5 else float(np.mean(er))
                            seed_means.append(val)
                    except Exception:
                        pass
            if seed_means:
                summary[env_name][algo] = {
                    "mean": float(np.mean(seed_means)),
                    "std": float(np.std(seed_means)),
                    "sem": float(np.std(seed_means) / max(len(seed_means)**0.5, 1)),
                    "n": len(seed_means),
                    "seeds": seed_means,
                }
    return summary


def compute_statistics(scores_a, scores_b, name_a, name_b):
    """Mann-Whitney U test + Cohen's d + Bootstrap CI."""
    try:
        from scipy import stats
    except ImportError:
        return {"mean_a": float(np.mean(scores_a)), "mean_b": float(np.mean(scores_b))}

    if len(scores_a) < 2 or len(scores_b) < 2:
        return {"mean_a": float(np.mean(scores_a)), "mean_b": float(np.mean(scores_b))}

    n_a, n_b = len(scores_a), len(scores_b)
    mean_a, mean_b = np.mean(scores_a), np.mean(scores_b)
    std_a = float(np.std(scores_a, ddof=1))
    std_b = float(np.std(scores_b, ddof=1))

    u_stat, p_value = stats.mannwhitneyu(scores_a, scores_b, alternative='two-sided')
    pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
    cohens_d = (mean_a - mean_b) / (pooled_std + 1e-8)

    rng = np.random.default_rng(42)
    diff_boot = [float(np.mean(rng.choice(scores_a, n_a, True)) - np.mean(rng.choice(scores_b, n_b, True)))
                 for _ in range(2000)]
    ci_low = float(np.percentile(diff_boot, 2.5))
    ci_high = float(np.percentile(diff_boot, 97.5))
    pct = (mean_a - mean_b) / (abs(mean_b) + 1e-8) * 100

    return {
        f"mean_{name_a}": float(mean_a), f"mean_{name_b}": float(mean_b),
        f"std_{name_a}": float(std_a), f"std_{name_b}": float(std_b),
        "mann_whitney_u": float(u_stat), "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "ci_low": ci_low, "ci_high": ci_high,
        "pct_improvement": float(pct),
        "significant_p05": bool(p_value < 0.05),
    }


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", default=None)
    parser.add_argument("--algos", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    envs = args.envs if args.envs else ENVS
    algos = args.algos if args.algos else ALGOS
    seeds = list(range(args.seeds))

    total_runs = len(envs) * len(algos) * len(seeds)
    run_idx = 0
    t0 = time.time()

    print(f"\n{'='*70}")
    print(f"  AutoSCR Experiment (Future Work 1)")
    print(f"  Envs: {envs}")
    print(f"  Algorithms: {algos}")
    print(f"  Seeds: {seeds} (n={len(seeds)})")
    print(f"  Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Total runs: {total_runs}")
    est_min = total_runs * 13 / 60
    print(f"  Estimated time: ~{est_min:.0f} min ({est_min/60:.1f} hours)")
    print(f"{'='*70}\n")

    for env_name in envs:
        for algo_name in algos:
            for seed in seeds:
                run_idx += 1
                print(f"\n[{run_idx}/{total_runs}] {env_name} | {algo_name} | seed={seed}")
                t1 = time.time()
                result = run_single(env_name, algo_name, seed)
                print(f"  -> elapsed: {time.time()-t1:.0f}s, final={result.get('final_reward', 0):.1f}")

    # Compute summary
    summary = compute_summary(RESULTS_DIR, envs, algos, seeds)

    print(f"\n{'='*70}")
    print(f"  Results Summary")
    print(f"{'='*70}")

    comparisons = []
    for env_name in envs:
        print(f"\n  {env_name}:")
        env_data = summary.get(env_name, {})
        for algo in algos:
            data = env_data.get(algo, {})
            if data:
                print(f"    {algo:<40} {data['mean']:8.1f} ± {data['std']:6.1f}  (n={data['n']})")

        # Key comparisons
        for algo_a, algo_b in [
            ("Optimal_HCGAE_v2", "Optimal_PPO"),
            ("Optimal_HCGAE_v2_AutoSCR", "Optimal_PPO"),
            ("Optimal_HCGAE_v2_AutoSCR", "Optimal_HCGAE_v2"),
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
            sig = "* p<0.05" if p < 0.05 else ("~ p<0.10" if p < 0.10 else "ns")
            print(f"    {algo_a} vs {algo_b}: {pct:+.1f}% (p={p:.3f} {sig}, d={d:+.2f})")

    # Save
    final = {"summary": summary, "comparisons": comparisons,
             "meta": {"envs": envs, "algos": algos, "n_seeds": len(seeds),
                      "total_time_s": time.time()-t0,
                      "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}}
    out_path = RESULTS_DIR / "autoscr_summary.json"
    with open(out_path, 'w') as f:
        json.dump(final, f, indent=2)
    print(f"\n  Results saved: {out_path}")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")
    return final


if __name__ == "__main__":
    main()

