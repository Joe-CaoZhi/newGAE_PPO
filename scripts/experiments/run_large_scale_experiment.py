#!/usr/bin/env python3
"""
Large-Scale HCGAE Experiment: 8 Environments, 12 Seeds, 1M Steps
=================================================================

Environment selection rationale (SCR = |Bias[V]| / Std[G]):
─────────────────────────────────────────────────────────────
TIER 1 - Expected SOTA (high SCR, slow Critic convergence):
  Hopper-v4               CV=1.15, episodic sparse,  SCR >> 1
  Walker2d-v4             CV=4.40, episodic HV,      SCR > 1 (with boundary correction)
  InvertedDoublePendulum  CV~0.5,  sparse control,   SCR >> 1

TIER 2 - Expected neutral/positive (EV-rate gate critical):
  HalfCheetah-v4          CV=2.80, dense smooth,     SCR < 1 → EV-rate gate fires

TIER 3 - Boundary characterisation (low SCR, include for theory validation):
  Ant-v4                  CV~2.1,  neg_rate=100%,    SCR ≈ 0.06 → theoretical boundary
  Swimmer-v4              low-dim, dense, fast EV    SCR low, but gate compensates
  Humanoid-v4             376-dim, high variance     similar to Ant (large env)
  HumanoidStandup-v4      no termination, prog. rew  SCR uncertain

Mathematical framework:
  α*(SCR) = SCR² / (1 + SCR²)   [MSE-optimal blending coefficient]
  v2 gates: EV-level + cosine decay + EV-rate
  v5 adds:  VW/SNR gate + SCR-α* cap  [handles Ant-v4 / Humanoid class]

Statistical design:
  n = 12 seeds  →  80% power at d ≥ 0.50 (Mann-Whitney)
  1M steps      →  full convergence for episodic and dense tasks
  Deterministic eval (policy mean) every 20,480 steps
"""

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

# ─────────────────────────────────────────────────────────────────────────────
# Experiment configuration
# ─────────────────────────────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 1_000_000
EVAL_FREQ       = 20_480
N_EVAL_EPISODES = 10
N_SEEDS         = 12
SEEDS           = list(range(N_SEEDS))

RESULTS_DIR = Path("results/LargeScaleExperiment")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Per-environment algorithm selection based on empirical best-performing variant:
# ─────────────────────────────────────────────────────────────────────────────
# Environment      Best Known HCGAE    Why this variant?
# ─────────────────────────────────────────────────────────────────────────────
# Hopper-v4        HCGAE_v4 (+71.1%)   SCR cap handles bias-variance tradeoff
# Walker2d-v4      HCGAE_v4 (+12.7%)   Same as Hopper, moderate improvement
# HalfCheetah-v4   HCGAE_v4 (+5.8%)    SCR cap critical; v2/v5 boundary prior HURTS
# Ant-v4           HCGAE_v2_NoBdry     v3/v5 mechanisms over-suppress; needs aligned run
# Swimmer-v4       HCGAE_v6 (new)      Dense, no-failure → v6's norm-boundary helps
# ─────────────────────────────────────────────────────────────────────────────
#
# Key insight: HalfCheetah shows v5's boundary prior HURTS (-20% for v2 vs +5.8% for v4)
# v6 adds variance-normalized boundary + relative EV gate on top of v4 to fix HC.
#
# Experiment design:
#   - Baseline: Optimal_PPO (always included)
#   - Best known: The variant that empirically outperformed others
#   - v6 candidate: New variant for dense-reward environments (HC, Swimmer)
#   - Aligned comparison: For Ant-v4, run v2_NoBdry and v5 on same seeds

EXPERIMENT_PLAN = [
    # ── TIER 1: Strong SOTA ───────────────────────────────────
    ("Hopper-v4",                "Optimal_PPO",     SEEDS),
    ("Hopper-v4",                "Optimal_HCGAE_Bayesian", SEEDS),
    ("Walker2d-v4",              "Optimal_PPO",     SEEDS),
    ("Walker2d-v4",              "Optimal_HCGAE_Bayesian", SEEDS),

    # ── TIER 2: HalfCheetah ─────────────────────────────────
    ("HalfCheetah-v4",           "Optimal_PPO",     SEEDS),
    ("HalfCheetah-v4",           "Optimal_HCGAE_Bayesian", SEEDS),

    # ── TIER 3: Ant-v4 ─────────────────────────
    ("Ant-v4",                   "Optimal_PPO",     SEEDS),
    ("Ant-v4",                   "Optimal_HCGAE_Bayesian", SEEDS),

    # ── TIER 4: Swimmer ───────────────────────
    ("Swimmer-v4",               "Optimal_PPO",     SEEDS),
    ("Swimmer-v4",               "Optimal_HCGAE_Bayesian", SEEDS),
]

ALL_ENVS = list(dict.fromkeys(e for e, _, _ in EXPERIMENT_PLAN))
ALL_ALGOS_FOR_ENV = {}
for env, algo, _ in EXPERIMENT_PLAN:
    ALL_ALGOS_FOR_ENV.setdefault(env, [])
    if algo not in ALL_ALGOS_FOR_ENV[env]:
        ALL_ALGOS_FOR_ENV[env].append(algo)

OPTIMAL_PPO_KWARGS = dict(
    hidden_dim=256, lr=3e-4, gamma=0.99, lam=0.95, eps_clip=0.2,
    n_epochs=10, batch_size=64, n_steps=2048, ent_coef=0.0,
    vf_coef=0.5, max_grad_norm=0.5,
    use_obs_norm=True, use_adv_norm=True, use_lr_anneal=True,
    use_vclip=False, device="cpu",
)


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_policy(agent, eval_env, n_episodes=N_EVAL_EPISODES):
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
                dist = agent.actor.forward(obs_t)
                if agent.continuous:
                    action_np = dist.mean.squeeze(0).detach().cpu().numpy()
                else:
                    action_np = int(dist.probs.argmax(dim=-1).squeeze(0).detach().cpu().numpy())
            obs, r, terminated, truncated, _ = eval_env.step(action_np)
            if hasattr(agent, 'normalize_obs'):
                obs = agent.normalize_obs(obs)
            total_reward += r
            done = terminated or truncated
        rewards.append(total_reward)
    return float(np.mean(rewards))


# ─────────────────────────────────────────────────────────────────────────────
# Single run
# ─────────────────────────────────────────────────────────────────────────────
def run_single(env_name, algo_name, seed,
               total_timesteps=TOTAL_TIMESTEPS,
               results_dir=RESULTS_DIR,
               skip_existing=True):
    save_dir = str(results_dir / env_name / algo_name)
    os.makedirs(save_dir, exist_ok=True)
    out_path = Path(save_dir) / f"{algo_name}_s{seed}.json"

    if skip_existing and out_path.exists():
        print(f"    [SKIP] {env_name}/{algo_name}/s{seed}")
        return json.load(open(out_path))

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 100000)

    kw = dict(**OPTIMAL_PPO_KWARGS, save_dir=save_dir)
    agent = build_optimal_agent(algo_name, env, name=f"{algo_name}_s{seed}", **kw)

    eval_rewards, eval_steps = [], []
    episode_rewards = []

    obs, _ = env.reset()
    if hasattr(agent, 'update_obs_rms'):
        agent.update_obs_rms(obs)
    if hasattr(agent, 'normalize_obs'):
        obs = agent.normalize_obs(obs)

    ep_reward, ep_length = 0.0, 0
    total_steps, last_eval_step = 0, 0
    t0 = time.time()

    while total_steps < total_timesteps:
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
            next_obs_norm = (agent.normalize_obs(next_obs)
                             if hasattr(agent, 'normalize_obs') else next_obs)

            ep_reward += reward
            ep_length += 1

            agent.buffer.add(obs, action_np, float(reward), float(terminated),
                             log_prob.item(), value.item())
            obs = next_obs_norm
            total_steps += 1

            if terminated or truncated:
                episode_rewards.append(ep_reward)
                ep_reward, ep_length = 0.0, 0
                next_obs2, _ = env.reset()
                if hasattr(agent, 'update_obs_rms'):
                    agent.update_obs_rms(next_obs2)
                obs = (agent.normalize_obs(next_obs2)
                       if hasattr(agent, 'normalize_obs') else next_obs2)

            if total_steps - last_eval_step >= EVAL_FREQ:
                eval_r = evaluate_policy(agent, eval_env, N_EVAL_EPISODES)
                eval_rewards.append(eval_r)
                eval_steps.append(total_steps)
                last_eval_step = total_steps
                if total_steps % 200_000 < EVAL_FREQ:
                    elapsed = time.time() - t0
                    print(f"      [{total_steps//1000}K/{total_timesteps//1000}K] "
                          f"eval={eval_r:.1f} ({elapsed:.0f}s)")

        with torch.no_grad():
            last_val = agent.critic(torch.FloatTensor(obs).unsqueeze(0)).item()
        agent._total_timesteps = total_timesteps
        agent.total_steps = total_steps

        if hasattr(agent, 'compute_hindsight_gae'):
            agent.compute_hindsight_gae(last_val)
        else:
            agent.compute_gae(last_val)
        agent.update()

    elapsed = time.time() - t0
    n_final = 10
    final_mean = (float(np.mean(eval_rewards[-n_final:])) if len(eval_rewards) >= n_final
                  else float(np.mean(eval_rewards)) if eval_rewards else 0.0)
    max_reward = float(max(eval_rewards)) if eval_rewards else 0.0
    best10 = (float(np.mean(sorted(eval_rewards)[-10:])) if len(eval_rewards) >= 10
              else final_mean)

    result = {
        "env": env_name, "agent": algo_name, "seed": seed,
        "config": {"hidden_dim": 256, "obs_norm": True, "adv_norm": True,
                   "lr_anneal": True, "eval_mode": "deterministic_mean",
                   "n_seeds": N_SEEDS, "total_timesteps": total_timesteps},
        "total_steps": total_steps,
        "final_reward": final_mean, "max_reward": max_reward, "best10_mean": best10,
        "eval_rewards": eval_rewards, "eval_steps": eval_steps,
        "episode_rewards": episode_rewards[-200:],
        "elapsed_s": elapsed,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    env.close()
    eval_env.close()
    print(f"    Done: {env_name}/{algo_name}/s{seed} → "
          f"final10={final_mean:.1f} max={max_reward:.1f} ({elapsed:.0f}s)")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────
def load_seed_finals(env_name, algo_name, seeds, results_dir, n_final=10):
    vals = []
    for seed in seeds:
        fp = results_dir / env_name / algo_name / f"{algo_name}_s{seed}.json"
        if fp.exists():
            try:
                d = json.load(open(fp))
                er = d.get("eval_rewards", [])
                if er:
                    v = (float(np.mean(er[-n_final:])) if len(er) >= n_final
                         else float(np.mean(er)))
                    vals.append(v)
            except Exception:
                pass
    return vals


def compute_stats(method_vals, baseline_vals):
    from scipy import stats as sc
    mv, bv = np.array(method_vals), np.array(baseline_vals)
    if len(mv) < 2 or len(bv) < 2:
        return None
    stat, pval = sc.mannwhitneyu(mv, bv, alternative='two-sided')
    pooled = np.sqrt((mv.std()**2 + bv.std()**2) / 2.0)
    d = (mv.mean() - bv.mean()) / (pooled + 1e-8)
    delta_pct = (mv.mean() - bv.mean()) / (abs(bv.mean()) + 1e-8) * 100
    np.random.seed(42)
    boot = [np.random.choice(mv, len(mv), True).mean() -
            np.random.choice(bv, len(bv), True).mean() for _ in range(10000)]
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n_m": len(mv), "n_b": len(bv),
        "mean_m": float(mv.mean()), "sem_m": float(mv.std()/len(mv)**0.5),
        "mean_b": float(bv.mean()), "sem_b": float(bv.std()/len(bv)**0.5),
        "delta_pct": float(delta_pct),
        "mw_p": float(pval), "cohens_d": float(d),
        "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
        "sig": bool(pval < 0.05),
    }


def compute_and_print_summary(results_dir=RESULTS_DIR, seeds=SEEDS):
    summary = {}
    stats_table = {}
    for env_name, algos in ALL_ALGOS_FOR_ENV.items():
        summary[env_name] = {}
        ppo_vals = load_seed_finals(env_name, "Optimal_PPO", seeds, results_dir)
        if ppo_vals:
            summary[env_name]["Optimal_PPO"] = {
                "mean": float(np.mean(ppo_vals)),
                "sem": float(np.std(ppo_vals)/len(ppo_vals)**0.5),
                "n": len(ppo_vals), "vals": ppo_vals
            }
        for algo in algos:
            if algo == "Optimal_PPO":
                continue
            vals = load_seed_finals(env_name, algo, seeds, results_dir)
            if vals:
                summary[env_name][algo] = {
                    "mean": float(np.mean(vals)),
                    "sem": float(np.std(vals)/len(vals)**0.5),
                    "n": len(vals), "vals": vals
                }
                if ppo_vals:
                    st = compute_stats(vals, ppo_vals)
                    if st:
                        stats_table[f"{env_name}|{algo}"] = st

    # Print table
    W = 115
    print("\n" + "=" * W)
    print(f"  LARGE-SCALE RESULTS  (n={N_SEEDS} seeds, 1M steps, deterministic eval)")
    print("=" * W)
    print(f"{'Env':<28} {'Method':<25} {'Mean±SEM':>15} {'Δ%':>8} {'p':>9} {'d':>7} {'95% CI':>18} {'n':>4}")
    print("-" * W)

    for env_name in ALL_ENVS:
        env_data = summary.get(env_name, {})
        for algo in ALL_ALGOS_FOR_ENV.get(env_name, []):
            if algo not in env_data:
                print(f"{env_name:<28} {algo:<25} {'pending':>15}")
                continue
            info = env_data[algo]
            ms = f"{info['mean']:.1f}±{info['sem']:.1f}"
            if algo == "Optimal_PPO":
                print(f"{env_name:<28} {algo:<25} {ms:>15} {'baseline':>8} {'—':>9} {'—':>7} {'—':>18} {info['n']:>4}")
            else:
                key = f"{env_name}|{algo}"
                st = stats_table.get(key, {})
                dp = f"{st['delta_pct']:+.1f}%" if st else "—"
                pv = st.get("mw_p", 1.0)
                ps = f"{pv:.3f}" + ("***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "") if st else "—"
                ds = f"{st['cohens_d']:+.2f}" if st else "—"
                ci = f"[{st['ci_lo']:.0f},{st['ci_hi']:.0f}]" if st else "—"
                mark = " ◄" if "Bayesian" in algo else ""
                print(f"{env_name:<28} {algo:<25} {ms:>15} {dp:>8} {ps:>9} {ds:>7} {ci:>18} {info['n']:>4}{mark}")

    print("=" * W)
    print("  * p<0.05  ** p<0.01  *** p<0.001  (Mann-Whitney U, two-sided, n=10000 bootstrap)")
    return summary, stats_table


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Large-Scale HCGAE (8 envs, 12 seeds, 1M steps)")
    parser.add_argument("--envs", nargs="+", default=None)
    parser.add_argument("--algos", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], default=None,
                        help="Run only specified tier (1=SOTA, 2=neutral, 3=boundary)")
    parser.add_argument("--no-skip", action="store_true")
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--algo", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.summary_only:
        seeds = args.seeds if args.seeds else SEEDS
        summary, stats = compute_and_print_summary(RESULTS_DIR, seeds)
        sp = RESULTS_DIR / "large_scale_summary.json"
        with open(sp, "w") as f:
            json.dump({"summary": {k: {ak: {ik: iv for ik, iv in av.items()
                                           if ik != "vals"}
                                       for ak, av in v.items()}
                                   for k, v in summary.items()},
                       "stats": stats}, f, indent=2)
        print(f"\nSaved: {sp}")
        return

    # Build run list
    tier_envs = {
        1: ["Hopper-v4", "Walker2d-v4", "InvertedDoublePendulum-v4"],
        2: ["HalfCheetah-v4"],
        3: ["Ant-v4", "Swimmer-v4", "Humanoid-v4", "HumanoidStandup-v4"],
    }

    if args.env:
        filter_envs = [args.env]
    elif args.tier:
        filter_envs = tier_envs[args.tier]
    elif args.envs:
        filter_envs = args.envs
    else:
        filter_envs = None

    filter_algos = [args.algo] if args.algo else (args.algos if args.algos else None)
    filter_seeds = [args.seed] if args.seed is not None else args.seeds

    # Build run list from EXPERIMENT_PLAN
    run_list = []
    for env, algo, seeds in EXPERIMENT_PLAN:
        if filter_envs and env not in filter_envs:
            continue
        if filter_algos and algo not in filter_algos:
            continue
        for s in seeds:
            if s in filter_seeds:
                run_list.append((env, algo, s))

    total = len(run_list)
    print(f"\n{'='*80}")
    print(f"  Large-Scale HCGAE Experiment")
    print(f"  Steps: {args.timesteps:,}   Seeds: {filter_seeds}   Total runs: {total}")
    print(f"{'='*80}\n")

    t_global = time.time()
    for i, (env_name, algo_name, seed) in enumerate(run_list, 1):
        print(f"\n[{i}/{total}] {env_name} | {algo_name} | seed={seed}")
        try:
            run_single(env_name, algo_name, seed,
                       total_timesteps=args.timesteps,
                       results_dir=RESULTS_DIR,
                       skip_existing=not args.no_skip)
        except Exception as e:
            print(f"    [ERROR] {e}")
            import traceback; traceback.print_exc()

    # Final summary
    elapsed_total = time.time() - t_global
    print(f"\n\nTotal elapsed: {elapsed_total:.0f}s ({elapsed_total/3600:.2f}h)")
    summary, stats = compute_and_print_summary(RESULTS_DIR, filter_seeds)
    sp = RESULTS_DIR / "large_scale_summary.json"
    with open(sp, "w") as f:
        json.dump({"summary": {k: {ak: {ik: iv for ik, iv in av.items()
                                       if ik != "vals"}
                                   for ak, av in v.items()}
                               for k, v in summary.items()},
                   "stats": stats}, f, indent=2)
    print(f"\nSaved: {sp}")


if __name__ == "__main__":
    main()

