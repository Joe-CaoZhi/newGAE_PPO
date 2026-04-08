"""
ADVANCE-PPO 消融实验
====================
对比 8 个变体 × 3 个环境（Hopper, Walker2d, HalfCheetah）× 3 seeds

目标：证明三项改进（A: 自适应信任域, B: Epoch-Decay IS, C: EV-Gated Critic）
      各自有效，组合更强，且在多环境中泛化。

实验设计：
  对照组：HCGAE_Imp12（HCGAE 的最佳变体，作为新实验的 Base）
  实验组：ADVANCE_{Base, ImpA, ImpB, ImpC, ImpAB, ImpAC, ImpBC, Full}

注意：ADVANCE_Base = HCGAE_Imp12 风格的 HCGAE GAE + 无新改进
     这样可以清晰看到每个新改进的增量贡献
"""
import os, sys, time, json
import numpy as np
import gymnasium as gym

sys.path.insert(0, os.path.dirname(__file__))
from gae_experiments.agents.advance_ppo import build_advance_agent, get_all_advance_variant_names
from gae_experiments.agents.base_ppo import BasePPO

SAVE_ROOT = "results/Advance-Ablation"
os.makedirs(SAVE_ROOT, exist_ok=True)

# ─────────────────────────────────────────────
# 实验配置
# ─────────────────────────────────────────────
ENVS = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]
SEEDS = [42, 123, 456]  # 每环境 3 seeds（快速验证）
TOTAL_TIMESTEPS = 300_000
EVAL_FREQ = 10_240
N_EVAL_EP = 10

COMMON_KWARGS = dict(
    hidden_dim=64,
    lr_actor=3e-4,
    lr_critic=1e-3,
    gamma=0.99,
    lam=0.95,
    eps_clip=0.2,
    n_epochs=10,
    batch_size=64,
    n_steps=2048,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    device="cpu",
    use_hcgae=True,  # 所有 ADVANCE 变体都基于 HCGAE GAE 改进
)

ALL_VARIANTS = get_all_advance_variant_names()


def run_one(env_id: str, variant: str, seed: int, save_dir: str) -> dict:
    """运行单个实验"""
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)

    env      = gym.make(env_id)
    eval_env = gym.make(env_id)
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 10000)

    os.makedirs(save_dir, exist_ok=True)
    agent = build_advance_agent(
        variant, env,
        name=f"{variant}_s{seed}",
        save_dir=save_dir,
        **COMMON_KWARGS,
    )

    t0 = time.time()
    logger = agent.train(
        total_timesteps=TOTAL_TIMESTEPS,
        eval_env=eval_env,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EP,
        verbose=False,
    )
    elapsed = time.time() - t0

    evals = logger.eval_rewards if logger.eval_rewards else [0.0]
    result = {
        "env": env_id,
        "variant": variant,
        "seed": seed,
        "final_reward": float(np.mean(evals[-3:])) if len(evals) >= 3 else float(evals[-1]),
        "best_reward": float(max(evals)),
        "mean_reward": float(np.mean(evals)),
        "all_eval_rewards": [float(x) for x in evals],
        "eval_steps": [int(x) for x in logger.eval_steps] if hasattr(logger, "eval_steps") else [],
        "elapsed_s": round(elapsed, 1),
    }
    env.close()
    eval_env.close()
    return result


def main():
    total_runs = len(ENVS) * len(SEEDS) * len(ALL_VARIANTS)
    run_idx = 0
    all_results = []

    t_total = time.time()
    print(f"\n{'='*75}")
    print(f"  ADVANCE-PPO 消融实验")
    print(f"  {len(ALL_VARIANTS)} 变体 × {len(ENVS)} 环境 × {len(SEEDS)} seeds = {total_runs} 次训练")
    print(f"  变体：{ALL_VARIANTS}")
    print(f"{'='*75}\n")

    for env_id in ENVS:
        env_save_dir = os.path.join(SAVE_ROOT, env_id)
        os.makedirs(env_save_dir, exist_ok=True)
        print(f"\n  ━━━ 环境: {env_id} ━━━")

        for variant in ALL_VARIANTS:
            seed_rewards = []
            for seed in SEEDS:
                run_idx += 1
                print(f"  [{run_idx:3d}/{total_runs}] {env_id}  {variant:<18}  seed={seed}  ", end="", flush=True)

                result_path = os.path.join(env_save_dir, f"{variant}_s{seed}.json")
                if os.path.exists(result_path):
                    with open(result_path) as f:
                        result = json.load(f)
                    print(f"(cached) final={result['final_reward']:.1f}")
                else:
                    result = run_one(env_id, variant, seed, env_save_dir)
                    with open(result_path, "w") as f:
                        json.dump(result, f, indent=2)
                    print(f"final={result['final_reward']:.1f}  best={result['best_reward']:.1f}  {result['elapsed_s']:.0f}s")

                seed_rewards.append(result["final_reward"])
                all_results.append(result)

            print(f"    → {variant}: mean={np.mean(seed_rewards):.1f} ± {np.std(seed_rewards):.1f}")

        # 环境汇总
        env_summary = {}
        for variant in ALL_VARIANTS:
            v_data = [r for r in all_results if r["env"] == env_id and r["variant"] == variant]
            rewards = [r["final_reward"] for r in v_data]
            env_summary[variant] = {
                "mean": float(np.mean(rewards)), "std": float(np.std(rewards)),
                "min": float(np.min(rewards)), "max": float(np.max(rewards)),
                "seeds": rewards,
            }
        with open(os.path.join(env_save_dir, "summary.json"), "w") as f:
            json.dump(env_summary, f, indent=2)

        print(f"\n  {env_id} 结果汇总:")
        print(f"  {'变体':<20} {'均值':>10} {'标准差':>8} {'最大值':>10}")
        print(f"  {'-'*55}")
        base_mean = env_summary.get("ADVANCE_Base", {}).get("mean", 1.0)
        for variant, stats in env_summary.items():
            gain = (stats['mean'] / (base_mean + 1e-8) - 1) * 100
            print(f"  {variant:<20} {stats['mean']:>10.1f} {stats['std']:>8.1f} {stats['max']:>10.1f}  ({gain:+.1f}% vs Base)")

    # 全局汇总
    global_summary = {"configs": {
        "envs": ENVS, "seeds": SEEDS, "variants": ALL_VARIANTS,
        "total_timesteps": TOTAL_TIMESTEPS,
    }, "results": {}}

    for env_id in ENVS:
        global_summary["results"][env_id] = {}
        for variant in ALL_VARIANTS:
            v_data = [r for r in all_results if r["env"] == env_id and r["variant"] == variant]
            rewards = [r["final_reward"] for r in v_data]
            if rewards:
                global_summary["results"][env_id][variant] = {
                    "mean": float(np.mean(rewards)),
                    "std": float(np.std(rewards)),
                    "seeds": rewards,
                }

    with open(os.path.join(SAVE_ROOT, "global_summary.json"), "w") as f:
        json.dump(global_summary, f, indent=2)

    total_elapsed = time.time() - t_total
    print(f"\n\n{'='*75}")
    print(f"  全部完成！总耗时: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print(f"  结果保存于: {SAVE_ROOT}/")
    print(f"{'='*75}\n")

    # 全局结果表
    print("  === ADVANCE-PPO 全局消融结果（均值 ± 标准差）===")
    header = f"  {'变体':<22}" + "".join(f"{e:>22}" for e in ENVS)
    print(header)
    print(f"  {'-'*90}")
    for variant in ALL_VARIANTS:
        row = f"  {variant:<22}"
        for env_id in ENVS:
            stats = global_summary["results"].get(env_id, {}).get(variant, {})
            if stats:
                row += f"  {stats['mean']:>8.0f}±{stats['std']:>5.0f}       "
            else:
                row += f"  {'N/A':>16}       "
        print(row)


if __name__ == "__main__":
    main()

