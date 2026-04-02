"""
补充实验：多环境 × 多随机种子 × 多基线 对比
============================================================
目的：为 ICML 投稿提供充分的统计证据

实验设计：
  - 环境：Hopper-v4, Walker2d-v4, HalfCheetah-v4, Ant-v4
  - 算法：Standard_PPO, GAE_Lambda1, HCGAE_Base, HCGAE_Best(Imp12)
  - 随机种子：[42, 123, 456, 789, 1234]（5 seeds）
  - 时间步：300k（所有环境统一）

结果保存：results/MultiEnv/
"""

import os, sys, time, json, copy
import numpy as np
import gymnasium as gym

sys.path.insert(0, os.path.dirname(__file__))
from gae_experiments.agents.base_ppo import BasePPO
from gae_experiments.agents.hindsight_ablation import build_ablation_agent

SAVE_ROOT = "results/MultiEnv"
os.makedirs(SAVE_ROOT, exist_ok=True)

# ─────────────────────────────────────────────
# 实验配置
# ─────────────────────────────────────────────
ENVS = [
    "Hopper-v4",
    "Walker2d-v4",
    "HalfCheetah-v4",
    "Ant-v4",
]
SEEDS = [42, 123, 456, 789, 1234]
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
)


class GAELambda1PPO(BasePPO):
    """
    GAE(λ=1) = MC Returns as advantages, 纯 Monte-Carlo 无偏优势。
    覆盖 compute_gae 使用 λ=1.0。
    """
    NAME = "GAE_Lambda1"

    def compute_gae(self, last_value: float) -> dict:
        self.buffer.compute_standard_gae(last_value, self.gamma, lam=1.0)  # λ=1 → MC return
        T  = self.buffer.pos
        nv = self.buffer._next_values(last_value)
        deltas = self.buffer.rewards[:T] + self.gamma * nv - self.buffer.values[:T]
        autocorr = float(np.corrcoef(deltas[:-1], deltas[1:])[0, 1]) if T > 2 else 0.0
        return {
            "delta_mean": float(deltas.mean()),
            "delta_std" : float(deltas.std()),
            "delta_autocorr": autocorr,
            "adv_mean" : float(self.buffer.advantages.mean()),
            "adv_std"  : float(self.buffer.advantages.std()),
        }


def run_one(env_id, agent_name, seed, save_dir):
    """运行单个实验，返回结果字典"""
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)

    env      = gym.make(env_id)
    eval_env = gym.make(env_id)
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 10000)

    os.makedirs(save_dir, exist_ok=True)

    if agent_name == "Standard_PPO":
        agent = BasePPO(env, save_dir=save_dir, **COMMON_KWARGS)
        agent.NAME = f"Standard_PPO_s{seed}"
    elif agent_name == "GAE_Lambda1":
        agent = GAELambda1PPO(env, save_dir=save_dir, **COMMON_KWARGS)
        agent.NAME = f"GAE_Lambda1_s{seed}"
    elif agent_name == "HCGAE_Base":
        agent = build_ablation_agent(
            "HCGAE_Base", env,
            name=f"HCGAE_Base_s{seed}",
            save_dir=save_dir,
            **COMMON_KWARGS,
        )
    elif agent_name == "HCGAE_Imp12":
        agent = build_ablation_agent(
            "HCGAE_Imp12", env,
            name=f"HCGAE_Imp12_s{seed}",
            save_dir=save_dir,
            **COMMON_KWARGS,
        )
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

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
        "agent": agent_name,
        "seed": seed,
        "final_reward": float(np.mean(evals[-3:])) if len(evals) >= 3 else float(evals[-1]),
        "best_reward": float(max(evals)),
        "mean_reward": float(np.mean(evals)),
        "all_eval_rewards": [float(x) for x in evals],
        "eval_steps": [int(x) for x in logger.eval_steps] if hasattr(logger, 'eval_steps') else [],
        "elapsed_s": round(elapsed, 1),
    }
    env.close()
    eval_env.close()
    return result


def main():
    algorithms = ["Standard_PPO", "GAE_Lambda1", "HCGAE_Base", "HCGAE_Imp12"]
    all_results = []
    total_runs = len(ENVS) * len(SEEDS) * len(algorithms)
    run_idx = 0

    t_total = time.time()
    print(f"\n{'='*70}")
    print(f"  多环境 × 多 Seed 对比实验")
    print(f"  总共 {total_runs} 次训练（{len(ENVS)} 环境 × {len(SEEDS)} seeds × {len(algorithms)} 算法）")
    print(f"{'='*70}\n")

    for env_id in ENVS:
        env_results = []
        print(f"\n  ━━━ 环境: {env_id} ━━━")
        env_save_dir = os.path.join(SAVE_ROOT, env_id)
        os.makedirs(env_save_dir, exist_ok=True)

        for alg in algorithms:
            alg_rewards = []
            for seed in SEEDS:
                run_idx += 1
                print(f"  [{run_idx:3d}/{total_runs}] {env_id}  {alg:<14}  seed={seed}  ", end="", flush=True)

                result_path = os.path.join(env_save_dir, f"{alg}_s{seed}.json")
                if os.path.exists(result_path):
                    with open(result_path) as f:
                        result = json.load(f)
                    print(f"(cached) final={result['final_reward']:.1f}")
                else:
                    result = run_one(env_id, alg, seed, env_save_dir)
                    with open(result_path, "w") as f:
                        json.dump(result, f, indent=2)
                    print(f"final={result['final_reward']:.1f}  best={result['best_reward']:.1f}  {result['elapsed_s']:.0f}s")

                alg_rewards.append(result["final_reward"])
                all_results.append(result)

            print(f"    → {alg}: mean={np.mean(alg_rewards):.1f} ± {np.std(alg_rewards):.1f}")

        # 保存环境汇总
        env_summary = {}
        for alg in algorithms:
            alg_data = [r for r in all_results if r["env"] == env_id and r["agent"] == alg]
            rewards = [r["final_reward"] for r in alg_data]
            env_summary[alg] = {
                "mean": float(np.mean(rewards)),
                "std": float(np.std(rewards)),
                "min": float(np.min(rewards)),
                "max": float(np.max(rewards)),
                "seeds": rewards,
            }
        with open(os.path.join(env_save_dir, "summary.json"), "w") as f:
            json.dump(env_summary, f, indent=2)

        # 打印对比表
        print(f"\n  {'算法':<16} {'均值':>10} {'标准差':>8} {'最大值':>10}")
        print(f"  {'-'*50}")
        for alg, stats in env_summary.items():
            print(f"  {alg:<16} {stats['mean']:>10.1f} {stats['std']:>8.1f} {stats['max']:>10.1f}")

    # 保存全局汇总
    global_summary = {"configs": {
        "envs": ENVS, "seeds": SEEDS, "algorithms": algorithms,
        "total_timesteps": TOTAL_TIMESTEPS,
    }, "results": {}}

    for env_id in ENVS:
        global_summary["results"][env_id] = {}
        for alg in algorithms:
            alg_data = [r for r in all_results if r["env"] == env_id and r["agent"] == alg]
            rewards = [r["final_reward"] for r in alg_data]
            if rewards:
                global_summary["results"][env_id][alg] = {
                    "mean": float(np.mean(rewards)),
                    "std": float(np.std(rewards)),
                    "seeds": rewards,
                }

    with open(os.path.join(SAVE_ROOT, "global_summary.json"), "w") as f:
        json.dump(global_summary, f, indent=2)

    total_elapsed = time.time() - t_total
    print(f"\n\n{'='*70}")
    print(f"  全部完成！总耗时: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print(f"  结果保存于: {SAVE_ROOT}/")
    print(f"{'='*70}\n")

    # 打印全局汇总表
    print("\n  === 全局结果汇总 (均值 ± 标准差) ===")
    header = f"  {'环境':<18}" + "".join(f"{a:>16}" for a in algorithms)
    print(header)
    print(f"  {'-'*80}")
    for env_id in ENVS:
        row = f"  {env_id:<18}"
        for alg in algorithms:
            stats = global_summary["results"].get(env_id, {}).get(alg, {})
            if stats:
                row += f"  {stats['mean']:>6.0f}±{stats['std']:>5.0f}"
            else:
                row += f"  {'N/A':>13}"
        print(row)


if __name__ == "__main__":
    main()

