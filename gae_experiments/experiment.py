"""
实验运行器：管理多个实验的配置、训练和对比
"""
import os
import random
import time
from dataclasses import dataclass, field
from typing import List, Dict

import gymnasium as gym
import numpy as np
import torch

from .agents import (
    BasePPO, ConservativeBootstrapPPO, AdaptiveLambdaPPO,
    ConfidenceWeightedPPO, CombinedPPO,
    HindsightPPO, MultiScalePPO, CausalAttentionPPO,
)
from .utils.logger import MetricLogger
from .utils.visualizer import (
    plot_all_results, plot_learning_curves_single, print_summary_table,
    plot_comprehensive_analysis,
)


@dataclass
class ExperimentConfig:
    """实验配置"""
    env_name: str = "CartPole-v1"
    total_timesteps: int = 200_000
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    lam: float = 0.95
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    eps_clip: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_dim: int = 64
    eval_freq: int = 5000
    n_eval_episodes: int = 10
    seed: int = 42
    device: str = "cpu"
    save_dir: str = "results"
    # 运行的 agent 列表
    agents: List[str] = field(default_factory=lambda: [
        "Standard_GAE",
        "Conservative_Bootstrap_GAE",
        "Adaptive_Lambda_GAE",
        "Confidence_Weighted_GAE",
        "Combined_GAE",
    ])
    verbose: bool = True


def set_seed(seed: int):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(env_name: str, seed: int = 42) -> gym.Env:
    """创建环境并设置种子"""
    env = gym.make(env_name)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def build_agent(agent_name: str, env: gym.Env, config: ExperimentConfig):
    """根据名称构建对应的 Agent"""
    common_kwargs = dict(
        env=env,
        hidden_dim=config.hidden_dim,
        lr_actor=config.lr_actor,
        lr_critic=config.lr_critic,
        gamma=config.gamma,
        eps_clip=config.eps_clip,
        n_epochs=config.n_epochs,
        batch_size=config.batch_size,
        n_steps=config.n_steps,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        device=config.device,
        save_dir=config.save_dir,
    )

    if agent_name == "Standard_GAE":
        return BasePPO(lam=config.lam, **common_kwargs)

    elif agent_name == "Conservative_Bootstrap_GAE":
        return ConservativeBootstrapPPO(lam=config.lam, **common_kwargs)

    elif agent_name == "Adaptive_Lambda_GAE":
        return AdaptiveLambdaPPO(lam_init=config.lam, **common_kwargs)

    elif agent_name == "Confidence_Weighted_GAE":
        return ConfidenceWeightedPPO(lam=config.lam, **common_kwargs)

    elif agent_name == "Combined_GAE":
        return CombinedPPO(**common_kwargs)

    elif agent_name == "Hindsight_GAE":
        return HindsightPPO(lam=config.lam, **common_kwargs)

    elif agent_name == "MultiScale_GAE":
        return MultiScalePPO(lam=config.lam, **common_kwargs)

    elif agent_name == "CausalAttn_GAE":
        return CausalAttentionPPO(lam=config.lam, **common_kwargs)

    else:
        raise ValueError(f"未知的 agent 类型: {agent_name}")


def run_single_experiment(
    agent_name: str,
    config: ExperimentConfig,
    exp_idx: int = 1,
    total_exps: int = 1,
) -> MetricLogger:
    """运行单个实验"""
    set_seed(config.seed)

    train_env = make_env(config.env_name, config.seed)
    eval_env  = make_env(config.env_name, config.seed + 1000)

    print(f"\n{'═'*68}")
    print(f"  实验 [{exp_idx}/{total_exps}]  方法: {agent_name}")
    print(f"  环境: {config.env_name}  |  总步数: {config.total_timesteps:,}  |  设备: {config.device}")
    print(f"  γ={config.gamma}  λ={config.lam}  n_steps={config.n_steps}  "
          f"batch={config.batch_size}  epochs={config.n_epochs}  seed={config.seed}")
    print(f"{'─'*68}")

    agent = build_agent(agent_name, train_env, config)

    start_time = time.time()
    logger = agent.train(
        total_timesteps=config.total_timesteps,
        eval_env=eval_env,
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        verbose=config.verbose,
    )
    elapsed = time.time() - start_time

    train_env.close()
    eval_env.close()

    final_reward = np.mean(logger.eval_rewards[-5:]) if logger.eval_rewards else 0.0
    best_reward  = max(logger.eval_rewards) if logger.eval_rewards else 0.0
    print(f"\n  ✔ 完成  耗时={elapsed:.1f}s  最终={final_reward:.1f}  最高={best_reward:.1f}  "
          f"episodes={len(logger.episode_rewards)}")
    print(f"{'═'*68}")

    return logger


def run_experiments(config: ExperimentConfig) -> Dict[str, MetricLogger]:
    """运行所有配置的实验"""
    os.makedirs(config.save_dir, exist_ok=True)

    loggers = {}
    total_start = time.time()
    n = len(config.agents)

    print(f"\n{'#'*68}")
    print(f"  GAE 对比实验  共 {n} 个方案  环境: {config.env_name}")
    print(f"  {config.agents}")
    print(f"{'#'*68}")

    for idx, agent_name in enumerate(config.agents, start=1):
        try:
            logger = run_single_experiment(agent_name, config,
                                           exp_idx=idx, total_exps=n)
            loggers[agent_name] = logger
        except Exception as e:
            print(f"\n\u274c {agent_name} 运行失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    total_elapsed = time.time() - total_start
    print(f"\n{'#'*68}")
    print(f"  全部完成  总耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    succ = len(loggers)
    fail = n - succ
    print(f"  成功: {succ}/{n}" + (f"  失败: {fail}" if fail else ""))
    print(f"{'#'*68}\n")

    return loggers


def run_and_visualize(config: ExperimentConfig) -> Dict[str, MetricLogger]:
    """运行实验并生成可视化结果"""
    # 运行所有实验
    loggers = run_experiments(config)

    if not loggers:
        print("⚠ 没有成功的实验，跳过可视化")
        return loggers

    # 打印汇总表
    print_summary_table(loggers)

    # 生成可视化图表
    plot_all_results(loggers, env_name=config.env_name, save_dir=config.save_dir)
    plot_learning_curves_single(loggers, env_name=config.env_name, save_dir=config.save_dir)
    # 新增：综合分析图（热力图 + 筱线图 + 收敛分析）
    plot_comprehensive_analysis(loggers, env_name=config.env_name, save_dir=config.save_dir)

    return loggers


def run_multi_seed_experiment(
    config: ExperimentConfig,
    seeds: List[int] = None,
) -> Dict[str, Dict[str, MetricLogger]]:
    """
    多种子实验：每个 agent 运行多次，计算均值和置信区间
    """
    if seeds is None:
        seeds = [42, 123, 456]

    all_loggers: Dict[str, List[MetricLogger]] = {name: [] for name in config.agents}

    for seed_idx, seed in enumerate(seeds):
        print(f"\n\n{'#' * 60}")
        print(f"  种子 {seed_idx + 1}/{len(seeds)}: seed={seed}")
        print(f"{'#' * 60}")

        seed_config = ExperimentConfig(**{
            **config.__dict__,
            "seed": seed,
            "save_dir": os.path.join(config.save_dir, f"seed_{seed}"),
        })

        seed_loggers = run_experiments(seed_config)
        for name, logger in seed_loggers.items():
            all_loggers[name].append(logger)

    # 生成多种子对比图
    plot_multi_seed_results(all_loggers, config)

    return all_loggers


def plot_multi_seed_results(
    all_loggers: Dict[str, List[MetricLogger]],
    config: ExperimentConfig,
):
    """绘制多种子均值 ± 标准差的学习曲线"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .utils.visualizer import AGENT_COLORS, AGENT_LABELS, AGENT_LINESTYLES

    os.makedirs(config.save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.rcParams.update({"font.size": 12})

    sorted_names = sorted(all_loggers.keys(), key=lambda n: (n != "Standard_GAE", n))

    for name in sorted_names:
        seed_loggers = all_loggers[name]
        if not seed_loggers:
            continue

        color = AGENT_COLORS.get(name, "#607D8B")
        label = AGENT_LABELS.get(name, name)
        ls = AGENT_LINESTYLES.get(name, "-")
        lw = 2.5 if name in ["Combined_GAE", "Standard_GAE"] else 1.8

        # 对所有种子的 eval_rewards 进行插值对齐
        all_steps = []
        all_rewards = []
        for logger in seed_loggers:
            if logger.eval_rewards:
                all_steps.append(np.array(logger.eval_steps))
                all_rewards.append(np.array(logger.eval_rewards))

        if not all_rewards:
            continue

        # 找公共 step 范围
        min_step = max(s[0] for s in all_steps)
        max_step = min(s[-1] for s in all_steps)
        common_steps = np.linspace(min_step, max_step, 50)

        # 插值对齐
        interp_rewards = []
        for steps, rewards in zip(all_steps, all_rewards):
            interp = np.interp(common_steps, steps, rewards)
            interp_rewards.append(interp)

        mean_r = np.mean(interp_rewards, axis=0)
        std_r = np.std(interp_rewards, axis=0)

        ax.plot(common_steps, mean_r, color=color, linestyle=ls, linewidth=lw, label=label)
        ax.fill_between(common_steps, mean_r - std_r, mean_r + std_r, color=color, alpha=0.15)

    ax.set_xlabel("Environment Steps", fontweight="bold")
    ax.set_ylabel("Mean Evaluation Reward", fontweight="bold")
    ax.set_title(
        f"GAE Methods Comparison (Multi-Seed: {len(list(all_loggers.values())[0])} seeds)\n"
        f"Environment: {config.env_name}",
        fontweight="bold"
    )
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle="--")

    save_path = os.path.join(config.save_dir, f"multi_seed_{config.env_name.replace('/', '_')}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n📊 多种子对比图已保存至: {save_path}")
    plt.close(fig)

