"""
GAE 改进方法实验对比
====================

实验方案：
  - 基线：Standard GAE (标准 PPO + 固定 λ)
  - 改进一：Conservative Bootstrap GAE（EMA target network 保守 bootstrap，减少过估计偏差）
  - 改进二：Adaptive Lambda GAE（状态相关 λ，自适应偏差-方差权衡）
  - 改进三：Confidence-Weighted GAE（置信度归一化加权）
  - 改进四：Combined GAE（组合改进，推荐方案）

测试环境：
  - CartPole-v1（离散，快速验证）
  - LunarLander-v3（离散，中等复杂度）
  - MountainCarContinuous-v0（连续，稀疏奖励挑战）

用法：
  python main.py                          # 默认实验（CartPole）
  python main.py --env CartPole-v1        # 指定环境
  python main.py --env LunarLander-v3 --steps 300000
  python main.py --multi-seed             # 多种子对比
  python main.py --agents Standard_GAE Conservative_Bootstrap_GAE  # 只跑指定 agent
  python main.py --agents Hindsight_GAE MultiScale_GAE CausalAttn_GAE  # 只跑新方法
  python main.py --visualize-only         # 只生成图（需已有结果）
"""
import argparse
import os
import sys

from gae_experiments.experiment import (
    ExperimentConfig,
    run_and_visualize,
    run_multi_seed_experiment,
)
from gae_experiments.utils.visualizer import (
    load_all_loggers,
    plot_all_results,
    plot_learning_curves_single,
    print_summary_table,
    plot_comprehensive_analysis,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="GAE 改进方法实验对比框架",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--env",
        type=str,
        default="CartPole-v1",
        help="Gymnasium 环境名称 (默认: CartPole-v1)\n"
             "推荐: CartPole-v1, LunarLander-v3, MountainCarContinuous-v0",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=150_000,
        help="总训练步数 (默认: 150000)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="每次 rollout 的步数 (默认: 2048)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="小批量大小 (默认: 64)",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="PPO 每次更新的 epoch 数 (默认: 10)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="折扣因子 (默认: 0.99)",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.95,
        help="GAE lambda (默认: 0.95)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="网络隐藏层维度 (默认: 64)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5000,
        help="评估频率（每多少步评估一次）(默认: 5000)",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=10,
        help="每次评估的 episode 数 (默认: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="计算设备 (默认: cpu)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results",
        help="结果保存目录 (默认: results)",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=None,
        choices=[
            "Standard_GAE",
            "Conservative_Bootstrap_GAE",
            "Adaptive_Lambda_GAE",
            "Confidence_Weighted_GAE",
            "Combined_GAE",
            "Hindsight_GAE",
            "MultiScale_GAE",
            "CausalAttn_GAE",
        ],
        help="指定运行的 agent（默认运行全部）",
    )
    parser.add_argument(
        "--multi-seed",
        action="store_true",
        help="运行多种子实验（seeds: 42, 123, 456）",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
        help="多种子实验的种子列表 (默认: 42 123 456)",
    )
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="仅生成可视化图表（需要已有实验结果）",
    )
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="关闭训练过程的详细输出",
    )
    return parser.parse_args()


def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════╗
║         GAE 改进方法实验对比框架                              ║
║         GAE Improvement Methods Benchmark                    ║
╠══════════════════════════════════════════════════════════════╣
║  方案对比：                                                   ║
║  ① Standard GAE              - 基线（固定 λ）                  ║
║  ② Conservative Bootstrap GAE - EMA 保守 Bootstrap              ║
║  ③ Adaptive λ GAE             - 状态相关自适应 λ                ║
║  ④ Confidence-Weighted GAE    - MAD+EMA 置信度加权            ║
║  ⑤ Combined GAE              - 组合改进                       ║
║  ⑥ Hindsight GAE   [革新]    - MC Hindsight 修正 Critic 偏差     ║
║  ⑦ MultiScale GAE  [革新]    - 多λ尺度可学习加权组合          ║
║  ⑧ CausalAttn GAE  [革新]    - 因果注意力替代几何衰减          ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def get_env_defaults(env_name: str) -> dict:
    """根据环境返回推荐的超参数"""
    defaults = {
        "CartPole-v1": {
            "total_timesteps": 150_000,
            "n_steps": 1024,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "lam": 0.95,
            "ent_coef": 0.01,
            "eval_freq": 5000,
        },
        "LunarLander-v3": {
            "total_timesteps": 500_000,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "lam": 0.95,
            "ent_coef": 0.01,
            "eval_freq": 10000,
        },
        "MountainCarContinuous-v0": {
            "total_timesteps": 300_000,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.999,
            "lam": 0.95,
            "ent_coef": 0.0,
            "eval_freq": 10000,
        },
        "Acrobot-v1": {
            "total_timesteps": 200_000,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "lam": 0.95,
            "ent_coef": 0.0,
            "eval_freq": 5000,
        },
        "Pendulum-v1": {
            "total_timesteps": 200_000,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.9,
            "lam": 0.95,
            "ent_coef": 0.0,
            "eval_freq": 5000,
        },
        "HalfCheetah-v4": {
            "total_timesteps": 1_000_000,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "lam": 0.95,
            "ent_coef": 0.0,
            "eval_freq": 10000,
        },
        "Hopper-v4": {
            "total_timesteps": 500_000,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "lam": 0.95,
            "ent_coef": 0.0,
            "eval_freq": 10000,
        },
    }
    return defaults.get(env_name, {})


def main():
    print_banner()
    args = parse_args()

    # 获取环境推荐默认值
    env_defaults = get_env_defaults(args.env)

    # 构建实验配置
    config = ExperimentConfig(
        env_name=args.env,
        total_timesteps=args.steps if args.steps != 150_000 else env_defaults.get("total_timesteps", 150_000),
        n_steps=args.n_steps if args.n_steps != 2048 else env_defaults.get("n_steps", 2048),
        batch_size=args.batch_size if args.batch_size != 64 else env_defaults.get("batch_size", 64),
        n_epochs=args.n_epochs if args.n_epochs != 10 else env_defaults.get("n_epochs", 10),
        gamma=args.gamma if args.gamma != 0.99 else env_defaults.get("gamma", 0.99),
        lam=args.lam if args.lam != 0.95 else env_defaults.get("lam", 0.95),
        lr_actor=3e-4,
        lr_critic=1e-3,
        eps_clip=0.2,
        ent_coef=env_defaults.get("ent_coef", 0.0),
        vf_coef=0.5,
        max_grad_norm=0.5,
        hidden_dim=args.hidden_dim,
        eval_freq=args.eval_freq if args.eval_freq != 5000 else env_defaults.get("eval_freq", 5000),
        n_eval_episodes=args.n_eval_episodes,
        seed=args.seed,
        device=args.device,
        save_dir=os.path.join(args.save_dir, args.env),
        agents=args.agents or [
            "Standard_GAE",
            "Conservative_Bootstrap_GAE",
            "Adaptive_Lambda_GAE",
            "Confidence_Weighted_GAE",
            "Combined_GAE",
            "Hindsight_GAE",
            "MultiScale_GAE",
            "CausalAttn_GAE",
        ],
        verbose=not args.no_verbose,
    )

    print(f"📋 实验配置:")
    print(f"   环境: {config.env_name}")
    print(f"   总步数: {config.total_timesteps:,}")
    print(f"   n_steps: {config.n_steps}, batch_size: {config.batch_size}, n_epochs: {config.n_epochs}")
    print(f"   γ={config.gamma}, λ={config.lam}, lr_actor={config.lr_actor}")
    print(f"   运行方案: {config.agents}")
    print(f"   结果目录: {config.save_dir}")
    print()

    # ─────────────────────────────────────────────────────────
    # 仅可视化模式
    # ─────────────────────────────────────────────────────────
    if args.visualize_only:
        print("📊 仅可视化模式，从结果目录加载数据...")
        loggers = load_all_loggers(config.save_dir)
        if not loggers:
            print(f"❌ 未找到结果文件，请先运行实验。目录: {config.save_dir}")
            sys.exit(1)
        print_summary_table(loggers)
        plot_all_results(loggers, env_name=config.env_name, save_dir=config.save_dir)
        plot_learning_curves_single(loggers, env_name=config.env_name, save_dir=config.save_dir)
        plot_comprehensive_analysis(loggers, env_name=config.env_name, save_dir=config.save_dir)
        return

    # ─────────────────────────────────────────────────────────
    # 多种子实验
    # ─────────────────────────────────────────────────────────
    if args.multi_seed:
        print(f"🔬 多种子实验 (seeds={args.seeds})")
        run_multi_seed_experiment(config, seeds=args.seeds)
        return

    # ─────────────────────────────────────────────────────────
    # 标准实验
    # ─────────────────────────────────────────────────────────
    loggers = run_and_visualize(config)

    if loggers:
        print("\n✅ 实验完成！")
        print(f"   结果文件保存在: {os.path.abspath(config.save_dir)}/")
        print("   生成的文件:")
        for fname in os.listdir(config.save_dir):
            fpath = os.path.join(config.save_dir, fname)
            size = os.path.getsize(fpath)
            print(f"   - {fname} ({size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
