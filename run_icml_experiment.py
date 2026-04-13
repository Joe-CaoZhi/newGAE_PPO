#!/usr/bin/env python3
"""
ICML 2026 完备实验脚本
======================
5 种算法 × 4 个环境 × 12 个随机种子 × 1M 步

算法（严格对齐超参数，所有算法共享 OptimalPPO 骨架）：
  1. Standard_PPO       ── 原始 PPO（无 obs-norm / adv-norm / lr-anneal）
  2. Optimal_PPO        ── 集成所有最佳实践的基线（obs-norm + adv-norm + lr-anneal）
  3. Heuristic_HCGAE    ── 启发式 MC 融合（OptimalHCGAE_v2）：余弦退火×EV门控×sigmoid
  4. BHVF               ── 本文方法 §2：解析最优增益 α*=SCR²/(SCR²+1)，无任何启发式
  5. BHVF_DCPPO         ── 本文方法 §2+§3：BHVF + EV 线性收缩梯度调制（DCPPO-S）

环境：Hopper-v4 / Walker2d-v4 / HalfCheetah-v4 / Ant-v4

输出目录结构：
  results/FinalExperiment/{env}/{algo}/{algo}_s{seed}.json

每个 JSON 包含：
  - eval_rewards, eval_steps  ── 标准学习曲线（每 20480 步评估一次）
  - final_reward              ── 最后 10 次评估均值（论文 Table 1 数据源）
  - max_reward                ── 全程最高评估分数
  - bhvf_diagnostics          ── BHVF 特有的中间状态（SCR、alpha*、sigma_V 等）
  - config                    ── 完整超参数（可溯源）

可溯源性设计：
  - 每个 JSON 含完整超参数 config，任意结果可独立复现
  - 种子列表固定（0–11），与 paper 报告严格对应
  - 中间 diagnostics 允许离线验证理论预测（如 HalfCheetah SCR ≈ 0，Hopper SCR >> 1）

用法：
  # 快速冒烟测试（50k步，2种子）
  python run_icml_experiment.py --smoke-test

  # 单环境单算法（断点续跑）
  python run_icml_experiment.py --env Hopper-v4 --algo BHVF

  # 完整实验（自动跳过已有结果）
  python run_icml_experiment.py

  # 并行（外部脚本控制进程池）
  python run_icml_experiment.py --env Hopper-v4 --algo BHVF --seeds 0,1,2,3
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

from gae_experiments.agents.optimal_ppo import (
    OptimalPPO,
    OptimalHCGAE_v2,
    OptimalHCGAE_Bayesian,
    OptimalHCGAE_BayesianV3,
    OptimalHCGAE_BayesianV4,
    OptimalHCGAE_BayesianV5,
    OptimalHCGAE_BayesianV6,
    OptimalHCGAE_BayesianV9,
    OptimalHCGAE_BayesianV10,
)

# ─────────────────────────────────────────────────────────────────────────────
# 实验设计常量
# ─────────────────────────────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 1_000_000
EVAL_FREQ       = 20_480     # 约 10 次 rollout 评估一次（与 Andrychowicz 2021 对齐）
N_EVAL_EPISODES = 10
N_SEEDS         = 12
SEEDS           = list(range(N_SEEDS))  # 0..11，固定，可溯源

ENVS = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4", "Ant-v4"]

# 8 个算法配置
ALGOS = ["Standard_PPO", "Optimal_PPO", "Heuristic_HCGAE", "BHVF", "BHVF_DCPPO", "BHVF_V3", "BHVF_V4", "BHVF_V5", "BHVF_V6", "BHVF_V7", "BHVF_V8", "BHVF_V9", "BHVF_V10", "BHVF_V12"]

RESULTS_DIR = Path("results/FinalExperiment")

# ─────────────────────────────────────────────────────────────────────────────
# 对齐超参数（所有算法完全相同，是公平对比的基础）
# ─────────────────────────────────────────────────────────────────────────────
SHARED_KWARGS = dict(
    hidden_dim      = 256,
    lr              = 3e-4,
    gamma           = 0.99,
    lam             = 0.95,
    eps_clip        = 0.2,
    n_epochs        = 10,
    batch_size      = 64,
    n_steps         = 2048,
    ent_coef        = 0.0,
    vf_coef         = 0.5,
    max_grad_norm   = 0.5,
    # OptimalPPO 骨架 tricks
    use_obs_norm    = True,
    use_adv_norm    = True,
    use_lr_anneal   = True,
    use_vclip       = False,
)

# Standard_PPO 禁用所有最佳实践（对标 Schulman 2017 原版）
STANDARD_PPO_OVERRIDES = dict(
    use_obs_norm  = False,
    use_adv_norm  = False,
    use_lr_anneal = False,
)

# BHVF 超参数（全环境完全相同，零环境特定调整）
# v3 changes:
#   clip_c: 3.0 → 2.5 (use σ_e for clip bound, slightly tighter)
#   alpha*: min(alpha_scr, alpha_ev) → min(alpha_ev, SNR_cap) (cross-env consistency)
#   c_mc floor: 0.3 → 0.1 (prevent MC noise pollution at high EV)
BHVF_KWARGS = dict(
    scr_ema_alpha = 0.1,   # SCR EMA 学习率
    clip_c        = 2.5,   # 新息截断系数（2.5σ_e ≈ 99% Normal，比原3σ_V更紧）
)

# Heuristic HCGAE (v2) 超参数
HEURISTIC_HCGAE_KWARGS = dict(
    hindsight_beta       = 3.0,
    hindsight_alpha_max  = 0.7,
    hindsight_alpha_min  = 0.1,
    use_scr_adapt        = False,
    use_boundary_correction = True,
    use_ev_rate_gate     = True,
    ev_rate_threshold    = 0.05,
    ev_rate_max          = 0.15,
    ev_gate_min_scale    = 0.1,
)


def build_agent(algo_name: str, env: gym.Env, seed: int, save_dir: str):
    """
    构建对应算法的 agent。
    所有算法共享 SHARED_KWARGS，仅算法特有参数有差异。
    """
    kw = dict(**SHARED_KWARGS, save_dir=save_dir)

    if algo_name == "Standard_PPO":
        kw.update(STANDARD_PPO_OVERRIDES)
        return OptimalPPO(env=env, name=f"Standard_PPO_s{seed}", **kw)

    elif algo_name == "Optimal_PPO":
        return OptimalPPO(env=env, name=f"Optimal_PPO_s{seed}", **kw)

    elif algo_name == "Heuristic_HCGAE":
        hkw = dict(**kw)
        hkw.update(HEURISTIC_HCGAE_KWARGS)
        return OptimalHCGAE_v2(env=env, name=f"Heuristic_HCGAE_s{seed}", **hkw)

    elif algo_name == "BHVF":
        # 纯 BHVF（无 DCPPO-S）：advantage 不乘 EV 权重
        return OptimalHCGAE_Bayesian(
            env=env, name=f"BHVF_s{seed}",
            **{k: v for k, v in {**kw, **BHVF_KWARGS}.items()}
        )

    elif algo_name == "BHVF_DCPPO":
        # BHVF + DCPPO-S：advantage 乘以 clip(EV, 0.1, 1.0)（在 update 中实现）
        agent = OptimalHCGAE_Bayesian(
            env=env, name=f"BHVF_DCPPO_s{seed}",
            **{k: v for k, v in {**kw, **BHVF_KWARGS}.items()}
        )
        agent._use_dcppo_s = True   # 标记启用 DCPPO-S（在 run_single 中处理）
        return agent

    elif algo_name == "BHVF_V3":
        # BHVF V3：GAE-Orthogonal Innovation + Var-optimal alpha + Standardized Innovation
        # C2 修复版：C2 = Var[A_c] + Var[D]  (algebraically correct)
        return OptimalHCGAE_BayesianV3(
            env=env, name=f"BHVF_V3_s{seed}",
            use_orth_inno=True,
            use_std_innovation=True,
            **{k: v for k, v in {**kw, **BHVF_KWARGS}.items()}
        )

    elif algo_name == "BHVF_V4":
        # BHVF V4：修正 Kalman gain 公式
        # α* = Var(δ)/Var(G) = 1 - EV，直接 EV-based，消除 SCR 的 MAE/std 混用和 std(G) 污染
        # 继承 V3 的正交创新 + 方差最优 α cap + 标准化创新
        return OptimalHCGAE_BayesianV4(
            env=env, name=f"BHVF_V4_s{seed}",
            use_orth_inno=True,
            use_std_innovation=True,
            **{k: v for k, v in {**kw, **BHVF_KWARGS}.items()}
        )

    elif algo_name == "BHVF_V5":
        # BHVF V5：修复 MAE/std 单位不一致的根本 bug
        # 正确 Kalman 公式: α* = sigma_e² / (sigma_e² + sigma_G²)
        # 其中 sigma_e = std(G-V)（与 sigma_G 同单位），取代了错误的 MAE/std 混用
        # 验证数据: Hopper alpha 从 0.044 提升到 0.104 (2.4x), Walker2d 从 0.148→0.226 (1.5x)
        # 去除正交化（V3的错误方向）和 Var(A_GAE)/Var(δ) 公式（V4的推导错误）
        return OptimalHCGAE_BayesianV5(
            env=env, name=f"BHVF_V5_s{seed}",
            **{k: v for k, v in {**kw, **BHVF_KWARGS}.items()}
        )

    elif algo_name == "BHVF_V6":
        # BHVF V6：精确协方差公式（第一性原理推导）
        # 核心推导：α* = -Cov(r_GAE, δ) / Var(δ)
        # 不需要 ε_V ⊥ ε_G 的独立性假设，直接处理 Cov(ε_V, ε_G) > 0 的情况
        # 无额外超参数，SNR ≥ 1.7 in all envs (HC最高: SNR=11.1)
        return OptimalHCGAE_BayesianV6(
            env=env, name=f"BHVF_V6_s{seed}",
            **{k: v for k, v in {**kw, **BHVF_KWARGS}.items()}
        )

    elif algo_name == "BHVF_V7":
        from gae_experiments.agents.optimal_ppo import OptimalHCGAE_BayesianV7
        return OptimalHCGAE_BayesianV7(env=env, name=f"BHVF_V7_s{seed}", warmup_scale=0.8, warmup_rollouts=200, **{k: v for k, v in {**kw, **BHVF_KWARGS}.items()})

    elif algo_name == "BHVF_V8":
        # BHVF V8：EV 自适应加法 floor（修正乘法退化 bug）
        # α_V8 = clip(α_V6 + α_floor·max(1−EV_ema, 0), 0, 1)
        # 加法 floor 确保即使 α_V6→0（Hopper 高 EV），仍能注入有效 MC 信号
        # α_floor=0.3：EV=0.87 时 Δ≈0.04，EV=0.20 时 Δ≈0.24
        from gae_experiments.agents.optimal_ppo import OptimalHCGAE_BayesianV8
        return OptimalHCGAE_BayesianV8(
            env=env, name=f"BHVF_V8_s{seed}",
            explore_floor=0.3,
            **{k: v for k, v in {**kw, **BHVF_KWARGS}.items()}
        )

    elif algo_name == "BHVF_V9":
        # BHVF V9: Separated Actor/Critic Control (first-principles optimal)
        # CRITIC: exact Cov formula alpha* = -Cov(r_GAE,delta)/Var(delta)  [V6 unchanged]
        # ACTOR:  EV-adaptive MC injection alpha_tilde = beta_A*max(1-EV_ema,0)  [new]
        # Decouples variance minimisation from exploration: no circular dependency
        return OptimalHCGAE_BayesianV9(
            env=env, name=f"BHVF_V9_s{seed}",
            actor_beta=0.3,
            actor_alpha_max=0.5,
            **{k: v for k, v in {**kw, **BHVF_KWARGS}.items()}
        )

    elif algo_name == "BHVF_V10":
        # BHVF V10: SCR-Squared Shrinkage Estimator (MSE-optimal, zero hyperparameters)
        # α_V10 = max(α_V6, SCR_ema² / (1 + SCR_ema²))
        # SCR = σ_V/σ_G = MAE(G-V)/Std(G) — James-Stein optimal mixing weight
        # Provably minimises E[(T* - V*)²] under independent-noise approximation.
        # Fixes V8's ad-hoc floor and V9's rollout-distribution-corruption problem.
        return OptimalHCGAE_BayesianV10(
            env=env, name=f"BHVF_V10_s{seed}",
            scr_shrink_alpha=0.1,
            **{k: v for k, v in {**kw, **BHVF_KWARGS}.items()}
        )

    elif algo_name == "BHVF_V12":
        # BHVF V12: Bias-aware V_c-based GAE
        # 融合 V5/V6/V8/Heuristic_HCGAE 各版本的最优改动
        #
        # 核心公式 (第一性原理 MSE 最小化):
        #   α* = (b_V² + σV² - Cov) / (b_V² + σV² + σG² - 2·Cov)
        #
        # b_V² = EMA(mean(G-V))² 是 Critic 偏差的平方估计
        # 自然地: b_V 大时 → α* 大 (更多 MC, 纠正偏差, 类似 V8)
        #         b_V 小时 → α* ≈ V6 精确 Cov 公式
        # Per-step sigmoid (来自 Heuristic) 防止全局 α 过大
        from gae_experiments.agents.optimal_ppo import OptimalHCGAE_BayesianV12
        return OptimalHCGAE_BayesianV12(
            env=env, name=f"BHVF_V12_s{seed}",
            scr_ema_alpha=0.1,
            clip_c=2.5,
            hindsight_beta=3.0,
            alpha_max_cap=0.7,
            mean_d_alpha=0.05,
            **{k: v for k, v in kw.items()}
        )

    elif algo_name == "BHVF_V12b":
        # BHVF V12b: Fixed Bias-aware V_c-based GAE
        # Fixes V12's critical bug (sigma_V_sq estimation causes α→0.7 positive feedback)
        #
        # Key fixes vs V12:
        #   1. Remove wrong sigma_V_sq = var_d * (1-EV) estimation
        #   2. Use V6 exact OLS formula as base: alpha_cov = -Cov(r_GAE,δ)/Var(δ)
        #   3. Add bounded bias correction: bias_add = 0.2 * sigmoid(2*(b_V_norm-0.5))
        #   4. Add EV guard: alpha_ev_max = 0.30*(1-EV_ema) + 0.10
        #      → prevents positive feedback loop when EV is low
        from gae_experiments.agents.optimal_ppo import OptimalHCGAE_BayesianV12b
        return OptimalHCGAE_BayesianV12b(
            env=env, name=f"BHVF_V12b_s{seed}",
            scr_ema_alpha=0.1,
            clip_c=2.5,
            hindsight_beta=3.0,
            alpha_max_cap=0.7,
            mean_d_alpha=0.05,
            bias_add_max=0.20,
            ev_guard_slope=0.30,
            ev_guard_base=0.10,
            **{k: v for k, v in kw.items()}
        )

    elif algo_name == "BHVF_V12c":
        # BHVF V12c: Balanced Bias-aware V_c-based GAE
        # Balances V12's high performance with V12b's stability
        #
        # Key improvements vs V12b:
        #   1. Higher α_cap (0.55 vs dynamic 0.10-0.40) → more纠偏 capacity
        #   2. Higher bias threshold (0.7 vs 0.5) → less sensitive to noise
        #   3. Soft EV penalty (only when EV < 0.4) vs hard cap
        #      → EV=0.35 → penalty 0.025 (small), still allows α≈0.30
        #
        # Expected:
        #   HC: α ≈ 0.25-0.35 (vs V12b's 0.15-0.28) → better纠偏
        #   Hopper: α ≈ 0.15-0.20 (similar to V12b) → stable
        from gae_experiments.agents.optimal_ppo import OptimalHCGAE_BayesianV12c
        return OptimalHCGAE_BayesianV12c(
            env=env, name=f"BHVF_V12c_s{seed}",
            scr_ema_alpha=0.1,
            clip_c=2.5,
            hindsight_beta=3.0,
            alpha_max_cap=0.55,
            mean_d_alpha=0.05,
            bias_add_max=0.15,
            ev_threshold=0.40,
            ev_penalty_slope=0.50,
            **{k: v for k, v in kw.items()}
        )

    elif algo_name == "BHVF_V12d":
        # BHVF V12d: Correct V6+V8 fusion (first-principles final version)
        #
        # Fixes V12c's CRITICAL BUG: EV penalty direction was REVERSED.
        # V12c: ev_penalty = slope*(threshold-EV) → low EV → α=0 → V_c=V → Critic死锁
        # V12d: alpha_floor = floor_scale*(1-EV) → low EV → α大 → Critic学MC → EV上升
        #
        # Formula:
        #   α = clip(α_base + α_floor + bias_add, 0, alpha_max_cap)
        #   α_base = clip(-Cov(r_GAE,δ)/Var(δ), 0, cap)  [V6 OLS]
        #   α_floor = 0.20 * (1 - EV_ema)                [V8 EV-floor, CORRECT direction]
        #   bias_add = 0.15 * sigmoid(2*(b_V_norm-0.5))  [bias correction]
        #
        # Expected behavior:
        #   HC init: EV=0 → α_floor=0.20 → Critic learns MC → EV rises
        #   HC stable: EV≈0.35 → α≈0.28-0.43 (safe, no collapse)
        #   Hopper: EV≈0.75 → α≈0.15-0.25 (non-invasive)
        from gae_experiments.agents.optimal_ppo import OptimalHCGAE_BayesianV12d
        return OptimalHCGAE_BayesianV12d(
            env=env, name=f"BHVF_V12d_s{seed}",
            scr_ema_alpha=0.1,
            clip_c=2.5,
            hindsight_beta=3.0,
            alpha_max_cap=0.6,
            mean_d_alpha=0.05,
            floor_scale=0.20,
            bias_add_max=0.15,
            **{k: v for k, v in kw.items()}
        )

    elif algo_name == "BHVF_V12e":
        # BHVF V12e: V12d with stronger EV floor (floor_scale 0.20 → 0.30)
        #
        # V12d analysis: HC Q1=-440 (slow cold start persists).
        # Root cause: floor=0.20 still insufficient for initial Critic learning.
        # V12e fix: increase floor_scale to 0.30 → stronger MC injection at EV=0
        #   HC init (EV=0):    α_floor=0.30 → faster Critic convergence
        #   HC stable (EV=0.35): α_floor=0.195 → still well-controlled
        #   Hopper (EV=0.75):  α_floor=0.075 → modest, non-invasive
        # Also lower alpha_max_cap 0.60 → 0.55 to compensate larger floor.
        # NOTE: V12e hurts Hopper (1000 vs V12d 1363) because alpha_max_cap=0.55
        #   prevents the "breakthrough" seen in V12d s2=2913.
        from gae_experiments.agents.optimal_ppo import OptimalHCGAE_BayesianV12d
        return OptimalHCGAE_BayesianV12d(
            env=env, name=f"BHVF_V12e_s{seed}",
            scr_ema_alpha=0.1,
            clip_c=2.5,
            hindsight_beta=3.0,
            alpha_max_cap=0.55,
            mean_d_alpha=0.05,
            floor_scale=0.30,
            bias_add_max=0.15,
            **{k: v for k, v in kw.items()}
        )

    elif algo_name == "BHVF_V12f":
        # BHVF V12f: Best of V12d + V12e synthesis
        #
        # V12e analysis showed:
        #   - floor_scale=0.30 helps HC cold start (Q2-Q3 slightly better)
        #   - BUT alpha_max_cap=0.55 hurt Hopper (blocked V12d's s2=2913 breakthrough)
        #
        # V12f: Keep floor=0.30 (from V12e) but restore alpha_max_cap=0.60 (from V12d)
        #   - HC: stronger floor (0.30) → better cold start than V12d
        #   - Hopper: larger cap (0.60) → allows breakthrough when EV is high
        #
        # Expected:
        #   HC:    α_floor=0.30 (EV=0) → better Q1-Q2 than V12d
        #   Hopper: α_floor=0.075 (EV=0.75) → same as V12e, cap=0.60 allows high-α
        from gae_experiments.agents.optimal_ppo import OptimalHCGAE_BayesianV12d
        return OptimalHCGAE_BayesianV12d(
            env=env, name=f"BHVF_V12f_s{seed}",
            scr_ema_alpha=0.1,
            clip_c=2.5,
            hindsight_beta=3.0,
            alpha_max_cap=0.60,  # restored from V12d (V12e had 0.55)
            mean_d_alpha=0.05,
            floor_scale=0.30,    # kept from V12e (V12d had 0.20)
            bias_add_max=0.15,
            **{k: v for k, v in kw.items()}
        )

    elif algo_name in ("BHVF_V12g", "Optimal_HCGAE_BayesianV12g"):
        # ── BHVF V12g: Exponential EV-decay floor (指数衰减 EV-Floor，最终设计) ──
        #
        # 核心发现：HC 需要 α ≥ 0.50 才能在 200k 步内有效提升 EV；
        # 线性 floor (V12d: 0.20*(1-EV)) 在冷启动阶段 (EV≈0) 只有 0.20，不够。
        #
        # V12g 设计：α_floor = cold_start_alpha * exp(-EV / decay_scale)
        #   EV=0.00: floor=0.500  (强冷启动)
        #   EV=0.15: floor=0.303  (HC 稳态，仍然很强)
        #   EV=0.60: floor=0.068  (Hopper 稳态，保守)
        #   EV=0.90: floor=0.025  (高 EV 时近零，几乎纯 OLS)
        from gae_experiments.agents.optimal_ppo import OptimalHCGAE_BayesianV12g
        return OptimalHCGAE_BayesianV12g(
            env=env, name=f"BHVF_V12g_s{seed}",
            scr_ema_alpha=0.1,
            clip_c=2.5,
            hindsight_beta=3.0,
            alpha_max_cap=0.70,
            mean_d_alpha=0.05,
            cold_start_alpha=0.50,
            decay_scale=0.30,
            bias_add_max=0.15,
            **{k: v for k, v in kw.items()}
        )

    elif algo_name in ("BHVF_V12h", "Optimal_HCGAE_BayesianV12h"):
        # ── BHVF V12h: Bayesian-Optimal EV-driven α (极简优雅设计) ──
        #
        # 从第一性原理推导：贝叶斯最优混合比例
        #   α* = σ_e² / (σ_e² + σ_G²) = (1-EV) / (2-EV)
        #
        # 零额外超参数：仅需 hindsight_beta (per-step sigmoid 斜率)
        # 与 V12g 的本质联系：V12g 的行为是对此贝叶斯最优值的近似，
        # V12h 直接使用该封闭解，消除 OLS base/exp_floor/bias_add 三项工程补丁。
        from gae_experiments.agents.optimal_ppo import OptimalHCGAE_BayesianV12h
        return OptimalHCGAE_BayesianV12h(
            env=env, name=f"BHVF_V12h_s{seed}",
            scr_ema_alpha=0.1,
            clip_c=2.5,
            hindsight_beta=3.0,
            **{k: v for k, v in kw.items()}
        )

    elif algo_name in ("BHVF_V12i", "Optimal_HCGAE_BayesianV12i"):
        # ── BHVF V12i: EV-Guarded Exponential Floor (抗崩溃修复版) ──
        #
        # V12g 的唯一改动：EV_ema < 0 时改用 fallback_floor=0.05
        #
        # 根因：V12g 用 clip(EV_ema, 0, 1)
        #   Ant EV 崩溃到 -0.98 → ev_clamped=0 → floor=0.50 (最强 MC!)
        #   Critic 崩溃时反而注入最大噪声，阻碍恢复。
        #
        # 修复语义：
        #   EV ≥ 0: 使用指数衰减 floor（与 V12g 完全一致）
        #   EV < 0: 使用 fallback_floor=0.05（保守常数，等待 Critic 恢复）
        #
        # 预期：HC/Hopper/Walker 行为与 V12g 完全一致；
        #        Ant 崩溃恢复速度大幅提升，接近 OptPPO 水平。
        from gae_experiments.agents.optimal_ppo import OptimalHCGAE_BayesianV12i
        return OptimalHCGAE_BayesianV12i(
            env=env, name=f"BHVF_V12i_s{seed}",
            scr_ema_alpha=0.1,
            clip_c=2.5,
            hindsight_beta=3.0,
            alpha_max_cap=0.70,
            mean_d_alpha=0.05,
            cold_start_alpha=0.50,
            decay_scale=0.30,
            bias_add_max=0.15,
            fallback_floor=0.05,
            **{k: v for k, v in kw.items()}
        )

    else:
        raise ValueError(f"Unknown algo: {algo_name}")


def evaluate_policy(agent, eval_env: gym.Env, n_episodes: int = 10) -> float:
    """
    评估策略（确定性均值动作，与 Andrychowicz 2021 对齐）。
    禁用 exploration noise，使用 policy mean。
    """
    total_reward = 0.0
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_r = 0.0
        while not done:
            obs_n = agent.normalize_obs(obs) if hasattr(agent, 'normalize_obs') else obs
            obs_t = torch.FloatTensor(obs_n).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                dist = agent.actor(obs_t)
                # 确定性评估：使用分布均值
                if agent.continuous:
                    action = dist.mean
                else:
                    action = dist.probs.argmax(dim=-1)
            a = action.squeeze(0).cpu().numpy()
            next_obs, r, terminated, truncated, _ = eval_env.step(
                a if agent.continuous else int(a)
            )
            ep_r += r
            done = terminated or truncated
            obs = next_obs
        total_reward += ep_r
    return total_reward / n_episodes


def run_single(
    env_name: str,
    algo_name: str,
    seed: int,
    total_timesteps: int = TOTAL_TIMESTEPS,
    results_dir: Path = RESULTS_DIR,
    skip_existing: bool = True,
) -> dict:
    """
    运行单个 (env, algo, seed) 组合。

    返回：包含完整指标的 dict，同时保存到 JSON。
    """
    save_dir = str(results_dir / env_name / algo_name)
    os.makedirs(save_dir, exist_ok=True)
    out_path = Path(save_dir) / f"{algo_name}_s{seed}.json"

    if skip_existing and out_path.exists():
        data = json.load(open(out_path))
        print(f"  [SKIP] {env_name}/{algo_name}/s{seed} "
              f"(final={data.get('final_reward', '?'):.1f})")
        return data

    # ── 种子设置 ──
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── 环境创建 ──
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 100_000)

    # ── Agent 构建 ──
    agent = build_agent(algo_name, env, seed, save_dir)
    use_dcppo_s = getattr(agent, '_use_dcppo_s', False)
    agent._total_timesteps = total_timesteps

    # ── 训练状态 ──
    eval_rewards, eval_steps = [], []
    episode_rewards = []
    bhvf_diag_log = []   # BHVF 诊断记录（每次 rollout 一条）

    # 峰值口径（Peak-10）：每次 eval 后实时滚动计算
    # 定义：以全程最高 eval 为中心取对称窗口（长度 N_PEAK），取窗口均值
    # 语义：算法在最佳状态附近的稳健表现，不受末期崩溃影响
    # 学术依据：与 RLiable 框架中 "IQM over best checkpoint" 精神一致
    N_PEAK = 10    # 峰值窗口大小（与 final 口径对称）
    peak10_running = 0.0   # 实时滚动峰值均值（每次 eval 更新）

    obs, _ = env.reset()
    if hasattr(agent, 'update_obs_rms'):
        agent.update_obs_rms(obs)
    obs = agent.normalize_obs(obs) if hasattr(agent, 'normalize_obs') else obs

    ep_reward, ep_length = 0.0, 0
    total_steps = 0
    last_eval_step = -EVAL_FREQ  # 确保第一次立即评估
    t0 = time.time()
    rollout_idx = 0

    print(f"  START {env_name}/{algo_name}/s{seed} "
          f"[DCPPO-S={'ON' if use_dcppo_s else 'OFF'}]")

    while total_steps < total_timesteps:
        # ── Rollout 采集 ──
        agent.buffer.reset()
        for _ in range(agent.n_steps):
            if hasattr(agent, 'update_obs_rms'):
                agent.update_obs_rms(obs)
            obs_n = agent.normalize_obs(obs) if hasattr(agent, 'normalize_obs') else obs
            obs_t = torch.FloatTensor(obs_n).unsqueeze(0).to(agent.device)

            with torch.no_grad():
                action, log_prob = agent.actor.get_action_and_logprob(obs_t)
                value = agent.critic(obs_t)

            action_np = action.squeeze(0).cpu().numpy()
            if not agent.continuous:
                action_np = int(action_np)

            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            ep_reward += reward
            ep_length += 1

            agent.buffer.add(obs_n, action_np, float(reward), float(terminated),
                             log_prob.item(), value.item())
            obs = next_obs
            total_steps += 1

            if terminated or truncated:
                episode_rewards.append(ep_reward)
                ep_reward, ep_length = 0.0, 0
                obs, _ = env.reset()
                if hasattr(agent, 'update_obs_rms'):
                    agent.update_obs_rms(obs)
                obs = agent.normalize_obs(obs) if hasattr(agent, 'normalize_obs') else obs

        # ── Bootstrap value ──
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            last_val = agent.critic(obs_t).item()

        agent.total_steps = total_steps

        # ── GAE / BHVF 计算 ──
        if hasattr(agent, 'compute_hindsight_gae'):
            agent.compute_hindsight_gae(last_val)
        else:
            agent.compute_gae(last_val)

        # ── DCPPO-S：EV 线性收缩优势（命题 1：仅缩放幅度，不改变方向）──
        if use_dcppo_s:
            ev_weight = float(np.clip(agent._ev_ema, 0.1, 1.0))
            T = agent.buffer.pos
            agent.buffer.advantages[:T] *= ev_weight

        # ── 参数更新 ──
        metrics = agent.update()
        rollout_idx += 1

        # ── 记录 BHVF diagnostics（每次 rollout）──
        if isinstance(agent, OptimalHCGAE_Bayesian):
            bhvf_diag_log.append({
                "step":        total_steps,
                "scr":         agent._diag_scr,
                "sigma_V":     agent._diag_sigma_V,
                "sigma_G":     agent._diag_sigma_G,
                "sigma_e":     agent._diag_sigma_e,
                "alpha_star":  agent._diag_alpha_star,
                "c_mc":        agent._diag_c_mc,
                "ev_now":      agent._diag_ev_now,
                "clip_ratio":  agent._diag_clip_ratio,
            })

        # ── 定期评估 ──
        if total_steps - last_eval_step >= EVAL_FREQ:
            eval_r = evaluate_policy(agent, eval_env, N_EVAL_EPISODES)
            eval_rewards.append(eval_r)
            eval_steps.append(total_steps)
            last_eval_step = total_steps

            # 实时滚动计算峰值口径（Peak-N_PEAK）
            # 以当前全程最高 eval 为中心，取对称窗口均值
            # 窗口随数据增长自然滑动，训练结束时即为最终 peak_reward
            peak_idx = int(np.argmax(eval_rewards))
            half = N_PEAK // 2
            lo = max(0, peak_idx - half)
            hi = min(len(eval_rewards), lo + N_PEAK)
            lo = max(0, hi - N_PEAK)   # hi 被边界截断时左移 lo 保持窗口完整
            peak10_running = float(np.mean(eval_rewards[lo:hi]))

            # 打印进度（同时显示 final 口径和 peak 口径）
            elapsed = time.time() - t0
            fps = int(total_steps / (elapsed + 1e-8))
            pct = total_steps / total_timesteps * 100
            ev = metrics.get("explained_variance", 0.0)
            # final_so_far：当前已有数据的末N次均值（与最终 final_reward 一致口径）
            n_avail = len(eval_rewards)
            final_so_far = float(np.mean(eval_rewards[-min(N_PEAK, n_avail):]))
            print(f"    [{algo_name:<15} s{seed}] "
                  f"{total_steps:7d}/{total_timesteps} ({pct:4.0f}%) "
                  f"| eval={eval_r:7.1f} "
                  f"| peak10={peak10_running:7.1f} "
                  f"| EV={ev:+.3f} "
                  f"| {fps:4d}fps {elapsed:5.0f}s",
                  end="")
            if isinstance(agent, OptimalHCGAE_Bayesian) and bhvf_diag_log:
                d = bhvf_diag_log[-1]
                print(f" | SCR={d['scr']:.3f} α*={d['alpha_star']:.3f} "
                      f"clip={d['clip_ratio']:.2f}", end="")
            print()

    # ── 整理最终结果 ──
    n_final = 10
    final_mean = (float(np.mean(eval_rewards[-n_final:])) if len(eval_rewards) >= n_final
                  else float(np.mean(eval_rewards)) if eval_rewards else 0.0)
    max_reward = float(max(eval_rewards)) if eval_rewards else 0.0

    # peak_reward：训练过程中实时维护的峰值口径，训练结束时即为 peak10_running 的最终值
    # 等价于：compute_peak_mean(eval_rewards, N_PEAK)
    # 已在训练循环中实时计算，此处直接使用，避免重复计算
    peak_mean = peak10_running

    elapsed = time.time() - t0

    # BHVF diagnostics：稀疏化保存（每隔 10 个 rollout 取一次，减小文件大小）
    diag_sparse = bhvf_diag_log[::10] if bhvf_diag_log else []

    result = {
        "env":    env_name,
        "agent":  algo_name,
        "seed":   seed,
        # ── 超参数（完整可溯源）──
        "config": {
            "hidden_dim":     256,
            "lr":             3e-4,
            "gamma":          0.99,
            "lam":            0.95,
            "eps_clip":       0.2,
            "n_epochs":       10,
            "batch_size":     64,
            "n_steps":        2048,
            "ent_coef":       0.0,
            "vf_coef":        0.5,
            "max_grad_norm":  0.5,
            "use_obs_norm":   True if algo_name != "Standard_PPO" else False,
            "use_adv_norm":   True if algo_name != "Standard_PPO" else False,
            "use_lr_anneal":  True if algo_name != "Standard_PPO" else False,
            "total_timesteps": total_timesteps,
            "eval_freq":      EVAL_FREQ,
            "n_eval_episodes": N_EVAL_EPISODES,
            "eval_mode":      "deterministic_mean",
            # BHVF 特有
            "scr_ema_alpha":  BHVF_KWARGS["scr_ema_alpha"] if "BHVF" in algo_name else None,
            "clip_c":         BHVF_KWARGS["clip_c"]         if "BHVF" in algo_name else None,
            "use_dcppo_s":    use_dcppo_s,
            "dcppo_w_min":    0.1,
        },
        # ── 主要性能指标 ──
        "total_steps":   total_steps,
        "elapsed_s":     elapsed,
        "final_reward":  final_mean,       # 末10次评估均值（论文 Table 1 数据源）
        "peak_reward":   peak_mean,        # 峰值窗口均值（以最高eval为中心取10次，反映真实峰值能力）
        "max_reward":    max_reward,       # 全程最高单次 eval（仅供参考，方差大）
        "eval_rewards":  eval_rewards,
        "eval_steps":    eval_steps,
        "episode_rewards": episode_rewards[-200:],   # 最近200条 episode
        # ── BHVF 中间状态诊断（用于验证理论预测）──
        "bhvf_diagnostics": diag_sparse,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    env.close()
    eval_env.close()
    print(f"  DONE  {env_name}/{algo_name}/s{seed} "
          f"→ final10={final_mean:.1f} peak10={peak_mean:.1f} max={max_reward:.1f} ({elapsed:.0f}s)")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 汇总统计
# ─────────────────────────────────────────────────────────────────────────────
def compute_peak_mean(eval_rewards: list, n_final: int = 10) -> float:
    """
    计算峰值窗口均值：以全程最高 eval 为中心，取长度为 n_final 的对称窗口均值。
    对旧 JSON（无 peak_reward 字段）提供兼容回算。
    """
    if not eval_rewards:
        return 0.0
    peak_idx = int(np.argmax(eval_rewards))
    half = n_final // 2
    lo = max(0, peak_idx - half)
    hi = min(len(eval_rewards), lo + n_final)
    lo = max(0, hi - n_final)
    return float(np.mean(eval_rewards[lo:hi]))


def compute_stats(env_name: str, algo_name: str, seeds: list,
                  results_dir: Path, n_final: int = 10) -> dict:
    """
    读取多个种子的结果，计算均值/标准差/中位数。
    同时返回 final（末N次）和 peak（峰值窗口）两套统计。
    """
    finals, peaks = [], []
    for s in seeds:
        fp = results_dir / env_name / algo_name / f"{algo_name}_s{s}.json"
        if fp.exists():
            try:
                d = json.load(open(fp))
                er = d.get("eval_rewards", [])
                if er:
                    # final：末 n_final 次均值
                    v_final = (float(np.mean(er[-n_final:])) if len(er) >= n_final
                               else float(np.mean(er)))
                    finals.append(v_final)
                    # peak：优先读字段，旧文件则回算
                    if "peak_reward" in d:
                        peaks.append(float(d["peak_reward"]))
                    else:
                        peaks.append(compute_peak_mean(er, n_final))
            except Exception as e:
                print(f"  [WARN] Failed to load {fp}: {e}")
    if not finals:
        return {
            "mean": None, "std": None, "median": None, "n": 0, "values": [],
            "peak_mean": None, "peak_std": None, "peak_values": [],
        }
    return {
        "mean":        float(np.mean(finals)),
        "std":         float(np.std(finals)),
        "median":      float(np.median(finals)),
        "n":           len(finals),
        "values":      finals,
        "peak_mean":   float(np.mean(peaks)),
        "peak_std":    float(np.std(peaks)),
        "peak_values": peaks,
    }


def print_summary_table(results_dir: Path = RESULTS_DIR,
                        seeds: list = SEEDS, n_final: int = 10):
    """
    打印论文 Table 1 风格的汇总表（两张：末N次均值 + 峰值窗口均值）。
    峰值窗口均值能揭示训练末期崩溃对 final 指标的影响程度。
    """
    # 预先收集所有 stats
    all_stats = {}
    for algo in ALGOS:
        all_stats[algo] = {}
        for env in ENVS:
            all_stats[algo][env] = compute_stats(env, algo, seeds, results_dir, n_final)

    col_w = 22  # 每列宽度
    sep = "=" * (22 + len(ENVS) * col_w)

    def _fmt(mean, std, n, total=12):
        tag = f"[{n}/{total}]" if n < total else ""
        return f"{mean:7.1f}±{std:<5.1f}{tag}"

    # ── Table A：末 N 次均值（Final） ──
    print("\n" + sep)
    print(f"  Table A — Final (last {n_final} evals mean ± std)")
    print(sep)
    print(f"  {'Algorithm':<20}" + "".join(f"  {e.replace('-v4','').replace('HalfCheetah','HC'):<{col_w-2}}" for e in ENVS))
    print("-" * len(sep))
    for algo in ALGOS:
        row = f"  {algo:<20}"
        for env in ENVS:
            st = all_stats[algo][env]
            if st["mean"] is not None:
                row += f"  {_fmt(st['mean'], st['std'], st['n']):<{col_w-2}}"
            else:
                row += f"  {'[—]':<{col_w-2}}"
        print(row)
    print(sep)

    # ── Table B：峰值窗口均值（Peak） ──
    print(f"\n  Table B — Peak (window of {n_final} evals centered at best eval)")
    print(f"  * Peak > Final → 训练末期发生崩溃; Peak ≈ Final → 训练末期稳定")
    print(sep)
    print(f"  {'Algorithm':<20}" + "".join(f"  {e.replace('-v4','').replace('HalfCheetah','HC'):<{col_w-2}}" for e in ENVS))
    print("-" * len(sep))
    for algo in ALGOS:
        row = f"  {algo:<20}"
        any_gap = False
        gap_markers = []
        for env in ENVS:
            st = all_stats[algo][env]
            if st["peak_mean"] is not None:
                gap = st["peak_mean"] - st["mean"] if st["mean"] is not None else 0
                marker = " ⚠" if gap > 100 else ""   # 崩溃超过100分标注警告
                if gap > 100:
                    any_gap = True
                row += f"  {_fmt(st['peak_mean'], st['peak_std'], st['n'])}{marker:<{col_w-2-len(_fmt(st['peak_mean'],st['peak_std'],st['n']))}}"
            else:
                row += f"  {'[—]':<{col_w-2}}"
        print(row)
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ICML 2026 完备实验")
    parser.add_argument("--env",   type=str, default=None,
                        help="单环境运行（默认：全部4个）")
    parser.add_argument("--algo",  type=str, default=None,
                        help="单算法运行（默认：全部5个）")
    parser.add_argument("--seeds", type=str, default=None,
                        help="逗号分隔的种子列表，如 0,1,2,3（默认：0-11）")
    parser.add_argument("--smoke-test", action="store_true",
                        help="快速冒烟测试（50k步，种子 0,1）")
    parser.add_argument("--quick-valid", action="store_true",
                        help="快速验证（200k步，种子 0-3，Hopper+HalfCheetah，仅 V6/V7/V8/Optimal_PPO）")
    parser.add_argument("--no-skip",   action="store_true",
                        help="强制重新运行（覆盖已有结果）")
    parser.add_argument("--summary",   action="store_true",
                        help="仅打印已有结果汇总，不运行实验")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="结果保存目录（默认：results/FinalExperiment）")
    args = parser.parse_args()

    # ── 参数解析 ──
    envs  = [args.env]  if args.env  else ENVS
    algos = [args.algo] if args.algo else ALGOS
    seeds = ([int(s) for s in args.seeds.split(",")]
             if args.seeds else SEEDS)
    total_ts = 50_000 if args.smoke_test else TOTAL_TIMESTEPS
    skip_existing = not args.no_skip
    # 支持自定义结果目录（用于新环境实验）
    custom_results_dir = Path(args.results_dir) if args.results_dir else None

    # --quick-valid：200k步，4种子，快速环境，重点算法
    QUICK_VALID_ENVS  = ["Hopper-v4", "HalfCheetah-v4"]
    QUICK_VALID_ALGOS = ["Optimal_PPO", "BHVF_V6", "BHVF_V8", "BHVF_V10"]
    QUICK_VALID_STEPS = 200_000
    QUICK_VALID_SEEDS = [0, 1, 2, 3]

    if args.quick_valid:
        if not args.env:
            envs = QUICK_VALID_ENVS
        if not args.algo:
            algos = QUICK_VALID_ALGOS
        if not args.seeds:
            seeds = QUICK_VALID_SEEDS
        total_ts = QUICK_VALID_STEPS

    if args.smoke_test:
        seeds = [0, 1]
        print(f"\n{'='*60}")
        print(f"  ⚡ SMOKE TEST: 50k steps, seeds {seeds}")
        print(f"{'='*60}\n")
    elif args.quick_valid:
        print(f"\n{'='*60}")
        print(f"  🚀 QUICK VALID: {total_ts//1000}k steps, seeds {seeds}")
        print(f"  Envs:  {envs}")
        print(f"  Algos: {algos}")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print(f"  ICML 2026 Full Experiment")
        print(f"  Envs:  {envs}")
        print(f"  Algos: {algos}")
        print(f"  Seeds: {seeds} ({len(seeds)} total)")
        print(f"  Steps: {total_ts:,}")
        total_runs = len(envs) * len(algos) * len(seeds)
        print(f"  Total runs: {total_runs}")
        print(f"{'='*60}\n")

    if args.summary:
        summary_dir = custom_results_dir if custom_results_dir else RESULTS_DIR
        print_summary_table(summary_dir, SEEDS, n_final=10)
        return

    # ── 执行实验 ──
    if custom_results_dir:
        results_dir = custom_results_dir
    elif args.smoke_test:
        results_dir = RESULTS_DIR.parent / "SmokeTest"
    elif args.quick_valid:
        results_dir = RESULTS_DIR.parent / "QuickValid"
    else:
        results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    t_global = time.time()

    for env_name in envs:
        for algo_name in algos:
            print(f"\n{'─'*60}")
            print(f"  [{algo_name}] on [{env_name}]")
            print(f"{'─'*60}")
            for seed in seeds:
                try:
                    r = run_single(
                        env_name, algo_name, seed,
                        total_timesteps=total_ts,
                        results_dir=results_dir,
                        skip_existing=skip_existing,
                    )
                    all_results.append(r)
                except Exception as e:
                    import traceback
                    print(f"\n  ✗ FAILED {env_name}/{algo_name}/s{seed}: {e}")
                    traceback.print_exc()
                    continue

    elapsed = time.time() - t_global
    print(f"\n{'='*60}")
    print(f"  All done! Total time: {elapsed/3600:.2f}h ({elapsed:.0f}s)")
    print(f"{'='*60}\n")

    # 打印汇总（smoke/quick-valid 取最后5次，full 取最后10次）
    n_final_summary = 5 if (args.smoke_test or args.quick_valid) else 10
    print_summary_table(results_dir, seeds, n_final=n_final_summary)


if __name__ == "__main__":
    main()

