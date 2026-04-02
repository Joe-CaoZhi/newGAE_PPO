#!/usr/bin/env python3
"""
DCPPO 消融实验脚本
=================
在 Hopper-v4 上对 DCPPO 的三项改进（G/A/S）进行全面消融实验，
并与 HCGAE_Imp12 基线进行对比。

运行方式：
  python run_dcppo.py

结果保存至：
  results/Hopper-v4-DCPPO/
"""
import os
import sys
import time
import json
import numpy as np
import random

# ── 路径处理 ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
from gae_experiments.agents import DCPPO, build_dcppo_agent, get_all_dcppo_variant_names
from gae_experiments.agents.hindsight_ablation import HindsightAblation

# ── 实验配置 ──────────────────────────────────────────────────────────
ENV_ID         = "Hopper-v4"
TOTAL_STEPS    = 500_000     # 比消融实验多，以展现后期稳定性差异
N_STEPS        = 2048
BATCH_SIZE     = 64
N_EPOCHS       = 10
EVAL_FREQ      = 10_000
N_EVAL_EPS     = 10
SEED           = 42
SAVE_DIR       = "results/Hopper-v4-DCPPO"

# 对比基线（HCGAE_Imp12）
INCLUDE_BASELINE = True

os.makedirs(SAVE_DIR, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)


def run_variant(variant_name: str, idx: int, total: int, is_baseline: bool = False):
    """运行单个变体并返回结果"""
    print(f"\n{'='*72}")
    print(f"  [{idx+1}/{total}] 运行变体: {variant_name}")
    print(f"{'='*72}")

    set_seed(SEED)

    train_env = gym.make(ENV_ID)
    eval_env  = gym.make(ENV_ID)
    train_env.reset(seed=SEED)
    eval_env.reset(seed=SEED + 1000)

    t0 = time.time()

    if is_baseline:
        # HCGAE_Imp12 基线（与消融实验相同配置）
        agent = HindsightAblation(
            env=train_env,
            name="HCGAE_Imp12_Baseline",
            use_imp1=True, use_imp2=True,
            use_imp3=False, use_imp4=False,
            hidden_dim=64,
            lr_actor=3e-4, lr_critic=1e-3,
            gamma=0.99, lam=0.95,
            eps_clip=0.2,
            n_epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            n_steps=N_STEPS,
            save_dir=SAVE_DIR,
        )
    else:
        agent = build_dcppo_agent(
            variant_name=variant_name,
            env=train_env,
            save_dir=SAVE_DIR,
            hidden_dim=64,
            lr_actor=3e-4, lr_critic=1e-3,
            gamma=0.99, lam=0.95,
            eps_clip=0.2,
            n_epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            n_steps=N_STEPS,
            # DCPPO 超参
            geo_blend=1.0,
            beta_strict=0.6,
            beta_loose=1.4,
            eps_max=0.4,
            snr_target=0.3,
            snr_gamma=0.5,
            snr_min_weight=0.2,
        )

    logger = agent.train(
        total_timesteps=TOTAL_STEPS,
        eval_env=eval_env,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPS,
        verbose=True,
    )

    elapsed = time.time() - t0
    train_env.close()
    eval_env.close()

    # 整理结果
    eval_rewards = logger.eval_rewards if logger.eval_rewards else [0.0]
    eval_steps   = logger.eval_steps   if logger.eval_steps   else [0]

    final_reward   = float(np.mean(eval_rewards[-5:])) if len(eval_rewards) >= 5 else float(np.mean(eval_rewards))
    best_reward    = float(np.max(eval_rewards))
    stability_std  = float(np.std(eval_rewards[-10:])) if len(eval_rewards) >= 10 else float(np.std(eval_rewards))
    peak_step      = int(eval_steps[int(np.argmax(eval_rewards))]) if eval_rewards else 0

    # 收敛步（达到最高奖励 90%）
    target_90 = best_reward * 0.9
    conv_step  = peak_step
    for s, r in zip(eval_steps, eval_rewards):
        if r >= target_90:
            conv_step = s
            break

    result = {
        "variant":      variant_name,
        "is_baseline":  is_baseline,
        "final_reward": final_reward,
        "best_reward":  best_reward,
        "peak_step":    peak_step,
        "conv_step_90": conv_step,
        "stability_std": stability_std,
        "final_ev_ema": float(getattr(agent, "_ev_ema", 0.0)),
        "elapsed_s":    round(elapsed, 1),
        "eval_steps":   [int(s) for s in eval_steps],
        "eval_rewards": [float(r) for r in eval_rewards],
    }

    # 保存单变体结果
    fname = os.path.join(SAVE_DIR, f"{variant_name}_summary.json")
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  ✓ 结果已保存: {fname}")
    print(f"    最终奖励: {final_reward:.1f}  最高奖励: {best_reward:.1f}  "
          f"稳定性σ: {stability_std:.1f}  耗时: {elapsed:.0f}s")

    return result


def print_comparison_table(all_results):
    """打印完整对比表"""
    print(f"\n{'═'*88}")
    print(f"  DCPPO 改进对比  (vs HCGAE_Imp12 基线)  — Hopper-v4, {TOTAL_STEPS//1000}K steps")
    print(f"{'═'*88}")

    # 找基线
    baseline = next((r for r in all_results if r.get("is_baseline")), None)
    base_val = baseline["final_reward"] if baseline else 0.0

    header = (f"  {'变体':<22} {'G':^3} {'A':^3} {'S':^3} "
              f"{'最终奖励':>10} {'最高奖励':>10} {'Δ基线':>10} "
              f"{'稳定σ':>9} {'EV':>7} {'耗时s':>7}")
    print(header)
    print(f"  {'─'*82}")

    from gae_experiments.agents.dcppo import DCPPO_VARIANTS

    for res in all_results:
        n = res["variant"]
        delta = res["final_reward"] - base_val
        flags = DCPPO_VARIANTS.get(n, {})
        g_str = "✓" if flags.get("use_imp_g") else ("✗" if not res.get("is_baseline") else "✓")
        a_str = "✓" if flags.get("use_imp_a") else ("✗" if not res.get("is_baseline") else "✓")
        s_str = "✓" if flags.get("use_imp_s") else ("✗" if not res.get("is_baseline") else "✓")
        if res.get("is_baseline"):
            g_str = a_str = s_str = "※"
        print(
            f"  {n:<22} {g_str:^3} {a_str:^3} {s_str:^3} "
            f"{res['final_reward']:>10.1f} {res['best_reward']:>10.1f} "
            f"{delta:>+10.1f} "
            f"{res['stability_std']:>9.1f} "
            f"{res['final_ev_ema']:>7.4f} "
            f"{res['elapsed_s']:>7.0f}"
        )

    print(f"  {'─'*82}")
    print(f"  ※ HCGAE_Imp12_Baseline: 使用 HCGAE①+② GAE，标准 PPO update")


def print_dcppo_math_analysis(all_results):
    """打印 DCPPO 数学分析"""
    print(f"\n{'═'*72}")
    print(f"  【DCPPO 数学分析】")
    print(f"{'═'*72}")

    from gae_experiments.agents.dcppo import DCPPO_VARIANTS

    baseline_r = next((r["final_reward"] for r in all_results if r.get("is_baseline")), 0.0)
    rf = {r["variant"]: r["final_reward"] for r in all_results}
    rb = {r["variant"]: r["best_reward"]  for r in all_results}
    rs = {r["variant"]: r["stability_std"] for r in all_results}

    print(f"\n  1. 单项改进效果（相对 HCGAE_Imp12 基线 = {baseline_r:.1f}）")
    print(f"  {'─'*52}")
    singles = {
        "G: 几何均值归一化": "DCPPO_ImpG",
        "A: 非对称裁剪":    "DCPPO_ImpA",
        "S: SNR梯度缩放":   "DCPPO_ImpS",
    }
    for desc, vname in singles.items():
        if vname in rf:
            delta = rf[vname] - baseline_r
            delta_pct = 100 * delta / (abs(baseline_r) + 1e-8)
            print(f"    {desc}: {baseline_r:.1f} → {rf[vname]:.1f}  "
                  f"Δ={delta:+.1f} ({delta_pct:+.1f}%)  "
                  f"σ={rs.get(vname, 0):.1f}")

    print(f"\n  2. 组合改进效果")
    print(f"  {'─'*52}")
    if "DCPPO_ImpG" in rf and "DCPPO_ImpA" in rf:
        d_g = rf["DCPPO_ImpG"] - baseline_r
        d_a = rf["DCPPO_ImpA"] - baseline_r
        d_s = rf.get("DCPPO_ImpS", baseline_r) - baseline_r

        combos = [
            ("G+A", "DCPPO_ImpGA", d_g + d_a),
            ("G+S", "DCPPO_ImpGS", d_g + d_s),
            ("A+S", "DCPPO_ImpAS", d_a + d_s),
            ("G+A+S (Full)", "DCPPO_Full", d_g + d_a + d_s),
        ]
        print(f"  {'组合':<16} {'实际Δ':>10} {'加性估计':>10} {'交互效应':>12}")
        print(f"  {'─'*50}")
        for desc, vname, additive in combos:
            if vname in rf:
                actual   = rf[vname] - baseline_r
                interact = actual - additive
                judge = "协同✓" if interact > 20 else ("拮抗✗" if interact < -20 else "≈加性")
                print(f"  {desc:<16} {actual:>+10.1f} {additive:>+10.1f} "
                      f"{interact:>+12.1f}  {judge}")

    print(f"\n  3. 维度分析：几何均值归一化的理论效果")
    print(f"  {'─'*52}")
    print(f"""
  标准 PPO: ratio = exp(Σ_d Δ_d) = exp(D × Δ_mean)，D=3 (Hopper-v4)
  DCPPO-G:  r_geo = exp(Δ_mean) = ratio^(1/D)

  方差分析（独立同分布假设）：
    Var[log(ratio)]     = D · Var[Δ_d]  ← 随维度线性增长
    Var[log(r_geo)]     = Var[Δ_mean] = Var[Δ_d] / D  ← 恒定

  信任域等价性：
    使用 r_geo 的 clip ε 等价于 "每维平均偏移 ε"
    使用 ratio 的 clip ε 等价于 "所有维度总偏移 D×ε"
    → 几何均值版本的信任域更均匀，不被高维放大
  """)

    print(f"  4. 非对称裁剪的方向分析")
    print(f"  {'─'*52}")
    print(f"""
  设 β_strict={0.6}, β_loose={1.4}, ε_base=0.2

  当 (ratio-1) 与 A 异号（远离坏区域）：
    ε_loose = 0.2 × 1.4 = 0.28   ← 允许更大的安全性改进
  当 (ratio-1) 与 A 同号（进入坏区域）：
    ε_strict = 0.2 × 0.6 = 0.12  ← 严格限制危险更新

  CPI 联系：Sham Kakade (2002) 证明若每步策略更新是
  "在好方向的保守线性步"，则可以保证单调改进。
  非对称裁剪是此原则的软版本：
    好方向允许更大步长，坏方向强制更小步长。
  """)


def plot_dcppo_results(all_results):
    """生成 DCPPO 综合可视化图表"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("  ⚠ matplotlib 未安装，跳过图表生成")
        return

    from gae_experiments.agents.dcppo import DCPPO_VARIANTS

    # ── 颜色方案 ───────────────────────────────────────────────────────
    COLORS = {
        "HCGAE_Imp12_Baseline": "#607D8B",
        "DCPPO_Base" : "#9E9E9E",
        "DCPPO_ImpG" : "#2196F3",
        "DCPPO_ImpA" : "#FF9800",
        "DCPPO_ImpS" : "#4CAF50",
        "DCPPO_ImpGA": "#E91E63",
        "DCPPO_ImpGS": "#9C27B0",
        "DCPPO_ImpAS": "#00BCD4",
        "DCPPO_Full" : "#F44336",
    }
    MARKERS = {
        "HCGAE_Imp12_Baseline": "s",
        "DCPPO_Base": "o",
        "DCPPO_ImpG": "^", "DCPPO_ImpA": "D", "DCPPO_ImpS": "p",
        "DCPPO_ImpGA": "h", "DCPPO_ImpGS": "8", "DCPPO_ImpAS": "*",
        "DCPPO_Full": "P",
    }

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor('#FAFAFA')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

    # ─ 子图1：学习曲线
    ax1 = fig.add_subplot(gs[0, :2])
    for res in all_results:
        n = res["variant"]
        steps, rews = res.get("eval_steps", []), res.get("eval_rewards", [])
        if not steps:
            continue
        lw = 2.5 if n in ("HCGAE_Imp12_Baseline", "DCPPO_Full") else 1.5
        ls = "--" if "Baseline" in n else "-"
        alpha = 1.0 if n in ("HCGAE_Imp12_Baseline", "DCPPO_Full") else 0.75
        color  = COLORS.get(n, "#888888")
        marker = MARKERS.get(n, "o")
        ax1.plot(steps, rews, color=color, lw=lw, ls=ls, alpha=alpha,
                 marker=marker, markevery=max(1, len(steps)//6), markersize=5,
                 label=n.replace("HCGAE_Imp12_", "HCGAE_").replace("DCPPO_", ""))
    ax1.set_xlabel("Steps", fontsize=10)
    ax1.set_ylabel("Eval Reward", fontsize=10)
    ax1.set_title("DCPPO Ablation — Learning Curves (Hopper-v4)", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    ax1.grid(True, alpha=0.25, ls="--")
    ax1.set_facecolor('#F8F9FA')

    # ─ 子图2：最终奖励柱状图
    ax2 = fig.add_subplot(gs[0, 2])
    names  = [r["variant"].replace("HCGAE_Imp12_", "HCGAE_").replace("DCPPO_", "") for r in all_results]
    finals = [r["final_reward"] for r in all_results]
    colors_b = [COLORS.get(r["variant"], "#888888") for r in all_results]
    ax2.barh(names, finals, color=colors_b, edgecolor="white", height=0.7)
    baseline_r = next((r["final_reward"] for r in all_results if r.get("is_baseline")), 0.0)
    ax2.axvline(baseline_r, color="#607D8B", ls="--", lw=1.5, label=f"Baseline={baseline_r:.0f}")
    ax2.set_xlabel("Final Reward", fontsize=9)
    ax2.set_title("Final Performance", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(axis="x", alpha=0.25, ls="--")
    ax2.set_facecolor('#F8F9FA')

    # ─ 子图3：Δ 基线柱状图
    ax3 = fig.add_subplot(gs[1, 0])
    deltas = [r["final_reward"] - baseline_r for r in all_results]
    c3 = ["#4CAF50" if d >= 0 else "#F44336" for d in deltas]
    bars3 = ax3.barh(names, deltas, color=c3, edgecolor="white", height=0.7)
    ax3.axvline(0, color="black", lw=0.8)
    ax3.set_xlabel("Δ vs HCGAE_Imp12 Baseline", fontsize=9)
    ax3.set_title("Improvement over Baseline", fontsize=10, fontweight="bold")
    ax3.grid(axis="x", alpha=0.3, ls="--")
    ax3.set_facecolor('#F8F9FA')

    # ─ 子图4：稳定性对比
    ax4 = fig.add_subplot(gs[1, 1])
    stabs = [r["stability_std"] for r in all_results]
    c4 = [COLORS.get(r["variant"], "#888888") for r in all_results]
    ax4.barh(names, stabs, color=c4, edgecolor="white", height=0.7)
    ax4.set_xlabel("Stability σ (lower = more stable)", fontsize=9)
    ax4.set_title("Training Stability", fontsize=10, fontweight="bold")
    ax4.grid(axis="x", alpha=0.3, ls="--")
    ax4.set_facecolor('#F8F9FA')

    # ─ 子图5：性能-稳定性散点
    ax5 = fig.add_subplot(gs[1, 2])
    for res in all_results:
        n = res["variant"]
        ax5.scatter(res["stability_std"], res["final_reward"],
                    color=COLORS.get(n, "#888888"), s=130,
                    marker=MARKERS.get(n, "o"), zorder=5,
                    edgecolors="white", linewidth=0.8)
        short_name = n.replace("HCGAE_Imp12_", "HCGAE_").replace("DCPPO_", "")
        ax5.annotate(short_name, (res["stability_std"] + 3, res["final_reward"]),
                     fontsize=7)
    ax5.set_xlabel("Stability σ (lower=better)", fontsize=9)
    ax5.set_ylabel("Final Reward", fontsize=9)
    ax5.set_title("Performance vs Stability", fontsize=10, fontweight="bold")
    ax5.grid(True, alpha=0.25, ls="--")
    ax5.set_facecolor('#F8F9FA')

    fig.suptitle(
        f"DCPPO: Dual-Control PPO Ablation Study (Hopper-v4, {TOTAL_STEPS//1000}K steps, seed={SEED})",
        fontsize=13, fontweight="bold", y=1.01,
    )

    save_path = os.path.join(SAVE_DIR, "dcppo_comprehensive.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor='#FAFAFA')
    plt.close(fig)
    print(f"\n  📊 综合图表已保存: {save_path}")


# ── 主函数 ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'╔'+'═'*70+'╗'}")
    print(f"║  DCPPO 消融实验 — Hopper-v4, {TOTAL_STEPS//1000}K 步, seed={SEED}")
    print(f"║  改进：G=几何均值Ratio  A=非对称Clip  S=SNR梯度缩放")
    print(f"║  基线：HCGAE_Imp12 (标准PPO update + HCGAE①+② GAE)")
    print(f"╚{'═'*70+'╝'}\n")

    variant_names = get_all_dcppo_variant_names()
    total_runs    = len(variant_names) + (1 if INCLUDE_BASELINE else 0)
    all_results   = []

    run_idx = 0

    # 运行基线
    if INCLUDE_BASELINE:
        result = run_variant("HCGAE_Imp12_Baseline", run_idx, total_runs, is_baseline=True)
        all_results.append(result)
        run_idx += 1

    # 运行所有 DCPPO 变体
    for vname in variant_names:
        result = run_variant(vname, run_idx, total_runs, is_baseline=False)
        all_results.append(result)
        run_idx += 1

    # 保存汇总
    summary_path = os.path.join(SAVE_DIR, "dcppo_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  📁 汇总结果已保存: {summary_path}")

    # 打印分析
    print_comparison_table(all_results)
    print_dcppo_math_analysis(all_results)

    # 生成图表
    plot_dcppo_results(all_results)

    print(f"\n{'═'*72}")
    print(f"  ✅ DCPPO 实验完成！")
    print(f"  结果目录: {SAVE_DIR}/")
    print(f"{'═'*72}\n")

