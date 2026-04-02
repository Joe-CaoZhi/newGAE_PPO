#!/usr/bin/env python3
"""
DCPPO 实验结果深度分析与可视化
================================
从 results/Hopper-v4-DCPPO/ 读取各变体的 JSON 结果，生成：
  1. 汇总对比表
  2. 数学分析（协同/拮抗效应）
  3. 综合可视化图表

用法：
  python analyze_dcppo_results.py
"""
import os
import sys
import json
import glob
import numpy as np

SAVE_DIR = "results/Hopper-v4-DCPPO"
PLOT_DIR = SAVE_DIR

# DCPPO 各变体的旗标配置（与 dcppo.py 保持同步）
DCPPO_VARIANTS = {
    "DCPPO_Base" : {"use_imp_g": False, "use_imp_a": False, "use_imp_s": False},
    "DCPPO_ImpG" : {"use_imp_g": True,  "use_imp_a": False, "use_imp_s": False},
    "DCPPO_ImpA" : {"use_imp_g": False, "use_imp_a": True,  "use_imp_s": False},
    "DCPPO_ImpS" : {"use_imp_g": False, "use_imp_a": False, "use_imp_s": True },
    "DCPPO_ImpGA": {"use_imp_g": True,  "use_imp_a": True,  "use_imp_s": False},
    "DCPPO_ImpGS": {"use_imp_g": True,  "use_imp_a": False, "use_imp_s": True },
    "DCPPO_ImpAS": {"use_imp_g": False, "use_imp_a": True,  "use_imp_s": True },
    "DCPPO_Full" : {"use_imp_g": True,  "use_imp_a": True,  "use_imp_s": True },
}

# 颜色方案
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

# 有序变体列表（对应论文表格顺序）
VARIANT_ORDER = [
    "HCGAE_Imp12_Baseline",
    "DCPPO_Base",
    "DCPPO_ImpG",
    "DCPPO_ImpA",
    "DCPPO_ImpS",
    "DCPPO_ImpGA",
    "DCPPO_ImpGS",
    "DCPPO_ImpAS",
    "DCPPO_Full",
]


def load_results():
    """加载所有已完成的变体结果"""
    results = {}
    pattern = os.path.join(SAVE_DIR, "*_summary.json")
    for fpath in sorted(glob.glob(pattern)):
        with open(fpath) as f:
            d = json.load(f)
        # 跳过列表格式的汇总文件（如 dcppo_summary.json）
        if isinstance(d, list):
            continue
        vname = d.get("variant", os.path.basename(fpath).replace("_summary.json", ""))
        results[vname] = d
    return results


def print_summary_table(results):
    """打印对比汇总表"""
    baseline_r = results.get("HCGAE_Imp12_Baseline", {}).get("final_reward", 0.0)

    print(f"\n{'═'*100}")
    print(f"  DCPPO 消融实验汇总  —  Hopper-v4, 500K 步, seed=42")
    print(f"{'═'*100}")
    header = (f"  {'变体':<26} {'G':^3} {'A':^3} {'S':^3} "
              f"{'最终奖励':>10} {'最高奖励':>10} {'Δ基线':>10} "
              f"{'Δ%基线':>8} {'稳定σ':>8} {'EV':>7} {'收敛步':>10}")
    print(header)
    print(f"  {'─'*94}")

    for vname in VARIANT_ORDER:
        if vname not in results:
            continue
        r = results[vname]
        flags = DCPPO_VARIANTS.get(vname, {})
        if r.get("is_baseline"):
            g_str = a_str = s_str = "※"
        elif vname == "DCPPO_Base":
            g_str = a_str = s_str = "✗"
        else:
            g_str = "✓" if flags.get("use_imp_g") else "✗"
            a_str = "✓" if flags.get("use_imp_a") else "✗"
            s_str = "✓" if flags.get("use_imp_s") else "✗"

        delta = r["final_reward"] - baseline_r
        delta_pct = 100 * delta / (abs(baseline_r) + 1e-8)
        conv = r.get("conv_step_90", 0)
        print(
            f"  {vname:<26} {g_str:^3} {a_str:^3} {s_str:^3} "
            f"{r['final_reward']:>10.1f} {r['best_reward']:>10.1f} "
            f"{delta:>+10.1f} {delta_pct:>+7.1f}% "
            f"{r['stability_std']:>8.1f} "
            f"{r.get('final_ev_ema', 0):>7.4f} "
            f"{conv:>10}"
        )

    print(f"  {'─'*94}")
    print(f"  ※ HCGAE_Imp12_Baseline: HCGAE①+② GAE + 标准 PPO update（对照组）")


def compute_interaction_effects(results):
    """计算交互效应（协同/拮抗分析）"""
    baseline_r = results.get("HCGAE_Imp12_Baseline", {}).get("final_reward", 0.0)
    rf = {v: r["final_reward"] for v, r in results.items()}

    # 单项效果
    d_g = rf.get("DCPPO_ImpG", baseline_r) - baseline_r
    d_a = rf.get("DCPPO_ImpA", baseline_r) - baseline_r
    d_s = rf.get("DCPPO_ImpS", baseline_r) - baseline_r

    print(f"\n{'═'*72}")
    print(f"  【DCPPO 交互效应分析（协同/拮抗）】")
    print(f"{'═'*72}")
    print(f"\n  单项改进效果（基线 = {baseline_r:.1f}）")
    print(f"  {'─'*60}")
    for desc, vname, delta in [
        ("G: 几何均值 Ratio 归一化", "DCPPO_ImpG", d_g),
        ("A: 方向感知非对称裁剪",    "DCPPO_ImpA", d_a),
        ("S: SNR 自适应梯度缩放",    "DCPPO_ImpS", d_s),
    ]:
        if vname in rf:
            pct = 100 * delta / (abs(baseline_r) + 1e-8)
            stab = results[vname]["stability_std"]
            print(f"  {desc:<30} Δ={delta:>+8.1f} ({pct:>+5.1f}%)  σ={stab:.1f}")

    print(f"\n  组合改进效果 & 交互效应")
    print(f"  {'─'*60}")
    print(f"  {'组合':<18} {'实际Δ':>10} {'加性估计':>10} {'交互量':>10} {'效应类型':>10}")
    print(f"  {'─'*60}")

    combos = [
        ("G+A",       "DCPPO_ImpGA", d_g + d_a,       d_g, d_a),
        ("G+S",       "DCPPO_ImpGS", d_g + d_s,       d_g, d_s),
        ("A+S",       "DCPPO_ImpAS", d_a + d_s,       d_a, d_s),
        ("G+A+S(Full)","DCPPO_Full", d_g + d_a + d_s, d_g + d_a, d_s),
    ]
    interaction_data = {}
    for desc, vname, additive, _, _ in combos:
        if vname in rf:
            actual    = rf[vname] - baseline_r
            interact  = actual - additive
            if interact > 50:     etype = "强协同 ↑↑"
            elif interact > 20:   etype = "协同 ↑"
            elif interact > -20:  etype = "≈ 加性"
            elif interact > -50:  etype = "拮抗 ↓"
            else:                 etype = "强拮抗 ↓↓"
            interaction_data[vname] = interact
            print(f"  {desc:<18} {actual:>+10.1f} {additive:>+10.1f} {interact:>+10.1f}  {etype}")
        else:
            print(f"  {desc:<18} {'(未完成)':>32}")

    return interaction_data


def print_dcppo_math_analysis(results):
    """打印数学分析"""
    print(f"\n{'═'*72}")
    print(f"  【DCPPO 数学原理分析】")
    print(f"{'═'*72}")

    baseline_r = results.get("HCGAE_Imp12_Baseline", {}).get("final_reward", 0.0)
    best_r = max((r["best_reward"] for r in results.values()), default=0.0)

    print(f"""
  ─── 1. 几何均值 Ratio 的方差控制（改进 G）─────────────────────────

  Hopper-v4: D=3 动作维度
  标准 PPO ratio = exp(Σ_d Δ_d)，其中 Δ_d = log π_d - log π_old_d

  方差分析（独立同分布假设）：
    Var[Σ Δ_d] = D × Var[Δ]  （随维度线性增长）
    → Var[ratio] ≈ D × Var[exp(Δ)] 量级

  几何均值归一化：
    r_geo = exp(Δ_mean) 其中 Δ_mean = (1/D) Σ Δ_d
    Var[r_geo] = Var[exp(Δ_mean)] = Var[exp(Δ/D)] ← 恒定，与 D 无关

  信任域等价性：
    标准 clip ε → 实际每维允许偏移 ε/D ≈ 0.067 (D=3)
    几何均值 clip ε → 直接对"平均维度偏移"施加 ε 约束
    → DCPPO-G 的信任域在每个动作维度上更均匀

  ─── 2. 非对称裁剪的方向性理论（改进 A）─────────────────────────

  β_strict=0.6, β_loose=1.4, ε_base=0.2

  危险场景（"进入坏区域"，严格限制）：
    - ratio > 1 且 A < 0：过度强化坏动作 → ε_strict = 0.12
    - ratio < 1 且 A > 0：急剧减少好动作概率 → ε_strict = 0.12

  安全场景（"离开坏区域"，宽松限制）：
    - ratio > 1 且 A > 0：强化好动作 → ε_loose = 0.28
    - ratio < 1 且 A < 0：减少坏动作概率 → ε_loose = 0.28

  理论保证：
    CPI 单调改进定理要求"策略更新不退步"
    非对称裁剪是 CPI 的软版本：
      好方向（减少 V^old(s) 的偏差）允许更大步长 ✓
      坏方向（增大 V^old(s) 的偏差）强制更小步长 ✓

  ─── 3. SNR 自适应梯度缩放的统计理论（改进 S）──────────────────

  Advantage 估计 A_hat = A_true + ε_noise
  其中 ε_noise ~ N(0, σ_critic²)（Critic 误差）

  信噪比：SNR = |E[A]| / Std[A]
          → 当 EV 低时，σ_critic 大，SNR 低
          → 当 EV 高时，σ_critic 小，SNR 高

  梯度缩放因子：w(SNR) = min(1, SNR/SNR_target)^γ

  期望梯度误差分析：
    E[∇L_noisy] = w × E[∇L_true] + w × E[∇L_noise]
    当 SNR < SNR_target：w < 1，噪声梯度被 w 衰减
    当 SNR = SNR_target：w = 1，恢复标准 PPO

  与 HCGAE 协同：
    HCGAE①+② 提升了 EV（Explained Variance），即减少了 Critic 误差
    EV ↑ → σ_critic ↓ → SNR ↑ → w ↑ → 策略更新更积极
    形成正向循环：更准确的 Advantage → 更大的梯度 → 更快学习
""")

    print(f"\n  实验结论（Hopper-v4, 500K 步）:")
    print(f"    基线 (HCGAE_Imp12) 最终奖励: {baseline_r:.1f}")
    print(f"    最优变体 最终奖励:           {best_r:.1f}")
    if abs(baseline_r) > 1:
        print(f"    最大改进幅度:               {best_r - baseline_r:+.1f} ({100*(best_r/baseline_r-1):+.1f}%)")


def plot_dcppo_results(results):
    """生成 DCPPO 综合可视化图表（6 子图）"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.patches as mpatches
    except ImportError:
        print("  ⚠ matplotlib 未安装，跳过图表生成")
        return

    # 按顺序筛选已有变体
    ordered = [v for v in VARIANT_ORDER if v in results]
    ordered_results = [results[v] for v in ordered]

    if not ordered_results:
        print("  ⚠ 无可用结果，跳过图表生成")
        return

    baseline_r = results.get("HCGAE_Imp12_Baseline", {}).get("final_reward",
                              ordered_results[0]["final_reward"])

    fig = plt.figure(figsize=(22, 15))
    fig.patch.set_facecolor('#FAFAFA')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.40)

    # ── 子图 1：学习曲线（占 2/3 宽）──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    for res in ordered_results:
        n = res["variant"]
        steps = res.get("eval_steps", [])
        rews  = res.get("eval_rewards", [])
        if not steps:
            continue
        lw    = 2.8 if n in ("HCGAE_Imp12_Baseline", "DCPPO_Full", "DCPPO_Base") else 1.8
        ls    = "--" if "Baseline" in n else "-"
        alpha = 1.0 if n in ("HCGAE_Imp12_Baseline", "DCPPO_Full") else 0.75
        color  = COLORS.get(n, "#888888")
        marker = MARKERS.get(n, "o")
        short  = n.replace("HCGAE_Imp12_", "HCGAE_").replace("DCPPO_", "")
        ax1.plot(steps, rews, color=color, lw=lw, ls=ls, alpha=alpha,
                 marker=marker, markevery=max(1, len(steps)//7), markersize=5,
                 label=short)

    ax1.set_xlabel("Training Steps", fontsize=11)
    ax1.set_ylabel("Eval Reward", fontsize=11)
    ax1.set_title("DCPPO Ablation — Learning Curves (Hopper-v4, 500K steps)",
                  fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8.5, ncol=2, framealpha=0.92)
    ax1.grid(True, alpha=0.25, ls="--")
    ax1.set_facecolor('#F8F9FA')

    # ── 子图 2：最终奖励条形图──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    names_short = [v.replace("HCGAE_Imp12_", "HCGAE_").replace("DCPPO_", "") for v in ordered]
    finals = [results[v]["final_reward"] for v in ordered]
    colors_b = [COLORS.get(v, "#888888") for v in ordered]
    bars = ax2.barh(names_short, finals, color=colors_b, edgecolor="white", height=0.7, zorder=3)
    ax2.axvline(baseline_r, color="#607D8B", ls="--", lw=2.0, label=f"Baseline\n={baseline_r:.0f}")
    for bar, val in zip(bars, finals):
        ax2.text(val + 30, bar.get_y() + bar.get_height() / 2, f"{val:.0f}",
                 va="center", fontsize=7.5)
    ax2.set_xlabel("Final Reward (avg last 5 evals)", fontsize=9)
    ax2.set_title("Final Performance", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8.5, loc="lower right")
    ax2.grid(axis="x", alpha=0.25, ls="--", zorder=0)
    ax2.set_facecolor('#F8F9FA')

    # ── 子图 3：Δ 基线条形图────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    deltas = [results[v]["final_reward"] - baseline_r for v in ordered]
    c3 = ["#4CAF50" if d >= 0 else "#EF5350" for d in deltas]
    ax3.barh(names_short, deltas, color=c3, edgecolor="white", height=0.7, zorder=3)
    ax3.axvline(0, color="black", lw=1.0)
    for i, d in enumerate(deltas):
        ax3.text(d + (5 if d >= 0 else -5), i, f"{d:+.0f}",
                 va="center", ha="left" if d >= 0 else "right", fontsize=7.5)
    ax3.set_xlabel("Δ Final Reward vs HCGAE_Imp12 Baseline", fontsize=9)
    ax3.set_title("Improvement over Baseline", fontsize=10, fontweight="bold")
    ax3.grid(axis="x", alpha=0.3, ls="--", zorder=0)
    ax3.set_facecolor('#F8F9FA')

    # ── 子图 4：稳定性（σ）──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    stabs = [results[v]["stability_std"] for v in ordered]
    c4 = [COLORS.get(v, "#888888") for v in ordered]
    ax4.barh(names_short, stabs, color=c4, edgecolor="white", height=0.7, zorder=3)
    for i, s in enumerate(stabs):
        ax4.text(s + 5, i, f"{s:.0f}", va="center", fontsize=7.5)
    ax4.set_xlabel("Std Dev of Last-10 Evals (lower = more stable)", fontsize=9)
    ax4.set_title("Training Stability (σ)", fontsize=10, fontweight="bold")
    ax4.grid(axis="x", alpha=0.3, ls="--", zorder=0)
    ax4.set_facecolor('#F8F9FA')

    # ── 子图 5：性能-稳定性散点──────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    for v in ordered:
        r = results[v]
        short = v.replace("HCGAE_Imp12_", "HCGAE_").replace("DCPPO_", "")
        ax5.scatter(r["stability_std"], r["final_reward"],
                    color=COLORS.get(v, "#888888"),
                    s=160, marker=MARKERS.get(v, "o"),
                    zorder=5, edgecolors="white", linewidth=1.0)
        # 标签偏移
        dx = r["stability_std"] * 0.03 + 5
        ax5.annotate(short, (r["stability_std"] + dx, r["final_reward"]),
                     fontsize=7.5, va="center")
    ax5.set_xlabel("Stability σ (lower = better)", fontsize=9)
    ax5.set_ylabel("Final Reward", fontsize=9)
    ax5.set_title("Performance vs Stability Trade-off", fontsize=10, fontweight="bold")
    ax5.grid(True, alpha=0.25, ls="--")
    ax5.set_facecolor('#F8F9FA')

    # ── 总标题 ─────────────────────────────────────────────────────────
    n_done = len(ordered)
    n_total = len(VARIANT_ORDER)
    status = f"({n_done}/{n_total} variants completed)"
    fig.suptitle(
        f"DCPPO: Dual-Control PPO Ablation Study  {status}\n"
        f"Hopper-v4 · 500K steps · seed=42 · G=Geo-Ratio  A=Asym-Clip  S=SNR-Scale",
        fontsize=12, fontweight="bold", y=1.01,
    )

    save_path = os.path.join(PLOT_DIR, "dcppo_comprehensive.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor='#FAFAFA')
    plt.close(fig)
    print(f"\n  📊 综合图表已保存: {save_path}")


def plot_interaction_heatmap(results, interaction_data):
    """生成协同效应热图（改进组合交互矩阵）"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    baseline_r = results.get("HCGAE_Imp12_Baseline", {}).get("final_reward", 0.0)
    rf = {v: r["final_reward"] for v, r in results.items()}

    # 构建 3x3 矩阵（G, A, S）
    labels = ["G", "A", "S"]
    mat = np.zeros((3, 3))
    np.fill_diagonal(mat, [
        rf.get("DCPPO_ImpG", baseline_r) - baseline_r,
        rf.get("DCPPO_ImpA", baseline_r) - baseline_r,
        rf.get("DCPPO_ImpS", baseline_r) - baseline_r,
    ])

    # 填充交互效应
    pairs = {
        (0, 1): interaction_data.get("DCPPO_ImpGA", 0),
        (0, 2): interaction_data.get("DCPPO_ImpGS", 0),
        (1, 2): interaction_data.get("DCPPO_ImpAS", 0),
    }
    for (i, j), val in pairs.items():
        mat[i, j] = val
        mat[j, i] = val

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor('#FAFAFA')

    vmax = max(abs(mat).max(), 50)
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect='auto')
    plt.colorbar(im, ax=ax, label="Δ Reward / Interaction Effect")

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels, fontsize=13, fontweight="bold")
    ax.set_yticklabels(labels, fontsize=13, fontweight="bold")

    for i in range(3):
        for j in range(3):
            val = mat[i, j]
            tag = ""
            if i == j:
                tag = f"单项\nΔ={val:+.0f}"
            else:
                if abs(val) < 20:   tag = f"≈加性\n{val:+.0f}"
                elif val > 0:        tag = f"协同↑\n{val:+.0f}"
                else:                tag = f"拮抗↓\n{val:+.0f}"
            ax.text(j, i, tag, ha="center", va="center", fontsize=9,
                    color="black" if abs(mat[i, j]) < vmax * 0.6 else "white")

    ax.set_title("DCPPO 改进交互效应矩阵\n(对角=单项效果, 非对角=协同/拮抗量)",
                 fontsize=11, fontweight="bold", pad=12)

    save_path = os.path.join(PLOT_DIR, "dcppo_interaction_matrix.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 交互效应热图已保存: {save_path}")


def plot_radar(results):
    """雷达图：各变体在多个指标上的综合表现"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    ordered = [v for v in VARIANT_ORDER if v in results]
    if len(ordered) < 3:
        print("  ⚠ 变体数量不足，跳过雷达图")
        return

    metrics_labels = ["最终奖励", "最高奖励", "稳定性\n(反σ)", "早期速度\n(10%)", "EV"]
    N = len(metrics_labels)

    # 归一化各指标到 [0, 1]
    all_final  = np.array([results[v]["final_reward"] for v in ordered])
    all_best   = np.array([results[v]["best_reward"]  for v in ordered])
    all_std    = np.array([results[v]["stability_std"] for v in ordered])
    all_ev     = np.array([results[v].get("final_ev_ema", 0.8) for v in ordered])

    # 早期速度（前 20% 步的最大评估奖励）
    all_early = []
    for v in ordered:
        r = results[v]
        steps = r.get("eval_steps", [])
        rews  = r.get("eval_rewards", [])
        if steps and rews:
            cutoff = max(steps) * 0.2
            early_r = [rw for s, rw in zip(steps, rews) if s <= cutoff]
            all_early.append(max(early_r) if early_r else rews[0])
        else:
            all_early.append(0)
    all_early = np.array(all_early, dtype=float)

    def norm01(arr):
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)

    normed = np.column_stack([
        norm01(all_final),
        norm01(all_best),
        1 - norm01(all_std),  # 稳定性：σ 越低越好，反转
        norm01(all_early),
        norm01(all_ev),
    ])

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#FAFAFA')

    # 只绘制部分关键变体
    key_variants = [v for v in ["HCGAE_Imp12_Baseline", "DCPPO_Base", "DCPPO_Full",
                                 "DCPPO_ImpG", "DCPPO_ImpA", "DCPPO_ImpS"]
                    if v in results]

    for v in key_variants:
        idx = ordered.index(v)
        vals = normed[idx].tolist()
        vals += vals[:1]
        short = v.replace("HCGAE_Imp12_", "HCGAE_").replace("DCPPO_", "")
        lw = 2.5 if v in ("HCGAE_Imp12_Baseline", "DCPPO_Full") else 1.8
        ls = "--" if "Baseline" in v else "-"
        ax.plot(angles, vals, color=COLORS.get(v, "#888"), lw=lw, ls=ls,
                marker="o", markersize=5, label=short)
        ax.fill(angles, vals, color=COLORS.get(v, "#888"), alpha=0.06)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_labels, fontsize=10)
    ax.set_yticklabels([])
    ax.set_title("DCPPO 变体多维度性能雷达图\n(归一化至 [0, 1])",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)

    save_path = os.path.join(PLOT_DIR, "dcppo_radar.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 雷达图已保存: {save_path}")


def main():
    print(f"\n{'╔'+'═'*68+'╗'}")
    print(f"║  DCPPO 结果深度分析  —  results/Hopper-v4-DCPPO/")
    print(f"╚{'═'*68+'╝'}\n")

    results = load_results()
    if not results:
        print(f"  ⚠ 未在 {SAVE_DIR}/ 找到任何 *_summary.json 文件")
        print("  请等待 run_dcppo.py 完成后再运行本脚本")
        return

    n_done = len([v for v in VARIANT_ORDER if v in results])
    print(f"  ✓ 已加载 {n_done}/{len(VARIANT_ORDER)} 个变体的结果")
    for v in VARIANT_ORDER:
        status = "✓ 已完成" if v in results else "○ 待完成"
        final  = f"最终={results[v]['final_reward']:.1f}" if v in results else ""
        print(f"      {status}  {v:<28}  {final}")

    # ── 打印分析 ─────────────────────────────────────────────────────
    print_summary_table(results)
    interaction_data = compute_interaction_effects(results)
    print_dcppo_math_analysis(results)

    # ── 生成图表 ─────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("  生成可视化图表...")
    plot_dcppo_results(results)
    plot_interaction_heatmap(results, interaction_data)
    plot_radar(results)

    # ── 汇总 JSON ────────────────────────────────────────────────────
    if n_done == len(VARIANT_ORDER):
        summary_path = os.path.join(SAVE_DIR, "dcppo_full_summary.json")
        export = []
        for v in VARIANT_ORDER:
            if v in results:
                export.append(results[v])
        with open(summary_path, "w") as f:
            json.dump(export, f, indent=2)
        print(f"\n  📁 完整汇总已保存: {summary_path}")
    else:
        print(f"\n  ℹ 还有 {len(VARIANT_ORDER) - n_done} 个变体未完成，")
        print("    再次运行本脚本以获得完整分析结果")

    print(f"\n{'═'*70}")
    print(f"  ✅ 分析完成！图表保存于: {SAVE_DIR}/")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()

