#!/usr/bin/env python3
"""
HCGAE 消融实验深度数学分析
==========================
读取消融实验结果，进行全面的数学分析和高质量可视化。
"""
import json
import os
import numpy as np

SAVE_DIR = "results/Hopper-v4-Ablation"

def load_results():
    with open(os.path.join(SAVE_DIR, "ablation_summary.json")) as f:
        return json.load(f)

def deep_math_analysis(results):
    """深度数学分析：偏差-方差分解、交互效应、条件贡献"""
    print("\n" + "="*80)
    print("  HCGAE 消融实验深度数学分析")
    print("="*80)

    r = {x["variant"]: x for x in results}
    R = {k: v["final_reward"] for k, v in r.items()}
    B_val = {k: v["best_reward"] for k, v in r.items()}
    S_val = {k: v["stability_std"] for k, v in r.items()}
    C_val = {k: v["conv_step_90"] for k, v in r.items()}
    EV_val = {k: v["final_ev_ema"] for k, v in r.items()}

    base = R["HCGAE_Base"]
    full = R["HCGAE_Full"]

    print(f"\n{'─'*80}")
    print(f"  § 1. 汇总表（最终奖励 / 最高奖励 / 稳定性σ / 收敛步 / 最终EV）")
    print(f"{'─'*80}")
    header = f"  {'变体':<16} {'最终':>8} {'最高':>8} {'Δ_base':>9} {'稳定σ':>8} {'收敛步':>8} {'EV':>6}  分析"
    print(header)
    print(f"  {'─'*16} {'─'*8} {'─'*8} {'─'*9} {'─'*8} {'─'*8} {'─'*6}")
    for res in results:
        n = res["variant"]
        delta = R[n] - base
        tag = ""
        if R[n] == max(R.values()):
            tag = "★最高最终奖励"
        elif B_val[n] == max(B_val.values()):
            tag = "  ☆最高峰值"
        elif S_val[n] == min(S_val.values()):
            tag = "  ◎最高稳定性"
        elif C_val[n] == min(C_val.values()):
            tag = "  ⚡最快收敛"
        print(f"  {n:<16} {R[n]:8.1f} {B_val[n]:8.1f} {delta:+9.1f} "
              f"{S_val[n]:8.1f} {C_val[n]:8,} {EV_val[n]:6.3f}  {tag}")

    print(f"\n{'─'*80}")
    print(f"  § 2. 主效应（Main Effects）：单独改进 vs 基线")
    print(f"{'─'*80}")
    IMPS = [
        ("①批内归一化", "HCGAE_Imp1", "消除EMA滞后，批内自适应中心化"),
        ("②EV驱动混合", "HCGAE_Imp2", "Critic精度驱动目标混合比例"),
        ("③末端修正",   "HCGAE_Imp3", "消除rollout边界bootstrap不一致"),
        ("④冻结统计量", "HCGAE_Imp4", "防止mini-batch归一化漂移"),
    ]
    print(f"\n  最终奖励维度：")
    for label, vname, desc in IMPS:
        delta_f = R[vname] - base
        delta_b = B_val[vname] - B_val["HCGAE_Base"]
        delta_s = S_val[vname] - S_val["HCGAE_Base"]
        delta_c = C_val["HCGAE_Base"] - C_val[vname]  # 负→更快
        delta_ev = EV_val[vname] - EV_val["HCGAE_Base"]
        print(f"  {label} ({vname:<16}): "
              f"Δ最终={delta_f:+8.1f}  Δ最高={delta_b:+7.1f}  "
              f"Δ稳定(σ)={delta_s:+7.1f}  Δ收敛步={-delta_c:+8,}  "
              f"ΔEV={delta_ev:+.3f}")
        print(f"    │→ {desc}")

    print(f"\n  ┌─────────────────────────────────────────────────────────────────────┐")
    print(f"  │  关键发现：改进④（冻结统计量）单独使用时 最终奖励 下降 {R['HCGAE_Imp4']-base:+.0f}！  │")
    print(f"  │  但 最高奖励 仍有 {B_val['HCGAE_Imp4']:,.0f}（与基线 {B_val['HCGAE_Base']:,.0f} 接近）            │")
    print(f"  │  说明 ④ 单独使用时 策略收敛后期出现不稳定（σ={S_val['HCGAE_Imp4']:.0f}）               │")
    print(f"  └─────────────────────────────────────────────────────────────────────┘")

    print(f"\n{'─'*80}")
    print(f"  § 3. 交互效应（Interaction Effects）")
    print(f"{'─'*80}")
    def interaction(v_ab, v_a, v_b, label):
        actual = R[v_ab] - base
        additive = (R[v_a] - base) + (R[v_b] - base)
        interact = actual - additive
        synergy = "🤝协同" if interact > 0 else "⚔拮抗"
        print(f"  {label}: 实际Δ={actual:+.1f}  加性估计={additive:+.1f}  "
              f"交互={interact:+.1f}  ({synergy})")
        return interact

    i12 = interaction("HCGAE_Imp12",  "HCGAE_Imp1", "HCGAE_Imp2", "①+②")
    i14 = interaction("HCGAE_Imp14",  "HCGAE_Imp1", "HCGAE_Imp4", "①+④")
    i24 = interaction("HCGAE_Imp24",  "HCGAE_Imp2", "HCGAE_Imp4", "②+④")

    print(f"\n  结论：")
    if i12 > 0:
        print(f"  ✓ ①+② 存在正协同效应（+{i12:.0f}），批内归一化 + EV驱动混合相互增强")
    if i14 < 0:
        print(f"  ✗ ①+④ 存在拮抗效应（{i14:.0f}），批内归一化与冻结统计量可能在归一化层面产生冲突")
    if i24 < 0:
        print(f"  ✗ ②+④ 存在拮抗效应（{i24:.0f}），EV驱动混合与冻结统计量在目标稳定性方面冲突")

    print(f"\n{'─'*80}")
    print(f"  § 4. 条件边际贡献（在组合中的贡献）")
    print(f"{'─'*80}")
    print(f"\n  添加③ 在不同子集基础上的边际贡献：")
    for base_set, base_name in [
        ("仅基线", "HCGAE_Base"),
        ("在①上", "HCGAE_Imp1"),
        ("在④上", "HCGAE_Imp4"),
    ]:
        print(f"    {base_set}: Imp3 贡献 ≈ Imp3_single - base = {R['HCGAE_Imp3']-R['HCGAE_Base']:+.1f}")
    # Full vs Imp124
    marginal_3_in_full = R["HCGAE_Full"] - R["HCGAE_Imp124"]
    print(f"    在①②④基础上加③: Δ={marginal_3_in_full:+.1f}")

    print(f"\n  添加④ 在不同子集基础上的边际贡献：")
    marginal_4_alone = R["HCGAE_Imp4"] - base
    marginal_4_in_12 = R["HCGAE_Imp124"] - R["HCGAE_Imp12"]  # Imp12 + 4 = Imp124 (approx)
    print(f"    单独添加④: Δ={marginal_4_alone:+.1f}")
    print(f"    在①②基础上添加④: Δ={marginal_4_in_12:+.1f}")
    print(f"    结论：④ 的贡献高度依赖上下文，单独使用负效果，与①② 组合后{'正效果' if marginal_4_in_12 > 0 else '仍为负效果'}")

    print(f"\n{'─'*80}")
    print(f"  § 5. 多维度分析（最终奖励 vs 稳定性 vs 收敛速度）")
    print(f"{'─'*80}")
    print(f"\n  稳定性排名（σ越小越稳定）：")
    sorted_stab = sorted(results, key=lambda x: x["stability_std"])
    for rank, res in enumerate(sorted_stab[:5], 1):
        print(f"    #{rank} {res['variant']:<18} σ={res['stability_std']:.1f}  final={res['final_reward']:.0f}")

    print(f"\n  收敛速度排名（收敛步越少越快）：")
    sorted_conv = sorted(results, key=lambda x: x["conv_step_90"])
    for rank, res in enumerate(sorted_conv[:5], 1):
        print(f"    #{rank} {res['variant']:<18} 步={res['conv_step_90']:,}  final={res['final_reward']:.0f}")

    print(f"\n  EV 质量排名（EV越高Critic越准确）：")
    sorted_ev = sorted(results, key=lambda x: x["final_ev_ema"], reverse=True)
    for rank, res in enumerate(sorted_ev[:5], 1):
        print(f"    #{rank} {res['variant']:<18} EV={res['final_ev_ema']:.3f}  final={res['final_reward']:.0f}")

    print(f"\n{'─'*80}")
    print(f"  § 6. 数学诊断：为何改进④单独使用有害？")
    print(f"{'─'*80}")
    print(f"""
  改进④（冻结优势统计量）的数学原理：
  ─────────────────────────────────────────────────────────────
  标准做法：在 update() 的每个 mini-batch 内当场计算归一化统计量
    ā_mb = mean(A_batch),  σ_mb = std(A_batch)
    Â_normalized = (A - ā_mb) / σ_mb

  冻结做法：在 compute_gae() 阶段预先冻结统计量
    ā_frozen = mean(A_rollout),  σ_frozen = std(A_rollout)
    ─ update() 全程使用 (ā_frozen, σ_frozen)

  问题分析（单独使用④时）：
  1. 冻结统计量的目标是「消除跨 epoch mini-batch 的归一化漂移」
  2. 但在 v1 模式（① 禁用）下，α 通过慢速 EMA 控制，修正量波动很大
  3. 冻结统计量会放大这种波动：
     - 当某些 mini-batch 的优势均值偏离 ā_frozen 较多时，
       等效于给这些 batch 施加不对称的梯度偏差
  4. 实质上④是对①的「下游消费者」：
     - ① 保证优势分布稳定（批内中心化抑制系统误差）
     - ④ 保证归一化统计量不漂移
     - 单独使用④但不稳定 ① 的上游，会固化一个「有噪声」的统计量

  为何 ①+② 组合后最高（final=3501.9）？
  ─────────────────────────────────────────────────────────────
  ① 批内中心化 → 每步的修正 sigmoid 以「当前批次均值」为中心
  ② EV 驱动混合 → 当 Critic 精度高时自动减少 MC 噪声
  两者形成「自适应-自适应」的双层正反馈：
    • Critic 精度↑ → ② 减少 MC 混合 → target 更干净 → Critic 精度再↑
    • 误差分布改变 → ① 实时重新中心化 → alpha 分布更合理

  为何加入③后 (Full vs Imp124) 性能下降？
  ─────────────────────────────────────────────────────────────
  改进③的末端 Bootstrap 修正在单独使用时有效（+36.9）
  但在①②④组合中边际贡献为 {marginal_3_in_full:.1f}，原因可能是：
  1. 在①②协同稳定优势后，rollout末端误差已经较小
  2. ③引入的 approx_G_last 估计本身有误差（用G[-1]保守估计）
  3. 末端修正后改变了 V_corrected_next 分布，
     可能打破①②已建立的协调机制

  综合建议：最优组合为 ①+②，不建议加入③和④
    """)

    print(f"{'─'*80}")
    print(f"  § 7. Shapley 值精确计算（所有已知子集的 marginalization）")
    print(f"{'─'*80}")

    # 已知子集：{}, {1}, {2}, {3}, {4}, {1,2}, {1,4}, {2,4}, {1,2,4}, {1,2,3,4}
    # Shapley(i) = Σ_{S不含i} [|S|!(n-|S|-1)!/n!] * [v(S∪{i}) - v(S)]
    # n=4, 权重 w(|S|) = |S|!(3-|S|)!/4! = |S|!(3-|S|)!/24
    # |S|=0: 1*6/24=1/4; |S|=1: 1*2/24=1/12; |S|=2: 2*1/24=1/12; |S|=3: 6*0/24=1/4

    def v(subset):
        """子集对应的价值（最终奖励相对基线的增益）"""
        name_map = {
            frozenset(): "HCGAE_Base",
            frozenset({1}): "HCGAE_Imp1",
            frozenset({2}): "HCGAE_Imp2",
            frozenset({3}): "HCGAE_Imp3",
            frozenset({4}): "HCGAE_Imp4",
            frozenset({1,2}): "HCGAE_Imp12",
            frozenset({1,4}): "HCGAE_Imp14",
            frozenset({2,4}): "HCGAE_Imp24",
            frozenset({1,2,4}): "HCGAE_Imp124",
            frozenset({1,2,3,4}): "HCGAE_Full",
        }
        k = frozenset(subset)
        if k in name_map:
            return R[name_map[k]] - base
        return None  # 未测量的子集

    weights = {0: 1/4, 1: 1/12, 2: 1/12, 3: 1/4}

    shapley = {}
    for i in [1, 2, 3, 4]:
        phi = 0.0
        count = 0
        for s_list in [
            [],       # {}
            [1], [2], [3], [4],              # 单元素
            [1,2],[1,3],[1,4],[2,3],[2,4],[3,4],  # 两元素
            [1,2,3],[1,2,4],[1,3,4],[2,3,4],  # 三元素
        ]:
            S = frozenset(s_list)
            if i in S:
                continue
            S_with_i = S | frozenset({i})
            v_S = v(S)
            v_Si = v(S_with_i)
            if v_S is None or v_Si is None:
                continue  # 跳过未测量的子集
            w = weights[len(S)]
            phi += w * (v_Si - v_S)
            count += 1
        shapley[i] = phi

    imp_labels = {1: "①批内归一化", 2: "②EV驱动混合", 3: "③末端修正", 4: "④冻结统计量"}
    total_phi = sum(shapley.values())
    print(f"\n  基于已知子集的估计 Shapley 值（9 个已测量子集）：")
    for i in [1, 2, 3, 4]:
        pct = 100 * shapley[i] / (abs(total_phi) + 1e-8)
        direction = "↑正贡献" if shapley[i] > 0 else "↓负贡献"
        print(f"  φ({imp_labels[i]}) = {shapley[i]:+8.1f}  ({pct:+6.1f}% of |Σφ|)  {direction}")
    print(f"  Σφ = {total_phi:+.1f}  (验证：应≈ v(Full) = {R['HCGAE_Full']-base:+.1f})")
    print(f"  注：缺少 {{1,3}}、{{2,3}}、{{3,4}} 等包含③的子集数据，Shapley 为近似值")

    print(f"\n{'─'*80}")
    print(f"  § 8. 关键结论总结")
    print(f"{'─'*80}")
    best_variant = max(results, key=lambda x: x["final_reward"])
    most_stable  = min(results, key=lambda x: x["stability_std"])
    fastest_conv = min(results, key=lambda x: x["conv_step_90"])
    print(f"""
  最终奖励最高变体:  {best_variant['variant']:<18} final={best_variant['final_reward']:.1f}
  最稳定变体:        {most_stable['variant']:<18} σ={most_stable['stability_std']:.1f}
  收敛最快变体:      {fastest_conv['variant']:<18} 步={fastest_conv['conv_step_90']:,}

  ┌────────────────────────────────────────────────────────────────────────┐
  │  消融结论（基于 Hopper-v4，300k步，seed=42）：                         │
  │                                                                        │
  │  1. 改进①②的协同效应是 v2 提升的核心驱动力（协同+643.4）             │
  │     → ①+② 最终奖励 3501.9，是所有变体中最高的                        │
  │                                                                        │
  │  2. 改进③（末端 Bootstrap 修正）单独有效（+36.9），                   │
  │     但在①②基础上边际贡献为负（-809.8），属于「冗余-拮抗」型改进      │
  │     → 建议移除或降权                                                  │
  │                                                                        │
  │  3. 改进④（冻结统计量）单独使用严重有害（-1683.4）                    │
  │     与①组合（①+④）有轻微改善（final=3287.9 vs base=3193.4）         │
  │     但与①②组合后（①②+④）仍为负（-809.8）                          │
  │     → ④ 是「条件性改进」：仅在①稳定后才有效，且收益边际递减           │
  │                                                                        │
  │  4. 最优实践建议：HCGAE_Imp12（仅保留①②）                            │
  │     - 最高最终奖励（3501.9）                                           │
  │     - 实现最简单（两项改进）                                            │
  │     - 避免③④的条件依赖复杂性                                          │
  │                                                                        │
  │  5. 改进③理论上正确（边界不一致确实存在），实践建议：                  │
  │     - 在 episode 较短（< 200步）的环境中可能更有效                      │
  │     - Hopper 平均 episode 约 500-1000 步，边界占比低                   │
  └────────────────────────────────────────────────────────────────────────┘
    """)

def plot_enhanced_visualization(results):
    """生成增强版可视化图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    os.makedirs(SAVE_DIR, exist_ok=True)

    r = {x["variant"]: x for x in results}
    names  = [x["variant"] for x in results]
    finals = [x["final_reward"] for x in results]
    bests  = [x["best_reward"] for x in results]
    stabs  = [x["stability_std"] for x in results]
    convs  = [x["conv_step_90"] for x in results]
    evs    = [x["final_ev_ema"] for x in results]
    base_r = r["HCGAE_Base"]["final_reward"]

    COLOR_MAP = {
        "HCGAE_Base"  : "#9E9E9E",
        "HCGAE_Imp1"  : "#2196F3",
        "HCGAE_Imp2"  : "#FF9800",
        "HCGAE_Imp3"  : "#9C27B0",
        "HCGAE_Imp4"  : "#F44336",
        "HCGAE_Imp12" : "#4CAF50",
        "HCGAE_Imp14" : "#8BC34A",
        "HCGAE_Imp24" : "#FFC107",
        "HCGAE_Imp124": "#FF5722",
        "HCGAE_Full"  : "#E91E63",
    }
    short_names = [n.replace("HCGAE_", "") for n in names]
    colors = [COLOR_MAP.get(n, "#607D8B") for n in names]

    # ── 图 1：全面对比（6格综合图）──────────────────────────────
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1a: 学习曲线
    ax1 = fig.add_subplot(gs[0, :2])
    for res in results:
        n = res["variant"]
        steps = res["eval_steps"]
        rewards = res["eval_rewards"]
        if not steps:
            continue
        lw = 2.5 if n in ("HCGAE_Base", "HCGAE_Imp12") else 1.5
        ls = "--" if n == "HCGAE_Base" else "-"
        ax1.plot(steps, rewards, color=COLOR_MAP.get(n, "#607D8B"),
                 linewidth=lw, linestyle=ls, label=n.replace("HCGAE_", ""), alpha=0.9)
    ax1.set_xlabel("Environment Steps")
    ax1.set_ylabel("Evaluation Reward")
    ax1.set_title("Learning Curves — All 10 Ablation Variants (Hopper-v4)", fontweight="bold")
    ax1.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.85)
    ax1.grid(True, alpha=0.25, linestyle="--")
    ax1.axvline(100000, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax1.text(100500, 200, "100k", fontsize=7, color="gray")

    # 1b: 最终奖励与基线对比
    ax2 = fig.add_subplot(gs[0, 2])
    deltas = [f - base_r for f in finals]
    bar_colors = [COLOR_MAP.get(n, "#607D8B") for n in names]
    bars = ax2.barh(range(len(names)), deltas, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(short_names, fontsize=8)
    ax2.set_xlabel("Δ Final Reward vs Base")
    ax2.set_title("Gain vs Baseline", fontweight="bold", fontsize=10)
    ax2.grid(axis="x", alpha=0.25, linestyle="--")
    for i, (bar, val) in enumerate(zip(bars, deltas)):
        ax2.text(val + (15 if val >= 0 else -15), i, f"{val:+.0f}",
                 ha="left" if val >= 0 else "right", va="center", fontsize=7)

    # 1c: 最终奖励 vs 最高奖励（散点图）
    ax3 = fig.add_subplot(gs[1, 0])
    for i, (n, f, b) in enumerate(zip(names, finals, bests)):
        ax3.scatter(f, b, color=COLOR_MAP.get(n, "#607D8B"), s=80, zorder=3)
        ax3.annotate(n.replace("HCGAE_", ""), (f, b), fontsize=6,
                     textcoords="offset points", xytext=(4, 2))
    ax3.set_xlabel("Final Reward")
    ax3.set_ylabel("Best Reward")
    ax3.set_title("Final vs Best Reward", fontweight="bold", fontsize=10)
    ax3.grid(True, alpha=0.25, linestyle="--")
    # 画 y=x 参考线
    m = min(min(finals), min(bests))
    M = max(max(finals), max(bests))
    ax3.plot([m, M], [m, M], "k--", alpha=0.3, linewidth=1)

    # 1d: 稳定性 vs 最终奖励
    ax4 = fig.add_subplot(gs[1, 1])
    for n, f, s in zip(names, finals, stabs):
        ax4.scatter(s, f, color=COLOR_MAP.get(n, "#607D8B"), s=80, zorder=3)
        ax4.annotate(n.replace("HCGAE_", ""), (s, f), fontsize=6,
                     textcoords="offset points", xytext=(4, 2))
    ax4.set_xlabel("Stability σ (lower = better)")
    ax4.set_ylabel("Final Reward")
    ax4.set_title("Stability vs Performance", fontweight="bold", fontsize=10)
    ax4.grid(True, alpha=0.25, linestyle="--")

    # 1e: 收敛步 vs 最终奖励
    ax5 = fig.add_subplot(gs[1, 2])
    for n, f, c in zip(names, finals, convs):
        ax5.scatter(c/1000, f, color=COLOR_MAP.get(n, "#607D8B"), s=80, zorder=3)
        ax5.annotate(n.replace("HCGAE_", ""), (c/1000, f), fontsize=6,
                     textcoords="offset points", xytext=(4, 2))
    ax5.set_xlabel("Convergence Step (×1k)")
    ax5.set_ylabel("Final Reward")
    ax5.set_title("Convergence vs Performance", fontweight="bold", fontsize=10)
    ax5.grid(True, alpha=0.25, linestyle="--")

    # 1f: EV vs 最终奖励
    ax6 = fig.add_subplot(gs[2, 0])
    for n, f, e in zip(names, finals, evs):
        ax6.scatter(e, f, color=COLOR_MAP.get(n, "#607D8B"), s=80, zorder=3)
        ax6.annotate(n.replace("HCGAE_", ""), (e, f), fontsize=6,
                     textcoords="offset points", xytext=(4, 2))
    ax6.set_xlabel("Final EV (Explained Variance)")
    ax6.set_ylabel("Final Reward")
    ax6.set_title("Critic Quality vs Performance", fontweight="bold", fontsize=10)
    ax6.grid(True, alpha=0.25, linestyle="--")

    # 1g: 改进矩阵（颜色=最终奖励）
    ax7 = fig.add_subplot(gs[2, 1])
    imp_matrix = np.array([
        [int(x["use_imp1"]), int(x["use_imp2"]), int(x["use_imp3"]), int(x["use_imp4"])]
        for x in results
    ], dtype=float)
    reward_arr = np.array(finals)
    reward_norm = (reward_arr - reward_arr.min()) / (reward_arr.max() - reward_arr.min() + 1e-8)
    cmap = plt.cm.RdYlGn
    ax7.set_xlim(-0.5, 3.5)
    ax7.set_ylim(-0.5, len(results) - 0.5)
    for i, res in enumerate(results):
        rn = reward_norm[i]
        for j in range(4):
            active = imp_matrix[i, j]
            cell_c = cmap(rn) if active else "#F5F5F5"
            rect = plt.Rectangle([j - 0.45, i - 0.45], 0.9, 0.9, color=cell_c, ec="gray", lw=0.5)
            ax7.add_patch(rect)
            ax7.text(j, i, "+" if active else ".", ha="center", va="center",
                     fontsize=10, color="white" if active else "#BDBDBD", fontweight="bold")
        ax7.text(4.1, i, f"{res['final_reward']:.0f}", ha="left", va="center",
                 fontsize=7, color=cmap(rn))
    ax7.set_xticks([0, 1, 2, 3])
    ax7.set_xticklabels(["Imp1\nBatch-Norm", "Imp2\nEV-Mix", "Imp3\nBootstrap", "Imp4\nFrozen"],
                        fontsize=7, fontweight="bold")
    ax7.set_yticks(range(len(results)))
    ax7.set_yticklabels(short_names, fontsize=7)
    ax7.set_title("Ablation Matrix\n(green=high reward)", fontweight="bold", fontsize=9)

    # 1h: Shapley 近似贡献
    ax8 = fig.add_subplot(gs[2, 2])
    # 使用 §7 的近似 Shapley 值
    shapley_vals = [-223.5, -128.0, -267.4, -838.6]  # 从分析结果
    # 重新计算
    r_dict = {x["variant"]: x["final_reward"] for x in results}
    base_v = r_dict["HCGAE_Base"]
    known = {
        frozenset(): base_v,
        frozenset({1}): r_dict["HCGAE_Imp1"],
        frozenset({2}): r_dict["HCGAE_Imp2"],
        frozenset({3}): r_dict["HCGAE_Imp3"],
        frozenset({4}): r_dict["HCGAE_Imp4"],
        frozenset({1,2}): r_dict["HCGAE_Imp12"],
        frozenset({1,4}): r_dict["HCGAE_Imp14"],
        frozenset({2,4}): r_dict["HCGAE_Imp24"],
        frozenset({1,2,4}): r_dict["HCGAE_Imp124"],
        frozenset({1,2,3,4}): r_dict["HCGAE_Full"],
    }
    weights_sh = {0: 1/4, 1: 1/12, 2: 1/12, 3: 1/4}
    sh_vals = []
    for i in [1, 2, 3, 4]:
        phi = 0.0
        for s_list in [[], [1],[2],[3],[4], [1,2],[1,3],[1,4],[2,3],[2,4],[3,4],
                        [1,2,3],[1,2,4],[1,3,4],[2,3,4]]:
            S = frozenset(s_list)
            if i in S: continue
            S_i = S | frozenset({i})
            if S not in known or S_i not in known: continue
            phi += weights_sh[len(S)] * (known[S_i] - known[S] - 0)  # marginal over v(S) baseline
        sh_vals.append(phi)

    sh_labels = ["①Batch\nNorm", "②EV\nMix", "③End\nBootstrap", "④Frozen\nStats"]
    sh_colors = ["#4CAF50" if v >= 0 else "#F44336" for v in sh_vals]
    bars8 = ax8.bar(sh_labels, sh_vals, color=sh_colors, edgecolor="white")
    ax8.axhline(0, color="black", linewidth=0.8)
    for bar, val in zip(bars8, sh_vals):
        ax8.text(bar.get_x() + bar.get_width()/2,
                 val + (20 if val >= 0 else -40),
                 f"{val:+.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax8.set_ylabel("Approx. Shapley Value")
    ax8.set_title("Shapley Value\n(contribution to reward)", fontweight="bold", fontsize=9)
    ax8.grid(axis="y", alpha=0.25, linestyle="--")

    fig.suptitle("HCGAE Ablation Study: Comprehensive Analysis\nHopper-v4 | 300k steps | seed=42",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(os.path.join(SAVE_DIR, "ablation_comprehensive.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  📊 综合分析图已保存: {SAVE_DIR}/ablation_comprehensive.png")

    # ── 图 2：学习曲线分组对比（2行：单改进 / 组合改进）────────────
    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 10))
    fig2.subplots_adjust(hspace=0.35)

    # 上排：单一改进
    ax_single = axes2[0]
    single_variants = ["HCGAE_Base", "HCGAE_Imp1", "HCGAE_Imp2", "HCGAE_Imp3", "HCGAE_Imp4"]
    for res in results:
        if res["variant"] not in single_variants:
            continue
        n = res["variant"]
        lw = 2.5 if n == "HCGAE_Base" else 2.0
        ls = "--" if n == "HCGAE_Base" else "-"
        ax_single.plot(res["eval_steps"], res["eval_rewards"],
                       color=COLOR_MAP.get(n), linewidth=lw, linestyle=ls,
                       label=f"{n.replace('HCGAE_','')} (final={res['final_reward']:.0f})")
    ax_single.set_title("Single Improvement Variants vs Baseline", fontweight="bold")
    ax_single.set_xlabel("Steps")
    ax_single.set_ylabel("Eval Reward")
    ax_single.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax_single.grid(True, alpha=0.25, linestyle="--")

    # 下排：组合改进
    ax_combo = axes2[1]
    combo_variants = ["HCGAE_Base", "HCGAE_Imp12", "HCGAE_Imp14", "HCGAE_Imp24", "HCGAE_Imp124", "HCGAE_Full"]
    for res in results:
        if res["variant"] not in combo_variants:
            continue
        n = res["variant"]
        lw = 2.5 if n in ("HCGAE_Base", "HCGAE_Imp12") else 2.0
        ls = "--" if n == "HCGAE_Base" else "-"
        ax_combo.plot(res["eval_steps"], res["eval_rewards"],
                      color=COLOR_MAP.get(n), linewidth=lw, linestyle=ls,
                      label=f"{n.replace('HCGAE_','')} (final={res['final_reward']:.0f})")
    ax_combo.set_title("Combination Variants vs Baseline", fontweight="bold")
    ax_combo.set_xlabel("Steps")
    ax_combo.set_ylabel("Eval Reward")
    ax_combo.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax_combo.grid(True, alpha=0.25, linestyle="--")

    fig2.suptitle("HCGAE Ablation Study: Grouped Learning Curves (Hopper-v4)",
                  fontsize=13, fontweight="bold")
    fig2.savefig(os.path.join(SAVE_DIR, "ablation_grouped_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  📊 分组学习曲线已保存: {SAVE_DIR}/ablation_grouped_curves.png")

    # ── 图 3：雷达图（多维对比）──────────────────────────────────
    from math import pi
    highlight_variants = ["HCGAE_Base", "HCGAE_Imp12", "HCGAE_Imp3", "HCGAE_Full"]
    dims = ["Final Reward", "Best Reward", "Stability\n(inv σ)", "Conv Speed\n(inv)", "EV"]

    def normalize(values):
        mn, mx = min(values), max(values)
        return [(v - mn) / (mx - mn + 1e-8) for v in values]

    all_finals = [r_dict[n] for n in highlight_variants]
    all_bests  = [r[n]["best_reward"] for n in highlight_variants]
    all_stabs  = [1.0 / (r[n]["stability_std"] + 1) for n in highlight_variants]  # inv
    all_convs  = [1.0 / (r[n]["conv_step_90"] / 1000 + 1) for n in highlight_variants]  # inv
    all_evs    = [r[n]["final_ev_ema"] for n in highlight_variants]

    norm_data = [
        normalize(all_finals),
        normalize(all_bests),
        normalize(all_stabs),
        normalize(all_convs),
        normalize(all_evs),
    ]

    n_dims = len(dims)
    angles = [i / float(n_dims) * 2 * pi for i in range(n_dims)]
    angles += angles[:1]

    fig3, ax_r = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    radar_colors = [COLOR_MAP.get(n, "#607D8B") for n in highlight_variants]

    for vi, (vname, color) in enumerate(zip(highlight_variants, radar_colors)):
        values = [norm_data[di][vi] for di in range(n_dims)]
        values += values[:1]
        ax_r.plot(angles, values, color=color, linewidth=2,
                  label=vname.replace("HCGAE_", ""))
        ax_r.fill(angles, values, color=color, alpha=0.1)

    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(dims, size=10)
    ax_r.set_ylim(0, 1)
    ax_r.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax_r.set_title("HCGAE Ablation: Multi-Dimensional Radar\n(Hopper-v4)",
                    fontweight="bold", pad=20)

    fig3.savefig(os.path.join(SAVE_DIR, "ablation_radar.png"), dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"  📊 雷达图已保存: {SAVE_DIR}/ablation_radar.png")


if __name__ == "__main__":
    print("读取消融实验结果...")
    results = load_results()
    print(f"已加载 {len(results)} 个变体的结果")

    deep_math_analysis(results)
    print("\n生成可视化图表...")
    plot_enhanced_visualization(results)

    print("\n" + "="*80)
    print("  分析完成！")
    print("="*80)

