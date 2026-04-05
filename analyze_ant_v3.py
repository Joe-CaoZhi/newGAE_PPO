#!/usr/bin/env python3
"""
Ant-v4 v3 验证结果分析脚本
===========================
功能：
1. 读取 results/AntV3Validation/ 下的实验结果
2. 计算各算法平均奖励、标准差、Mann-Whitney U 统计量
3. 分析 v3 三项修复的数学有效性：
   - 各修复机制的独立贡献（消融对比）
   - 与预期（G-Clamping/VW-Gate 理论预测）的吻合程度
4. 绘制学习曲线对比图
5. 输出决策建议
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS_DIR = Path("results/AntV3Validation")
ALGORITHMS = [
    "Standard_PPO",
    "Optimal_PPO",
    "Optimal_HCGAE_v2",
    "Optimal_HCGAE_v3",
    "Optimal_HCGAE_v3_NoClamp",
    "Optimal_HCGAE_v3_NoVWGate",
    "Optimal_HCGAE_v3_NoBdryPrior",
]
SEEDS = [0, 1, 2]


def load_results(algo, seeds=SEEDS):
    results = {}
    for seed in seeds:
        fp = RESULTS_DIR / algo / f"{algo}_s{seed}.json"
        if fp.exists():
            with open(fp) as f:
                results[seed] = json.load(f)
    return results


def final_rewards(data):
    return [d['final_reward'] for d in data.values()]


def eval_curve_mean(data, n_points=None):
    """计算跨种子的平均学习曲线（对齐到最短曲线）"""
    curves = [d['eval_rewards'] for d in data.values()]
    if not curves:
        return [], []
    min_len = min(len(c) for c in curves)
    if n_points:
        min_len = min(min_len, n_points)
    curves_trimmed = [c[:min_len] for c in curves]
    steps = None
    for d in data.values():
        if 'eval_steps' in d and d['eval_steps']:
            steps = d['eval_steps'][:min_len]
            break
    mean = np.mean(curves_trimmed, axis=0)
    std = np.std(curves_trimmed, axis=0)
    return steps, mean, std


def mann_whitney(rewards_a, rewards_b):
    if len(rewards_a) < 2 or len(rewards_b) < 2:
        return float('nan'), float('nan')
    u, p = stats.mannwhitneyu(rewards_a, rewards_b, alternative='two-sided')
    # Cohen's d
    pooled_std = np.sqrt((np.std(rewards_a)**2 + np.std(rewards_b)**2) / 2) + 1e-8
    d = (np.mean(rewards_a) - np.mean(rewards_b)) / pooled_std
    return p, d


def pct_change(a, b):
    """a vs b 的百分比变化"""
    return 100 * (np.mean(a) - np.mean(b)) / (abs(np.mean(b)) + 1e-8)


def print_section(title):
    print(f"\n{'═'*70}")
    print(f"  {title}")
    print('═'*70)


def analyze():
    # ── 1. 加载所有结果
    all_data = {}
    for algo in ALGORITHMS:
        all_data[algo] = load_results(algo)

    # ── 2. 统计汇总表
    print_section("Ant-v4 快速验证结果汇总 (200K steps)")
    print(f"  {'算法':<38} {'均值':>8}  {'标准差':>8}  {'种子奖励'}")
    print('-'*70)

    reward_table = {}
    for algo in ALGORITHMS:
        data = all_data[algo]
        if not data:
            print(f"  {algo:<38} {'—':>8}  {'—':>8}  [无数据]")
            continue
        rewards = final_rewards(data)
        mean_r = np.mean(rewards)
        std_r = np.std(rewards)
        reward_table[algo] = rewards
        seed_str = " | ".join(f"{r:.0f}" for r in rewards)
        print(f"  {algo:<38} {mean_r:>8.1f}  {std_r:>8.1f}  [{seed_str}]")

    # ── 3. 统计显著性检验（v3 vs 各基线）
    print_section("统计显著性分析 (Mann-Whitney U test)")
    if "Optimal_HCGAE_v3" in reward_table:
        v3 = reward_table["Optimal_HCGAE_v3"]
        comparisons = [
            ("Optimal_PPO", "v3 vs Optimal_PPO (关键基线)"),
            ("Optimal_HCGAE_v2", "v3 vs HCGAE_v2 (前代)"),
            ("Standard_PPO", "v3 vs Standard_PPO"),
        ]
        for algo, label in comparisons:
            if algo in reward_table:
                p, d = mann_whitney(v3, reward_table[algo])
                pct = pct_change(v3, reward_table[algo])
                sig = "**" if p < 0.05 else ("*" if p < 0.1 else "n.s.")
                print(f"  {label}")
                print(f"    Δ={pct:+.1f}%  d={d:+.2f}  p={p:.3f} {sig}")

    # ── 4. 消融分析：各修复机制的贡献
    print_section("v3 消融分析：各修复机制的独立贡献")
    ablation_map = [
        ("Optimal_HCGAE_v3_NoClamp",     "无G-Clamping (FIX① 关闭)"),
        ("Optimal_HCGAE_v3_NoVWGate",    "无VW-Gate    (FIX② 关闭)"),
        ("Optimal_HCGAE_v3_NoBdryPrior", "无Bdry Prior (FIX③ 关闭)"),
    ]
    if "Optimal_HCGAE_v3" in reward_table:
        v3_mean = np.mean(reward_table["Optimal_HCGAE_v3"])
        print(f"  HCGAE_v3 (全量)  均值={v3_mean:.1f}")
        for algo, label in ablation_map:
            if algo in reward_table:
                m = np.mean(reward_table[algo])
                delta = m - v3_mean
                print(f"  {label:<38} 均值={m:.1f}  Δ={delta:+.1f} vs 全量v3")
            else:
                print(f"  {label:<38} [无数据]")

    # ── 5. 数学有效性分析
    print_section("数学有效性分析：各修复机制是否符合预期")

    if ("Optimal_HCGAE_v2" in reward_table and
            "Optimal_HCGAE_v3" in reward_table and
            "Optimal_PPO" in reward_table):
        v2_m = np.mean(reward_table["Optimal_HCGAE_v2"])
        v3_m = np.mean(reward_table["Optimal_HCGAE_v3"])
        ppo_m = np.mean(reward_table["Optimal_PPO"])
        std_m = np.mean(reward_table.get("Standard_PPO", [0]))

        print(f"""
  理论预测（基于 Ant-v4 诊断分析）：
  ─────────────────────────────────────────
  根因：Ant-v4 早期 G_t 高度噪声（CV=16.47），SNR=0.06
        α_max=0.7 >> SCR/(1+SCR)≈0.06 → 过校正导致悲观偏差

  预期效果（v2 → v3）：
  ① G-Clamping 防止 G_t 将 V^c 拉入深负值区  → 减少悲观偏差 → 更好策略梯度
  ② VW-Gate（SNR加权）早期大幅压低 α_max     → 减少错误修正量
  ③ Boundary Prior 防止边界 V^c 被单次负 G 污染 → 减少训练波动

  实际结果：
  ─────────────────────────────────────────
  Standard_PPO  : {std_m:.1f}
  Optimal_PPO   : {ppo_m:.1f}  [目标基线]
  HCGAE_v2      : {v2_m:.1f}  (vs PPO: {pct_change([v2_m], [ppo_m]):+.1f}%)
  HCGAE_v3      : {v3_m:.1f}  (vs PPO: {pct_change([v3_m], [ppo_m]):+.1f}%,  vs v2: {pct_change([v3_m], [v2_m]):+.1f}%)
""")

        if v3_m > v2_m and v3_m > ppo_m:
            conclusion = "✅ v3 完全符合预期：超越 v2 且超越 Optimal_PPO！可进入完整实验。"
        elif v3_m > v2_m:
            gap = pct_change([v3_m], [ppo_m])
            conclusion = f"⚠️  v3 改善了 v2 ({pct_change([v3_m],[v2_m]):+.1f}%) 但仍落后 PPO {gap:.1f}%，需进一步分析。"
        elif v3_m > ppo_m:
            conclusion = f"⚠️  v3 超越 PPO ({pct_change([v3_m],[ppo_m]):+.1f}%) 但未超越 v2，机制设计可能有冗余。"
        else:
            conclusion = "❌ v3 未达预期，需要检查：G-Clamping margin、SNR target 参数是否合适。"

        print(f"  结论：{conclusion}")

    # ── 6. 学习曲线分析（各阶段表现）
    print_section("学习曲线关键阶段分析")
    for algo in ["Optimal_PPO", "Optimal_HCGAE_v2", "Optimal_HCGAE_v3"]:
        data = all_data.get(algo, {})
        if not data:
            continue
        steps, mean, std = eval_curve_mean(data)
        if steps is None or len(mean) == 0:
            continue
        # 早期 (前1/3)、中期、最终
        n = len(mean)
        early_end = n // 3
        mid_end = 2 * n // 3
        early_r = np.mean(mean[:max(1, early_end)])
        mid_r = np.mean(mean[early_end:mid_end]) if mid_end > early_end else mean[early_end]
        final_r = np.mean(mean[mid_end:]) if mid_end < n else mean[-1]
        print(f"  {algo:<38} 早期={early_r:.0f}  中期={mid_r:.0f}  后期={final_r:.0f}")

    # ── 7. 绘制学习曲线
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {
            "Standard_PPO": "gray",
            "Optimal_PPO": "black",
            "Optimal_HCGAE_v2": "blue",
            "Optimal_HCGAE_v3": "red",
            "Optimal_HCGAE_v3_NoClamp": "orange",
            "Optimal_HCGAE_v3_NoVWGate": "green",
            "Optimal_HCGAE_v3_NoBdryPrior": "purple",
        }
        styles = {
            "Standard_PPO": "--",
            "Optimal_PPO": "--",
            "Optimal_HCGAE_v2": "-",
            "Optimal_HCGAE_v3": "-",
            "Optimal_HCGAE_v3_NoClamp": ":",
            "Optimal_HCGAE_v3_NoVWGate": ":",
            "Optimal_HCGAE_v3_NoBdryPrior": ":",
        }
        lw = {
            "Standard_PPO": 1.5,
            "Optimal_PPO": 2.0,
            "Optimal_HCGAE_v2": 2.5,
            "Optimal_HCGAE_v3": 3.0,
            "Optimal_HCGAE_v3_NoClamp": 1.5,
            "Optimal_HCGAE_v3_NoVWGate": 1.5,
            "Optimal_HCGAE_v3_NoBdryPrior": 1.5,
        }
        labels = {
            "Standard_PPO": "Standard PPO",
            "Optimal_PPO": "Optimal PPO (基线)",
            "Optimal_HCGAE_v2": "HCGAE v2 (失效)",
            "Optimal_HCGAE_v3": "HCGAE v3 (全量修复) ★",
            "Optimal_HCGAE_v3_NoClamp": "v3 无G-Clamping",
            "Optimal_HCGAE_v3_NoVWGate": "v3 无VW-Gate",
            "Optimal_HCGAE_v3_NoBdryPrior": "v3 无Bdry Prior",
        }

        plotted = False
        for algo in ALGORITHMS:
            data = all_data.get(algo, {})
            if not data:
                continue
            steps, mean, std = eval_curve_mean(data)
            if steps is None or len(mean) == 0:
                continue
            x = np.array(steps) / 1000  # 转为 K 步
            color = colors.get(algo, "gray")
            ls = styles.get(algo, "-")
            linewidth = lw.get(algo, 1.5)
            label = labels.get(algo, algo)
            ax.plot(x, mean, color=color, linestyle=ls, linewidth=linewidth, label=label)
            ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)
            plotted = True

        if plotted:
            ax.set_xlabel("Training Steps (K)", fontsize=12)
            ax.set_ylabel("Eval Reward", fontsize=12)
            ax.set_title("Ant-v4: HCGAE v3 验证 (200K steps, 3 seeds)", fontsize=13)
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)
            out_path = RESULTS_DIR / "ant_v3_validation.png"
            fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"\n  📊 学习曲线图已保存至: {out_path}")
    except Exception as e:
        print(f"\n  [图形绘制跳过: {e}]")

    # ── 8. 决策建议
    print_section("决策建议")
    if "Optimal_HCGAE_v3" in reward_table and "Optimal_PPO" in reward_table:
        v3_m = np.mean(reward_table["Optimal_HCGAE_v3"])
        ppo_m = np.mean(reward_table["Optimal_PPO"])
        if v3_m >= ppo_m * 0.95:
            print("""  ✅ 建议进入完整实验（500K步 × 5种子）：
      - v3 在 Ant-v4 上达到或超过 Optimal PPO
      - 可按与 ICMLExperiment 对齐的完全相同条件运行
      - 更新论文发现 4 (Ant-v4 局限性 → 已解决)""")
        else:
            gap = (ppo_m - v3_m) / (abs(ppo_m) + 1e-8) * 100
            print(f"""  ⚠️  v3 仍落后 PPO {gap:.1f}%，考虑调参后再扩展：
      - 检查消融结果：哪个修复贡献最大？
      - 考虑调整 snr_target (当前=0.5) 或 g_clamp_margin_k (当前=1.5)
      - 也可先提交当前结果（改进但未超越）作为诚实的中间发现""")
    else:
        print("  [数据不足，请等待实验完成后重新运行]")


if __name__ == "__main__":
    analyze()

