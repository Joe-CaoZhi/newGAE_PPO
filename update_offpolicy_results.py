"""
等 SAC/TD3 实验完成后，自动将结果填入论文 §4.8
使用方式：
    python3 update_offpolicy_results.py
"""
import json
import re
from pathlib import Path

SUMMARY_PATH = Path("results/OffPolicyComparison/offpolicy_comparison_summary.json")
PAPER_PATH = Path("docs/paper_draft_zh.md")


def load_summary() -> dict:
    if not SUMMARY_PATH.exists():
        print(f"ERROR: {SUMMARY_PATH} 不存在，请先运行 run_sac_td3_comparison.py")
        return {}
    with open(SUMMARY_PATH) as f:
        return json.load(f)


def load_baseline() -> dict:
    bl_path = Path("results/BaselineComparison/baseline_comparison_summary.json")
    if not bl_path.exists():
        return {}
    with open(bl_path) as f:
        return json.load(f)


def format_result(summary: dict, env: str, algo: str) -> str:
    """将 (env, algo) 的结果格式化为 'mean ± std' 字符串"""
    info = summary.get(env, {}).get(algo, {})
    seeds = info.get("seeds", [])
    if not seeds:
        return "*待测*"
    import numpy as np
    m = float(np.mean(seeds))
    s = float(np.std(seeds))
    n = len(seeds)
    sem = s / (n ** 0.5)
    return f"**{m:.0f} ± {sem:.0f}** (n={n})"


def build_table(summary: dict, baseline: dict) -> str:
    """构建论文 §4.8 表 7 的更新版本"""
    envs = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]
    algos_online = [
        ("标准 PPO", "在线策略", "1M"),
        ("HCGAE_Imp12", "在线策略", "1M"),
    ]
    algos_offline = [
        ("SAC", "离线策略", "1M"),
        ("TD3", "离线策略", "1M"),
    ]

    lines = []
    lines.append("| 方法 | 类型 | 步数 | Hopper-v4 | Walker2d-v4 | HalfCheetah-v4 |")
    lines.append("|---|---|:---:|:---:|:---:|:---:|")

    # 在线策略（从 BaselineComparison 读取 n=5 数据）
    for algo_display, algo_type, steps in algos_online:
        algo_key = algo_display
        row_vals = []
        for env in envs:
            info = baseline.get(env, {}).get(algo_key, {})
            seeds = info.get("seeds", [])
            if seeds:
                import numpy as np
                m = float(np.mean(seeds))
                s = float(np.std(seeds))
                n = len(seeds)
                sem = s / (n ** 0.5)
                row_vals.append(f"{m:.0f} ± {sem:.0f}")
            else:
                row_vals.append("*待测*")
        lines.append(f"| {algo_display} | {algo_type} | {steps} | " +
                     " | ".join(row_vals) + " |")

    # 离线策略（从 OffPolicyComparison 读取）
    for algo_display, algo_type, steps in algos_offline:
        row_vals = []
        for env in envs:
            result = format_result(summary, env, algo_display)
            row_vals.append(result)
        lines.append(f"| {algo_display} | {algo_type} | {steps} | " +
                     " | ".join(row_vals) + " |")

    return "\n".join(lines)


def compute_stats(summary: dict) -> dict:
    """计算 SAC/TD3 的统计摘要"""
    import numpy as np

    results = {}
    envs = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]

    for algo in ["SAC", "TD3"]:
        results[algo] = {}
        for env in envs:
            seeds = summary.get(env, {}).get(algo, {}).get("seeds", [])
            if len(seeds) >= 2:
                m = float(np.mean(seeds))
                s = float(np.std(seeds))
                sem = s / (len(seeds) ** 0.5)
                results[algo][env] = {
                    "mean": m,
                    "std": s,
                    "sem": sem,
                    "n": len(seeds),
                    "seeds": seeds,
                }
            else:
                results[algo][env] = None

    return results


def update_paper(summary: dict, baseline: dict):
    """将 SAC/TD3 结果填入论文"""

    paper_text = PAPER_PATH.read_text(encoding="utf-8")

    # 构建新的表格
    new_table = build_table(summary, baseline)

    # 替换旧的"待测"表格
    # 寻找表 7 的位置
    old_table_pattern = r"\| 方法 \| 类型 \| 步数 \| Hopper-v4 \| Walker2d-v4 \| HalfCheetah-v4 \|.*?\| TD3 \| 离线策略 \| 1M \|.*?\|"

    match = re.search(old_table_pattern, paper_text, re.DOTALL)
    if not match:
        print("WARNING: 在论文中未找到表 7 的位置，跳过自动更新")
        return

    # 替换
    new_paper = paper_text[:match.start()] + new_table + paper_text[match.end():]

    # 更新注脚
    stats = compute_stats(summary)

    # 构建统计摘要段落
    stat_lines = []
    stat_lines.append("\n*数据来源（已完成）：`results/OffPolicyComparison/offpolicy_comparison_summary.json`。*\n")
    stat_lines.append("\n**结果分析（基于实验数据）：**\n")

    for algo in ["SAC", "TD3"]:
        for env in ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]:
            d = stats.get(algo, {}).get(env)
            if d:
                stat_lines.append(f"- {algo} @ {env}: {d['mean']:.0f} ± {d['sem']:.0f}（n={d['n']}）")

    # 保存
    PAPER_PATH.write_text(new_paper, encoding="utf-8")
    print(f"论文 §4.8 已更新：{PAPER_PATH}")
    print("\n统计摘要：")
    for line in stat_lines:
        print(line)


def main():
    print("=== SAC/TD3 结果自动更新脚本 ===\n")
    summary = load_summary()
    baseline = load_baseline()

    if not summary:
        return

    # 检查完成度
    envs = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]
    for algo in ["SAC", "TD3"]:
        for env in envs:
            seeds = summary.get(env, {}).get(algo, {}).get("seeds", [])
            print(f"  {algo} @ {env}: {len(seeds)}/5 seeds 完成")

    print()
    print("当前结果：")
    table = build_table(summary, baseline)
    print(table)

    # 检查是否有任何数据可以更新
    has_data = any(
        len(summary.get(env, {}).get(algo, {}).get("seeds", [])) > 0
        for algo in ["SAC", "TD3"]
        for env in envs
    )

    if has_data:
        update_paper(summary, baseline)
    else:
        print("\n暂无数据可更新（实验仍在运行中）")


if __name__ == "__main__":
    main()

