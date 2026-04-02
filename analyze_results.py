"""
综合分析两个环境的实验结果：收敛速度、稳定性、最终性能
"""
import json
import os

import numpy as np

ENVS = {
    "CartPole-v1":  {"results_dir": "results/CartPole-v1",  "threshold": 450, "max_score": 500},
    "Acrobot-v1":   {"results_dir": "results/Acrobot-v1",   "threshold": -100, "max_score": -75},
}

AGENT_ORDER = [
    "Standard_GAE",
    "Conservative_Bootstrap_GAE",
    "Adaptive_Lambda_GAE",
    "Confidence_Weighted_GAE",
    "Combined_GAE",
    "Hindsight_GAE",
    "MultiScale_GAE",
    "CausalAttn_GAE",
]

SHORT_NAMES = {
    "Standard_GAE":               "Standard",
    "Conservative_Bootstrap_GAE": "Conservative",
    "Adaptive_Lambda_GAE":        "Adaptive-λ",
    "Confidence_Weighted_GAE":    "Confidence",
    "Combined_GAE":               "Combined",
    "Hindsight_GAE":              "Hindsight★",
    "MultiScale_GAE":             "MultiScale★",
    "CausalAttn_GAE":             "CausalAttn★",
}


def load_metrics(path):
    with open(path) as f:
        return json.load(f)


def find_convergence_step(eval_steps, eval_rewards, threshold, window=3):
    """找到评估奖励连续 window 次超过 threshold 的第一步"""
    count = 0
    for step, r in zip(eval_steps, eval_rewards):
        if r >= threshold:
            count += 1
            if count >= window:
                return step
        else:
            count = 0
    return None


def analyze_env(env_name, cfg):
    results_dir = cfg["results_dir"]
    threshold = cfg["threshold"]
    max_score = cfg["max_score"]

    print(f"\n{'=' * 70}")
    print(f"  环境: {env_name}")
    print(f"  收敛阈值: {threshold}  满分: {max_score}")
    print(f"{'=' * 70}")
    print(f"{'方法':<26} {'最终(avg5)':<12} {'最高':<10} {'收敛步数':<12} {'稳定性(std)':<12}")
    print(f"{'─' * 70}")

    results = {}
    for agent in AGENT_ORDER:
        # 尝试多个可能的文件名
        for fname in [f"{agent}_metrics.json", f"Double_Critic_GAE_metrics.json" if agent == "Conservative_Bootstrap_GAE" else None]:
            if fname is None:
                continue
            path = os.path.join(results_dir, fname)
            if os.path.exists(path):
                break
        else:
            # fallback: 遍历目录
            files = os.listdir(results_dir) if os.path.isdir(results_dir) else []
            matched = [f for f in files if agent.lower() in f.lower() and f.endswith(".json")]
            if not matched:
                # 特殊处理 Conservative_Bootstrap_GAE 可能存为 Double_Critic_GAE
                if agent == "Conservative_Bootstrap_GAE":
                    matched = [f for f in files if "double_critic" in f.lower() and f.endswith(".json")]
            if not matched:
                continue
            path = os.path.join(results_dir, matched[0])

        try:
            data = load_metrics(path)
        except Exception:
            continue

        eval_rewards = data.get("eval_rewards", [])
        eval_steps = data.get("eval_steps", [])
        if not eval_rewards:
            continue

        final5 = float(np.mean(eval_rewards[-5:])) if len(eval_rewards) >= 5 else float(np.mean(eval_rewards))
        best = float(np.max(eval_rewards))

        # CartPole 收敛判断：越大越好
        # Acrobot 收敛判断：越大（越接近0）越好
        conv_step = find_convergence_step(eval_steps, eval_rewards, threshold)

        # 稳定性：后半段评估奖励的标准差
        mid = len(eval_rewards) // 2
        stability_std = float(np.std(eval_rewards[mid:])) if mid > 0 else 0.0

        results[agent] = {
            "final5": final5,
            "best": best,
            "conv_step": conv_step,
            "stability_std": stability_std,
        }

        name = SHORT_NAMES.get(agent, agent)
        conv_str = f"{conv_step:,}" if conv_step else "未收敛"
        marker = " ★" if agent in ["Hindsight_GAE", "MultiScale_GAE", "CausalAttn_GAE"] else "  "
        print(f"  {name:<24}{marker} {final5:>8.1f}      {best:>7.1f}    {conv_str:<12} {stability_std:>8.2f}")

    return results


def print_convergence_ranking(all_results):
    print(f"\n{'=' * 70}")
    print("  收敛速度排名（两个环境汇总）")
    print(f"{'=' * 70}")

    # 计算每个方法的平均收敛步数排名
    env_names = list(all_results.keys())
    scores = {}
    for agent in AGENT_ORDER:
        conv_steps = []
        for env_name, results in all_results.items():
            if agent in results and results[agent]["conv_step"] is not None:
                # 归一化到总步数
                max_steps = {"CartPole-v1": 150000, "Acrobot-v1": 200000}
                conv_steps.append(results[agent]["conv_step"] / max_steps.get(env_name, 200000))
        if conv_steps:
            scores[agent] = float(np.mean(conv_steps))

    sorted_agents = sorted(scores.items(), key=lambda x: x[1])
    print(f"  {'排名':<4} {'方法':<26} {'平均收敛进度':<15}")
    print(f"  {'─' * 50}")
    for rank, (agent, score) in enumerate(sorted_agents, 1):
        name = SHORT_NAMES.get(agent, agent)
        marker = "★" if agent in ["Hindsight_GAE", "MultiScale_GAE", "CausalAttn_GAE"] else " "
        print(f"  #{rank:<3} {marker}{name:<25} {score*100:>8.1f}% of budget")


if __name__ == "__main__":
    all_results = {}
    for env_name, cfg in ENVS.items():
        if os.path.isdir(cfg["results_dir"]):
            all_results[env_name] = analyze_env(env_name, cfg)
        else:
            print(f"  跳过 {env_name}（无结果目录）")

    if len(all_results) >= 2:
        print_convergence_ranking(all_results)

    print("\n✅ 分析完成")

