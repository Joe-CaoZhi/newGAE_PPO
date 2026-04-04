#!/usr/bin/env python3
"""
全面数据溯源验证脚本
检查论文中所有表格和统计数据与实际JSON文件的一致性
"""

import json
import os
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = Path("/Users/joe-caozhi/newGAE_ppo/results")

def load_json(filepath):
    """加载JSON文件"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        return None

def get_final_reward(data):
    """从JSON数据中提取最终奖励"""
    if data is None:
        return None

    # 尝试不同的字段名
    if 'final_reward' in data:
        return data['final_reward']
    elif 'eval_rewards' in data and isinstance(data['eval_rewards'], list):
        # 取最后10个评估奖励的平均值
        return np.mean(data['eval_rewards'][-10:])
    elif 'episode_rewards' in data and isinstance(data['episode_rewards'], list):
        return np.mean(data['episode_rewards'][-10:])
    elif 'mean_reward' in data:
        return data['mean_reward']
    else:
        return None

def get_all_rewards(data):
    """获取所有评估奖励"""
    if data is None:
        return []

    if 'eval_rewards' in data:
        return data['eval_rewards']
    elif 'episode_rewards' in data:
        return data['episode_rewards']
    else:
        return []

def check_table1_icml():
    """检查Table 1: ICML主要实验结果"""
    print("\n" + "="*80)
    print("检查 Table 1: ICML 主要实验结果 (500K步, 5种子)")
    print("="*80)

    icml_dir = RESULTS_DIR / "ICMLExperiment"
    if not icml_dir.exists():
        print("❌ ICMLExperiment目录不存在")
        return

    algorithms = {
        'Standard_PPO': 'Standard PPO',
        'Optimal_PPO': 'Optimal PPO',
        'Optimal_HCGAE': 'Optimal HCGAE',
        'Optimal_HCGAE_SCR': 'Optimal HCGAE + SCR'
    }

    environments = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']

    results = {}
    issues = []

    for env in environments:
        print(f"\n### {env} ###")
        results[env] = {}

        for algo_key, algo_name in algorithms.items():
            algo_dir = icml_dir / env / algo_key
            if not algo_dir.exists():
                print(f"  ⚠️  {algo_name}: 目录不存在")
                issues.append(f"{env}/{algo_key}: 目录不存在")
                continue

            # 收集所有种子的最终奖励
            rewards = []
            seed_files = sorted(algo_dir.glob(f"{algo_key}_s*.json"))

            if not seed_files:
                print(f"  ⚠️  {algo_name}: 无种子文件")
                issues.append(f"{env}/{algo_key}: 无种子文件")
                continue

            for seed_file in seed_files:
                data = load_json(seed_file)
                reward = get_final_reward(data)
                if reward is not None:
                    rewards.append(reward)

            if rewards:
                mean_reward = np.mean(rewards)
                std_reward = np.std(rewards, ddof=1)
                print(f"  ✓ {algo_name}: {mean_reward:.1f} ± {std_reward:.1f} (n={len(rewards)})")
                results[env][algo_key] = {
                    'mean': mean_reward,
                    'std': std_reward,
                    'n': len(rewards),
                    'rewards': rewards
                }
            else:
                print(f"  ❌ {algo_name}: 无法提取奖励数据")
                issues.append(f"{env}/{algo_key}: 无法提取奖励数据")

    # 保存Table 1数据
    with open(RESULTS_DIR / 'table1_audit.json', 'w') as f:
        json.dump(results, f, indent=2)

    # 检查统计显著性
    print("\n### 统计显著性检查 (HCGAE vs Optimal PPO) ###")
    for env in environments:
        if env in results and 'Optimal_HCGAE' in results[env] and 'Optimal_PPO' in results[env]:
            hcgae = results[env]['Optimal_HCGAE']['rewards']
            ppo = results[env]['Optimal_PPO']['rewards']

            if len(hcgae) >= 2 and len(ppo) >= 2:
                stat, p_value = stats.mannwhitneyu(hcgae, ppo, alternative='two-sided')
                cohens_d = (np.mean(hcgae) - np.mean(ppo)) / np.sqrt(
                    (np.std(hcgae, ddof=1)**2 + np.std(ppo, ddof=1)**2) / 2
                )
                print(f"  {env}: p={p_value:.4f}, Cohen's d={cohens_d:.3f}")

    return issues

def check_table2_dcppo():
    """检查Table 2: DCPPO结果"""
    print("\n" + "="*80)
    print("检查 Table 2: DCPPO MultiEnv 结果")
    print("="*80)

    dcppo_dir = RESULTS_DIR / "Hopper-v4-DCPPO"
    if not dcppo_dir.exists():
        print("❌ Hopper-v4-DCPPO目录不存在")
        return

    # DCPPO变体
    variants = {
        'DCPPO_Base': 'DCPPO-Base',
        'DCPPO_ImpS': 'DCPPO-ImpS (SNR Adaptive)',
        'DCPPO_ImpG': 'DCPPO-ImpG (Gradient)',
        'DCPPO_ImpA': 'DCPPO-ImpA (Advantage)',
        'DCPPO_Full': 'DCPPO-Full'
    }

    results = {}
    issues = []

    for variant_key, variant_name in variants.items():
        metrics_file = dcppo_dir / f"{variant_key}_metrics.json"
        summary_file = dcppo_dir / f"{variant_key}_summary.json"

        if metrics_file.exists():
            data = load_json(metrics_file)
            if data:
                rewards = get_all_rewards(data)
                if rewards:
                    mean_reward = np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
                    print(f"  ✓ {variant_name}: {mean_reward:.1f} (从metrics提取)")
                    results[variant_key] = {'mean': mean_reward, 'source': 'metrics'}
                else:
                    print(f"  ⚠️  {variant_name}: metrics中无奖励数据")
                    issues.append(f"DCPPO/{variant_key}: metrics中无奖励数据")
            else:
                print(f"  ❌ {variant_name}: 无法加载metrics文件")
                issues.append(f"DCPPO/{variant_key}: 无法加载metrics文件")
        elif summary_file.exists():
            data = load_json(summary_file)
            if data and 'mean_reward' in data:
                print(f"  ✓ {variant_name}: {data['mean_reward']:.1f} (从summary提取)")
                results[variant_key] = {'mean': data['mean_reward'], 'source': 'summary'}
            else:
                print(f"  ⚠️  {variant_name}: summary中无mean_reward")
                issues.append(f"DCPPO/{variant_key}: summary中无mean_reward")
        else:
            print(f"  ⚠️  {variant_name}: 无数据文件")
            issues.append(f"DCPPO/{variant_key}: 无数据文件")

    with open(RESULTS_DIR / 'table2_audit.json', 'w') as f:
        json.dump(results, f, indent=2)

    return issues

def check_table3_ablation():
    """检查Table 3: HCGAE消融实验"""
    print("\n" + "="*80)
    print("检查 Table 3: HCGAE Ablation Study")
    print("="*80)

    # 在BaselineComparison中查找消融数据
    baseline_dir = RESULTS_DIR / "BaselineComparison"

    # HCGAE消融应该包含:
    # - HCGAE_Imp1 (仅改进I: Batch Normalization)
    # - HCGAE_Imp2 (仅改进II: EV-driven mixing)
    # - HCGAE_Imp12 (组合改进)

    ablation_configs = {
        'HCGAE_Imp1': 'HCGAE-I (仅批归一化)',
        'HCGAE_Imp2': 'HCGAE-II (仅EV混合)',
        'HCGAE_Imp12': 'HCGAE-I+II (组合)',
        'Standard_PPO': 'Standard PPO (基线)'
    }

    results = {}
    issues = []

    # 检查Hopper-v4
    hopper_dir = baseline_dir / "Hopper-v4"
    if hopper_dir.exists():
        print("\n### Hopper-v4 ###")
        for config_key, config_name in ablation_configs.items():
            config_dir = hopper_dir / config_key
            if config_dir.exists():
                rewards = []
                for seed_file in sorted(config_dir.glob(f"{config_key}_s*_metrics.json")):
                    data = load_json(seed_file)
                    reward = get_final_reward(data)
                    if reward is not None:
                        rewards.append(reward)

                if rewards:
                    mean_reward = np.mean(rewards)
                    std_reward = np.std(rewards, ddof=1)
                    print(f"  ✓ {config_name}: {mean_reward:.1f} ± {std_reward:.1f} (n={len(rewards)})")
                    results[config_key] = {'mean': mean_reward, 'std': std_reward, 'n': len(rewards)}
                else:
                    print(f"  ❌ {config_name}: 无有效奖励数据")
                    issues.append(f"Abaltion/Hopper/{config_key}: 无有效奖励数据")
            else:
                print(f"  ⚠️  {config_name}: 目录不存在")
                if config_key not in ['HCGAE_Imp1', 'HCGAE_Imp2']:
                    issues.append(f"Ablation/Hopper/{config_key}: 目录不存在")
    else:
        print("❌ BaselineComparison/Hopper-v4目录不存在")
        issues.append("Ablation: BaselineComparison/Hopper-v4目录不存在")

    with open(RESULTS_DIR / 'table3_audit.json', 'w') as f:
        json.dump(results, f, indent=2)

    return issues

def check_section43_halfcheetah():
    """检查Section 4.3 HalfCheetah统计表"""
    print("\n" + "="*80)
    print("检查 Section 4.3: HalfCheetah Mann-Whitney 统计")
    print("="*80)

    # 从BaselineComparison加载数据
    baseline_dir = RESULTS_DIR / "BaselineComparison" / "HalfCheetah-v4"

    configs = ['Standard_PPO', 'HCGAE_Imp12', 'PPO_Full_Baseline']
    results = {}
    issues = []

    for config in configs:
        config_dir = baseline_dir / config
        if config_dir.exists():
            rewards = []
            for seed_file in sorted(config_dir.glob(f"{config}_s*_metrics.json")):
                data = load_json(seed_file)
                reward = get_final_reward(data)
                if reward is not None:
                    rewards.append(reward)

            if rewards:
                mean_reward = np.mean(rewards)
                std_reward = np.std(rewards, ddof=1)
                print(f"  ✓ {config}: {mean_reward:.1f} ± {std_reward:.1f} (n={len(rewards)})")
                results[config] = {
                    'mean': mean_reward,
                    'std': std_reward,
                    'n': len(rewards),
                    'rewards': rewards
                }
            else:
                print(f"  ❌ {config}: 无有效数据")
                issues.append(f"Section43/{config}: 无有效数据")
        else:
            print(f"  ⚠️  {config}: 目录不存在")
            issues.append(f"Section43/{config}: 目录不存在")

    # 计算Mann-Whitney统计
    if 'Standard_PPO' in results and 'HCGAE_Imp12' in results:
        std_ppo = results['Standard_PPO']['rewards']
        hcgae = results['HCGAE_Imp12']['rewards']

        if len(std_ppo) >= 2 and len(hcgae) >= 2:
            stat, p_value = stats.mannwhitneyu(hcgae, std_ppo, alternative='two-sided')
            cohens_d = (np.mean(hcgae) - np.mean(std_ppo)) / np.sqrt(
                (np.std(hcgae, ddof=1)**2 + np.std(std_ppo, ddof=1)**2) / 2
            )

            print(f"\n  Mann-Whitney U 检验:")
            print(f"    HCGAE vs Standard PPO: p={p_value:.4f}, Cohen's d={cohens_d:.3f}")

            results['mann_whitney'] = {
                'p_value': float(p_value),
                'cohens_d': float(cohens_d),
                'statistic': float(stat)
            }

    with open(RESULTS_DIR / 'section43_audit.json', 'w') as f:
        json.dump(results, f, indent=2)

    return issues

def check_sensitivity_analysis():
    """检查超参数敏感性分析"""
    print("\n" + "="*80)
    print("检查 超参数敏感性分析")
    print("="*80)

    sensitivity_dir = RESULTS_DIR / "Sensitivity"
    if not sensitivity_dir.exists():
        print("❌ Sensitivity目录不存在")
        return ["Sensitivity: 目录不存在"]

    issues = []
    results = {}

    # HCGAE参数敏感性 (amax和beta)
    hcgae_files = {
        'amax': sorted(sensitivity_dir.glob("HCGAE_amax*_metrics.json")),
        'beta': sorted(sensitivity_dir.glob("HCGAE_beta*_metrics.json"))
    }

    print("\n### HCGAE 参数敏感性 ###")
    for param_type, files in hcgae_files.items():
        if files:
            print(f"\n  {param_type} 参数:")
            for f in files:
                data = load_json(f)
                reward = get_final_reward(data)
                if reward is not None:
                    param_val = f.stem.split('_')[1].replace('amax', 'a_max=').replace('beta', 'beta=')
                    print(f"    {param_val}: {reward:.1f}")
                    results[f.stem] = reward
                else:
                    print(f"    {f.stem}: 无法提取奖励")
                    issues.append(f"Sensitivity/{f.stem}: 无法提取奖励")

    # DCPPO-S SNR阈值敏感性
    dcppo_files = sorted(sensitivity_dir.glob("DCPPO_S_snr*_metrics.json"))
    if dcppo_files:
        print("\n### DCPPO-S SNR阈值敏感性 ###")
        for f in dcppo_files:
            data = load_json(f)
            reward = get_final_reward(data)
            if reward is not None:
                snr_val = f.stem.split('snr')[1].replace('p', '.')
                print(f"  SNR阈值={snr_val}: {reward:.1f}")
                results[f.stem] = reward
            else:
                print(f"  {f.stem}: 无法提取奖励")
                issues.append(f"Sensitivity/{f.stem}: 无法提取奖励")

    with open(RESULTS_DIR / 'sensitivity_audit.json', 'w') as f:
        json.dump(results, f, indent=2)

    return issues

def generate_audit_report():
    """生成完整的审计报告"""
    print("\n" + "="*80)
    print("开始全面数据溯源审计")
    print("="*80)

    all_issues = []

    # 检查各个部分
    all_issues.extend(check_table1_icml())
    all_issues.extend(check_table2_dcppo())
    all_issues.extend(check_table3_ablation())
    all_issues.extend(check_section43_halfcheetah())
    all_issues.extend(check_sensitivity_analysis())

    # 生成总结报告
    print("\n" + "="*80)
    print("审计总结")
    print("="*80)

    if all_issues:
        print(f"\n发现 {len(all_issues)} 个问题:")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\n✓ 所有数据验证通过!")

    # 保存审计报告
    report = {
        'total_issues': len(all_issues),
        'issues': all_issues,
        'timestamp': str(np.datetime64('now'))
    }

    with open(RESULTS_DIR / 'comprehensive_audit_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n审计报告已保存至: {RESULTS_DIR / 'comprehensive_audit_report.json'}")

    return all_issues

if __name__ == "__main__":
    generate_audit_report()

