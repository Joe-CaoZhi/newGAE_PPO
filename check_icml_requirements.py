#!/usr/bin/env python3
"""
ICML标准实验检查清单
根据ICML 2024-2025审稿标准，检查论文是否符合要求
"""

import json
from pathlib import Path

RESULTS_DIR = Path("/Users/joe-caozhi/newGAE_ppo/results")

print("="*80)
print("ICML标准实验检查清单")
print("="*80)

# 定义ICML标准检查项
checks = {
    "1. 核心实验完整性": {
        "checks": [
            ("多环境验证 (≥3个)", True, "Hopper-v4, Walker2d-v4, HalfCheetah-v4 ✓"),
            ("多种子实验 (≥5 seeds)", True, "每个环境5个种子 ✓"),
            ("基线对比完整", True, "Standard PPO, Optimal PPO ✓"),
            ("统计显著性检验", True, "Mann-Whitney U + Cohen's d ✓"),
            ("效应量报告", True, "Cohen's d + 95% CI ✓"),
        ]
    },

    "2. 消融实验": {
        "checks": [
            ("单一改进I消融", True, "HCGAE_Imp1 存在 ✓"),
            ("单一改进II消融", True, "HCGAE_Imp2 存在 ✓"),
            ("组合改进消融", True, "HCGAE_Imp12 存在 ✓"),
            ("多种子消融验证", True, "5 seeds × 300K steps ✓"),
            ("协同效应分析", True, "Synergy = +660 ✓"),
        ]
    },

    "3. 超参数敏感性": {
        "checks": [
            ("HCGAE a_max敏感性", True, "0.3, 0.5, 0.7, 0.9 测试 ✓"),
            ("HCGAE beta敏感性", True, "1-5 测试 ✓"),
            ("DCPPO-S SNR阈值敏感性", True, "0.1-0.7 测试 ✓"),
            ("敏感性图表", False, "⚠️ 需要在论文中添加敏感性曲线图"),
        ]
    },

    "4. 计算开销分析": {
        "checks": [
            ("训练时间对比", True, "论文中报告 +2% ✓"),
            ("内存开销对比", False, "⚠️ 缺少内存使用量对比"),
            ("样本效率分析", False, "⚠️ 缺少样本效率曲线"),
        ]
    },

    "5. 负向结果报告": {
        "checks": [
            ("HalfCheetah负向结果", True, "论文中明确报告 -16.0% ✓"),
            ("DCPPO-Full失败分析", True, "论文中详细讨论 ✓"),
            ("Optimal PPO Hopper失败", True, "论文中讨论 -11.4% ✓"),
            ("失败原因分析", True, "提供了机制性解释 ✓"),
        ]
    },

    "6. 可复现性": {
        "checks": [
            ("代码开源", True, "项目代码完整 ✓"),
            ("超参数详细记录", True, "论文中有详细表格 ✓"),
            ("随机种子记录", True, "所有实验记录种子 ✓"),
            ("环境版本记录", True, "MuJoCo v4 ✓"),
            ("硬件配置记录", False, "⚠️ 缺少GPU/CPU型号和内存"),
        ]
    },

    "7. 额外环境 (ICML强烈建议)": {
        "checks": [
            ("Ant-v4", False, "⚠️ 缺少Ant-v4实验 (ICML常见要求)"),
            ("Humanoid-v4", False, "⚠️ 缺少Humanoid实验 (高维任务)"),
            ("Atari游戏", False, "⚠️ 缺少Atari实验 (视觉输入)"),
        ]
    },

    "8. 更强基线对比": {
        "checks": [
            ("SAC对比", False, "⚠️ 缺少SAC (Soft Actor-Critic) 对比"),
            ("TD3对比", False, "⚠️ 缺少TD3对比"),
            ("PPO-LM对比", False, "⚠️ 缺少最新PPO变体对比"),
        ]
    },

    "9. 统计功效分析": {
        "checks": [
            ("事后功效分析", True, "论文中报告需要n≥258 ✓"),
            ("置信区间", True, "Bootstrap 95% CI ✓"),
            ("多重比较校正", False, "⚠️ 缺少Bonferroni校正"),
        ]
    },

    "10. 理论分析": {
        "checks": [
            ("命题与证明", True, "4个命题 + 严格证明 ✓"),
            ("收敛性分析", False, "⚠️ 缺少收敛性理论分析"),
            ("计算复杂度", False, "⚠️ 缺少O()复杂度分析"),
        ]
    }
}

# 打印检查结果
total_checks = 0
passed_checks = 0
missing_items = []

for category, data in checks.items():
    print(f"\n### {category} ###\n")

    for check_name, is_passed, note in data['checks']:
        total_checks += 1
        if is_passed:
            passed_checks += 1
            print(f"  ✓ {check_name}")
        else:
            print(f"  ✗ {check_name}: {note}")
            missing_items.append(f"{category} - {check_name}")
        print(f"    {note}")

# 生成缺失项清单
print("\n" + "="*80)
print(f"检查总结: {passed_checks}/{total_checks} 项通过")
print("="*80)

print("\n### 缺失/需要补充的项目 ###\n")
for i, item in enumerate(missing_items, 1):
    print(f"{i}. {item}")

# 按优先级分类
print("\n### 按优先级分类 ###\n")

high_priority = [
    "计算开销分析 - 内存开销对比",
    "可复现性 - 硬件配置记录",
    "统计功效分析 - 多重比较校正",
]

medium_priority = [
    "超参数敏感性 - 敏感性图表",
    "计算开销分析 - 样本效率分析",
    "额外环境 - Ant-v4",
    "理论分析 - 收敛性分析",
    "理论分析 - 计算复杂度",
]

low_priority = [
    "额外环境 - Humanoid-v4",
    "额外环境 - Atari游戏",
    "更强基线对比 - SAC对比",
    "更强基线对比 - TD3对比",
    "更强基线对比 - PPO-LM对比",
]

print("#### 高优先级 (建议补充) ####")
for item in high_priority:
    if item in missing_items:
        print(f"  • {item}")

print("\n#### 中优先级 (ICML建议) ####")
for item in medium_priority:
    if item in missing_items:
        print(f"  • {item}")

print("\n#### 低优先级 (加分项) ####")
for item in low_priority:
    if item in missing_items:
        print(f"  • {item}")

# 保存检查结果
report = {
    'total_checks': total_checks,
    'passed_checks': passed_checks,
    'pass_rate': f"{passed_checks/total_checks*100:.1f}%",
    'missing_items': missing_items,
    'high_priority': [i for i in missing_items if any(h in i for h in ['内存', '硬件', '多重'])],
    'medium_priority': [i for i in missing_items if any(m in i for m in ['敏感性图表', '样本效率', 'Ant', '收敛', '复杂度'])],
    'low_priority': [i for i in missing_items if any(l in i for l in ['Humanoid', 'Atari', 'SAC', 'TD3', 'PPO-LM'])]
}

with open(RESULTS_DIR / 'icml_checklist_report.json', 'w') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"\n检查报告已保存至: {RESULTS_DIR / 'icml_checklist_report.json'}")

