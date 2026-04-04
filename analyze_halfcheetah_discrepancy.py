"""分析 HalfCheetah 在不同步数/种子下的数据差异"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

# 数据集1: BaselineComparison (300K步, 种子 42, 123, 456, 789, 1234)
print("=" * 80)
print("分析 HalfCheetah-v4 在不同实验数据集中的表现差异")
print("=" * 80)

# 读取 BaselineComparison 数据
print("\n1. BaselineComparison (300K步, 种子42/123/456/789/1234):")
bl_ppo = []
bl_hcgae = []
for s in [42, 123, 456, 789, 1234]:
    for algo, lst in [('Standard_PPO', bl_ppo), ('HCGAE_Imp12', bl_hcgae)]:
        fp = Path(f'results/BaselineComparison/HalfCheetah-v4/{algo}/{algo}_s{s}_metrics.json')
        if fp.exists():
            with open(fp) as f:
                d = json.load(f)
            er = d.get('eval_rewards', [])
            v = float(np.mean(er[-5:])) if len(er) >= 5 else d.get('final_reward', 0)
            lst.append(v)

print(f"   Standard_PPO: {np.mean(bl_ppo):.0f} ± {np.std(bl_ppo)/np.sqrt(len(bl_ppo)):.0f} (n={len(bl_ppo)})")
print(f"   HCGAE:        {np.mean(bl_hcgae):.0f} ± {np.std(bl_hcgae)/np.sqrt(len(bl_hcgae)):.0f} (n={len(bl_hcgae)})")
print(f"   Individual PPO: {[f'{v:.0f}' for v in bl_ppo]}")
print(f"   Individual HCGAE: {[f'{v:.0f}' for v in bl_hcgae]}")

# 读取 UnifiedComparison 数据
print("\n2. UnifiedComparison (1M步, 种子42/123/456/789/1234):")
uni_ppo = []
uni_hcgae = []
for s in [42, 123, 456, 789, 1234]:
    for algo, lst in [('Standard_PPO', uni_ppo), ('HCGAE_Imp12', uni_hcgae)]:
        fp = Path(f'results/UnifiedComparison/HalfCheetah-v4/{algo}/{algo}_s{s}.json')
        if fp.exists():
            with open(fp) as f:
                d = json.load(f)
            er = d.get('eval_rewards', [])
            v = float(np.mean(er[-5:])) if len(er) >= 5 else d.get('final_reward', 0)
            lst.append(v)

print(f"   Standard_PPO: {np.mean(uni_ppo):.0f} ± {np.std(uni_ppo)/np.sqrt(len(uni_ppo)):.0f} (n={len(uni_ppo)})")
print(f"   HCGAE:        {np.mean(uni_hcgae):.0f} ± {np.std(uni_hcgae)/np.sqrt(len(uni_hcgae)):.0f} (n={len(uni_hcgae)})")
print(f"   Individual PPO: {[f'{v:.0f}' for v in uni_ppo]}")
print(f"   Individual HCGAE: {[f'{v:.0f}' for v in uni_hcgae]}")

# Mann-Whitney 检验
u, p = stats.mannwhitneyu(uni_hcgae, uni_ppo, alternative='two-sided')
print(f"   Mann-Whitney U={u:.0f}, p={p:.3f}")

# 读取 MultiSeedPower 数据 (300K步, 10种子)
print("\n3. MultiSeedPower (300K步, 种子1-10):")
ms_ppo = []
ms_hcgae = []
for s in range(1, 11):
    for algo, lst in [('Standard_PPO', ms_ppo), ('HCGAE_Imp12', ms_hcgae)]:
        fp = Path(f'results/MultiSeedPower/HalfCheetah-v4/{algo}/{algo}_s{s}_metrics.json')
        if fp.exists():
            with open(fp) as f:
                d = json.load(f)
            er = d.get('eval_rewards', [])
            v = float(np.mean(er[-5:])) if len(er) >= 5 else d.get('final_reward', 0)
            lst.append(v)

print(f"   Standard_PPO: {np.mean(ms_ppo):.0f} ± {np.std(ms_ppo)/np.sqrt(len(ms_ppo)):.0f} (n={len(ms_ppo)})")
print(f"   HCGAE:        {np.mean(ms_hcgae):.0f} ± {np.std(ms_hcgae)/np.sqrt(len(ms_hcgae)):.0f} (n={len(ms_hcgae)})")
print(f"   Individual PPO: {[f'{v:.0f}' for v in ms_ppo]}")
print(f"   Individual HCGAE: {[f'{v:.0f}' for v in ms_hcgae]}")
if ms_ppo and ms_hcgae:
    u, p = stats.mannwhitneyu(ms_hcgae, ms_ppo, alternative='two-sided')
    print(f"   Mann-Whitney U={u:.0f}, p={p:.3f}")

# 检查 UnifiedComparison 评测时机
print("\n4. UnifiedComparison HCGAE HalfCheetah 各时段数据 (seed=42):")
fp = Path('results/UnifiedComparison/HalfCheetah-v4/HCGAE_Imp12/HCGAE_Imp12_s42.json')
if fp.exists():
    with open(fp) as f:
        d = json.load(f)
    er = d.get('eval_rewards', [])
    es = d.get('eval_steps', [])
    print(f"   总评测次数: {len(er)}")
    checkpoints = [100_000, 300_000, 500_000, 700_000, 1_000_000]
    for ckpt in checkpoints:
        close = sorted([(abs(step-ckpt), step, rew) for step, rew in zip(es, er)])
        _, actual, rew = close[0]
        print(f"   @{ckpt//1000:>4}K: eval={rew:.0f}")

print("\n5. BaselineComparison HCGAE HalfCheetah 各时段数据 (seed=42):")
fp = Path('results/BaselineComparison/HalfCheetah-v4/HCGAE_Imp12/HCGAE_Imp12_s42_metrics.json')
if fp.exists():
    with open(fp) as f:
        d = json.load(f)
    er = d.get('eval_rewards', [])
    es = d.get('eval_steps', [])
    print(f"   总评测次数: {len(er)}")
    checkpoints = [50_000, 100_000, 200_000, 300_000]
    for ckpt in checkpoints:
        if es:
            close = sorted([(abs(step-ckpt), step, rew) for step, rew in zip(es, er)])
            _, actual, rew = close[0]
            print(f"   @{ckpt//1000:>4}K: eval={rew:.0f}")
        else:
            print(f"   No eval_steps data, final_reward={d.get('final_reward', 'N/A')}")

print("\n" + "=" * 80)
print("分析结论:")
print("="*80)
print("""
UnifiedComparison (1M步) vs BaselineComparison (300K步) vs MultiSeedPower (300K步, n=10):
- BaselineComparison (n=5, 300K步): PPO=902, HCGAE=828 (-8.2%)
- MultiSeedPower (n=10, 300K步):    PPO=950, HCGAE=757 (-20.3%, p=0.026)
- UnifiedComparison (n=5, 1M步):    PPO=956, HCGAE=1209 (+26.5%)

关键差异原因:
1. n=5 时统计不稳定（HalfCheetah方差大）
2. UnifiedComparison使用不同的种子（42/123/456/789/1234），而MultiSeedPower使用1-10
3. 可能HCGAE在HalfCheetah上的表现与训练时间非单调相关
4. 1M步时HCGAE表现更好可能是因为：Critic在长训练后终于收敛，此时HCGAE校正已关闭

→ 核心统计真相：n=10种子显示p=0.026（显著劣于），n=5时结论不稳定。
  论文应以n=10多种子数据（MultiSeedPower）为准，报告300K步时的显著负向结果。
""")

