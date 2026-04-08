"""分析 UnifiedComparison 学习曲线：是否存在性能崩溃 + 与 BaselineComparison 超参对比"""
import json
from pathlib import Path

import numpy as np

SEEDS = [42, 123, 456, 789, 1234]


def load_uni(env, algo, seed):
    fp = Path('results/UnifiedComparison') / env / algo / f'{algo}_s{seed}.json'
    if not fp.exists():
        return None
    with open(fp) as f:
        return json.load(f)


def load_bl(env, algo, seed):
    fp = Path('results/BaselineComparison') / env / algo / f'{algo}_s{seed}_metrics.json'
    if not fp.exists():
        return None
    with open(fp) as f:
        return json.load(f)


# ── 1. 检查 PPO 在 1M 步的完整曲线是否有崩溃现象 ──────────────────────────────
print('=' * 80)
print('1. Standard_PPO @ Hopper-v4 — 各时间段性能（全部5个种子）')
print('   验证是否存在训练后期性能崩溃')
print('=' * 80)

checkpoints = [100_000, 200_000, 300_000, 500_000, 700_000, 1_000_000]

for s in SEEDS:
    d = load_uni('Hopper-v4', 'Standard_PPO', s)
    if not d:
        continue
    er = d['eval_rewards']
    es = d['eval_steps']
    print(f'\n  Seed={s}:')
    for ckpt in checkpoints:
        # 取最接近 ckpt 的 eval 点
        close = [(abs(step - ckpt), step, rew) for step, rew in zip(es, er)]
        close.sort()
        _, actual_step, rew = close[0]
        print(f'    @{ckpt//1000:>4}K steps → eval={rew:>8.1f} (actual={actual_step})')

print()
print('=' * 80)
print('2. HCGAE_Imp12 @ Hopper-v4 — 各时间段性能（全部5个种子）')
print('=' * 80)

for s in SEEDS:
    d = load_uni('Hopper-v4', 'HCGAE_Imp12', s)
    if not d:
        continue
    er = d['eval_rewards']
    es = d['eval_steps']
    print(f'\n  Seed={s}:')
    for ckpt in checkpoints:
        close = [(abs(step - ckpt), step, rew) for step, rew in zip(es, er)]
        close.sort()
        _, actual_step, rew = close[0]
        print(f'    @{ckpt//1000:>4}K steps → eval={rew:>8.1f}')

print()
print('=' * 80)
print('3. 汇总：UnifiedComparison 各时段平均性能（5种子均值±SEM）')
print('   Hopper-v4, Standard_PPO vs HCGAE_Imp12')
print('=' * 80)
print(f'{"Checkpoint":<12} | {"Standard_PPO":>20} | {"HCGAE_Imp12":>20} | {"Delta":>10}')
print('-' * 70)

for ckpt in checkpoints:
    ppo_vals, hcgae_vals = [], []
    for s in SEEDS:
        for algo, bucket in [('Standard_PPO', ppo_vals), ('HCGAE_Imp12', hcgae_vals)]:
            d = load_uni('Hopper-v4', algo, s)
            if d:
                er, es = d['eval_rewards'], d['eval_steps']
                close = [(abs(step - ckpt), rew) for step, rew in zip(es, er)]
                close.sort()
                bucket.append(close[0][1])

    if ppo_vals and hcgae_vals:
        pm, psem = np.mean(ppo_vals), np.std(ppo_vals)/len(ppo_vals)**0.5
        hm, hsem = np.mean(hcgae_vals), np.std(hcgae_vals)/len(hcgae_vals)**0.5
        delta = hm - pm
        pct = delta / pm * 100 if pm > 0 else 0
        print(f'@{ckpt//1000:>5}K   | {pm:>9.0f}±{psem:>5.0f}       | {hm:>9.0f}±{hsem:>5.0f}       | {delta:>+7.0f}({pct:+.1f}%)')

print()
print('=' * 80)
print('4. BaselineComparison 参考值（300K步，同超参数）')
print('   为了对比，显示 BL 数据')
print('=' * 80)

bl_ppo, bl_hcgae = [], []
for s in SEEDS:
    for algo, bucket in [('Standard_PPO', bl_ppo), ('HCGAE_Imp12', bl_hcgae)]:
        d = load_bl('Hopper-v4', algo, s)
        if d:
            er = d.get('eval_rewards', [])
            fr = d.get('final_reward', 0.0)
            v = float(np.mean(er[-5:])) if len(er) >= 5 else fr
            bucket.append(v)

print(f'  BL Standard_PPO @300K: {np.mean(bl_ppo):.0f} ± {np.std(bl_ppo)/len(bl_ppo)**0.5:.0f}')
print(f'  BL HCGAE_Imp12  @300K: {np.mean(bl_hcgae):.0f} ± {np.std(bl_hcgae)/len(bl_hcgae)**0.5:.0f}')

print()
print('=' * 80)
print('5. 关键结论：论文需要使用哪组数据？')
print('=' * 80)
print("""
  情况诊断：
  - BaselineComparison (300K步): PPO=2735, HCGAE=2873 (+5.1%)
  - UnifiedComparison  (1M步):  PPO=1858, HCGAE=1997 (+7.5%)

  两组实验的超参数相同，但存在以下差异：
  1. UnifiedComparison 训练更长 (1M步)，PPO 出现性能崩溃（峰值约 3000，最终约 1858）
  2. UnifiedComparison 的 eval 在 rollout 收集中途执行（不是更新后），可能低估性能
  3. BaselineComparison 的 eval 在每轮 update 后立即执行

  SAC/TD3 对比应使用哪组？
  → 必须与 SAC/TD3 用完全相同的步数（1M步）
  → 应使用 UnifiedComparison 的1M步数据（PPO=1858, HCGAE=1997）
  → 或者重新为 PPO/HCGAE 跑1M步实验，并修复 eval 时机
""")

