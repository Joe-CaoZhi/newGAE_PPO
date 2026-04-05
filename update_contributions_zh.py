#!/usr/bin/env python3
"""Update Contributions section in paper_draft_zh.md to include decoupling experiment."""

# Read the file
with open('docs/paper_draft_zh.md', 'r', encoding='utf-8') as f:
    content = f.read()

# The old contributions item 3
old_item3 = '''3. **多种子实证分析**（§4）：4 个环境、4 种算法、5 个种子，带 Mann-Whitney 统计检验；60 次实验运行的组件级消融（表 G.2），刻画各 v2 组件的环境依赖作用；与 5 种独立实现的 PPO 变体对比。结果包含一个重要的负面发现：值函数裁剪（PPO-VClip）在 Hopper-v4 和 Walker2d-v4 上显著有害（$d > 6.0$，$p = 0.008$），复现并从机制上解释了 Engstrom 等人 (2020) 的结论。'''

# The new contributions item 3
new_item3 = '''3. **多种子实证分析**（§4）：4 个环境、4 种算法、5 个种子，带 Mann-Whitney 统计检验；60 次实验运行的组件级消融（表 G.2），刻画各 v2 组件的环境依赖作用；与 5 种独立实现的 PPO 变体对比。结果包含一个重要的负面发现：值函数裁剪（PPO-VClip）在 Hopper-v4 和 Walker2d-v4 上显著有害（$d > 6.0$，$p = 0.008$），复现并从机制上解释了 Engstrom 等人 (2020) 的结论。

4. **解耦实验**（§4.6.1）：在 Standard PPO 基线（无实现技巧）与 Optimal PPO 基线（观测归一化、优势归一化、学习率退火）上系统对比 HCGAE v2，揭示**环境依赖的耦合效应**——HCGAE v2 的增益在 Hopper-v4 上基本独立于 Optimal 技巧（+7.6% vs. +10.1%），但在 Walker2d-v4（−0.6% vs. +25.2%）和 HalfCheetah-v4（−31.6% vs. +4.3%）上强耦合。该发现表明观测归一化创造了 EV 驱动门控按预期工作的条件。

5. **SCR 框架与诚实的局限性刻画**（§5 和 §7）：'''

# Replace
if old_item3 in content:
    content = content.replace(old_item3, new_item3)
    # Also need to update item 4 to item 5
    content = content.replace(
        '4. **SCR 框架与诚实的局限性刻画**（§5 和 §7）：信号-校正比',
        '信号-校正比'
    )
    with open('docs/paper_draft_zh.md', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully updated Contributions section in paper_draft_zh.md")
else:
    print("Old item 3 not found")

