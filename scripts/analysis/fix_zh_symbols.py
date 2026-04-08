#!/usr/bin/env python3
"""清理中文论文草稿中的非正式符号，替换为正式学术符号"""

with open('docs/paper_draft_zh.md', 'r', encoding='utf-8') as f:
    content = f.read()

original = content

# ========================
# 1. 附录 G 中的代码对照表（✅ ⚠️ ❌ 在技术比较表格中）
#    这些是"技术状态符号"，在学术附录中需要替换为文字
# ========================

# 附录G中代码实现状态表格内的替换
# 将 ✅ 替换为 "是" 或 "[匹配]"，⚠️ 替换为 "[注]"，❌ 替换为 "否"
# 只在附录G的代码对照文本中
content = content.replace('— ✅ **完全匹配**', '— **完全匹配**')
content = content.replace('— ⚠️ **实现偏差**', '— **[注] 实现偏差**')
content = content.replace('— ✅ **与论文一致**', '— **与论文一致**')
content = content.replace('同样修正，✅ **下界 = 0.1**', '同样修正，**下界 = 0.1**')
content = content.replace('`last_value_corrected`。✅ **完整实现**', '`last_value_corrected`。**完整实现**')
content = content.replace('（与 HindsightPPO 匹配）。✅ **v2 中完整实现**', '（与 HindsightPPO 匹配）。**v2 中完整实现**')
content = content.replace('中计算，在 `update()` 中重用。✅ **完整', '中计算，在 `update()` 中重用。**完整')

# 附录G表格中的状态列
# | I：... | ✅ | ✅ | ✅ |  -> 用"是/否/注"
content = content.replace('| I：批内中心化 sigmoid | ✅ | ✅ | ✅ |',
                           '| I：批内中心化 sigmoid | 是 | 是 | 是 |')
content = content.replace('| II：EV 驱动目标混合（c_mc ≥ 0.1） | ✅ | ⚠️ 0.0 下界（已修正）→ ✅ 现为 0.1 | ✅ |',
                           '| II：EV 驱动目标混合（c_mc ≥ 0.1） | 是 | [注] 0.0 下界（已修正）改为 0.1 | 是 |')
content = content.replace('| III：边界 bootstrap 校正 | ✅ 完整 | ❌ 未应用 | ✅ 完整 |',
                           '| III：边界 bootstrap 校正 | 完整 | 未应用 | 完整 |')
content = content.replace('| IV：EV 增长率门控（新增） | ❌ 不适用 | ❌ 不适用 | ✅ v2 新增 |',
                           '| IV：EV 增长率门控（新增） | 不适用 | 不适用 | v2 新增 |')
content = content.replace('| V：冻结优势归一化 | ✅ | ⚠️ 按 mini-batch（有意设计） | ⚠️ 按 mini-batch（有意设计） |',
                           '| V：冻结优势归一化 | 是 | [注] 按 mini-batch（有意设计） | [注] 按 mini-batch（有意设计） |')

# 附录G结果汇总表格
content = content.replace('| **Optimal HCGAE v2** ✅ | **1760 ± 380** | **1999 ± 785** | 1550 ± 389 | 677 ± 201 |',
                           '| **Optimal HCGAE v2** | **1760 ± 380** | **1999 ± 785** | 1550 ± 389 | 677 ± 201 |')
content = content.replace('| Δ v2 vs v1 | +0.5% | **+6.8%** ✅ | **+24.1%** ✅ | **+20.5%** ✅ |',
                           '| Δ v2 vs v1 | +0.5% | **+6.8%** | **+24.1%** | **+20.5%** |')
content = content.replace('| Δ v2 vs Opt. PPO | **+10.1%** | **+25.2%** | **+4.3%** | −14.6% ⚠️ |',
                           '| Δ v2 vs Opt. PPO | **+10.1%** | **+25.2%** | **+4.3%** | −14.6%$^\\dagger$ |')

# 附录G消融表格
content = content.replace('| **v2_NoBdry** | 仅 EV 门控（无边界校正） | **1766 ± 755** ✅ | **1545 ± 153** | 1475 ± 643 | **711 ± 73** ✅ |',
                           '| **v2_NoBdry** | 仅 EV 门控（无边界校正） | **1766 ± 755** | **1545 ± 153** | 1475 ± 643 | **711 ± 73** |')
content = content.replace('| **v2 完整版** | EV 门控 + 边界校正 | 1550 ± 389 ✅ | **1760 ± 380** ✅ | **1999 ± 785** ✅ | 677 ± 201 |',
                           '| **v2 完整版** | EV 门控 + 边界校正 | 1550 ± 389 | **1760 ± 380** | **1999 ± 785** | 677 ± 201 |')

# 附录G环境影响表格
content = content.replace('| HalfCheetah | +41.4% ✅ | +20.3% ✅ | 稳定方差 |',
                           '| HalfCheetah | +41.4% | +20.3% | 稳定方差 |')
content = content.replace('| Ant | +26.5% ✅ | -7.1% ⚠️ | 有害 |',
                           '| Ant | +26.5% | $-$7.1%$^\\dagger$ | 有害 |')
content = content.replace('| Hopper | -11.8% ⚠️ | -6.6% ⚠️ | 必需（vs NoBdry +12.3%）|',
                           '| Hopper | $-$11.8%$^\\dagger$ | $-$6.6%$^\\dagger$ | 必需（vs NoBdry +12.3%）|')
content = content.replace('| Walker2d | -21.2% ⚠️ | -16.5% ⚠️ | 必需（vs NoBdry +35.5%）|',
                           '| Walker2d | $-$21.2%$^\\dagger$ | $-$16.5%$^\\dagger$ | 必需（vs NoBdry +35.5%）|')

# ========================
# 2. 主体表格中的 ⚠ 注释符号（表示统计显著的劣化）
# ========================
# 表格内的 ⚠ -> $^†$（LaTeX 脚注符号）
content = content.replace('757 ± 47 ⚠ |', '757 ± 47$^\\dagger$ |')
content = content.replace('709 ± 59 ⚠ |', '709 ± 59$^\\dagger$ |')
content = content.replace('412 ± 16 ⚠ |', '412 ± 16$^\\dagger$ |')
# 脚注说明中的 ⚠
content = content.replace('*⚠ 相比 Standard PPO 统计显著的劣化', '*$^\\dagger$ 相比 Standard PPO 统计显著的劣化')

# EV诊断表格中的 ⚠
content = content.replace('412 ± 16 ⚠ |', '412 ± 16$^\\dagger$ |')

# ========================
# 3. 文本中的 → 符号
#    (A) LaTeX 数学公式中的 → 应改为 \to（但这些通常已在 $$ 内）
#    (B) 因果链/流程描述中的 → 改为文字"→"在中文版本中作为标点是可接受的
#    但作为正式格式，改为用破折号+文字或直接描述
# ========================

# 图引用中的 → （指向文件路径，改为破折号）
content = content.replace(') → `results/', ')，见 `results/')

# 自强化循环描述中的 →（保留为数学文本的→，或改为文字"使得"/"导致"）
# 这类 → 在中文论文中是惯用的逻辑推导符号，ICML 中文版本可保留为 \Rightarrow 或文字，
# 但在 Markdown 草稿中可保留 →，因为论文会重新排版
# 主要改掉"非数学文本"语境中的孤立箭头

# 摘要和引言中讨论SCR的 → （"→ SCR ≫ 1"这类）: 改为破折号
content = content.replace('奖励 → SCR ≫ 1，', '奖励，因此 SCR ≫ 1，')

# 域适用性表格中的 ✓ ✗ -> 是/否/适用/不适用
content = content.replace('| RTB / 竞价 | ✓（稀疏，短期） | ✓（非平稳） | 需要离线策略校正 |',
                           '| RTB / 竞价 | 适用（稀疏，短期） | 适用（非平稳） | 需要离线策略校正 |')
content = content.replace('| 机器人操作（D≫1） | ✓（SCR 临界） | ✓✓（ImpG 在 D≥10 有益） | SCR 实证临界 |',
                           '| 机器人操作（D≫1） | 适用（SCR 临界） | 适用（ImpG 在 D≥10 有益） | SCR 实证临界 |')
content = content.replace('| RLHF / LLM（token 级） | ✓（早期 RM 噪声） | ✓（不一致 RM） | 不同噪声来源 |',
                           '| RLHF / LLM（token 级） | 适用（早期 RM 噪声） | 适用（不一致 RM） | 不同噪声来源 |')
content = content.replace('| 密集奖励（HalfCheetah 类） | ✗（§5.1 分析） | ✓ | HCGAE 禁用；DCPPO-S 独立有效 |',
                           '| 密集奖励（HalfCheetah 类） | 不适用（§5.1 分析） | 适用 | HCGAE 禁用；DCPPO-S 独立有效 |')

# ========================
# 4. 正文中的 → 在因果链描述中
#    在正式学术中文论文中，这类→通常保留（尤其在摘要/贡献描述的因果列表中）
#    但要统一：正文段落中的孤立 → 改为文字，表格/列表中可保留
# ========================
# 流程描述中的 → 改为"从而"/"进而"（仅选择明确的不适合用箭头的情形）

# 步数→50% 这是专业术语，保留
# 6.7 → 13.4 ms 这是数值范围，改为破折号
content = content.replace('（6.7 → 13.4 ms）', '（6.7 ms 增至 13.4 ms）')
content = content.replace('c_MC → 0.1（纯 TD 目标）', '$c_{\\rm MC} \\to 0.1$（纯 TD 目标）')
content = content.replace('c_mc → 0.1（纯 TD 目标）', '$c_{\\rm MC} \\to 0.1$（纯 TD 目标）')
# 0.0→0.1
content = content.replace('（c_mc 下界修正（0.0→0.1）本身', '（$c_{\\rm MC}$ 下界修正（0.0 改为 0.1）本身')

# ========================
# 5. 剩余的 ✅ ⚠️ ❌ 符号（如果还有的话）
# ========================
# 最后一道清扫：将所有剩余的这类符号替换
content = content.replace(' ✅ ', ' ')
content = content.replace('✅ ', '')
content = content.replace(' ✅', '')
content = content.replace('✅', '')
content = content.replace(' ⚠️ ', '$^\\dagger$')
content = content.replace('⚠️', '$^\\dagger$')
content = content.replace(' ⚠ ', '$^\\dagger$')
content = content.replace('⚠', '$^\\dagger$')
content = content.replace(' ❌ ', ' ')
content = content.replace('❌', '')

print(f"原始长度: {len(original)} 字符")
print(f"修改后长度: {len(content)} 字符")
print(f"共修改 {sum(1 for a, b in zip(original, content) if a != b)} 处字符差异")

# 验证：还有多少目标符号？
remaining = []
for sym in ['✅', '⚠️', '⚠', '✓', '✗', '❌']:
    count = content.count(sym)
    if count > 0:
        remaining.append(f"  {sym}: {count} 处")
if remaining:
    print("\n[警告] 以下符号仍未清理：")
    for r in remaining:
        print(r)
else:
    print("\n[OK] 所有目标符号已清理完毕")

with open('docs/paper_draft_zh.md', 'w', encoding='utf-8') as f:
    f.write(content)

print("文件已保存")

