#!/usr/bin/env python3
# Temporary script to fix v3 ablation table
with open('docs/paper_draft_zh.md', 'r', encoding='utf-8') as f:
    content = f.read()

replacements = [
    ('| 消融变体 | 均值（200K相对） | vs v3（全量） |',
     '| 消融变体 | 均值（200K相对） | vs HCGAE-Dense（全量） |'),
    ('| v3 全量（G-Clamp+VW-Gate+BdryPrior） | −18.3 | — |',
     '| HCGAE-Dense 全量（G-Clamp+VW-Gate+BdryPrior） | −18.3 | — |'),
    ('| v3 无G-Clamping | −36.0 | −17.7 |',
     '| HCGAE-Dense 无G-Clamping | −36.0 | −17.7 |'),
    ('| v3 无VW-Gate | −29.3 | −11.0 |',
     '| HCGAE-Dense 无VW-Gate | −29.3 | −11.0 |'),
    ('| v3 无Boundary Prior | −26.1 | −7.8 |',
     '| HCGAE-Dense 无Boundary Prior | −26.1 | −7.8 |'),
    ('**解读：** v3 相比 v2 在 Ant-v4 上实现 **+4.9%** 均值提升，更重要的是**方差缩小 35%**（117.5 vs 201.0）',
     '**解读：** HCGAE-Dense 相比 HCGAE 在 Ant-v4 上实现 **+4.9%** 均值提升，更重要的是**方差缩小 35%**（117.5 vs 180.0）'),
    ('G-Clamping（FIX①）贡献最大。v3 将 Ant-v4 的差距从 −14.6% 缩窄至 **−10.4%**',
     'G-Clamping（FIX①）贡献最大。HCGAE-Dense 将 Ant-v4 的差距从 −14.6% 缩窄至 **−10.4%**'),
]

for old, new in replacements:
    if old in content:
        content = content.replace(old, new, 1)
        print(f"Replaced: {old[:50]}...")
    else:
        print(f"NOT FOUND: {old[:50]}...")

with open('docs/paper_draft_zh.md', 'w', encoding='utf-8') as f:
    f.write(content)
print("Done!")

