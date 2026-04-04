#!/usr/bin/env python3
# Fix cross-references in paper_draft_zh.md
import re

filepath = 'docs/paper_draft_zh.md'
content = open(filepath, 'r', encoding='utf-8').read()

# Fix 1: Line 558 - §4.5 -> §4.2 (Walker2d context)
old1 = '结合 §4.5 的 300K 步分析（HCGAE vs. PPO：d=−0.272，n.s.）'
new1 = '结合 §4.2 的 300K 步分析（HCGAE vs. PPO：d=−0.272，n.s.）'
if old1 in content:
    content = content.replace(old1, new1, 1)
    print("Fix 1 applied: Walker2d §4.5 -> §4.2")
else:
    print("Fix 1 NOT FOUND (may already be applied)")

# Fix 2: Line 560 - §4.5 -> §4.2 (HalfCheetah note context)
old2 = '与 §4.5 的 n=10、300K 步显著负向结果（p=0.026）'
new2 = '与 §4.2 的 n=10、300K 步显著负向结果（p=0.026）'
if old2 in content:
    content = content.replace(old2, new2, 1)
    print("Fix 2 applied: HalfCheetah §4.5 -> §4.2")
else:
    print("Fix 2 NOT FOUND (may already be applied)")

open(filepath, 'w', encoding='utf-8').write(content)
print("Done.")

