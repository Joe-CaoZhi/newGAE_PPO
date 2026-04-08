#!/usr/bin/env python3

with open('docs/paper_draft_zh.md', 'r', encoding='utf-8') as f:
    content = f.read()

original = content

# 数字变化范围
content = content.replace('（677\u2192710.5）', '（从 677 增至 710.5）')
content = content.replace('v1: \u221229.2% \u2192 v2: \u221214.6% \u2192 v3: \u221210.4%',
                          'v1: \u221229.2%，v2: \u221214.6%，v3: \u221210.4%')

# 因果链中的→
content = content.replace(
    '\u201c\u4f4e EV\u2192\u566a\u58f0\u68af\u5ea6\u2192\u7b56\u7565\u65e0\u6cd5\u6539\u5584\u2192EV \u7ee7\u7eed\u4f4e\u8ff7\u201d',
    '\u201c\u4f4e EV $\\Rightarrow$ \u566a\u58f0\u68af\u5ea6 $\\Rightarrow$ \u7b56\u7565\u65e0\u6cd5\u6539\u5584 $\\Rightarrow$ EV \u7ee7\u7eed\u4f4e\u8ff7\u201d'
)
content = content.replace(
    '\u6539\u8fdb I \u7a33\u5b9a\u6821\u6b63\u5206\u5e03\u2192EV \u52a0\u901f\u2192\u6539\u8fdb II \u5b89\u5168\u63d0\u9ad8 MC \u6743\u91cd\u7684\u81ea\u5f3a\u5316\u5faa\u73af',
    '\u6539\u8fdb I \u7a33\u5b9a\u6821\u6b63\u5206\u5e03\uff0cEV \u52a0\u901f\u63d0\u5347\uff0c\u6539\u8fdb II \u53ef\u5b89\u5168\u63d0\u9ad8 MC \u6743\u91cd\u7684\u81ea\u5f3a\u5316\u5faa\u73af'
)

# SCR分析
content = content.replace(
    '\uff08\u9ad8\u5956\u52b1\u4e00\u81f4\u6027 \u2192 SCR < 1\uff09',
    '\uff08\u9ad8\u5956\u52b1\u4e00\u81f4\u6027\uff0c\u56e0\u6b64 SCR < 1\uff09'
)
content = content.replace(
    'Hopper\uff08SCR\u226b1 \u2192 \u6709\u76ca\uff09\u3001Walker2d\uff08\u4e34\u754c \u2192 \u6709\u76ca\uff09\u3001HalfCheetah\uff08SCR<1 \u2192 v1 \u6709\u5bb3\uff0cv2 \u901a\u8fc7\u68c0\u6d4b\u6536\u655b\u901f\u7387\u90e8\u5206\u9006\u8f6c\uff09',
    'Hopper\uff08SCR\u226b1\uff0c\u6709\u76ca\uff09\u3001Walker2d\uff08\u4e34\u754c\uff0c\u6709\u76ca\uff09\u3001HalfCheetah\uff08SCR<1\uff0cv1 \u6709\u5bb3\uff1bv2 \u901a\u8fc7\u68c0\u6d4b\u6536\u655b\u901f\u7387\u90e8\u5206\u9006\u8f6c\uff09'
)

# HCGAE链式描述
content = content.replace(
    'HCGAE\u2192EV\u2192SNR_eff\u2192\u68af\u5ea6\u7684\u94fe\u5f0f\u53cd\u5e94',
    'HCGAE \u63d0\u5347 EV\uff0cEV \u6539\u5584 SNR_eff\uff0c\u8fdb\u800c\u6539\u5584\u68af\u5ea6\u7684\u94fe\u5f0f\u53cd\u5e94'
)

# 42.6%的 HCGAE→SCR 回升
content = content.replace(
    '42.6% \u7684 HCGAE\u2192SCR \u56de\u5347',
    '\u4ece HCGAE \u5230 HCGAE+SCR \u7684 42.6% \u56de\u5347'
)

print(f"修改了 {sum(1 for a, b in zip(original, content) if a != b)} 处字符")

# 统计剩余的→
remaining = []
for i, line in enumerate(content.split('\n'), 1):
    if '\u2192' in line:
        remaining.append((i, line[:100]))

print(f"剩余→共 {len(remaining)} 处（含数学公式内的）")

with open('docs/paper_draft_zh.md', 'w', encoding='utf-8') as f:
    f.write(content)
print("保存完成")

