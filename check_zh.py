with open('docs/paper_draft_zh.md', 'r', encoding='utf-8') as f:
    content = f.read()
for sym in ['✅', '⚠️', '⚠', '✓', '✗', '❌']:
    count = content.count(sym)
    if count > 0:
        print(f'{repr(sym)}: {count}处')
        # 找到这些行
        for i, line in enumerate(content.split('\n'), 1):
            if sym in line:
                print(f'  行{i}: {line[:80]}')
print('检查完毕')

