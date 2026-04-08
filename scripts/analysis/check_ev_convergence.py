import json
import os

ev_path = 'results/EVConvergenceStudy'
files = sorted([f for f in os.listdir(ev_path) if f.endswith('.json')])

# Categorize files
series_files = [f for f in files if 'series' in f]
summary_files = [f for f in files if 'summary' in f or 'report' in f]
other_files = [f for f in files if f not in series_files and f not in summary_files]

print(f'Total JSON files: {len(files)}')
print(f'Series files: {len(series_files)}')
print(f'Summary files: {len(summary_files)}')
print(f'Other files: {len(other_files)}')

# Check summary files
print('\n=== Summary files ===')
for f in summary_files[:5]:
    fpath = os.path.join(ev_path, f)
    with open(fpath) as fp:
        d = json.load(fp)
    print(f'\n{f}:')
    if isinstance(d, dict):
        for k, v in list(d.items())[:10]:
            print(f'  {k}: {str(v)[:100]}')

# Check for EV timing data - look at series files
print('\n=== Sample series file ===')
if series_files:
    sf = os.path.join(ev_path, series_files[0])
    with open(sf) as fp:
        d = json.load(fp)
    print(f'{series_files[0]}:')
    for k, v in list(d.items())[:5]:
        if isinstance(v, list):
            print(f'  {k}: list len={len(v)}, first few: {v[:5]}')
        else:
            print(f'  {k}: {v}')

# Check other files for timing data
print('\n=== Other files sample ===')
for f in other_files[:5]:
    fpath = os.path.join(ev_path, f)
    with open(fpath) as fp:
        d = json.load(fp)
    print(f'\n{f}:')
    for k, v in list(d.items())[:5]:
        print(f'  {k}: {str(v)[:80]}')

# Look for AULC data
print('\n=== Looking for AULC data ===')
for f in files:
    fpath = os.path.join(ev_path, f)
    with open(fpath) as fp:
        d = json.load(fp)
    if isinstance(d, dict):
        for k in d.keys():
            if 'aulc' in k.lower() or 'auc' in k.lower() or 'steps_to' in k.lower() or 'convergence' in k.lower():
                print(f'  {f}: key={k}, value={str(d[k])[:100]}')
                break

