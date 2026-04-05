import json

with open('results/ICMLExperiment/Ant-v4/Optimal_HCGAE_v2/Optimal_HCGAE_v2_s0.json') as f:
    d = json.load(f)

print('Top-level keys:', list(d.keys()))

for k in list(d.keys()):
    v = d[k]
    if isinstance(v, list) and len(v) > 0:
        print(f'\n  {k}: list of {len(v)} items')
        if isinstance(v[0], dict):
            print(f'    Item[0] keys: {list(v[0].keys())}')
            print(f'    Item[0] sample: {v[0]}')
        else:
            print(f'    Item[0]: {v[0]}')
    elif isinstance(v, dict):
        print(f'\n  {k}: dict with {len(v)} keys')
        print(f'    Keys: {list(v.keys())[:10]}')
        if v:
            first_key = list(v.keys())[0]
            first_val = v[first_key]
            if isinstance(first_val, list):
                print(f'    {first_key}: list of {len(first_val)}, first 3: {first_val[:3]}')
            else:
                print(f'    {first_key}: {first_val}')
    elif isinstance(v, (int, float, str)):
        print(f'\n  {k}: {v}')

