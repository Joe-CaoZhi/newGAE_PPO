"""快速检查已有实验数据中的格式，并找 metrics 文件"""
import json
import os

# 检查 ICMLExperiment 数据格式
with open('results/ICMLExperiment/Hopper-v4/Optimal_PPO/Optimal_PPO_s0.json') as f:
    d = json.load(f)

print('=== ICMLExperiment data format ===')
print('Keys:', list(d.keys()))
print('eval_rewards:', d.get('eval_rewards', [])[:5])
print('eval_steps:', d.get('eval_steps', [])[:5])
print()

# 搜索包含 explained_variances 的文件
print('=== Searching for metrics files with EV data ===')
for root, dirs, files in os.walk('results'):
    for f in files:
        if f.endswith('_metrics.json'):
            path = os.path.join(root, f)
            try:
                with open(path) as fh:
                    dd = json.load(fh)
                if 'explained_variances' in dd and len(dd['explained_variances']) > 0:
                    print(f'Found: {path}')
                    print(f'  Keys: {list(dd.keys())[:8]}')
                    evs = dd['explained_variances']
                    steps = dd['total_steps']
                    print(f'  EV len: {len(evs)}, steps len: {len(steps)}')
                    print(f'  First 3 steps: {steps[:3]}, EVs: {[round(e,3) for e in evs[:3]]}')
                    # find when EV > 0.9
                    for s, e in zip(steps, evs):
                        if e > 0.9:
                            print(f'  => EV>0.9 first at step {s} (EV={e:.3f})')
                            break
                    break
            except Exception as ex:
                pass
    else:
        continue
    break

