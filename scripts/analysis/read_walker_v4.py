import json
import numpy as np
import os

env = 'Walker2d-v4'
algo = 'Optimal_HCGAE_v4'
vals = []
for s in range(5):
    path = f'results/V4FullExperiment/{env}/{algo}/{algo}_s{s}.json'
    if os.path.exists(path):
        d = json.load(open(path))
        v = np.mean(d['eval_rewards'][-5:])
        vals.append(v)
        print(f's{s}: last5={v:.1f}')
print(f'Mean: {np.mean(vals):.1f} ± {np.std(vals)/np.sqrt(len(vals)):.1f} (n={len(vals)})')

