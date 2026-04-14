import json, numpy as np; from pathlib import Path
for algo in ['Optimal_PPO', 'Optimal_HCGAE_Optimal']:
    d = Path(f'results/FinalOptimal/Ant-v4/{algo}')
    scores = [np.mean(json.load(open(f))['eval_rewards'][-10:]) for f in sorted(d.glob('*.json'))]
    print(f'Ant-v4/{algo}: n={len(scores)} mean={np.mean(scores):.0f} std={np.std(scores):.0f} min={np.min(scores):.0f} max={np.max(scores):.0f}')

