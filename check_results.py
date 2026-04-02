"""检查已有实验结果"""
import json
import os
import sys

sys.path.insert(0, '.')

results_dir = 'results/CartPole-v1'
for f in sorted(os.listdir(results_dir)):
    if f.endswith('.json'):
        d = json.load(open(f'{results_dir}/{f}'))
        evals = d.get('eval_rewards', [])
        steps = d.get('total_steps', [])
        ep_rew = d.get('episode_rewards', [])
        name = d.get('agent_name', f)
        last5 = sum(evals[-5:]) / min(5, len(evals)) if evals else 0
        print(f"{name:<30}: eval_count={len(evals):3d}, max_step={max(steps) if steps else 0:7d}, "
              f"last_eval={evals[-1] if evals else 0:7.1f}, last5_mean={last5:7.1f}, ep_count={len(ep_rew)}")

