import json, os, numpy as np

base = 'results/ICMLExperiment'
envs = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4', 'Ant-v4']
algos = ['Standard_PPO', 'Optimal_PPO', 'Optimal_HCGAE', 'Optimal_HCGAE_SCR',
         'Optimal_HCGAE_v2', 'Optimal_HCGAE_v2_NoBdry', 'Optimal_HCGAE_v2_NoGate']

for env in envs:
    print(f'\n=== {env} ===')
    for algo in algos:
        d = f'{base}/{env}/{algo}'
        if not os.path.exists(d):
            print(f'  {algo}: MISSING')
            continue
        files = [f for f in os.listdir(d) if f.endswith('.json')]
        rewards = []
        for f in files:
            try:
                data = json.load(open(f'{d}/{f}'))
                evr = data.get('eval_rewards', [])
                if evr:
                    rewards.append(float(np.mean(evr[-5:])))
                elif 'final_reward' in data:
                    rewards.append(float(data['final_reward']))
            except:
                pass
        if rewards:
            print(f'  {algo}: n={len(rewards)}, mean={np.mean(rewards):.1f} +/- {np.std(rewards):.1f}')
        else:
            print(f'  {algo}: n=0 (no data), files={len(files)}')

