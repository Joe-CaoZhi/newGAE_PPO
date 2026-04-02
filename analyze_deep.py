"""深度分析实验结果，找出各方法的数学缺陷和改进方向"""
import json
import os

import numpy as np


def analyze_results(results_dir, env_name="CartPole-v1"):
    print(f"\n{'='*70}")
    print(f"  深度分析: {env_name}")
    print(f"{'='*70}")

    agent_data = {}
    for f in sorted(os.listdir(results_dir)):
        if not f.endswith('.json'):
            continue
        d = json.load(open(f'{results_dir}/{f}'))
        evals = np.array(d.get('eval_rewards', []))
        steps = np.array(d.get('eval_steps', []))
        name = d.get('agent_name', f.replace('_metrics.json', ''))
        if len(evals) == 0:
            continue
        agent_data[name] = {'evals': evals, 'steps': steps, 'data': d}

    # 1. 收敛速度分析
    print("\n--- 收敛速度分析 ---")
    thresholds = {
        'CartPole-v1': [200, 350, 450, 490],
        'LunarLander-v3': [0, 50, 100, 150, 200],
        'Acrobot-v1': [-300, -200, -150, -120, -100],
    }
    thresh = thresholds.get(env_name, [0])

    for name, ad in sorted(agent_data.items()):
        evals, steps = ad['evals'], ad['steps']
        row = f"{name:<35}"
        for t in thresh:
            if env_name == 'Acrobot-v1':
                mask = evals >= t
            else:
                mask = evals >= t
            if np.any(mask):
                idx = np.argmax(mask)
                row += f"  >{t}@{steps[idx]//1000}k"
            else:
                row += f"  >{t}@--"
        row += f"  | final={evals[-1]:.1f} best={evals.max():.1f}"
        print(row)

    # 2. 训练稳定性分析
    print("\n--- 训练稳定性分析 ---")
    for name, ad in sorted(agent_data.items()):
        evals = ad['evals']
        if len(evals) < 5:
            continue
        # 计算方差、最大回撤、最终5轮均值
        variance = evals.std()
        max_drop = 0.0
        for i in range(1, len(evals)):
            peak = evals[:i].max()
            drop = peak - evals[i]
            if drop > max_drop:
                max_drop = drop
        final5 = evals[-5:].mean()
        final10 = evals[-10:].mean()
        print(f"  {name:<35}: std={variance:.1f}  max_drop={max_drop:.1f}  final5={final5:.1f}  final10={final10:.1f}")

    # 3. 关键训练指标分析
    print("\n--- 训练指标分析（最后5次更新平均）---")
    for name, ad in sorted(agent_data.items()):
        d = ad['data']
        updates = d.get('updates', [])
        if not updates:
            # 尝试其他键
            vlosses = d.get('value_losses', [])
            evs = d.get('explained_variances', [])
            if vlosses:
                print(f"  {name:<35}: VL={np.mean(vlosses[-5:]):.3f}  EV={np.mean(evs[-5:]) if evs else 'N/A':.3f}")
            continue
        last_n = updates[-5:]
        vl = np.mean([u.get('value_loss', 0) for u in last_n])
        ev = np.mean([u.get('explained_variance', 0) for u in last_n])
        kl = np.mean([u.get('approx_kl', 0) for u in last_n])
        clip = np.mean([u.get('clip_frac', 0) for u in last_n])
        lam = np.mean([u.get('mean_lambda', 0.95) or 0.95 for u in last_n])
        print(f"  {name:<35}: VL={vl:.3f}  EV={ev:.3f}  KL={kl:.4f}  clip={clip:.2f}  λ={lam:.3f}")

    # 4. 数据结构探查
    print("\n--- 数据结构 ---")
    for name, ad in sorted(agent_data.items()):
        d = ad['data']
        keys = list(d.keys())
        print(f"  {name:<35}: keys={keys}")
        break  # 只打印一个


if __name__ == '__main__':
    import sys
    env = sys.argv[1] if len(sys.argv) > 1 else 'CartPole-v1'
    results_dir = f'results/{env}'
    if os.path.exists(results_dir):
        analyze_results(results_dir, env)
    else:
        print(f"目录不存在: {results_dir}")
        # 列出可用目录
        for d in os.listdir('results'):
            print(f"  可用: results/{d}")

