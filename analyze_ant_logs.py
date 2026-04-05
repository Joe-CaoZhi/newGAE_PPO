"""
Ant-v4 实验日志分析脚本
分析各算法的中间状态，找出HCGAE在Ant上失效的原因
"""
import json
import os

import numpy as np


def read_seeds(base_path, algo, n=5):
    data = []
    for i in range(n):
        p = os.path.join(base_path, algo, f'{algo}_s{i}.json')
        if os.path.exists(p):
            with open(p) as f:
                d = json.load(f)
            data.append(d)
    return data

base = 'results/ICMLExperiment/Ant-v4/'
algos = ['Standard_PPO', 'Optimal_PPO', 'Optimal_HCGAE', 'Optimal_HCGAE_v2']

print("=" * 80)
print("ANT-v4 实验日志深度分析")
print("=" * 80)

for algo in algos:
    seeds = read_seeds(base, algo)
    if not seeds:
        print(f"[{algo}] No data found!")
        continue

    print(f"\n{'='*60}")
    print(f"[{algo}]")
    print(f"{'='*60}")

    final_rewards = []
    all_ev = []
    all_alpha = []
    all_bias = []
    all_c_mc = []
    all_value_loss = []
    all_policy_loss = []
    all_clip_frac = []

    for s, d in enumerate(seeds):
        if 'eval_rewards' in d and d['eval_rewards']:
            final_rewards.append(d['eval_rewards'][-1])

        if 'updates' in d and d['updates']:
            updates = d['updates']
            n_upd = len(updates)

            ev_seq = [u.get('explained_variance', 0) for u in updates]
            alpha_seq = [u.get('mean_alpha', 0) for u in updates]
            bias_seq = [u.get('bias_proxy', 0) for u in updates]
            c_mc_seq = [u.get('c_mc', 0) for u in updates]
            vl_seq = [u.get('value_loss', 0) for u in updates]
            pl_seq = [u.get('policy_loss', 0) for u in updates]
            cf_seq = [u.get('clip_frac', 0) for u in updates]

            all_ev.append(ev_seq)
            all_alpha.append(alpha_seq)
            all_bias.append(bias_seq)
            all_c_mc.append(c_mc_seq)
            all_value_loss.append(vl_seq)
            all_policy_loss.append(pl_seq)
            all_clip_frac.append(cf_seq)

    # Final performance
    if final_rewards:
        print(f"  Final rewards: {[round(r,1) for r in final_rewards]}")
        print(f"  Mean: {np.mean(final_rewards):.1f} ± {np.std(final_rewards):.1f}")

    # EV trajectory analysis
    if all_ev:
        ev_arr = np.array([e[:min(len(e), 50)] for e in all_ev if len(e) >= 10])
        if len(ev_arr) > 0:
            n_pts = ev_arr.shape[1]
            checkpoints = [0, 4, 9, 19, min(29, n_pts-1), min(49, n_pts-1)]
            checkpoints = [c for c in checkpoints if c < n_pts]
            print("\n  EV 轨迹（Rollout索引 → EV均值 ± 标准差）:")
            for c in checkpoints:
                vals = ev_arr[:, c]
                print(f"    Rollout {c+1:2d}: EV = {vals.mean():.4f} ± {vals.std():.4f}")

            # EV收敛速度
            ev_final = ev_arr[:, -1].mean()
            ev_half = np.argmax(ev_arr.mean(axis=0) > 0.5) if (ev_arr.mean(axis=0) > 0.5).any() else -1
            print(f"  EV 最终值均值: {ev_final:.4f}")
            print(f"  到达EV=0.5的rollout索引: {ev_half+1 if ev_half >= 0 else 'N/A'}")

    # Alpha analysis (HCGAE specific)
    if all_alpha and any(any(a) for a in all_alpha):
        alpha_arr = np.array([a[:min(len(a), 50)] for a in all_alpha if len(a) >= 10])
        if len(alpha_arr) > 0:
            n_pts = alpha_arr.shape[1]
            checkpoints = [0, 4, 9, 19, min(49, n_pts-1)]
            checkpoints = [c for c in checkpoints if c < n_pts]
            print("\n  Mean Alpha (MC混合强度) 轨迹:")
            for c in checkpoints:
                vals = alpha_arr[:, c]
                print(f"    Rollout {c+1:2d}: alpha = {vals.mean():.4f} ± {vals.std():.4f}")
            print(f"  全程平均alpha: {alpha_arr.mean():.4f}")

    # Bias proxy analysis
    if all_bias and any(any(b) for b in all_bias):
        bias_arr = np.array([b[:min(len(b), 50)] for b in all_bias if len(b) >= 10])
        if len(bias_arr) > 0:
            n_pts = bias_arr.shape[1]
            checkpoints = [0, 4, 9, 19, min(49, n_pts-1)]
            checkpoints = [c for c in checkpoints if c < n_pts]
            print("\n  Bias Proxy (V - G均值) 轨迹:")
            for c in checkpoints:
                vals = bias_arr[:, c]
                print(f"    Rollout {c+1:2d}: bias = {vals.mean():.4f} ± {vals.std():.4f}")

    # c_mc analysis
    if all_c_mc and any(any(c) for c in all_c_mc):
        cmc_arr = np.array([c[:min(len(c), 50)] for c in all_c_mc if len(c) >= 10])
        if len(cmc_arr) > 0:
            n_pts = cmc_arr.shape[1]
            checkpoints = [0, 4, 9, 19, min(49, n_pts-1)]
            checkpoints = [c for c in checkpoints if c < n_pts]
            print("\n  c_mc (MC混合权重) 轨迹:")
            for c in checkpoints:
                vals = cmc_arr[:, c]
                print(f"    Rollout {c+1:2d}: c_mc = {vals.mean():.4f}")

    # Value loss analysis
    if all_value_loss:
        vl_arr = np.array([v[:min(len(v), 50)] for v in all_value_loss if len(v) >= 10])
        if len(vl_arr) > 0:
            print(f"\n  Value Loss:")
            print(f"    初期(前5): {vl_arr[:, :5].mean():.4f}")
            print(f"    中期(10-20): {vl_arr[:, 10:20].mean():.4f}")
            print(f"    末期(最后10): {vl_arr[:, -10:].mean():.4f}")

    # Clip frac analysis
    if all_clip_frac:
        cf_arr = np.array([c[:min(len(c), 50)] for c in all_clip_frac if len(c) >= 10])
        if len(cf_arr) > 0:
            print(f"\n  Clip Fraction:")
            print(f"    初期(前5): {cf_arr[:, :5].mean():.4f}")
            print(f"    末期(最后10): {cf_arr[:, -10:].mean():.4f}")


# === 关键比较：Ant的SCR特性分析 ===
print("\n" + "=" * 80)
print("HCGAE 在 Ant-v4 失效的关键指标对比")
print("=" * 80)

# 读取全部更新数据，计算关键比率
for algo in ['Optimal_PPO', 'Optimal_HCGAE', 'Optimal_HCGAE_v2']:
    seeds = read_seeds(base, algo)
    if not seeds:
        continue

    # 收集v(s) - G的分布特性
    all_bias_vals = []
    all_var_G = []
    all_ev_series = []

    for d in seeds:
        if 'updates' not in d:
            continue
        for u in d['updates']:
            bp = u.get('bias_proxy', None)
            if bp is not None:
                all_bias_vals.append(bp)
            ev = u.get('explained_variance', 0)
            all_ev_series.append(ev)

    print(f"\n{algo}:")
    if all_bias_vals:
        print(f"  Bias proxy (V-G): mean={np.mean(all_bias_vals):.4f}, std={np.std(all_bias_vals):.4f}")
        print(f"  Bias proxy range: [{min(all_bias_vals):.4f}, {max(all_bias_vals):.4f}]")
    if all_ev_series:
        print(f"  EV: final_mean={np.mean(all_ev_series[-len(seeds)*5:]):.4f}")


# === 对比 Ant vs Hopper 的关键统计 ===
print("\n" + "=" * 80)
print("Ant-v4 vs Hopper-v4 环境特性对比 (Optimal_HCGAE_v2)")
print("=" * 80)

hopper_base = 'results/ICMLExperiment/Hopper-v4/'
ant_base = 'results/ICMLExperiment/Ant-v4/'

for env_name, env_base in [('Hopper-v4', hopper_base), ('Ant-v4', ant_base)]:
    seeds = read_seeds(env_base, 'Optimal_HCGAE_v2')
    if not seeds:
        continue

    final_rewards = [d['eval_rewards'][-1] for d in seeds if 'eval_rewards' in d and d['eval_rewards']]

    all_alpha = []
    all_ev = []
    all_c_mc = []
    all_bias = []

    for d in seeds:
        if 'updates' not in d:
            continue
        for u in d['updates']:
            all_alpha.append(u.get('mean_alpha', 0))
            all_ev.append(u.get('explained_variance', 0))
            all_c_mc.append(u.get('c_mc', 0))
            all_bias.append(u.get('bias_proxy', 0))

    print(f"\n{env_name}:")
    print(f"  Final: {np.mean(final_rewards):.1f} ± {np.std(final_rewards):.1f}")
    print(f"  EV (全程均值): {np.mean(all_ev):.4f}")
    print(f"  EV (末期均值): {np.mean(all_ev[-len(seeds)*20:]):.4f}")
    print(f"  Mean Alpha (全程): {np.mean(all_alpha):.4f}")
    print(f"  c_mc (全程均值): {np.mean(all_c_mc):.4f}")
    print(f"  Bias proxy (全程均值): {np.mean(all_bias):.4f}")
    print(f"  Bias proxy (末期均值): {np.mean(all_bias[-len(seeds)*20:]):.4f}")

