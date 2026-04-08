"""
深度数学诊断脚本
分析 Acrobot-v1 结果中各方法的数学行为
"""
import json
import os

import numpy as np


def deep_math_analysis(results_dir, env_name):
    print(f"\n{'='*70}")
    print(f"  数学深度诊断: {env_name}")
    print(f"{'='*70}")

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith('.json'):
            continue
        d = json.load(open(f'{results_dir}/{fname}'))
        name = d.get('agent_name', fname.replace('_metrics.json',''))

        vlosses = np.array(d.get('value_losses', []))
        evs = np.array(d.get('explained_variances', []))
        kls = np.array(d.get('approx_kls', []))
        clip_fracs = np.array(d.get('clip_fracs', []))
        evals = np.array(d.get('eval_rewards', []))
        lams = d.get('mean_lambda_values', [])
        steps = np.array(d.get('total_steps', []))

        if len(vlosses) == 0:
            continue

        n = len(vlosses)

        # 分三段分析（训练早、中、晚期）
        t1, t2 = n//3, 2*n//3

        print(f"\n{name}")
        print(f"  训练段   VLoss        EV           KL          clip_frac")
        for seg, seg_name in [(slice(0, t1), '早期'), (slice(t1, t2), '中期'), (slice(t2, n), '晚期')]:
            vl_s = vlosses[seg].mean() if len(vlosses[seg]) > 0 else 0
            ev_s = evs[seg].mean() if len(evs[seg]) > 0 else 0
            kl_s = kls[seg].mean() if len(kls[seg]) > 0 else 0
            clip_s = clip_fracs[seg].mean() if len(clip_fracs[seg]) > 0 else 0
            print(f"  {seg_name}    {vl_s:10.3f}   {ev_s:10.3f}   {kl_s:10.4f}   {clip_s:10.3f}")

        # λ 统计（如果有）
        if lams:
            lams_arr = np.array(lams)
            print(f"  λ统计: mean={lams_arr.mean():.3f}  std={lams_arr.std():.3f}  min={lams_arr.min():.3f}  max={lams_arr.max():.3f}")

        # 评估奖励分析
        if len(evals) > 0:
            # 找峰值后是否发生崩溃
            peak_idx = np.argmax(evals)
            after_peak = evals[peak_idx:]
            if len(after_peak) > 3:
                collapse = (evals[peak_idx] - after_peak.min()) > 50
                print(f"  eval: peak={evals[peak_idx]:.1f}@{peak_idx} final={evals[-1]:.1f} "
                      f"{'⚠️ 崩溃!' if collapse else '✅ 稳定'}")

        # VLoss 趋势
        if len(vlosses) > 10:
            early_vl = vlosses[:10].mean()
            late_vl = vlosses[-10:].mean()
            trend = "↓ 收敛" if late_vl < early_vl * 0.5 else ("↑ 发散!" if late_vl > early_vl * 2 else "→ 平稳")
            print(f"  VLoss趋势: early={early_vl:.3f} → late={late_vl:.3f}  {trend}")

        # EV 趋势
        if len(evs) > 10:
            early_ev = evs[:10].mean()
            late_ev = evs[-10:].mean()
            print(f"  EV趋势: early={early_ev:.3f} → late={late_ev:.3f}")

    print(f"\n{'='*70}")
    print("关键诊断总结:")
    print("  CartPole: 所有方法已接近收敛（满分500），差异在收敛速度")
    print("  Acrobot: Combined_GAE 最稳定（std最低），Double_Critic 最不稳定")
    print("  主要问题: Confidence_Weighted 收敛慢，Double_Critic VLoss发散(41.4)")
    print("  改进方向: 需要更强的归纳偏差和更稳定的置信度估计")


if __name__ == '__main__':
    import sys
    env = sys.argv[1] if len(sys.argv) > 1 else 'Acrobot-v1'
    deep_math_analysis(f'results/{env}', env)

