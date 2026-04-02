"""分析训练日志中暴露的问题"""
import json, numpy as np, os

results_dir = "results/Hopper-v4-Ablation"
names = ["HCGAE_Base", "HCGAE_Imp12", "HCGAE_Full", "HCGAE_Imp1", "HCGAE_Imp2", "HCGAE_Imp3", "HCGAE_Imp4"]

print("=" * 80)
print("  训练过程中暴露的关键问题分析")
print("=" * 80)

for name in names:
    path = os.path.join(results_dir, f"{name}_metrics.json")
    if not os.path.exists(path):
        continue
    d = json.load(open(path))
    kls = d.get("approx_kls", [])
    clips = d.get("clip_fracs", [])
    evs = d.get("explained_variances", [])
    vls = d.get("value_losses", [])
    pls = d.get("policy_losses", [])
    evals = d.get("eval_rewards", [])

    if not kls:
        continue

    kls_arr = np.array(kls)
    clips_arr = np.array(clips)
    evs_arr = np.array(evs)
    vls_arr = np.array(vls)

    # 检测训练不稳定指标
    high_kl_frac = float(np.mean(kls_arr > 0.02))   # KL > 0.02 视为过大
    high_clip_frac = float(np.mean(clips_arr > 0.3))  # clip>30% 视为过激进
    low_ev_frac = float(np.mean(evs_arr < 0.3))      # EV<0.3 视为 Critic 差
    neg_ev_frac = float(np.mean(evs_arr < 0.0))      # 负 EV 极差

    # 晚期稳定性
    n_last = max(1, len(evals) // 3)
    late_std = float(np.std(evals[-n_last:])) if evals else 0

    print(f"\n  {name}:")
    print(f"    KL: mean={np.mean(kls_arr):.4f}  max={max(kls_arr):.4f}  >0.02比例={high_kl_frac:.1%}")
    print(f"    clip_frac: mean={np.mean(clips_arr):.3f}  max={max(clips_arr):.3f}  >0.3比例={high_clip_frac:.1%}")
    print(f"    EV: mean={np.mean(evs_arr):.3f}  min={min(evs_arr):.3f}  <0比例={neg_ev_frac:.1%}")
    print(f"    value_loss: mean={np.mean(vls_arr):.3f}  max={max(vls_arr):.3f}")
    print(f"    晚期评估奖励: mean={np.mean(evals[-n_last:]):.1f}  std={late_std:.1f}")

print("\n\n  === 关键问题总结 ===")
print("  1. [clip_frac] 当 clip_frac 高时（>0.3），说明 policy 更新幅度超过信任域限制，")
print("     但 KL 散度未被惩罚，可能导致策略崩溃")
print("  2. [EV下降] EV 在训练中期可能出现大幅下跌，Critic 在快速更新后出现 overfitting")
print("  3. [固定 ε=0.2] 信任域在不同训练阶段应该动态调整：早期可宽松，后期应收紧")
print("  4. [重要性采样比率] 多轮 epoch 更新后，ratio 偏离 1.0 过多，但 clip 无法完全阻止")
print("  5. [批内方差] 同一 batch 内的优势方差极大，影响梯度方向的一致性")
print("  6. [on-policy 样本浪费] 当前固定 n_steps×n_epochs 复用，不管样本新鲜度")

