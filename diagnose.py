"""
深度诊断脚本：hook 进训练过程，观察以下数学量的实时分布
  1. TD 残差 δ 的分布（均值、标准差、偏度）
  2. 优势函数 A 的分布
  3. 置信度 c 的分布
  4. 自适应 λ 的分布
  5. V1 vs V2 的相关性（双 Critic）
  6. 训练稳定性：grad_norm、value_loss 方差
运行：python3 diagnose.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import gymnasium as gym
from scipy import stats as scipy_stats

from gae_experiments.utils.rollout_buffer import RolloutBuffer
from gae_experiments.utils.networks import ActorNetwork, CriticNetwork, LambdaNetwork

ENV_NAME = "Acrobot-v1"
N_STEPS  = 2048
HIDDEN   = 64
N_ROLLOUTS = 5   # 观察前 N 个 rollout 的统计量

def make_env():
    return gym.make(ENV_NAME)

def fmt_dist(arr, name, indent=4):
    """格式化打印数组的分布统计"""
    a = np.array(arr)
    sp = " " * indent
    skew = scipy_stats.skew(a) if len(a) > 2 else 0.0
    kurt = scipy_stats.kurtosis(a) if len(a) > 2 else 0.0
    print(f"{sp}{name:<18}: mean={a.mean():+8.4f}  std={a.std():7.4f}  "
          f"min={a.min():+8.4f}  max={a.max():+8.4f}  "
          f"skew={skew:+5.2f}  kurt={kurt:+5.2f}")

def analyze_standard_gae(rollout_idx: int):
    """分析标准 GAE 的 δ 和 A 分布"""
    env = make_env()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device("cpu")

    actor  = ActorNetwork(obs_dim, action_dim, HIDDEN, False)
    critic = CriticNetwork(obs_dim, HIDDEN)
    buffer = RolloutBuffer(N_STEPS, obs_dim, action_dim, device, False)

    # 先训练一段再观察（让网络有基本能力）
    # 这里用随机初始化网络做演示
    buffer.reset()
    obs, _ = env.reset()
    for step in range(N_STEPS):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action, log_prob = actor.get_action_and_logprob(obs_t)
            value = critic(obs_t)
        action_np = action.squeeze(0).cpu().numpy()
        next_obs, reward, terminated, truncated, _ = env.step(int(action_np))
        buffer.add(obs, action_np, reward, float(terminated), log_prob.item(), value.item())
        obs = next_obs if not (terminated or truncated) else env.reset()[0]

    with torch.no_grad():
        last_v = critic(torch.FloatTensor(obs).unsqueeze(0)).item()

    buffer.compute_standard_gae(last_v, gamma=0.99, lam=0.95)

    T = buffer.pos
    nv = buffer._next_values(last_v)
    deltas = buffer.rewards[:T] + 0.99 * nv - buffer.values[:T]

    print(f"\n{'='*70}")
    print(f"  Standard GAE — Rollout #{rollout_idx+1} (T={T})")
    print(f"{'='*70}")
    fmt_dist(buffer.values[:T],  "V(s)")
    fmt_dist(nv,                  "next_V(s')")
    fmt_dist(deltas,              "δ (TD residual)")
    fmt_dist(buffer.advantages,   "A (GAE adv)")
    fmt_dist(buffer.returns,      "returns (targets)")

    # 检验 δ 的自相关性（衡量价值函数的时序偏差）
    autocorr = np.corrcoef(deltas[:-1], deltas[1:])[0, 1]
    print(f"    δ 一阶自相关: {autocorr:+.4f}  (近0=好，说明残差无时序结构)")

    # 检验 A 与 V 的相关性（不应该相关，否则优势被 Critic 污染）
    corr_av = np.corrcoef(buffer.advantages, buffer.values[:T])[0, 1]
    print(f"    corr(A, V)  : {corr_av:+.4f}  (应接近0，否则优势含有价值信息)")

    env.close()
    return deltas, buffer.advantages

def analyze_adaptive_lambda(n_rollouts=3):
    """分析自适应λ分布与δ的关系：λ是否真的在高δ状态降低？"""
    env = make_env()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device("cpu")

    actor      = ActorNetwork(obs_dim, action_dim, HIDDEN, False)
    critic     = CriticNetwork(obs_dim, HIDDEN)
    lambda_net = LambdaNetwork(obs_dim, 32)
    buffer     = RolloutBuffer(N_STEPS, obs_dim, action_dim, device, False)
    lambda_buf = np.zeros(N_STEPS, dtype=np.float32)

    print(f"\n{'='*70}")
    print(f"  Adaptive Lambda GAE — Lambda Network Analysis")
    print(f"{'='*70}")

    all_deltas = []
    all_lambdas = []

    for r in range(n_rollouts):
        buffer.reset()
        obs, _ = env.reset()
        for step in range(N_STEPS):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, log_prob = actor.get_action_and_logprob(obs_t)
                value = critic(obs_t)
                lam_v = lambda_net(obs_t)
            action_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, _ = env.step(int(action_np))
            buffer.add(obs, action_np, reward, float(terminated), log_prob.item(), value.item())
            lambda_buf[step] = lam_v.item()
            obs = next_obs if not (terminated or truncated) else env.reset()[0]

        T = buffer.pos
        with torch.no_grad():
            last_v = critic(torch.FloatTensor(obs).unsqueeze(0)).item()
        nv = buffer._next_values(last_v)
        deltas = buffer.rewards[:T] + 0.99 * nv - buffer.values[:T]
        lambdas = lambda_buf[:T]

        all_deltas.extend(deltas.tolist())
        all_lambdas.extend(lambdas.tolist())

        # 检验 λ 是否与 |δ| 负相关（核心假设）
        corr = np.corrcoef(np.abs(deltas), lambdas)[0, 1]
        fmt_dist(lambdas, f"  λ (rollout {r+1})")
        print(f"      corr(|δ|, λ) = {corr:+.4f}  "
              f"(期望<0: |δ|大→λ小; 当前网络随机初始化，相关性应接近0)")

    print(f"\n  综合分析 ({n_rollouts} rollouts):")
    corr_total = np.corrcoef(np.abs(all_deltas), all_lambdas)[0, 1]
    print(f"    corr(|δ|_all, λ_all) = {corr_total:+.4f}")
    fmt_dist(all_lambdas, "  λ (total)")

    env.close()

def analyze_confidence(n_rollouts=3):
    """分析置信度的分布，以及置信度低的状态是否真的 |δ| 大"""
    env = make_env()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device("cpu")

    actor  = ActorNetwork(obs_dim, action_dim, HIDDEN, False)
    critic = CriticNetwork(obs_dim, HIDDEN)
    buffer = RolloutBuffer(N_STEPS, obs_dim, action_dim, device, False)

    # Welford stats
    wn, wmean, wM2 = 0, 0.0, 1.0

    print(f"\n{'='*70}")
    print(f"  Confidence-Weighted GAE — Confidence Distribution Analysis")
    print(f"{'='*70}")

    for r in range(n_rollouts):
        buffer.reset()
        obs, _ = env.reset()
        for step in range(N_STEPS):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, lp = actor.get_action_and_logprob(obs_t)
                value = critic(obs_t)
            action_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, _ = env.step(int(action_np))
            buffer.add(obs, action_np, reward, float(terminated), lp.item(), value.item())
            obs = next_obs if not (terminated or truncated) else env.reset()[0]

        T = buffer.pos
        with torch.no_grad():
            last_v = critic(torch.FloatTensor(obs).unsqueeze(0)).item()
        nv = buffer._next_values(last_v)
        deltas = buffer.rewards[:T] + 0.99 * nv - buffer.values[:T]

        # Welford update
        for x in deltas:
            wn += 1
            d1 = x - wmean; wmean += d1 / wn
            d2 = x - wmean; wM2 += d1 * d2
        run_var = max(wM2 / max(wn - 1, 1), 1e-8)
        confidence = 1.0 / (1.0 + 1.0 * deltas**2 / run_var)

        fmt_dist(confidence, f"  c_t (rollout {r+1})")
        # 分位数
        p25, p50, p75 = np.percentile(confidence, [25, 50, 75])
        print(f"      Q25={p25:.4f}  Q50={p50:.4f}  Q75={p75:.4f}  "
              f"run_var={run_var:.4f}")

        # 验证：低置信度区域的 |δ| 是否确实更大
        low_c   = deltas[confidence < np.median(confidence)]
        high_c  = deltas[confidence >= np.median(confidence)]
        print(f"      |δ| mean: low_c={np.abs(low_c).mean():.4f}  "
              f"high_c={np.abs(high_c).mean():.4f}  "
              f"(期望 low_c > high_c)")

    env.close()

def analyze_v1v2_correlation():
    """分析双 Critic 的 V1/V2 相关性与差异"""
    env = make_env()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device("cpu")

    actor   = ActorNetwork(obs_dim, action_dim, HIDDEN, False)
    critic1 = CriticNetwork(obs_dim, HIDDEN)
    critic2 = CriticNetwork(obs_dim, HIDDEN)
    buffer  = RolloutBuffer(N_STEPS, obs_dim, action_dim, device, False)
    v2_buf  = np.zeros(N_STEPS, dtype=np.float32)

    buffer.reset()
    obs, _ = env.reset()
    for step in range(N_STEPS):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action, lp = actor.get_action_and_logprob(obs_t)
            v1 = critic1(obs_t); v2 = critic2(obs_t)
        action_np = action.squeeze(0).cpu().numpy()
        next_obs, reward, terminated, truncated, _ = env.step(int(action_np))
        buffer.add(obs, action_np, reward, float(terminated), lp.item(), v1.item())
        v2_buf[step] = v2.item()
        obs = next_obs if not (terminated or truncated) else env.reset()[0]

    T = buffer.pos
    V1 = buffer.values[:T]
    V2 = v2_buf[:T]
    diff = V1 - V2
    min_v = np.minimum(V1, V2)
    mean_v = (V1 + V2) / 2.0

    print(f"\n{'='*70}")
    print(f"  Double Critic — V1/V2 Correlation Analysis")
    print(f"{'='*70}")
    fmt_dist(V1,    "V1(s)")
    fmt_dist(V2,    "V2(s)")
    fmt_dist(diff,  "V1 - V2")
    fmt_dist(min_v, "min(V1,V2)")

    corr_v1v2 = np.corrcoef(V1, V2)[0, 1]
    print(f"    corr(V1, V2) = {corr_v1v2:+.4f}  "
          f"(随机初始化时低，训练后应升高但不应=1)")

    # 保守偏差：min vs mean
    bias = min_v - mean_v
    fmt_dist(bias, "min-mean bias")
    print(f"    保守偏差 mean={bias.mean():+.6f}  "
          f"(应<0: min < mean，起到下界保守作用)")

    env.close()

def print_training_instability_diagnosis():
    """
    根据已有实验数据诊断训练不稳定性
    关注：Double_Critic 的 VLoss=58 问题
    """
    import json
    print(f"\n{'='*70}")
    print(f"  Training Instability Diagnosis from Saved Results")
    print(f"{'='*70}")

    for agent in ['Standard_GAE', 'Double_Critic_GAE', 'Adaptive_Lambda_GAE',
                  'Confidence_Weighted_GAE', 'Combined_GAE']:
        path = f"results/Acrobot-v1/{agent}_metrics.json"
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        vls  = d['value_losses']
        kls  = d['approx_kls']
        evs  = d['explained_variances']
        evals = d['eval_rewards']

        # 稳定性指标：VLoss 的变异系数 (CV)
        vl_cv = np.std(vls) / (np.mean(vls) + 1e-8)
        # KL 超限率（KL > 0.02 视为策略更新过大）
        kl_exceed = np.mean(np.array(kls) > 0.02)
        # EV 最低点（捕获灾难性遗忘）
        ev_min = min(evs)
        # 评估奖励的波动
        eval_cv = np.std(evals) / (np.abs(np.mean(evals)) + 1e-8)

        print(f"\n  {agent}:")
        print(f"    VLoss: mean={np.mean(vls):.2f}  std={np.std(vls):.2f}  CV={vl_cv:.2f}")
        print(f"    KL:    mean={np.mean(kls):.5f}  max={max(kls):.5f}  exceed_rate={kl_exceed:.2%}")
        print(f"    EV:    mean={np.mean(evs):.3f}  min={ev_min:.3f}  final={evs[-1]:.3f}")
        print(f"    Eval:  mean={np.mean(evals):.1f}  std={np.std(evals):.1f}  CV={eval_cv:.2f}")

        # 诊断
        if vl_cv > 1.5:
            print(f"    ⚠️  VLoss 变异系数={vl_cv:.2f} 过高，Critic 训练不稳定")
        if kl_exceed > 0.05:
            print(f"    ⚠️  KL 超限率={kl_exceed:.2%}，策略更新过于激进")
        if ev_min < 0.5:
            print(f"    ⚠️  EV 最低={ev_min:.3f}，出现价值函数崩溃")
        if eval_cv > 0.5:
            print(f"    ⚠️  评估奖励变异系数={eval_cv:.2f}，训练不稳定/策略退化")


if __name__ == "__main__":
    print("\n" + "★"*70)
    print("  GAE 改进方案深度诊断报告")
    print("★"*70)

    # 1. 标准 GAE 的数学量分析
    analyze_standard_gae(0)

    # 2. 自适应 λ 分析
    analyze_adaptive_lambda(n_rollouts=3)

    # 3. 置信度分布分析
    analyze_confidence(n_rollouts=3)

    # 4. 双 Critic 相关性
    analyze_v1v2_correlation()

    # 5. 从已有结果诊断训练稳定性
    print_training_instability_diagnosis()

    print("\n" + "★"*70)
    print("  诊断完成")
    print("★"*70 + "\n")

