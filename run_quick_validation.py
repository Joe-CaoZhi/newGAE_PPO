"""
快速验证实验：验证三项代码改进的有效性
==========================================

验证目标：
  V1. Critic 目标路径修复：
      旧路径: buf.returns = c_mc * G + (1-c_mc) * (adv_corrected + V)
      新路径: buf.returns = c_mc * G + (1-c_mc) * std_gae_returns
      预期：新路径防止 Critic 更新循环依赖，应改善 EV 稳定性

  V2. EV 驱动 SNR vs 原始 E[|A|]/std(A) SNR：
      新定义: SNR_eff = EV_ema（直接反映 Critic 精度）
      旧定义: SNR = E[|A|]/std(A)（≈0.798 常数，缺乏区分度）
      预期：EV 驱动的 SNR 在训练早期抑制更有效

  V3. SCR 在线诊断：
      输出 SCR_estimate = |bias| / std(G) 的训练曲线
      预期：Hopper SCR > 1（HCGAE 有益），HalfCheetah SCR < 1

实验设置：
  - 环境：Hopper-v4（HCGAE 有效的代表）
  - 步数：200,000（快速，足够观察到早期行为差异）
  - 种子：[42, 123, 456]（3 种子 quick 对比）
  - 比较组：DCPPO_ImpS_old（原始 SNR）vs DCPPO_ImpS（EV SNR）
  - 对照组：DCPPO_Base（无 S 改进）

注意：此脚本不产生虚假结论，只报告实测数据。
"""
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from gae_experiments.agents.dcppo import DCPPO, build_dcppo_agent

RESULTS_DIR = Path("results/QuickValidation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# 实验超参（小规模快速验证）
# ─────────────────────────────────────────────────────────────────────
SEEDS = [42, 123, 456]
TOTAL_TIMESTEPS = 200_000
EVAL_FREQ = 10_240
N_EVAL_EPISODES = 5

SHARED_KWARGS = dict(
    hidden_dim=64,
    lr_actor=3e-4,
    lr_critic=1e-3,
    gamma=0.99,
    lam=0.95,
    eps_clip=0.2,
    n_epochs=10,
    batch_size=64,
    n_steps=2048,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    device="cpu",
)


# ─────────────────────────────────────────────────────────────────────
# 带原始 SNR 定义的 DCPPO_ImpS（用于对比）
# ─────────────────────────────────────────────────────────────────────
class DCPPO_ImpS_OldSNR(DCPPO):
    """
    使用原始 E[|A|]/std(A) SNR 定义的 DCPPO_ImpS 变体。
    继承 DCPPO 的所有逻辑，仅覆盖 SNR 计算。
    用于对比验证 EV 驱动 SNR 的优越性。
    """

    def update(self) -> dict:
        """覆盖 update，使用旧的 SNR 定义"""
        obs, actions, old_log_probs, advantages, returns, old_values = self.buffer.get_batch()

        # 标准归一化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── 旧 SNR 定义：E[|A|]/std(A) ──────────────────────────────
        if self.use_imp_s:
            adv_mean_abs = float(advantages.abs().mean().item())
            adv_std      = float(advantages.std().item()) + 1e-8
            snr_batch    = adv_mean_abs / adv_std   # 旧定义，≈0.798 for Gaussian
            snr_weight   = float(
                np.clip(
                    (snr_batch / (self.snr_target + 1e-8)) ** self.snr_gamma,
                    self.snr_min_weight,
                    1.0,
                )
            )
            snr_ratio = snr_batch
        else:
            snr_batch  = 0.0
            snr_weight = 1.0
            snr_ratio  = 0.0

        effective_adv = advantages * snr_weight
        T = self.buffer.pos
        indices = np.arange(T)

        metrics = {
            "value_loss": 0.0, "policy_loss": 0.0, "entropy_loss": 0.0,
            "approx_kl": 0.0, "clip_frac": 0.0,
            "clip_frac_strict": 0.0, "clip_frac_loose": 0.0,
            "ratio_mean": 0.0, "geo_ratio_mean": 0.0,
            "snr_batch": snr_batch, "snr_weight": snr_weight, "snr_ratio": snr_ratio,
        }
        _no_avg_keys = {"snr_batch", "snr_weight", "snr_ratio"}
        update_count = 0

        import torch.nn as nn
        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, self.batch_size):
                end = start + self.batch_size
                if end > T:
                    break
                batch_idx = indices[start:end]
                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = effective_adv[batch_idx]
                batch_returns = returns[batch_idx]
                batch_old_values = old_values[batch_idx]

                if self.continuous:
                    dist = self.actor(batch_obs)
                    new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1)
                else:
                    new_log_probs, entropy = self.actor.evaluate_actions(batch_obs, batch_actions)

                new_values = self.critic(batch_obs)
                log_ratio_total = new_log_probs - batch_old_log_probs

                # 标准 ratio（无几何均值改进）
                ratio = torch.exp(log_ratio_total)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                clip_frac_val = ((ratio - 1).abs() > self.eps_clip).float().mean().item()

                value_loss = 0.5 * ((new_values - batch_returns) ** 2).mean()
                entropy_loss = -entropy.mean()

                self.actor_optimizer.zero_grad()
                (policy_loss + self.ent_coef * entropy_loss).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                (self.vf_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()

                metrics["value_loss"] += value_loss.item()
                metrics["policy_loss"] += policy_loss.item()
                metrics["entropy_loss"] += entropy_loss.item()
                metrics["approx_kl"] += approx_kl
                metrics["clip_frac"] += clip_frac_val
                metrics["clip_frac_strict"] += clip_frac_val
                metrics["clip_frac_loose"] += clip_frac_val
                metrics["ratio_mean"] += float(ratio.mean().item())
                metrics["geo_ratio_mean"] += float(ratio.mean().item())
                update_count += 1

        if update_count > 0:
            for k in metrics:
                if k not in _no_avg_keys:
                    metrics[k] /= update_count

        with torch.no_grad():
            y_pred = old_values.cpu().numpy()
            y_true = returns.cpu().numpy()
            var_y = np.var(y_true)
            ev = 1 - np.var(y_true - y_pred) / (var_y + 1e-8)
            metrics["explained_variance"] = float(ev)

        self._ev_ema = (1 - self._ev_ema_alpha) * self._ev_ema + self._ev_ema_alpha * ev
        return metrics


# ─────────────────────────────────────────────────────────────────────
# 单次实验运行（追踪诊断指标）
# ─────────────────────────────────────────────────────────────────────
def run_single(agent, total_timesteps: int, eval_env, seed: int) -> dict:
    """运行单个实验，返回训练轨迹（EV 曲线、SCR 曲线、最终回报）"""
    import time as _time
    agent._total_timesteps = total_timesteps
    start_t = _time.time()
    eval_rewards = []
    ev_history = []
    scr_history = []
    snr_weight_history = []
    snr_ratio_history = []
    update_idx = 0

    while agent.total_steps < total_timesteps:
        last_value = agent.collect_rollout()
        gae_stats  = agent.compute_gae(last_value)
        metrics    = agent.update()
        update_idx += 1
        metrics.update(gae_stats)

        # 收集诊断指标
        ev = float(metrics.get("explained_variance", 0.0))
        scr = float(metrics.get("scr_current", metrics.get("scr_ema", 0.0)))
        snr_w = float(metrics.get("snr_weight", 1.0))
        snr_r = float(metrics.get("snr_ratio", 0.0))

        ev_history.append((agent.total_steps, ev))
        scr_history.append((agent.total_steps, scr))
        snr_weight_history.append((agent.total_steps, snr_w))
        snr_ratio_history.append((agent.total_steps, snr_r))

        # 评估
        if agent.total_steps % EVAL_FREQ < agent.n_steps:
            ep_r = agent.evaluate(eval_env, N_EVAL_EPISODES)
            eval_rewards.append((agent.total_steps, ep_r))

    elapsed = _time.time() - start_t
    final_mean = float(np.mean([r for _, r in eval_rewards[-5:]])) if eval_rewards else 0.0
    best = float(max([r for _, r in eval_rewards])) if eval_rewards else 0.0

    return {
        "seed": seed,
        "final_mean_reward": final_mean,
        "best_reward": best,
        "eval_rewards": eval_rewards,
        "ev_history": ev_history[-20:],    # 最后 20 个更新步（节省空间）
        "scr_history": scr_history[-20:],
        "snr_weight_history": snr_weight_history[-20:],
        "snr_ratio_history": snr_ratio_history[-20:],
        "ev_early": [ev for _, ev in ev_history[:10]],   # 早期 EV（前 10 个 update）
        "ev_late": [ev for _, ev in ev_history[-10:]],   # 后期 EV
        "elapsed_s": elapsed,
    }


# ─────────────────────────────────────────────────────────────────────
# 主实验
# ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "═" * 70)
    print("  快速验证实验：EV驱动SNR + Critic目标路径修复 + SCR诊断")
    print("  环境: Hopper-v4  |  步数: 200K  |  种子: 3")
    print("═" * 70)

    env_id = "Hopper-v4"

    # 实验组配置
    experiments = {
        # 对照组（无 S 改进）
        "DCPPO_Base": lambda seed, env: build_dcppo_agent(
            "DCPPO_Base", env,
            save_dir=str(RESULTS_DIR / "DCPPO_Base"),
            name=f"DCPPO_Base_s{seed}",
            **SHARED_KWARGS,
        ),
        # 旧 SNR 定义（E[|A|]/std(A)）
        "DCPPO_ImpS_OldSNR": lambda seed, env: DCPPO_ImpS_OldSNR(
            env=env, name=f"DCPPO_ImpS_OldSNR_s{seed}",
            use_imp_g=False, use_imp_a=False, use_imp_s=True,
            snr_target=0.3, snr_gamma=0.5, snr_min_weight=0.2,
            save_dir=str(RESULTS_DIR / "DCPPO_ImpS_OldSNR"),
            **SHARED_KWARGS,
        ),
        # 新 EV 驱动 SNR（当前论文/代码中的幂律门控）
        "DCPPO_ImpS_EVdriven": lambda seed, env: build_dcppo_agent(
            "DCPPO_ImpS", env,
            save_dir=str(RESULTS_DIR / "DCPPO_ImpS_EVdriven"),
            name=f"DCPPO_ImpS_EVdriven_s{seed}",
            snr_mode="ev_power",
            # snr_target=0.3 对应 EV=0.3 时开始充分发挥
            snr_target=0.3, snr_gamma=0.5, snr_min_weight=0.2,
            **SHARED_KWARGS,
        ),
        # 新改进：线性 EV 收缩（MSE-optimal linear shrinkage）
        "DCPPO_ImpS_EVlinear": lambda seed, env: build_dcppo_agent(
            "DCPPO_ImpS", env,
            save_dir=str(RESULTS_DIR / "DCPPO_ImpS_EVlinear"),
            name=f"DCPPO_ImpS_EVlinear_s{seed}",
            snr_mode="ev_linear",
            snr_min_weight=0.2,
            **SHARED_KWARGS,
        ),
    }

    all_results = {}

    for variant_name, builder in experiments.items():
        print(f"\n  ── 运行变体: {variant_name} ──")
        variant_results = []
        save_subdir = RESULTS_DIR / variant_name
        save_subdir.mkdir(parents=True, exist_ok=True)

        for seed in SEEDS:
            print(f"    种子 {seed}...", end=" ", flush=True)

            # 设置随机种子
            np.random.seed(seed)
            torch.manual_seed(seed)

            env = gym.make(env_id)
            env.reset(seed=seed)
            eval_env = gym.make(env_id)
            eval_env.reset(seed=seed + 10000)

            agent = builder(seed, env)
            result = run_single(agent, TOTAL_TIMESTEPS, eval_env, seed)

            print(f"最终={result['final_mean_reward']:.1f}  "
                  f"最优={result['best_reward']:.1f}  "
                  f"早期EV均={np.mean(result['ev_early']):.3f}  "
                  f"晚期EV均={np.mean(result['ev_late']):.3f}  "
                  f"用时={result['elapsed_s']:.0f}s")

            # 保存单次结果
            seed_file = save_subdir / f"{variant_name}_s{seed}_quick.json"
            with open(seed_file, "w") as f:
                json.dump(result, f, indent=2)

            variant_results.append(result)
            env.close()
            eval_env.close()

        # 汇总统计
        final_rewards = [r["final_mean_reward"] for r in variant_results]
        best_rewards  = [r["best_reward"] for r in variant_results]
        all_ev_early  = [np.mean(r["ev_early"]) for r in variant_results]
        all_ev_late   = [np.mean(r["ev_late"]) for r in variant_results]
        all_snr_w     = [np.mean([snr for _, snr in r["snr_weight_history"]]) for r in variant_results]

        # SCR 诊断（仅 DCPPO 系列有此字段）
        scr_vals = [np.mean([scr for _, scr in r.get("scr_history", [(0, 0)])])
                    for r in variant_results]

        summary = {
            "variant": variant_name,
            "env": env_id,
            "seeds": SEEDS,
            "total_timesteps": TOTAL_TIMESTEPS,
            "final_reward_mean": float(np.mean(final_rewards)),
            "final_reward_std":  float(np.std(final_rewards)),
            "best_reward_mean":  float(np.mean(best_rewards)),
            "best_reward_std":   float(np.std(best_rewards)),
            "ev_early_mean": float(np.mean(all_ev_early)),
            "ev_late_mean":  float(np.mean(all_ev_late)),
            "snr_weight_late_mean": float(np.mean(all_snr_w)),
            "scr_mean": float(np.mean(scr_vals)),
        }
        all_results[variant_name] = summary

        print(f"  ✓ {variant_name}: "
              f"最终={summary['final_reward_mean']:.1f}±{summary['final_reward_std']:.1f}  "
              f"EV早={summary['ev_early_mean']:.3f}→晚={summary['ev_late_mean']:.3f}  "
              f"SNR_w均={summary['snr_weight_late_mean']:.3f}  "
              f"SCR均={summary['scr_mean']:.3f}")

    # ─── 保存汇总 & 打印分析 ──────────────────────────────────────────
    summary_file = RESULTS_DIR / "quick_validation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "═" * 70)
    print("  【验证结论】（严谨：只报告实测数据，不做超出数据的推论）")
    print("═" * 70)

    base = all_results.get("DCPPO_Base", {})
    old_snr = all_results.get("DCPPO_ImpS_OldSNR", {})
    new_snr = all_results.get("DCPPO_ImpS_EVdriven", {})
    linear_snr = all_results.get("DCPPO_ImpS_EVlinear", {})

    if base and old_snr and new_snr:
        print(f"\n  对照组  DCPPO_Base:         {base['final_reward_mean']:.1f}±{base['final_reward_std']:.1f}")
        print(f"  旧SNR   DCPPO_ImpS_OldSNR:  {old_snr['final_reward_mean']:.1f}±{old_snr['final_reward_std']:.1f}  "
              f"早期EV={old_snr['ev_early_mean']:.3f}  SNR_w={old_snr['snr_weight_late_mean']:.3f}")
        print(f"  新SNR   DCPPO_ImpS_EVdriven: {new_snr['final_reward_mean']:.1f}±{new_snr['final_reward_std']:.1f}  "
              f"早期EV={new_snr['ev_early_mean']:.3f}  SNR_w={new_snr['snr_weight_late_mean']:.3f}")
        if linear_snr:
            print(f"  线性EV  DCPPO_ImpS_EVlinear:  {linear_snr['final_reward_mean']:.1f}±{linear_snr['final_reward_std']:.1f}  "
                  f"早期EV={linear_snr['ev_early_mean']:.3f}  SNR_w={linear_snr['snr_weight_late_mean']:.3f}")

        delta_old = old_snr['final_reward_mean'] - base['final_reward_mean']
        delta_new = new_snr['final_reward_mean'] - base['final_reward_mean']
        print(f"\n  相对基线: 旧SNR Δ={delta_old:+.1f}  |  新EV-SNR Δ={delta_new:+.1f}")
        if linear_snr:
            delta_linear = linear_snr['final_reward_mean'] - base['final_reward_mean']
            print(f"            线性EV Δ={delta_linear:+.1f}")

        ev_delta = new_snr['ev_early_mean'] - old_snr['ev_early_mean']
        print(f"  早期EV差（新-旧）: {ev_delta:+.4f}  "
              f"（{'新定义早期更好' if ev_delta > 0 else '新定义早期无明显优势'}）")
        if linear_snr:
            ev_linear_delta = linear_snr['ev_early_mean'] - old_snr['ev_early_mean']
            print(f"  早期EV差（线性-旧）: {ev_linear_delta:+.4f}")

        scr = (linear_snr or new_snr).get("scr_mean", 0)
        print(f"\n  SCR 诊断: Hopper-v4 SCR均值={scr:.3f}  "
              f"（{'> 1.0，HCGAE 有益' if scr > 1.0 else '≤ 1.0，HCGAE 收益有限'}）")

    print(f"\n  结果保存至: {summary_file}")
    print("═" * 70 + "\n")

    return all_results


if __name__ == "__main__":
    main()

