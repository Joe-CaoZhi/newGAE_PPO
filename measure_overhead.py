#!/usr/bin/env python3
"""
测量 HCGAE 和 DCPPO-S 相对于标准 PPO 的计算开销。
结果用于 paper_draft 中的 Computational Overhead 章节。
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gymnasium as gym


def make_env(env_id="Hopper-v4", seed=42):
    env = gym.make(env_id)
    env.reset(seed=seed)
    return env

def measure_gae_time(agent_cls, kwargs, n_steps=2048, n_repeats=20, label=""):
    """测量 compute_gae() 的平均耗时"""
    env = make_env()
    agent = agent_cls(env=env, n_steps=n_steps, **kwargs)
    # 预热
    for _ in range(3):
        lv = agent.collect_rollout()
        agent.compute_gae(lv)
    # 正式测量
    times = []
    for _ in range(n_repeats):
        lv = agent.collect_rollout()
        t0 = time.perf_counter()
        agent.compute_gae(lv)
        times.append(time.perf_counter() - t0)
    env.close()
    mean_ms = np.mean(times) * 1000
    std_ms  = np.std(times) * 1000
    print(f"  {label:<30}: GAE  {mean_ms:7.3f} ± {std_ms:.3f} ms")
    return mean_ms, std_ms

def measure_update_time(agent_cls, kwargs, n_steps=2048, n_repeats=20, label=""):
    """测量 update() 的平均耗时"""
    env = make_env()
    agent = agent_cls(env=env, n_steps=n_steps, **kwargs)
    # 预热
    for _ in range(3):
        lv = agent.collect_rollout()
        agent.compute_gae(lv)
        agent.update()
    # 正式测量
    times = []
    for _ in range(n_repeats):
        lv = agent.collect_rollout()
        agent.compute_gae(lv)
        t0 = time.perf_counter()
        agent.update()
        times.append(time.perf_counter() - t0)
    env.close()
    mean_ms = np.mean(times) * 1000
    std_ms  = np.std(times) * 1000
    print(f"  {label:<30}: Update {mean_ms:7.3f} ± {std_ms:.3f} ms")
    return mean_ms, std_ms

def main():
    print("=" * 62)
    print("  计算开销测量 (Hopper-v4, n_steps=2048, n_repeats=20)")
    print("=" * 62)

    from gae_experiments.agents.base_ppo import BasePPO
    from gae_experiments.agents.hindsight_ppo import HindsightPPO
    from gae_experiments.agents.dcppo import DCPPO

    base_kwargs   = dict(hidden_dim=64, n_epochs=10, batch_size=64)
    hcgae_kwargs  = dict(hidden_dim=64, n_epochs=10, batch_size=64)
    dcppo_s_kwargs = dict(hidden_dim=64, n_epochs=10, batch_size=64,
                          use_imp_g=False, use_imp_a=False, use_imp_s=True,
                          use_hcgae=True, name="DCPPO_ImpS")

    print("\n  GAE 计算时间:")
    g_base, _   = measure_gae_time(BasePPO,      base_kwargs,   label="Standard GAE (Base)")
    g_hcgae, _  = measure_gae_time(HindsightPPO, hcgae_kwargs,  label="HCGAE_Imp12")
    g_dcppo, _  = measure_gae_time(DCPPO,        dcppo_s_kwargs, label="DCPPO-S (HCGAE+ImpS)")

    print("\n  Update 时间:")
    u_base, _   = measure_update_time(BasePPO,      base_kwargs,   label="Standard GAE (Base)")
    u_hcgae, _  = measure_update_time(HindsightPPO, hcgae_kwargs,  label="HCGAE_Imp12")
    u_dcppo, _  = measure_update_time(DCPPO,        dcppo_s_kwargs, label="DCPPO-S (HCGAE+ImpS)")

    print("\n  相对开销 (相对 Standard GAE Base):")
    print(f"  {'方法':<30} GAE overhead   Update overhead")
    print(f"  {'-'*60}")
    print(f"  {'Standard GAE (Base)':<30} 1.00×          1.00×")
    print(f"  {'HCGAE_Imp12':<30} {g_hcgae/g_base:.2f}×          {u_hcgae/u_base:.2f}×")
    print(f"  {'DCPPO-S':<30} {g_dcppo/g_base:.2f}×          {u_dcppo/u_base:.2f}×")

    # 保存结果
    import json
    result = {
        "gae_ms":    {"Standard_GAE": g_base,  "HCGAE_Imp12": g_hcgae, "DCPPO_S": g_dcppo},
        "update_ms": {"Standard_GAE": u_base,  "HCGAE_Imp12": u_hcgae, "DCPPO_S": u_dcppo},
        "gae_ratio":    {"HCGAE_Imp12": g_hcgae/g_base, "DCPPO_S": g_dcppo/g_base},
        "update_ratio": {"HCGAE_Imp12": u_hcgae/u_base, "DCPPO_S": u_dcppo/u_base},
    }
    os.makedirs("results", exist_ok=True)
    with open("results/overhead_measurement.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  结果保存至 results/overhead_measurement.json")
    return result

if __name__ == "__main__":
    main()

