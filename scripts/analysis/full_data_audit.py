#!/usr/bin/env python3
"""
全面的论文数据溯源验证脚本
验证所有表格数据与实际 JSON 文件的一致性
"""
import json
import os

import numpy as np
from scipy import stats

BASE = "/Users/joe-caozhi/newGAE_ppo/results"

def load_json(path):
    with open(path) as f:
        return json.load(f)

def get_final_reward(filepath):
    """从 metrics JSON 获取最终奖励"""
    data = load_json(filepath)
    if "final_reward" in data:
        return data["final_reward"]
    if "all_eval_rewards" in data:
        rewards = data["all_eval_rewards"]
        return float(np.mean(rewards[-5:]))
    if "eval_rewards" in data:
        rewards = data["eval_rewards"]
        return float(np.mean(rewards[-5:]))
    return None

def compute_stats(values):
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1) if len(arr) > 1 else 0),
        "n": len(arr)
    }

def mann_whitney(a, b):
    u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    d = (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0
    pct = (np.mean(a) - np.mean(b)) / np.mean(b) * 100
    return {"u": float(u), "p": float(p), "d": float(d), "pct": float(pct)}

def label_d(d):
    d = abs(d)
    if d < 0.2: return "negligible"
    elif d < 0.5: return "small"
    elif d < 0.8: return "medium"
    else: return "large"

print("=" * 80)
print("论文数据溯源验证报告")
print("=" * 80)

issues = []

# ========================
# TABLE 1: ICMLExperiment (5 seeds x 500K steps)
# ========================
print("\n### TABLE 1 验证: ICMLExperiment (5 seeds × 500K steps)")
print("Source: results/ICMLExperiment/")

paper_table1 = {
    "Hopper-v4": {
        "Standard_PPO":      (1804, 69),
        "Optimal_PPO":       (1598, 149),
        "Optimal_HCGAE":     (1752, 81),
        "Optimal_HCGAE_SCR": (1366, 133),
    },
    "Walker2d-v4": {
        "Standard_PPO":      (1425, 223),
        "Optimal_PPO":       (1596, 417),
        "Optimal_HCGAE":     (1872, 547),
        "Optimal_HCGAE_SCR": (1896, 682),
    },
    "HalfCheetah-v4": {
        "Standard_PPO":      (1051, 134),
        "Optimal_PPO":       (1487, 61),
        "Optimal_HCGAE":     (1250, 53),
        "Optimal_HCGAE_SCR": (1254, 78),
    }
}

# Load summary json
summary_path = os.path.join(BASE, "ICMLExperiment", "icml_stats_report.json")
icml_summary = load_json(summary_path)["summary"]

for env in ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]:
    print(f"\n  {env}:")
    for alg in ["Standard_PPO", "Optimal_PPO", "Optimal_HCGAE", "Optimal_HCGAE_SCR"]:
        actual_mean = icml_summary[env][alg]["mean"]
        actual_std = icml_summary[env][alg]["std"]
        paper_mean, paper_std = paper_table1[env][alg]

        mean_diff = abs(actual_mean - paper_mean)
        std_diff = abs(actual_std - paper_std)
        status = "✅" if mean_diff < 1 and std_diff < 2 else "⚠️"
        if status == "⚠️":
            issues.append(f"Table1 {env}/{alg}: paper={paper_mean}±{paper_std}, actual={actual_mean:.0f}±{actual_std:.0f}")
        print(f"    {alg}: paper={paper_mean}±{paper_std}, actual={actual_mean:.0f}±{actual_std:.0f} {status}")

# ========================
# TABLE 1 统计对比验证
# ========================
print("\n### TABLE 1 统计对比验证 (Mann-Whitney)")

paper_stats = {
    ("Hopper-v4", "HCGAE vs Optimal PPO"): (9.6, 0.222, +1.28),
    ("Walker2d-v4", "HCGAE vs Optimal PPO"): (17.3, 0.841, +0.57),
    ("HalfCheetah-v4", "HCGAE vs Optimal PPO"): (-16.0, 0.008, -4.14),
    ("Hopper-v4", "HCGAE vs Standard PPO"): (-2.9, 0.421, -0.69),
    ("Walker2d-v4", "HCGAE vs Standard PPO"): (31.4, 0.151, +1.07),
    ("HalfCheetah-v4", "HCGAE vs Standard PPO"): (18.9, 0.008, +1.95),
    ("Hopper-v4", "Optimal PPO vs Standard PPO"): (-11.4, 0.032, -1.78),
    ("Walker2d-v4", "Optimal PPO vs Standard PPO"): (12.0, 0.548, +0.51),
    ("HalfCheetah-v4", "Optimal PPO vs Standard PPO"): (41.5, 0.008, +4.18),
}

for env in ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]:
    hcgae = icml_summary[env]["Optimal_HCGAE"]["seeds"]
    std_ppo = icml_summary[env]["Standard_PPO"]["seeds"]
    opt_ppo = icml_summary[env]["Optimal_PPO"]["seeds"]

    r1 = mann_whitney(hcgae, opt_ppo)
    r2 = mann_whitney(hcgae, std_ppo)
    r3 = mann_whitney(opt_ppo, std_ppo)

    p_pct1, p_p1, p_d1 = paper_stats[(env, "HCGAE vs Optimal PPO")]
    p_pct2, p_p2, p_d2 = paper_stats[(env, "HCGAE vs Standard PPO")]
    p_pct3, p_p3, p_d3 = paper_stats[(env, "Optimal PPO vs Standard PPO")]

    def check_stat(actual_p, paper_p, actual_d, paper_d, name):
        p_ok = abs(actual_p - paper_p) < 0.01
        d_ok = abs(actual_d - paper_d) < 0.1
        status = "✅" if (p_ok and d_ok) else "⚠️"
        if status == "⚠️":
            issues.append(f"Stats {env}/{name}: paper p={paper_p},d={paper_d}, actual p={actual_p:.3f},d={actual_d:.2f}")
        print(f"    {name}: p={actual_p:.3f}(paper:{paper_p}), d={actual_d:.2f}(paper:{paper_d}) {status}")

    print(f"\n  {env}:")
    check_stat(r1["p"], p_p1, r1["d"], p_d1, "HCGAE vs OptPPO")
    check_stat(r2["p"], p_p2, r2["d"], p_d2, "HCGAE vs StdPPO")
    check_stat(r3["p"], p_p3, r3["d"], p_d3, "OptPPO vs StdPPO")

# ========================
# TABLE 2: DCPPO MultiEnv
# ========================
print("\n### TABLE 2 验证: DCPPO Variant Multi-Env (5 seeds × 500K)")
print("Source: results/MultiEnv_DCPPO/")

# Load the global_summary or per-seed metrics
def load_dcppo_seeds(env, variant):
    """Load 5-seed data for DCPPO variants"""
    env_dir = os.path.join(BASE, "MultiEnv_DCPPO", env, variant)
    seeds = [42, 123, 456, 789, 1234]
    rewards = []
    for s in seeds:
        fn = os.path.join(env_dir, f"{variant}_s{s}_metrics.json")
        if os.path.exists(fn):
            data = load_json(fn)
            if "final_reward" in data:
                rewards.append(data["final_reward"])
            elif "all_eval_rewards" in data:
                r = data["all_eval_rewards"]
                rewards.append(float(np.mean(r[-5:])))
    return rewards

hopper_base = load_dcppo_seeds("Hopper-v4", "DCPPO_Base")
hopper_imps = load_dcppo_seeds("Hopper-v4", "DCPPO_ImpS")
hopper_full = load_dcppo_seeds("Hopper-v4", "DCPPO_Full")
walker_base = load_dcppo_seeds("Walker2d-v4", "DCPPO_Base")
walker_imps = load_dcppo_seeds("Walker2d-v4", "DCPPO_ImpS")
walker_full = load_dcppo_seeds("Walker2d-v4", "DCPPO_Full")

def report_dcppo(name, rewards, paper_mean, paper_std):
    if len(rewards) == 0:
        print(f"    {name}: ❌ NO DATA FOUND! Paper: {paper_mean}±{paper_std}")
        issues.append(f"Table2 {name}: No data found, paper claims {paper_mean}±{paper_std}")
        return
    actual_mean = float(np.mean(rewards))
    actual_std = float(np.std(rewards, ddof=1))
    mean_diff = abs(actual_mean - paper_mean)
    std_diff = abs(actual_std - paper_std)
    status = "✅" if mean_diff < 10 and std_diff < 20 else "⚠️"
    if status == "⚠️":
        issues.append(f"Table2 {name}: paper={paper_mean}±{paper_std}, actual={actual_mean:.0f}±{actual_std:.0f}")
    print(f"    {name}: paper={paper_mean}±{paper_std}, actual={actual_mean:.0f}±{actual_std:.0f} (n={len(rewards)}) {status}")

print("\n  Hopper-v4:")
report_dcppo("DCPPO_Base", hopper_base, 2958, 397)
report_dcppo("DCPPO_ImpS", hopper_imps, 3056, 420)
report_dcppo("DCPPO_Full", hopper_full, 1192, 461)

print("\n  Walker2d-v4:")
report_dcppo("DCPPO_Base", walker_base, 1895, 632)
report_dcppo("DCPPO_ImpS", walker_imps, 1895, 632)
report_dcppo("DCPPO_Full", walker_full, 610, 205)

# ========================
# TABLE 3: Ablation MultiSeed (5 seeds × 300K)
# ========================
print("\n### TABLE 3 验证: HCGAE Ablation (5 seeds × 300K, Hopper-v4)")
print("Source: results/Hopper-v4-Ablation-MultiSeed/")

ablation_summary = load_json(os.path.join(BASE, "Hopper-v4-Ablation-MultiSeed", "multiseed_summary.json"))

paper_table3 = {
    "HCGAE_Base":  (2653, 627),
    "HCGAE_Imp1":  (2406, 787),
    "HCGAE_Imp2":  (2425, 615),
    "HCGAE_Imp12": (2839, 543),
}

for variant in ["HCGAE_Base", "HCGAE_Imp1", "HCGAE_Imp2", "HCGAE_Imp12"]:
    if variant in ablation_summary:
        actual_mean = ablation_summary[variant]["mean"]
        actual_std = ablation_summary[variant]["std"]
        paper_mean, paper_std = paper_table3[variant]
        mean_diff = abs(actual_mean - paper_mean)
        std_diff = abs(actual_std - paper_std)
        status = "✅" if mean_diff < 2 and std_diff < 2 else "⚠️"
        if status == "⚠️":
            issues.append(f"Table3 {variant}: paper={paper_mean}±{paper_std}, actual={actual_mean:.0f}±{actual_std:.0f}")
        print(f"    {variant}: paper={paper_mean}±{paper_std}, actual={actual_mean:.0f}±{actual_std:.0f} {status}")
    else:
        print(f"    {variant}: ❌ Not found in summary")
        issues.append(f"Table3 {variant}: Missing from summary")

# Verify synergy calculation
base_mean = ablation_summary.get("HCGAE_Base", {}).get("mean", 0)
imp1_mean = ablation_summary.get("HCGAE_Imp1", {}).get("mean", 0)
imp2_mean = ablation_summary.get("HCGAE_Imp2", {}).get("mean", 0)
imp12_mean = ablation_summary.get("HCGAE_Imp12", {}).get("mean", 0)

imp1_delta = imp1_mean - base_mean
imp2_delta = imp2_mean - base_mean
additive_pred = base_mean + imp1_delta + imp2_delta
actual_gain = imp12_mean - base_mean
synergy = imp12_mean - additive_pred

print(f"\n    Synergy verification:")
print(f"      Base={base_mean:.0f}, Imp1={imp1_mean:.0f}(Δ={imp1_delta:.0f}), Imp2={imp2_mean:.0f}(Δ={imp2_delta:.0f})")
print(f"      Additive pred={additive_pred:.0f}, Imp12={imp12_mean:.0f}")
print(f"      Synergy (Imp12 - additive_pred) = {synergy:.0f} (paper claims +661)")
if abs(synergy - 661) < 5:
    print("      ✅ Synergy matches paper claim")
else:
    issues.append(f"Table3 Synergy: paper=+661, actual={synergy:.0f}")
    print(f"      ⚠️ Synergy mismatch: paper=+661, actual={synergy:.0f}")

# ========================
# TABLE 5: Overhead
# ========================
print("\n### TABLE 5 验证: Computational Overhead")
print("Source: results/overhead_measurement.json")

overhead = load_json(os.path.join(BASE, "overhead_measurement.json"))

paper_overhead = {
    "Standard_GAE": {"gae": (6.7, 0.2), "update": (304.5, 22.5)},
    "HCGAE_Imp12": {"gae": (13.4, 0.2), "update": (278.2, 4.2)},
    "DCPPO_S": {"gae": (7.1, 0.2), "update": (281.7, 5.3)},
}

for method in ["Standard_GAE", "HCGAE_Imp12", "DCPPO_S"]:
    actual_gae = overhead["gae_ms"].get(method, None)
    actual_update = overhead["update_ms"].get(method, None)
    paper_gae = paper_overhead[method]["gae"][0]
    paper_update = paper_overhead[method]["update"][0]

    if actual_gae is not None:
        gae_diff = abs(actual_gae - paper_gae)
        upd_diff = abs(actual_update - paper_update)
        status = "✅" if gae_diff < 0.5 and upd_diff < 5 else "⚠️"
        if status == "⚠️":
            issues.append(f"Table5 {method}: paper_gae={paper_gae}, actual_gae={actual_gae:.1f}")
        print(f"    {method}: gae={actual_gae:.1f}ms(paper:{paper_gae}), update={actual_update:.1f}ms(paper:{paper_update}) {status}")
    else:
        print(f"    {method}: ❌ Not found in overhead_measurement.json")
        issues.append(f"Table5 {method}: Missing from overhead_measurement.json")

# ========================
# TABLE 6: MultiSeedPower n=10
# ========================
print("\n### TABLE 6 验证: n=10 Statistical Power (300K steps)")
print("Source: results/MultiSeedPower/final_statistical_report_n10.json")

n10_report = load_json(os.path.join(BASE, "MultiSeedPower", "final_statistical_report_n10.json"))

paper_table6 = {
    "Hopper-v4": {
        "Standard_PPO":      (2524, 167),  # mean ± SEM
        "HCGAE_Imp12":       (2663, 150),
        "HCGAE_Imp12_SCR":   (2834, 155),
    },
    "Walker2d-v4": {
        "Standard_PPO":      (1252, 228),
        "HCGAE_Imp12":       (1063, 212),
        "HCGAE_Imp12_SCR":   (1516, 298),
    },
    "HalfCheetah-v4": {
        "Standard_PPO":      (950, 56),
        "HCGAE_Imp12":       (757, 47),
        "HCGAE_Imp12_SCR":   (709, 59),
    }
}

n10_summary = n10_report["summary"]

for env in ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]:
    print(f"\n  {env}:")
    for alg in ["Standard_PPO", "HCGAE_Imp12", "HCGAE_Imp12_SCR"]:
        if env in n10_summary and alg in n10_summary[env]:
            actual_mean = n10_summary[env][alg]["mean"]
            actual_sem = n10_summary[env][alg]["sem"]
            paper_mean, paper_sem = paper_table6[env][alg]
            mean_diff = abs(actual_mean - paper_mean)
            sem_diff = abs(actual_sem - paper_sem)
            status = "✅" if mean_diff < 2 and sem_diff < 2 else "⚠️"
            if status == "⚠️":
                issues.append(f"Table6 {env}/{alg}: paper={paper_mean}±{paper_sem}(SEM), actual={actual_mean:.0f}±{actual_sem:.0f}")
            print(f"    {alg}: paper={paper_mean}±{paper_sem}(SEM), actual={actual_mean:.0f}±{actual_sem:.0f} {status}")
        else:
            print(f"    {alg}: ❌ Not found")
            issues.append(f"Table6 {env}/{alg}: Missing data")

# Verify n=10 statistical tests
print("\n  Statistical test verification (n=10):")

paper_n10_stats = {
    "Hopper-v4 HCGAE vs Std": ("+5.5%", 0.571, 0.277),
    "Hopper-v4 SCR vs Std": ("+12.3%", 0.241, 0.609),
    "Walker2d-v4 HCGAE vs Std": ("-15.1%", 0.427, -0.272),
    "Walker2d-v4 SCR vs Std": ("+21.1%", 0.970, 0.315),
    "HalfCheetah HCGAE vs Std": ("-20.3%", 0.026, -1.169),
    "HalfCheetah SCR vs Std": ("-25.3%", 0.011, -1.324),
}

for comp in n10_report["comparisons"]:
    env = comp["env"]
    label = comp["label"]
    p = comp["stats"]["p_value"]
    d = comp["stats"]["cohens_d"]
    pct = comp["stats"]["pct_improvement"]

    key = None
    if env == "Hopper-v4" and "HCGAE_Imp12 vs Standard_PPO" in label:
        key = "Hopper-v4 HCGAE vs Std"
    elif env == "Hopper-v4" and "HCGAE_Imp12_SCR vs Standard_PPO" in label:
        key = "Hopper-v4 SCR vs Std"
    elif env == "Walker2d-v4" and "HCGAE_Imp12 vs Standard_PPO" in label:
        key = "Walker2d-v4 HCGAE vs Std"
    elif env == "Walker2d-v4" and "HCGAE_Imp12_SCR vs Standard_PPO" in label:
        key = "Walker2d-v4 SCR vs Std"
    elif env == "HalfCheetah-v4" and "HCGAE_Imp12 vs Standard_PPO" in label:
        key = "HalfCheetah HCGAE vs Std"
    elif env == "HalfCheetah-v4" and "HCGAE_Imp12_SCR vs Standard_PPO" in label:
        key = "HalfCheetah SCR vs Std"

    if key and key in paper_n10_stats:
        paper_pct, paper_p, paper_d = paper_n10_stats[key]
        p_ok = abs(p - paper_p) < 0.01
        d_ok = abs(d - paper_d) < 0.01
        status = "✅" if (p_ok and d_ok) else "⚠️"
        if status == "⚠️":
            issues.append(f"Table6 stats {key}: paper p={paper_p},d={paper_d}, actual p={p:.3f},d={d:.3f}")
        print(f"    {key}: p={p:.3f}(paper:{paper_p}), d={d:.3f}(paper:{paper_d}) {status}")

# ========================
# TABLE S1-S3: Sensitivity
# ========================
print("\n### TABLE S1/S2/S3 验证: Sensitivity Analysis")
print("Source: results/Sensitivity/sensitivity_summary.json")

sens = load_json(os.path.join(BASE, "Sensitivity", "sensitivity_summary.json"))

print("\n  Beta sensitivity (Table S1):")
paper_beta = {1.0: 3202, 2.0: 1849, 3.0: 3457, 4.0: 1203, 5.0: 2556}
for item in sens["beta_sensitivity"]:
    beta = item["beta"]
    actual = round(item["final_reward"])
    paper = paper_beta.get(beta, None)
    if paper is not None:
        diff = abs(actual - paper)
        status = "✅" if diff < 2 else "⚠️"
        if status == "⚠️":
            issues.append(f"TableS1 beta={beta}: paper={paper}, actual={actual}")
        print(f"    beta={beta}: paper={paper}, actual={actual} {status}")

print("\n  Alpha_max sensitivity (Table S2):")
paper_amax = {0.3: 3287, 0.5: 2607, 0.7: 3457, 0.9: 2178}
for item in sens["amax_sensitivity"]:
    amax = item["alpha_max0"]
    actual = round(item["final_reward"])
    paper = paper_amax.get(amax, None)
    if paper is not None:
        diff = abs(actual - paper)
        status = "✅" if diff < 2 else "⚠️"
        if status == "⚠️":
            issues.append(f"TableS2 amax={amax}: paper={paper}, actual={actual}")
        print(f"    amax={amax}: paper={paper}, actual={actual} {status}")

print("\n  SNR sensitivity (Table S3):")
paper_snr = {0.1: 2601, 0.2: 2601, 0.3: 2945, 0.5: 3240, 0.7: 2460}
for item in sens["snr_sensitivity"]:
    snr = item["snr_target"]
    actual = round(item["final_reward"])
    paper = paper_snr.get(snr, None)
    if paper is not None:
        diff = abs(actual - paper)
        status = "✅" if diff < 2 else "⚠️"
        if status == "⚠️":
            issues.append(f"TableS3 snr={snr}: paper={paper}, actual={actual}")
        print(f"    snr={snr}: paper={paper}, actual={actual} {status}")

# ========================
# Appendix F: Power vs Linear
# ========================
print("\n### Appendix F 验证: DCPPO_ImpS_Power vs Linear")
print("Source: results/MultiEnv_DCPPO/dcppo_multiseed_summary.json")

dcppo_ms = load_json(os.path.join(BASE, "MultiEnv_DCPPO", "dcppo_multiseed_summary.json"))

for env in ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]:
    if env in dcppo_ms:
        for variant in ["DCPPO_ImpS_Power", "DCPPO_ImpS_Linear"]:
            if variant in dcppo_ms[env]:
                mean = dcppo_ms[env][variant]["mean"]
                std = dcppo_ms[env][variant]["std"]
                n = len(dcppo_ms[env][variant]["seeds"])
                print(f"    {env}/{variant}: mean={mean:.0f}±{std:.0f} (n={n})")

# ========================
# Table 4.3 HalfCheetah Mann-Whitney table (checking OLD vs NEW data issue)
# ========================
print("\n### TABLE 4.3 验证: HalfCheetah Mann-Whitney (HCGAE vs baselines)")
print("NOTE: This table refers to old UnifiedComparison data, NOT the new ICMLExperiment data")
print("Checking if these numbers are from UnifiedComparison or ICMLExperiment...")

# The table in 4.3 shows HCGAE (828 ± 113) vs Standard PPO (902 ± 90)
# This is OLD data - not from ICMLExperiment
# New ICMLExperiment: HCGAE=1250±53, Standard PPO=1051±134
print(f"\n  Paper section 4.3 states: 'HCGAE (828 ± 113) vs Standard PPO (902 ± 90)'")
print(f"  New ICMLExperiment data: HCGAE={1250}±{53}, Standard PPO={1051}±{134}")
print(f"  ⚠️ CRITICAL: Section 4.3 HalfCheetah table uses OLD data, contradicts Section 4.2 (new data)!")
issues.append("CRITICAL: Section 4.3 HalfCheetah table uses old UnifiedComparison data (HCGAE=828) that contradicts the new ICMLExperiment data in Table 1 (HCGAE=1250)")

# Also check if UnifiedComparison HCGAE data matches those numbers
def load_unified_seeds(env, alg):
    alg_dir = os.path.join(BASE, "UnifiedComparison", env, alg)
    seeds = [42, 123, 456, 789, 1234]
    rewards = []
    for s in seeds:
        fn = os.path.join(alg_dir, f"{alg}_s{s}.json")
        if os.path.exists(fn):
            data = load_json(fn)
            if "final_reward" in data:
                rewards.append(data["final_reward"])
            elif "all_eval_rewards" in data:
                r = data["all_eval_rewards"]
                rewards.append(float(np.mean(r[-5:])))
    return rewards

hc_unified_hcgae = load_unified_seeds("HalfCheetah-v4", "HCGAE_Imp12")
hc_unified_std = load_unified_seeds("HalfCheetah-v4", "Standard_PPO")

if hc_unified_hcgae:
    m_h = float(np.mean(hc_unified_hcgae))
    s_h = float(np.std(hc_unified_hcgae, ddof=1))
    print(f"\n  UnifiedComparison HalfCheetah HCGAE: mean={m_h:.0f}±{s_h:.0f}")
if hc_unified_std:
    m_s = float(np.mean(hc_unified_std))
    s_s = float(np.std(hc_unified_std, ddof=1))
    print(f"  UnifiedComparison HalfCheetah Standard PPO: mean={m_s:.0f}±{s_s:.0f}")
    print(f"  (Paper section 4.3 claims HCGAE=828±113, StdPPO=902±90 — cross-checking...)")
    if abs(m_h - 828) < 50:
        print(f"  → UnifiedComparison data matches section 4.3 numbers (old data)")
    else:
        print(f"  → UnifiedComparison data does NOT match section 4.3 numbers either")

# ========================
# FINAL SUMMARY
# ========================
print("\n" + "=" * 80)
print("ISSUES FOUND (requires fix):")
print("=" * 80)

if not issues:
    print("✅ No issues found! All data matches.")
else:
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")

print(f"\nTotal issues: {len(issues)}")

# Save report
report = {
    "issues": issues,
    "n_issues": len(issues),
    "tables_checked": ["Table1", "Table2", "Table3", "Table4", "Table5", "Table6", "TableS1", "TableS2", "TableS3", "AppF"],
}
with open(os.path.join(BASE, "data_audit_report.json"), "w") as f:
    json.dump(report, f, indent=2)
print(f"\nReport saved to: results/data_audit_report.json")

