"""
Check all MultiSeedPower metrics files and rebuild multiseed_summary_n10.json
"""
import json
import os
import glob
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("results/MultiSeedPower")

def get_final_mean(eval_rewards, n=5):
    """Get mean of last n eval rewards."""
    if len(eval_rewards) == 0:
        return float('nan')
    return float(np.mean(eval_rewards[-n:]))

def get_scr_ema_last(data):
    """Get last SCR EMA value from metrics."""
    scr = data.get('scr_ema_history', [])
    if scr:
        return float(scr[-1])
    return None

def check_all_seeds():
    """Check all available metrics files."""
    envs = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']
    algos = ['Standard_PPO', 'HCGAE_Imp12', 'HCGAE_Imp12_SCR']

    print("=" * 70)
    print("  Checking all metrics files")
    print("=" * 70)

    for env in envs:
        print(f"\n{env}:")
        for algo in algos:
            # Map algo name to file prefix
            if algo == 'HCGAE_Imp12_SCR':
                prefix = 'HCGAE_Imp12'
            else:
                prefix = algo

            base = RESULTS_DIR / env / algo
            if not base.exists():
                print(f"  {algo}: DIR NOT FOUND")
                continue

            seeds_found = []
            for i in range(1, 11):
                fn = base / f"{prefix}_s{i}_metrics.json"
                if fn.exists():
                    with open(fn) as f:
                        d = json.load(f)
                    er = d.get('eval_rewards', [])
                    final = get_final_mean(er)
                    seeds_found.append((i, final))
                else:
                    pass

            if seeds_found:
                vals = [v for _, v in seeds_found]
                print(f"  {algo}: n={len(seeds_found)} seeds found, mean={np.mean(vals):.1f} ± {np.std(vals, ddof=1):.1f}")
                for s, v in seeds_found:
                    print(f"    s{s}: {v:.1f}")
            else:
                print(f"  {algo}: NO DATA")


def rebuild_summary():
    """Rebuild multiseed_summary_n10.json from metrics files."""
    envs = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']
    algos = ['Standard_PPO', 'HCGAE_Imp12', 'HCGAE_Imp12_SCR']

    # Load existing summary to preserve already computed data
    summary_path = RESULTS_DIR / "multiseed_summary_n10.json"
    if summary_path.exists():
        with open(summary_path) as f:
            existing = json.load(f)
    else:
        existing = {}

    new_summary = {}

    for env in envs:
        new_summary[env] = {}
        for algo in algos:
            # Map algo name to file prefix
            if algo == 'HCGAE_Imp12_SCR':
                prefix = 'HCGAE_Imp12'
            else:
                prefix = algo

            base = RESULTS_DIR / env / algo
            if not base.exists():
                # Keep existing if available
                if env in existing and algo in existing[env]:
                    new_summary[env][algo] = existing[env][algo]
                continue

            seeds = []
            seed_list = []
            scr_ema_list = []

            for i in range(1, 11):
                fn = base / f"{prefix}_s{i}_metrics.json"
                if fn.exists():
                    with open(fn) as f:
                        d = json.load(f)
                    er = d.get('eval_rewards', [])
                    final = get_final_mean(er)
                    if not np.isnan(final):
                        seeds.append(final)
                        seed_list.append(i)
                        scr = get_scr_ema_last(d)
                        if scr is not None:
                            scr_ema_list.append(scr)

            if seeds:
                new_summary[env][algo] = {
                    "seeds": seeds,
                    "seed_list": seed_list,
                    "scr_ema_list": scr_ema_list,
                    "mean": float(np.mean(seeds)),
                    "std": float(np.std(seeds, ddof=1)) if len(seeds) > 1 else 0.0,
                    "n_seeds": len(seeds)
                }
                print(f"  {env}/{algo}: n={len(seeds)}, mean={np.mean(seeds):.1f}")
            elif env in existing and algo in existing[env]:
                new_summary[env][algo] = existing[env][algo]
                print(f"  {env}/{algo}: kept existing (n={existing[env][algo].get('n_seeds', '?')})")

    # Remove empty envs
    new_summary = {env: data for env, data in new_summary.items() if data}

    with open(summary_path, 'w') as f:
        json.dump(new_summary, f, indent=2)
    print(f"\nSaved updated summary to {summary_path}")
    return new_summary


if __name__ == "__main__":
    check_all_seeds()
    print("\n\nRebuilding summary...")
    rebuild_summary()

