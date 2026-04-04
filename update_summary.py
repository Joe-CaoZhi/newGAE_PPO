"""
Rebuild multiseed_summary_n10.json from all completed *_metrics.json files.
Uses final_mean = mean of last 5 eval_rewards (consistent with run_multiseed_power.py).
"""
import json
import re
import numpy as np
from pathlib import Path

base = Path("results/MultiSeedPower")
summary = {}

for env_dir in sorted(base.iterdir()):
    if not env_dir.is_dir():
        continue
    env_name = env_dir.name
    if env_name.startswith("."):
        continue
    summary[env_name] = {}
    for algo_dir in sorted(env_dir.iterdir()):
        if not algo_dir.is_dir():
            continue
        algo_name = algo_dir.name
        seeds = []
        seed_ids = []
        scr_list = []
        for f in sorted(algo_dir.glob("*_metrics.json")):
            try:
                d = json.loads(f.read_text())

                # Extract final_mean = mean of last 5 eval_rewards
                eval_rewards = d.get("eval_rewards", [])
                if not eval_rewards:
                    print(f"  SKIP {f.name}: no eval_rewards")
                    continue
                final_rewards_tail = eval_rewards[-5:]
                final_mean = float(np.mean(final_rewards_tail))

                # Extract seed from filename
                stem = f.stem  # e.g. Standard_PPO_s3_metrics or HCGAE_Imp12_s4_metrics
                m_seed = re.search(r"_s(\d+)(?:_metrics)?$", stem)
                if not m_seed:
                    print(f"  SKIP {f.name}: cannot parse seed")
                    continue
                seed = int(m_seed.group(1))

                # Extract SCR EMA (snr_history last 10 values)
                scr_ema_hist = d.get("scr_ema_history", d.get("snr_history", []))
                scr_ema = float(np.mean(scr_ema_hist[-10:])) if scr_ema_hist else None

                seeds.append(final_mean)
                seed_ids.append(seed)
                if scr_ema is not None and algo_name != "Standard_PPO":
                    scr_list.append(float(scr_ema))

                print(f"    {f.name}: seed={seed} final_mean={final_mean:.1f}" +
                      (f" scr={scr_ema:.3f}" if scr_ema else ""))
            except Exception as e:
                print(f"  ERROR reading {f}: {e}")

        summary[env_name][algo_name] = {
            "seeds": seeds,
            "seed_list": seed_ids,
            "scr_ema_list": scr_list,
            "mean": float(np.mean(seeds)) if seeds else 0.0,
            "std": float(np.std(seeds, ddof=1)) if len(seeds) > 1 else 0.0,
            "n_seeds": len(seeds),
        }
        n = len(seeds)
        mean_val = np.mean(seeds) if seeds else float("nan")
        print(f"  >>> {env_name}/{algo_name}: n={n}  mean={mean_val:.1f}")

output = base / "multiseed_summary_n10.json"
output.write_text(json.dumps(summary, indent=2))
print(f"\nSaved -> {output}")

