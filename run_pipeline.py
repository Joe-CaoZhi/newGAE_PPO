#!/usr/bin/env python3
"""
全自动实验流水线（Pipeline）
============================

自动按序执行以下两个阶段，前一阶段完成后自动启动下一阶段，无需人工干预：

  阶段 1 — GRPO 补全
  ─────────────────────────────────────────────────────────────
  目标：results/GRPO/{env}/{algo}/{algo}_s{seed}.json  全部完成
  规模：4 环境 × 4 算法 × 15 seeds × 1.5M steps
  跳过：已有完整结果的 trial（断点续跑）
  当前缺口（约）：
    Ant-v4/Optimal_GRPO        seeds 13-14  (已运行过，仅补剩余)
    Ant-v4/HCGAE_Optimal_GRPO  seeds 0-14   (全部)

  阶段 2 — 消融实验
  ─────────────────────────────────────────────────────────────
  目标：results/Ablation/{env}/{algo}/{algo}_s{seed}.json  全部完成
  规模：4 环境 × 4 算法 × 8 seeds × 1M steps
  算法：
    Optimal_HCGAE_Optimal     (Full 对照)
    Optimal_HCGAE_NoFixSCR    (-FixSCR 消融)
    Optimal_HCGAE_NoMCSigmoid (-per-step sigmoid 消融)
    Optimal_HCGAE_NoEVGate    (-EV 自适应门控消融)

用法:
  nohup python3 run_pipeline.py > logs/pipeline.log 2>&1 &
  nohup python3 run_pipeline.py --workers 6 > logs/pipeline.log 2>&1 &
  python3 run_pipeline.py --status           # 只查看当前进度，不运行

实现方式:
  - 每个 trial 用独立 subprocess 启动（与 run_grpo/run_ablation 一致）
  - skip_existing=True（已完成的 trial 自动跳过）
  - 循环轮询，阶段1全部完成后才启动阶段2
  - 每个阶段结束后打印汇总
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 阶段定义
# ─────────────────────────────────────────────────────────────────────────────
ENVS = ["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4", "Ant-v4"]

PHASE1 = {
    "name":        "阶段1: GRPO 补全",
    "script":      "run_grpo_experiment.py",
    "results_dir": "results/GRPO",
    "steps":       1_500_000,
    "min_steps":   1_400_000,   # 95% 视为完成
    "algos": [
        "Standard_GRPO_NoTrick",
        "Optimal_GRPO",
        "HCGAE_Standard_GRPO",
        "HCGAE_Optimal_GRPO",
    ],
    "seeds": list(range(15)),  # 0-14
}

PHASE2 = {
    "name":        "阶段2: 消融实验",
    "script":      "run_ablation.py",
    "results_dir": "results/Ablation",
    "steps":       1_000_000,
    "min_steps":   950_000,    # 95% 视为完成
    "algos": [
        "Optimal_HCGAE_Optimal",
        "Optimal_HCGAE_NoFixSCR",
        "Optimal_HCGAE_NoMCSigmoid",
        "Optimal_HCGAE_NoEVGate",
    ],
    "seeds": list(range(8)),   # 0-7
}

PHASES = [PHASE1, PHASE2]


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────
def is_done(results_dir: Path, env: str, algo: str, seed: int, min_steps: int) -> bool:
    """判断某个 trial 是否已完成。"""
    jf = results_dir / env / algo / f"{algo}_s{seed}.json"
    if not jf.exists():
        return False
    try:
        d = json.load(open(jf))
        return d.get("total_steps", 0) >= min_steps
    except Exception:
        return False


def get_pending(phase: dict) -> list:
    """返回该阶段所有未完成的 (env, algo, seed) 列表。"""
    results_dir = Path(phase["results_dir"])
    pending = []
    for env in ENVS:
        for algo in phase["algos"]:
            for seed in phase["seeds"]:
                if not is_done(results_dir, env, algo, seed, phase["min_steps"]):
                    pending.append((env, algo, seed))
    return pending


def phase_status(phase: dict) -> tuple:
    """返回 (done_count, total_count, pending_list)。"""
    results_dir = Path(phase["results_dir"])
    total = len(ENVS) * len(phase["algos"]) * len(phase["seeds"])
    pending = get_pending(phase)
    done = total - len(pending)
    return done, total, pending


def print_status(phases=None):
    """打印所有阶段的当前进度。"""
    if phases is None:
        phases = PHASES
    print("\n" + "=" * 70)
    print("  实验流水线进度")
    print("=" * 70)
    for ph in phases:
        done, total, pending = phase_status(ph)
        pct = done / total * 100 if total > 0 else 0
        status = "✅ 完成" if done >= total else f"🔄 进行中 ({pct:.0f}%)"
        print(f"\n  {ph['name']}")
        print(f"    进度: {done}/{total}  {status}")
        if pending:
            # 按 env/algo 分组显示缺口
            gaps = {}
            for env, algo, seed in pending:
                key = f"{env}/{algo}"
                gaps.setdefault(key, 0)
                gaps[key] += 1
            print(f"    缺口:")
            for key, cnt in sorted(gaps.items()):
                print(f"      {key}: 缺 {cnt} seeds")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 阶段执行器（并行 subprocess）
# ─────────────────────────────────────────────────────────────────────────────
def run_phase(phase: dict, n_workers: int, base_dir: Path):
    """运行一个阶段，并行调度所有未完成的 trial。"""
    pending = get_pending(phase)
    if not pending:
        print(f"\n  [{phase['name']}] 全部已完成，跳过。")
        return True

    total_trials = len(ENVS) * len(phase["algos"]) * len(phase["seeds"])
    done_before  = total_trials - len(pending)

    script = str(base_dir / phase["script"])
    results_dir = str(base_dir / phase["results_dir"])

    print(f"\n{'='*70}")
    print(f"  {phase['name']}")
    print(f"  待完成: {len(pending)} trials  (已完成: {done_before}/{total_trials})")
    print(f"  并行度: {n_workers} workers")
    print(f"  步数:   {phase['steps']:,}")
    print(f"  输出:   {results_dir}")
    print(f"{'='*70}\n")

    running = {}    # pid → (proc, env, algo, seed)
    done_count = 0
    pending_queue = list(pending)  # copy
    n_to_do = len(pending_queue)
    t0 = time.time()

    while pending_queue or running:
        # 补充空位
        while len(running) < n_workers and pending_queue:
            env, algo, seed = pending_queue.pop(0)
            cmd = [
                sys.executable, script,
                "--single-env",  env,
                "--single-algo", algo,
                "--single-seed", str(seed),
                "--steps",       str(phase["steps"]),
                "--results-dir", results_dir,
            ]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(base_dir),
            )
            running[proc.pid] = (proc, env, algo, seed)
            print(f"  → 启动 {env}/{algo}/s{seed}  (PID {proc.pid})", flush=True)

        # 轮询完成
        for pid in list(running.keys()):
            proc, env, algo, seed = running[pid]
            ret = proc.poll()
            if ret is not None:
                out, _ = proc.communicate()
                # 只打印 DONE / START / SKIP 行，过滤进度条噪音
                for line in out.strip().splitlines():
                    stripped = line.strip()
                    if any(k in stripped for k in ("DONE", "START", "SKIP", "ERROR", "WARN")):
                        print(f"    {stripped}", flush=True)
                done_count += 1
                elapsed = time.time() - t0
                eta = (elapsed / done_count * (n_to_do - done_count)) if done_count > 0 else 0
                print(
                    f"  [{done_count}/{n_to_do}] 完成 {env}/{algo}/s{seed}  "
                    f"exit={ret}  "
                    f"已用={elapsed/3600:.1f}h  ETA={eta/3600:.1f}h",
                    flush=True,
                )
                del running[pid]

        if running:
            time.sleep(5)

    elapsed_total = time.time() - t0
    print(f"\n  ✅ {phase['name']} 全部完成！"
          f"  共 {n_to_do} trials  耗时 {elapsed_total/3600:.2f}h")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 打印阶段汇总
# ─────────────────────────────────────────────────────────────────────────────
def summarize_phase(phase: dict, base_dir: Path):
    """调用对应脚本的 --summarize 模式打印汇总。"""
    script = str(base_dir / phase["script"])
    seeds_str = [str(s) for s in phase["seeds"]]
    cmd = [
        sys.executable, script,
        "--summarize",
        "--results-dir", str(base_dir / phase["results_dir"]),
        "--seeds", *seeds_str,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                cwd=str(base_dir), timeout=120)
        print(result.stdout)
        if result.stderr.strip():
            for line in result.stderr.strip().splitlines():
                if "DeprecationWarning" not in line:
                    print(f"  [STDERR] {line}")
    except subprocess.TimeoutExpired:
        print("  [WARN] 汇总超时")
    except Exception as e:
        print(f"  [WARN] 汇总失败: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description="全自动实验流水线")
    parser.add_argument("--workers", type=int, default=None,
                        help="并行 worker 数（默认 CPU 核数）")
    parser.add_argument("--status",  action="store_true",
                        help="只打印当前进度，不运行")
    parser.add_argument("--phase",   type=int, default=None, choices=[1, 2],
                        help="只运行指定阶段（1=GRPO补全，2=消融实验）")
    parser.add_argument("--summarize", action="store_true",
                        help="只打印所有阶段的汇总结果")
    args = parser.parse_args()

    base_dir  = Path(__file__).parent.resolve()
    n_workers = args.workers or mp.cpu_count()

    # ── 状态查看模式 ──────────────────────────────────────────────────────────
    if args.status:
        print_status()
        sys.exit(0)

    # ── 汇总模式 ──────────────────────────────────────────────────────────────
    if args.summarize:
        for ph in PHASES:
            done, total, _ = phase_status(ph)
            if done > 0:
                print(f"\n{'─'*70}")
                summarize_phase(ph, base_dir)
        sys.exit(0)

    # ── 流水线执行模式 ────────────────────────────────────────────────────────
    phases_to_run = PHASES if args.phase is None else [PHASES[args.phase - 1]]

    print(f"\n{'='*70}")
    print(f"  全自动实验流水线启动")
    print(f"  并行度: {n_workers} workers")
    print(f"  阶段数: {len(phases_to_run)}")
    print(f"{'='*70}")
    print_status(phases_to_run)

    t_pipeline = time.time()

    for i, phase in enumerate(phases_to_run):
        print(f"\n{'#'*70}")
        print(f"#  开始 {phase['name']}  ({i+1}/{len(phases_to_run)})")
        print(f"{'#'*70}")

        # 确保日志目录存在
        (base_dir / "logs").mkdir(exist_ok=True)

        # 运行阶段（阻塞，直到全部完成）
        run_phase(phase, n_workers, base_dir)

        # 阶段汇总
        print(f"\n── {phase['name']} 汇总 ──")
        summarize_phase(phase, base_dir)

    elapsed_pipeline = time.time() - t_pipeline
    print(f"\n{'='*70}")
    print(f"  🎉 所有实验阶段完成！  总耗时: {elapsed_pipeline/3600:.2f}h")
    print(f"{'='*70}")
    print_status()

