#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# ICML 2026 并行实验启动脚本
# 4 环境 × 5 算法 × 12 种子 = 240 次运行
# 使用进程池控制并发数（默认 4 并发，避免内存溢出）
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT="/Users/joe-caozhi/newGAE_ppo/run_icml_experiment.py"
PYTHON="python3"
MAX_PARALLEL=${1:-4}   # 默认 4 并发（可通过第一个参数调整）
LOG_DIR="/Users/joe-caozhi/newGAE_ppo/logs/FinalExperiment"
mkdir -p "$LOG_DIR"

ENVS=("Hopper-v4" "Walker2d-v4" "HalfCheetah-v4" "Ant-v4")
ALGOS=("Standard_PPO" "Optimal_PPO" "Heuristic_HCGAE" "BHVF" "BHVF_DCPPO")

echo "=== ICML 2026 Full Experiment ==="
echo "Envs:     ${ENVS[@]}"
echo "Algos:    ${ALGOS[@]}"
echo "Seeds:    0-11"
echo "Steps:    1,000,000"
echo "Parallel: $MAX_PARALLEL"
echo ""
echo "Logs: $LOG_DIR"
echo ""

# 生成所有任务（env × algo × seeds_batch）
# 每批 seeds: 0,1,2,3 | 4,5,6,7 | 8,9,10,11（每批启动一个进程）
SEED_BATCHES=("0,1,2,3" "4,5,6,7" "8,9,10,11")

active=0
pids=()

for env in "${ENVS[@]}"; do
    for algo in "${ALGOS[@]}"; do
        for batch in "${SEED_BATCHES[@]}"; do
            # 等待空闲槽
            while [ "$active" -ge "$MAX_PARALLEL" ]; do
                for i in "${!pids[@]}"; do
                    if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                        unset "pids[$i]"
                        ((active--))
                    fi
                done
                sleep 2
            done

            log_file="$LOG_DIR/${env}_${algo}_s${batch//,/-}.log"
            echo "[START] $env/$algo seeds=$batch → $log_file"
            $PYTHON "$SCRIPT" --env "$env" --algo "$algo" --seeds "$batch" \
                > "$log_file" 2>&1 &
            pid=$!
            pids+=("$pid")
            ((active++))
            sleep 0.5   # 错开启动，避免同时初始化
        done
    done
done

# 等待所有剩余任务完成
echo ""
echo "Waiting for remaining tasks..."
wait
echo ""
echo "=== All done! ==="
echo ""
$PYTHON "$SCRIPT" --summary

