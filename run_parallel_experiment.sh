#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# ICML 2026 并行实验启动脚本
# 支持模式:
#   ./run_parallel_experiment.sh complete [N]  补全 V3~V10 所有缺失seed (自动skip已完成)
#   ./run_parallel_experiment.sh v9only [N]    仅跑 V9 全量 (4env×12seed)
#   ./run_parallel_experiment.sh v10only [N]   仅跑 V10 全量 (4env×12seed)
#   ./run_parallel_experiment.sh v6v8 [N]      补跑 V6 缺失 + V8/V9 全量
#   ./run_parallel_experiment.sh v8only [N]    仅跑 V8 全量 (4env×12seed)
#   ./run_parallel_experiment.sh               全量实验 (4env×所有algo×12seed)
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT="/Users/joe-caozhi/newGAE_ppo/run_icml_experiment.py"
PYTHON="python3"
MODE=${1:-"full"}
MAX_PARALLEL=${2:-4}
LOG_DIR="/Users/joe-caozhi/newGAE_ppo/logs/FinalExperiment"
mkdir -p "$LOG_DIR"

ENVS=("Hopper-v4" "Walker2d-v4" "HalfCheetah-v4" "Ant-v4")
pids=()

wait_for_slot() {
    while true; do
        running=0
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                ((running++))
            fi
        done
        [ "$running" -lt "$MAX_PARALLEL" ] && break
        sleep 3
    done
}

launch() {
    local env="$1" algo="$2" seeds="$3"
    local tag="${seeds//,/-}"
    local log_file="$LOG_DIR/${env}_${algo}_s${tag}.log"
    echo "[START] $env / $algo  seeds=$seeds"
    $PYTHON "$SCRIPT" --env "$env" --algo "$algo" --seeds "$seeds" \
        > "$log_file" 2>&1 &
    pids+=($!)
    sleep 0.5
}

if [ "$MODE" = "complete" ]; then
    # ── complete 模式：补全 V3~V10 所有缺失 seed ──
    # run_icml_experiment.py 默认 skip_existing=True，已完成的 seed 自动跳过
    # 每个 batch 覆盖全部 12 个 seed，脚本内部自动跳过已有 JSON
    echo "=== Complete Mode: V3~V10 全量补全 (4env × 8algo × 12seed, Parallel=$MAX_PARALLEL) ==="
    echo "NOTE: 已完成的 seed 自动跳过（skip_existing=True）"
    echo ""

    # V4: HC 缺 s7-11(5个), Ant 缺 s6-11(6个)
    # 全批次投喂，由脚本内部 skip 已完成的
    for batch in "0,1,2,3" "4,5,6,7" "8,9,10,11"; do
        wait_for_slot; launch "HalfCheetah-v4" "BHVF_V4" "$batch"
        wait_for_slot; launch "Ant-v4"         "BHVF_V4" "$batch"
    done

    # V5: Walker 全缺, HC 缺 s2-11(10), Ant 缺 s8-11(4), Hopper 缺 s10-11(2)
    for env in "${ENVS[@]}"; do
        for batch in "0,1,2,3" "4,5,6,7" "8,9,10,11"; do
            wait_for_slot; launch "$env" "BHVF_V5" "$batch"
        done
    done

    # V6: Hopper 缺 s10-11(2), Walker 缺 s10-11(2), Ant 缺 s2-11(10)
    for batch in "0,1,2,3" "4,5,6,7" "8,9,10,11"; do
        wait_for_slot; launch "Hopper-v4"   "BHVF_V6" "$batch"
        wait_for_slot; launch "Walker2d-v4" "BHVF_V6" "$batch"
        wait_for_slot; launch "Ant-v4"      "BHVF_V6" "$batch"
    done

    # V7: Hopper/Walker/Ant 全缺, HC 缺 s4-11(8)
    for env in "${ENVS[@]}"; do
        for batch in "0,1,2,3" "4,5,6,7" "8,9,10,11"; do
            wait_for_slot; launch "$env" "BHVF_V7" "$batch"
        done
    done

    # V8/V9/V10: 全批次投喂，正在运行的由 skip 保护，新的按序启动
    for algo in "BHVF_V8" "BHVF_V9" "BHVF_V10"; do
        for env in "${ENVS[@]}"; do
            for batch in "0,1,2,3" "4,5,6,7" "8,9,10,11"; do
                wait_for_slot; launch "$env" "$algo" "$batch"
            done
        done
    done

elif [ "$MODE" = "v10only" ]; then
    echo "=== V10 全量实验 (4env × 12seed × 1M steps, Parallel=$MAX_PARALLEL) ==="
    echo "V10: SCR-Squared Shrinkage (James-Stein optimal, zero hyperparameters)"
    echo ""
    for env in "${ENVS[@]}"; do
        for batch in "0,1,2,3" "4,5,6,7" "8,9,10,11"; do
            wait_for_slot; launch "$env" "BHVF_V10" "$batch"
        done
    done

elif [ "$MODE" = "v6v8" ]; then
    echo "=== V6/V8 补跑模式 (Parallel=$MAX_PARALLEL) ==="
    echo ""
    # V6 缺失
    for batch in "6,7,8,9,10,11"; do
        for env in "Hopper-v4" "Walker2d-v4"; do
            wait_for_slot; launch "$env" "BHVF_V6" "$batch"
        done
    done
    wait_for_slot; launch "HalfCheetah-v4" "BHVF_V6" "9,10,11"
    for batch in "0,1,2,3" "4,5,6,7" "8,9,10,11"; do
        wait_for_slot; launch "Ant-v4" "BHVF_V6" "$batch"
    done
    # V8 全量
    for env in "${ENVS[@]}"; do
        for batch in "0,1,2,3" "4,5,6,7" "8,9,10,11"; do
            wait_for_slot; launch "$env" "BHVF_V8" "$batch"
        done
    done
    # V9 全量
    for env in "${ENVS[@]}"; do
        for batch in "0,1,2,3" "4,5,6,7" "8,9,10,11"; do
            wait_for_slot; launch "$env" "BHVF_V9" "$batch"
        done
    done

elif [ "$MODE" = "v8only" ]; then
    echo "=== V8 全量实验 (4env × 12seed × 1M steps, Parallel=$MAX_PARALLEL) ==="
    echo ""
    for env in "${ENVS[@]}"; do
        for batch in "0,1,2,3" "4,5,6,7" "8,9,10,11"; do
            wait_for_slot; launch "$env" "BHVF_V8" "$batch"
        done
    done

elif [ "$MODE" = "v9only" ]; then
    echo "=== V9 全量实验 (4env × 12seed × 1M steps, Parallel=$MAX_PARALLEL) ==="
    echo "V9: Separated Actor/Critic — Actor EV-adaptive MC injection + V6 Critic"
    echo ""
    for env in "${ENVS[@]}"; do
        for batch in "0,1,2,3" "4,5,6,7" "8,9,10,11"; do
            wait_for_slot; launch "$env" "BHVF_V9" "$batch"
        done
    done

else
    ALGOS=("Standard_PPO" "Optimal_PPO" "Heuristic_HCGAE" "BHVF" "BHVF_DCPPO" "BHVF_V6" "BHVF_V8" "BHVF_V9")
    echo "=== ICML 2026 Full Experiment (Parallel=$MAX_PARALLEL) ==="
    echo "Envs: ${ENVS[@]}"
    echo "Algos: ${ALGOS[@]}"
    echo ""
    for env in "${ENVS[@]}"; do
        for algo in "${ALGOS[@]}"; do
            for batch in "0,1,2,3" "4,5,6,7" "8,9,10,11"; do
                wait_for_slot; launch "$env" "$algo" "$batch"
            done
        done
    done
fi

echo ""
echo "Waiting for all tasks to finish..."
wait
echo ""
echo "=== All done! ==="
echo ""
$PYTHON "$SCRIPT" --summary

