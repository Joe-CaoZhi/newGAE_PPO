#!/bin/bash
# V6/V8 补跑脚本 — 补齐 V6 缺失种子 + V8 全量 (4env × 12seed × 1M steps)
# 用法: ./run_v6v8_experiment.sh [并发数，默认4]

SCRIPT="/Users/joe-caozhi/newGAE_ppo/run_icml_experiment.py"
PYTHON="python3"
MAX_PARALLEL=${1:-4}
LOG_DIR="/Users/joe-caozhi/newGAE_ppo/logs/FinalExperiment"
mkdir -p "$LOG_DIR"

pids=()

wait_for_slot() {
    while true; do
        running=0
        for pid in "${pids[@]}"; do
            kill -0 "$pid" 2>/dev/null && ((running++))
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
    sleep 0.3
}

echo "=== V6/V8 全面实验 (Parallel=$MAX_PARALLEL) ==="
echo "V6: 补跑缺失种子"
echo "V8: 全量 4env × 12seed × 1M steps"
echo ""

# ── V6 补跑缺失种子 ───────────────────────────────────────────────────────
# Hopper: 缺 6-11
wait_for_slot; launch "Hopper-v4" "BHVF_V6" "6,7,8,9,10,11"
# Walker2d: 缺 6-11
wait_for_slot; launch "Walker2d-v4" "BHVF_V6" "6,7,8,9,10,11"
# HalfCheetah: 缺 9-11
wait_for_slot; launch "HalfCheetah-v4" "BHVF_V6" "9,10,11"
# Ant: 全缺 0-11
wait_for_slot; launch "Ant-v4" "BHVF_V6" "0,1,2,3"
wait_for_slot; launch "Ant-v4" "BHVF_V6" "4,5,6,7"
wait_for_slot; launch "Ant-v4" "BHVF_V6" "8,9,10,11"

# ── V8 全量 (4 env × 3 batch) ────────────────────────────────────────────
for env in "Hopper-v4" "Walker2d-v4" "HalfCheetah-v4" "Ant-v4"; do
    wait_for_slot; launch "$env" "BHVF_V8" "0,1,2,3"
    wait_for_slot; launch "$env" "BHVF_V8" "4,5,6,7"
    wait_for_slot; launch "$env" "BHVF_V8" "8,9,10,11"
done

# ── V9 全量 (4 env × 3 batch) ────────────────────────────────────────────
# V9: Separated Actor/Critic Control — Actor EV-adaptive MC injection + V6 Critic
for env in "Hopper-v4" "Walker2d-v4" "HalfCheetah-v4" "Ant-v4"; do
    wait_for_slot; launch "$env" "BHVF_V9" "0,1,2,3"
    wait_for_slot; launch "$env" "BHVF_V9" "4,5,6,7"
    wait_for_slot; launch "$env" "BHVF_V9" "8,9,10,11"
done

echo ""
echo "All tasks launched. Waiting for completion..."
wait
echo ""
echo "=== All done! ==="
echo ""
$PYTHON "$SCRIPT" --summary

