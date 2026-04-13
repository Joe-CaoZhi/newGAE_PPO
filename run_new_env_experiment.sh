#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# 新环境 V4 扩展实验启动脚本
#
# 实验设计:
#   第一轮 (4 seeds): Standard_PPO / Optimal_PPO / BHVF_V4
#   若 V4 > Optimal_PPO，扩展到 12 seeds
#
# 目标环境:
#   Humanoid-v4  ── 376-dim obs, 17-dim act，预测 V4 有优势（类似 Walker2d）
#   Swimmer-v4   ── 8-dim obs, 2-dim act，快速验证
#
# 用法:
#   bash run_new_env_experiment.sh              # 4 seeds 快速验证
#   bash run_new_env_experiment.sh --full       # 12 seeds 完整实验
#   bash run_new_env_experiment.sh --env Humanoid-v4  # 单环境
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT="/Users/joe-caozhi/newGAE_ppo/run_icml_experiment.py"
PYTHON="python3"
LOG_DIR="/Users/joe-caozhi/newGAE_ppo/logs/NewEnvExperiment"
RESULTS_DIR="results/NewEnvExperiment"

mkdir -p "$LOG_DIR"

# ── 参数解析 ──────────────────────────────────────────────────────────────────
FULL_MODE=false
SINGLE_ENV=""
SEEDS="0,1,2,3"   # 默认 4 seeds (第一轮)
ALGOS=("Standard_PPO" "Optimal_PPO" "BHVF_V4")

for arg in "$@"; do
    case $arg in
        --full)
            FULL_MODE=true
            SEEDS="0,1,2,3,4,5,6,7,8,9,10,11"
            echo "[INFO] Full mode: 12 seeds"
            ;;
        --env)
            shift
            SINGLE_ENV="$1"
            ;;
        --env=*)
            SINGLE_ENV="${arg#*=}"
            ;;
        --seeds=*)
            SEEDS="${arg#*=}"
            ;;
    esac
done

if [ -n "$SINGLE_ENV" ]; then
    ENVS=("$SINGLE_ENV")
else
    ENVS=("Humanoid-v4" "Swimmer-v4")
fi

# ── 打印配置 ──────────────────────────────────────────────────────────────────
echo "========================================================"
echo "  BHVF V4 新环境扩展实验"
echo "========================================================"
echo "  环境:    ${ENVS[@]}"
echo "  算法:    ${ALGOS[@]}"
echo "  Seeds:   $SEEDS"
echo "  Steps:   1,000,000"
echo "  Results: $RESULTS_DIR"
echo "  Logs:    $LOG_DIR"
echo "========================================================"
echo ""

# ── 启动实验 ──────────────────────────────────────────────────────────────────
PIDS=()

for env in "${ENVS[@]}"; do
    for algo in "${ALGOS[@]}"; do
        log_file="$LOG_DIR/${env}_${algo}_s${SEEDS//,/-}.log"
        echo "[START] $env / $algo  seeds=$SEEDS"
        nohup $PYTHON "$SCRIPT" \
            --env "$env" \
            --algo "$algo" \
            --seeds "$SEEDS" \
            --results-dir "$RESULTS_DIR" \
            > "$log_file" 2>&1 &
        PID=$!
        PIDS+=("$PID")
        echo "        PID=$PID  log=$log_file"
        sleep 0.3
    done
done

echo ""
echo "  已启动 ${#PIDS[@]} 个进程，后台运行中..."
echo "  查看进度: python3 check_new_env_progress.py"
echo ""

