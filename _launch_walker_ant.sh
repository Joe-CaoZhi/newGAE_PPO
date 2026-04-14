#!/bin/bash
# 等待 Hopper HCGAE_Standard_GRPO 当前 8 个 seed 完成后，
# 启动 Walker2d 和 Ant 的全套 GRPO 实验
# 用法：nohup bash _launch_walker_ant.sh > logs/walker_ant_launch.log 2>&1 &

cd /Users/joe-caozhi/newGAE_ppo

echo "=== [$(date)] 开始监控 Hopper HCGAE_Standard_GRPO seeds 5-12 ==="

# 等待当前 8 个 Hopper HCGAE_Standard_GRPO 进程全部结束
while true; do
    RUNNING=$(ps aux | grep "HCGAE_Standard_GRPO" | grep -v grep | wc -l)
    if [ "$RUNNING" -eq 0 ]; then
        echo "[$(date)] Hopper HCGAE_Standard_GRPO 全部完成"
        break
    fi
    echo "[$(date)] 仍有 $RUNNING 个 Hopper 进程运行中，等待 5 分钟..."
    sleep 300
done

echo ""
echo "=== [$(date)] 启动 Walker2d-v4 全套 GRPO 实验 (4算法×15seeds×1.5M步) ==="
python3 run_grpo_experiment.py \
    --envs Walker2d-v4 \
    --algos Standard_GRPO_NoTrick Optimal_GRPO HCGAE_Standard_GRPO HCGAE_Optimal_GRPO \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 \
    --steps 1500000 \
    --workers 8 \
    --results-dir results/GRPO

echo ""
echo "=== [$(date)] Walker2d 完成，启动 Ant-v4 全套 GRPO 实验 ==="
python3 run_grpo_experiment.py \
    --envs Ant-v4 \
    --algos Standard_GRPO_NoTrick Optimal_GRPO HCGAE_Standard_GRPO HCGAE_Optimal_GRPO \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 \
    --steps 1500000 \
    --workers 8 \
    --results-dir results/GRPO

echo ""
echo "=== [$(date)] 全部 Walker2d + Ant GRPO 实验完成 ==="

