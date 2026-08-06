#!/bin/bash

# 配置参数
BASH_PID=1338409                    # 要终止的旧脚本进程ID
CONFIG_NAME="train-5.7.yaml"        # 监控的训练任务名
NEW_SCRIPT="/mnt/efs/chenran/ttt/infworld/configs/runs/train/256/train-5.7and5.8-add.sh"  # 后续运行的新脚本

echo "===== 监控脚本启动 $(date) ====="

# 1. 等待指定训练任务开始
echo "等待 ${CONFIG_NAME} 训练开始..."
while ! pgrep -f "${CONFIG_NAME}" > /dev/null; do
    sleep 10
done

# 2. 等待训练任务完成
echo "[$(date)] ${CONFIG_NAME} 已启动，等待完成..."
while pgrep -f "${CONFIG_NAME}" > /dev/null; do
    sleep 10
done

# 3. 训练完成后，终止旧脚本（防止启动 gpu_burn）
echo "[$(date)] ${CONFIG_NAME} 已完成！终止旧脚本进程 ${BASH_PID}..."
kill ${BASH_PID} 2>/dev/null
sleep 2
kill -9 ${BASH_PID} 2>/dev/null  # 确保终止

# 4. 启动新脚本
echo "[$(date)] 启动新脚本: ${NEW_SCRIPT}"
bash "${NEW_SCRIPT}"

echo "===== 监控脚本完成 $(date) ====="
