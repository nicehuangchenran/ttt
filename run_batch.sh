#!/bin/bash
# =============================================================================
# 批量运行 run_eval.sh
# 每行一组参数，按顺序串行执行（共用同一块 GPU，串行最安全）
#
# 用法: bash run_batch.sh
# =============================================================================
set -o pipefail

cd /root/autodl-tmp/ttt

# 每行格式: "MODEL NUM_GPUS NUM_CASES ONLINE GEN_SCRIPT"
#   MODEL      : 模型名 / 输出目录名
#   NUM_GPUS   : GPU 数量 (1=python, >1=torchrun)
#   NUM_CASES  : 取前 N 个 case，0=全部
#   ONLINE     : online (test-time) training 开关 on/off
#   GEN_SCRIPT : 生成阶段运行的 python 文件
JOBS=(
    "infworld-online-cut21   3 6 on  generate_video_cut21.py"
    "infworld-offline-cut21  3 6 off generate_video_cut21.py"
)

TOTAL=${#JOBS[@]}
i=0
for job in "${JOBS[@]}"; do
    i=$((i + 1))
    read -r MODEL NUM_GPUS NUM_CASES ONLINE GEN_SCRIPT <<< "$job"
    echo "###################################################"
    echo "### [$i/$TOTAL] 开始: MODEL=$MODEL GPUS=$NUM_GPUS CASES=$NUM_CASES ONLINE=$ONLINE GEN=$GEN_SCRIPT"
    echo "###################################################"
    bash run_eval.sh "$MODEL" "$NUM_GPUS" "$NUM_CASES" "$ONLINE" "$GEN_SCRIPT"
    echo "### [$i/$TOTAL] 完成: $MODEL (exit=$?)"
    echo ""
done

echo "全部任务完成 ($TOTAL 个)"
