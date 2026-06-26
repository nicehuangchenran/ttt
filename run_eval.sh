#!/bin/bash
# =============================================================================
# 一体化脚本: 生成 Infinite-World 视频 → WBench 评测
#
# 两个阶段共用同一个模型名 MODEL，保证评测读取的正是刚生成的视频
# (落盘到 WBench/work_dirs/<MODEL>/videos/)。
#
# 用法: bash run_eval.sh [model_name] [num_gpus] [num_cases] [online] [gen_script]
#   model_name : 模型名 / 输出目录名，默认 infworld-online-cut21
#   num_gpus   : GPU 数量，默认 1（=1 直接 python；>1 用 torchrun）
#   num_cases  : 取 dataset 前 N 个 case，默认 0 表示全部
#   online     : online (test-time) training 开关 on/off，默认 off
#   gen_script : 生成阶段运行的 python 文件，默认 generate_video.py
#
# 示例:
#   bash run_eval.sh                              # 默认模型，单 GPU，全部 case
#   bash run_eval.sh my-model 8                   # 8 GPU，全部 case
#   bash run_eval.sh my-model 1 2 on              # 单 GPU，前 2 个 case，开 online
# 多 GPU 端口冲突(EADDRINUSE)时: export MASTER_PORT=29500
# =============================================================================

set -o pipefail

# ----------------------------- 参数 ------------------------------------------
MODEL=${1:-infworld-online-cut21}
NUM_GPUS=${2:-1}
NUM_CASES=${3:-6}
ONLINE=${4:-off}
# 生成阶段运行的 python 文件
GEN_SCRIPT=${5:-generate_video.py}


TTT_ROOT=/root/autodl-tmp/ttt
INFWORLD_DIR="$TTT_ROOT/infworld"
WBENCH_DIR="$TTT_ROOT/WBench"
LOG_DIR="$TTT_ROOT/logs"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${MODEL}_$(date +%Y-%m-%d_%H-%M-%S).log"

echo "=============================================="
echo "Infinite-World → WBench  一体化运行+评测"
echo "=============================================="
echo "model: $MODEL | GPUs: $NUM_GPUS | num_cases: $NUM_CASES (0=all) | online: $ONLINE"
echo "gen:   $GEN_SCRIPT"
echo "logs:  $LOG_FILE"
echo "=============================================="

# 准备 conda
source /root/miniconda3/etc/profile.d/conda.sh

# ======================= 阶段 1: 生成视频 (infworld) ==========================
echo "" | tee  "$LOG_FILE"
echo "########## 阶段 1/2: 生成视频 (infworld) ##########" | tee -a "$LOG_FILE"

conda activate infworld
cd "$INFWORLD_DIR"

export NUM_CASES
export OUTPUT_MODEL_NAME="$MODEL"
export ONLINE_TRAINING="$ONLINE"

if [ "$NUM_GPUS" -eq 1 ]; then
    python "$GEN_SCRIPT" 2>&1 | tee -a "$LOG_FILE"
else
    MASTER_PORT=${MASTER_PORT:-29400}
    echo "MASTER_PORT: $MASTER_PORT" | tee -a "$LOG_FILE"
    torchrun --nnodes=1 --nproc_per_node="$NUM_GPUS" \
        --rdzv_id=100 --rdzv_backend=c10d \
        --rdzv_endpoint=localhost:"$MASTER_PORT" \
        "$GEN_SCRIPT" 2>&1 | tee -a "$LOG_FILE"
fi

GEN_STATUS=${PIPESTATUS[0]}
if [ "$GEN_STATUS" -ne 0 ]; then
    echo "生成阶段失败 (exit $GEN_STATUS)，终止评测" | tee -a "$LOG_FILE"
    exit "$GEN_STATUS"
fi
echo "视频生成完成" | tee -a "$LOG_FILE"

# ======================= 阶段 2: 评测 (WBench) ===============================
echo "" | tee -a "$LOG_FILE"
echo "########## 阶段 2/2: 评测 (WBench) ##########" | tee -a "$LOG_FILE"

conda activate wbench-main
cd "$WBENCH_DIR"

python main.py --model "$MODEL" --phase precompute --skip_da3 --skip_sam2 2>&1 | tee -a "$LOG_FILE"
echo "precompute 运行完成" | tee -a "$LOG_FILE"

python main.py --model "$MODEL" --phase gpu --metrics consistency --skip_da3 --skip_sam2 --skip_megasam 2>&1 | tee -a "$LOG_FILE"
echo "gpu 运行完成" | tee -a "$LOG_FILE"

python main.py --model "$MODEL" --phase report 2>&1 | tee -a "$LOG_FILE"
echo "生成 report.json" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "全部完成: $MODEL" | tee -a "$LOG_FILE"
