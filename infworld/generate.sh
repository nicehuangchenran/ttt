#!/bin/bash
# 自己写的，当时想批量读取 wbench 形成的dataset 中的数据
# 为 WBench 评测产出 Infinite-World 视频 (单/多 GPU)
# 用法: bash generate.sh [num_gpus] [num_cases] [model_name] [online]
#   num_gpus   : GPU 数量，默认 1（=1 时直接 python，避免 torchrun 端口冲突；>1 用 torchrun）
#   num_cases  : 取 dataset 前 N 个 case（按 case id 升序），默认 0 表示全部
#   model_name : 输出模型名，落盘到 WBench/work_dirs/<model_name>/videos/，默认 infworld
#   online     : online (test-time) training 开关 on/off，默认 off
#
# 示例:
#   bash generate.sh 1                    # 单 GPU，全部 case
#   bash generate.sh 8                    # 8 GPU，全部 case
#   bash generate.sh 8 10 infworld        # 8 GPU，仅前 10 个 case
#   bash generate.sh 1 2 infworld on      # 单 GPU，前 2 个 case，开启 online training
# 多 GPU 端口冲突(EADDRINUSE)时: export MASTER_PORT=29500

NUM_GPUS=${1:-1}
NUM_CASES=${2:-0}
MODEL_NAME=${3:-infworld-online06-26}
ONLINE=${4:-off}
INPUT_DATASET=${5:-dataset/long_case}

export NUM_CASES
export OUTPUT_MODEL_NAME="$MODEL_NAME"
export ONLINE_TRAINING="$ONLINE"
export INPUT_DATASET

echo "=============================================="
echo "Infinite World - Generate for WBench"
echo "=============================================="
echo "GPUs: $NUM_GPUS | num_cases: $NUM_CASES (0=all) | model: $MODEL_NAME | online: $ONLINE"

if [ -n "${CONDA_PREFIX:-}" ] && [ -f "$CONDA_PREFIX/etc/profile.d/conda.sh" ]; then
    source "$CONDA_PREFIX/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "conda not found. Please activate the infworld environment before running this script."
    exit 1
fi
conda activate infworld

if [ "$NUM_GPUS" -eq 1 ]; then
    python generate_video.py 2>&1 | tee logs/infworld-online06-26.log
else
    MASTER_PORT=${MASTER_PORT:-29400}
    echo "MASTER_PORT: $MASTER_PORT"
    torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS \
        --rdzv_id=100 --rdzv_backend=c10d \
        --rdzv_endpoint=localhost:$MASTER_PORT \
        generate_video.py
fi
