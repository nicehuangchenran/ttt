#!/bin/bash
# =============================================================================
# 阶段 1: 生成 Infinite-World 视频 (infworld)
#
# 视频落盘到 WBench/work_dirs/<RUN_MODEL>/videos/，供 eval.sh 评测读取。
#
# 用法: bash gen.sh [gen_script] [num_gpus] [num_cases] [online]
#   gen_script : 生成阶段运行的 python 文件，默认 generate_video.py；
#                其去掉 .py 的基名同时作为模型名 / 输出目录名
#   num_gpus   : GPU 数量，默认 1（=1 直接 python；>1 用 torchrun）
#   num_cases  : 取 dataset 前 N 个 case，默认 6，0 表示全部
#   online     : online (test-time) training 开关 on/off，默认 off
#
# 由 run_eval.sh 调用时，可通过环境变量 OUT_DIR / LOG_FILE 复用同一
# 带时间戳的目录名与日志，使评测读取的正是刚生成的视频；单独运行时会
# 自行生成 <gen_script 基名>_<时间戳>。
#
# 单独运行示例:
#   bash gen.sh generate_video.py 1 2 on
# 多 GPU 端口冲突(EADDRINUSE)时: export MASTER_PORT=29500
# =============================================================================

set -o pipefail # 只要有命令失败，整个管道失败
# ----------------------------- 参数 ------------------------------------------
GEN_SCRIPT=${1:-generate_video.py}
NUM_GPUS=${2:-1}
NUM_CASES=${3:-6}
ONLINE=${4:-off}


# ----------------------------- 变量准备 ------------------------------------------
TTT_ROOT=/root/autodl-tmp/ttt
INFWORLD_DIR="$TTT_ROOT/infworld"
LOG_DIR="$TTT_ROOT/logs"
MODEL=$(basename "$GEN_SCRIPT" .py)  # 模型名 / 输出目录名：取 gen_script 去掉 .py 的基名
INPUT_DATASET=${INPUT_DATASET:-$INFWORLD_DIR/dataset/wbench}
# 时间戳只算一次，保证 log 名与 model 名带的日期完全一致
TS=$(date +%Y-%m-%d_%H-%M-%S)
# 带日期的 model 名：优先复用 run_eval.sh 传入的环境变量，保持生成/评测一致
OUT_DIR=${OUT_DIR:-${MODEL}_${TS}}
LOG_FILE=${LOG_FILE:-$LOG_DIR/${OUT_DIR}.log}

mkdir -p "$LOG_DIR"
echo "==============================================" | tee -a "$LOG_FILE"
echo "阶段 1/2: 生成视频 (infworld)" | tee -a "$LOG_FILE"
echo "==============================================" | tee -a "$LOG_FILE"
echo "input_dataset: $INPUT_DATASET | out_dir: $OUT_DIR" | tee -a "$LOG_FILE"
echo "gen:   $GEN_SCRIPT | GPUs: $NUM_GPUS | num_cases: $NUM_CASES (0=all) | online: $ONLINE" | tee -a "$LOG_FILE"
echo "==============================================" | tee -a "$LOG_FILE"

# 准备 conda
source /root/miniconda3/etc/profile.d/conda.sh
conda activate infworld
cd "$INFWORLD_DIR" # 当前cwd
# 确保无论 GEN_SCRIPT 放在哪，都能导入 infworld / scripts 等包
export PYTHONPATH="$INFWORLD_DIR:$PYTHONPATH"

# 生成阶段 CLI 参数（generate_video.py 从命令行读取，不再用环境变量覆盖）
GEN_ARGS=(--input_dataset "$INPUT_DATASET" \
          --output_model_name "$OUT_DIR" \
          --num_cases "$NUM_CASES" \
          --online "$ONLINE")

if [ "$NUM_GPUS" -eq 1 ]; then
    # 运行 python <脚本> 时，Python 会把脚本所在目录（这里就是 $INFWORLD_DIR）自动加入sys.path[0]，不是运行python的cli目录（cd "$INFWORLD_DIR"）
    python "$GEN_SCRIPT" "${GEN_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
else
    MASTER_PORT=${MASTER_PORT:-29400}
    echo "MASTER_PORT: $MASTER_PORT" | tee -a "$LOG_FILE"
    torchrun --nnodes=1 --nproc_per_node="$NUM_GPUS" \
        --rdzv_id=100 --rdzv_backend=c10d \
        --rdzv_endpoint=localhost:"$MASTER_PORT" \
        "$GEN_SCRIPT" "${GEN_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
fi

GEN_STATUS=${PIPESTATUS[0]}
if [ "$GEN_STATUS" -ne 0 ]; then
    echo "生成阶段失败 (exit $GEN_STATUS)" | tee -a "$LOG_FILE"
    exit "$GEN_STATUS"
fi
echo "视频生成完成: $OUT_DIR" | tee -a "$LOG_FILE"
