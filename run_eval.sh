#!/bin/bash
# =============================================================================
# 一体化脚本: 生成 Infinite-World 视频 → WBench 评测
#
# 只负责编排：计算一次带时间戳的 OUT_DIR / LOG_FILE，依次调用
#   gen.sh  (阶段 1: 生成视频)  和  eval.sh (阶段 2: 评测)，
# 两阶段共用同一 OUT_DIR，保证评测读取的正是刚生成的视频
# (WBench/work_dirs/<OUT_DIR>/videos/)。两个子脚本也可单独运行。
#
# 用法: bash run_eval.sh [gen_script] [num_gpus] [num_cases] [online]
#   gen_script : 生成阶段运行的 python 文件，默认 generate_video.py；
#                其去掉 .py 的基名同时作为模型名 / 输出目录名
#   num_gpus   : GPU 数量，默认 1（=1 直接 python；>1 用 torchrun）
#   num_cases  : 取 dataset 前 N 个 case，默认 6，0 表示全部
#   online     : online (test-time) training 开关 on/off，默认 off
#
# 示例:
#   bash run_eval.sh                              # 默认脚本，单 GPU
#   bash run_eval.sh generate_video.py 8          # 8 GPU
#   bash run_eval.sh generate_video.py 1 2 on     # 单 GPU，前 2 个 case，开 online
# 多 GPU 端口冲突(EADDRINUSE)时: export MASTER_PORT=29500
# =============================================================================

set -o pipefail

# ----------------------------- 参数 ------------------------------------------
GEN_SCRIPT=${1:-generate_video.py}
NUM_GPUS=${2:-1}
NUM_CASES=${3:-6}
ONLINE=${4:-off}
# 模型名 / 输出目录名：取 gen_script 去掉 .py 的基名
MODEL=$(basename "$GEN_SCRIPT" .py)

TTT_ROOT=/root/autodl-tmp/ttt
LOG_DIR="$TTT_ROOT/logs"
mkdir -p "$LOG_DIR"

# 时间戳只算一次，OUT_DIR 与 LOG_FILE 由两阶段共享
TS=$(date +%Y-%m-%d_%H-%M-%S)
OUT_DIR="${MODEL}_${TS}"
LOG_FILE="$LOG_DIR/${OUT_DIR}.log"

echo "==============================================" | tee "$LOG_FILE"
echo "Infinite-World → WBench  一体化运行+评测" | tee -a "$LOG_FILE"
echo "==============================================" | tee -a "$LOG_FILE"
echo "model: $MODEL | GPUs: $NUM_GPUS | num_cases: $NUM_CASES (0=all) | online: $ONLINE" | tee -a "$LOG_FILE"
echo "gen:   $GEN_SCRIPT | out_dir: $OUT_DIR" | tee -a "$LOG_FILE"
echo "logs:  $LOG_FILE" | tee -a "$LOG_FILE"
echo "==============================================" | tee -a "$LOG_FILE"

# ======================= 阶段 1: 生成视频 ====================================
# 通过命令行内联环境变量把共享的 OUT_DIR / LOG_FILE 传给 gen.sh（不使用 export）
OUT_DIR="$OUT_DIR" LOG_FILE="$LOG_FILE" \
    bash "$TTT_ROOT/gen.sh" "$GEN_SCRIPT" "$NUM_GPUS" "$NUM_CASES" "$ONLINE"
GEN_STATUS=$?
if [ "$GEN_STATUS" -ne 0 ]; then
    echo "生成阶段失败 (exit $GEN_STATUS)，终止评测" | tee -a "$LOG_FILE"
    exit "$GEN_STATUS"
fi

# ======================= 阶段 2: 评测 ========================================
# OUT_DIR / LOG_FILE 作为参数传给 eval.sh
bash "$TTT_ROOT/eval.sh" "$OUT_DIR" "$LOG_FILE"
EVAL_STATUS=$?
if [ "$EVAL_STATUS" -ne 0 ]; then
    echo "评测阶段失败 (exit $EVAL_STATUS)" | tee -a "$LOG_FILE"
    exit "$EVAL_STATUS"
fi

echo "" | tee -a "$LOG_FILE"
echo "全部完成: $OUT_DIR" | tee -a "$LOG_FILE"
