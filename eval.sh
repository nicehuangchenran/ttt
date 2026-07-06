#!/bin/bash
# =============================================================================
# 阶段 2: WBench 评测
#
# 读取 WBench/work_dirs/<OUT_DIR>/videos/ 下的视频进行评测。
#
# 用法: bash eval.sh <out_dir> [log_file]
#   out_dir  : 待评测的（带时间戳的）输出目录名；缺失则报错退出。
#   log_file : 可选，日志文件路径，缺省自动生成。
#
# 单独运行示例:
#   bash eval.sh infworld-online-cut21_2026-07-05_17-04-00
# =============================================================================

set -o pipefail

TTT_ROOT=/root/autodl-tmp/ttt
WBENCH_DIR="$TTT_ROOT/WBench"
LOG_DIR="$TTT_ROOT/logs"

OUT_DIR=$1
if [ -z "$OUT_DIR" ]; then
    echo "错误: 未提供 out_dir，用法: bash eval.sh <out_dir> [log_file]" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
LOG_FILE=${2:-$LOG_DIR/${OUT_DIR}.log}

echo "==============================================" | tee -a "$LOG_FILE"
echo "阶段 2/2: 评测 (WBench)" | tee -a "$LOG_FILE"
echo "==============================================" | tee -a "$LOG_FILE"
echo "out_dir: $OUT_DIR" | tee -a "$LOG_FILE"
echo "logs:    $LOG_FILE" | tee -a "$LOG_FILE"
echo "==============================================" | tee -a "$LOG_FILE"

# 准备 conda
source /root/miniconda3/etc/profile.d/conda.sh
conda activate wbench-main
cd "$WBENCH_DIR"

python main.py --model "$OUT_DIR" --phase precompute --skip_da3 --skip_sam2 2>&1 | tee -a "$LOG_FILE"

python main.py --model "$OUT_DIR" --phase gpu --metrics consistency --skip_da3 --skip_sam2 --skip_megasam 2>&1 | tee -a "$LOG_FILE"

python main.py --model "$OUT_DIR" --phase report 2>&1 | tee -a "$LOG_FILE"

echo "评测完成: $OUT_DIR" | tee -a "$LOG_FILE"
