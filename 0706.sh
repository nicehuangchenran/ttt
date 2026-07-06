#!/bin/bash
# =============================================================================
# 0706.sh: ① eval.sh 评测已有视频 → ② gen.sh 生成新视频 → ③ eval.sh 评测新视频
# 无传参，要改就直接改下面几个变量。任一步失败即终止（set -e）。
# =============================================================================
set -eo pipefail

TTT_ROOT=/root/autodl-tmp/ttt
EVAL_DIR=generate_video_2026-07-06_10-39-50   # 任务① 待评测的已有目录
GEN_SCRIPT=generate_video.py
NUM_GPUS=1
NUM_CASES=6      # 0=全部
ONLINE=off

# ① 评测已有视频
bash "$TTT_ROOT/eval.sh" "$EVAL_DIR"

# ②③ 生成新视频并评测：共用同一 OUT_DIR（gen.sh 的日志名也随之一致）
OUT_DIR="$(basename "$GEN_SCRIPT" .py)_$(date +%Y-%m-%d_%H-%M-%S)"
OUT_DIR="$OUT_DIR" bash "$TTT_ROOT/gen.sh" "$GEN_SCRIPT" "$NUM_GPUS" "$NUM_CASES" "$ONLINE"
bash "$TTT_ROOT/eval.sh" "$OUT_DIR"

echo "全部完成: 已评 $EVAL_DIR + 新生成并评测 $OUT_DIR"
