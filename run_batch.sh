#!/bin/bash
# =============================================================================
# 批量运行 run_eval.sh
# 每行一组参数，按顺序串行执行（共用同一块 GPU，串行最安全）
#
# 用法: bash run_batch.sh
# =============================================================================
set -o pipefail

cd /root/autodl-tmp/ttt
# 动态探测当前可用 CPU 核心数（容器/cgroup 限制下 nproc 已是真实可用数，会随机器变化）
CPUS=$(nproc)
LOG_DIR=/root/autodl-tmp/ttt/logs
mkdir -p "$LOG_DIR"

# 每行格式: "MODEL NUM_GPUS NUM_CASES ONLINE GEN_SCRIPT"
#   MODEL      : 模型名 , work_dir输出目录名加时间后缀
#   NUM_GPUS   : GPU 数量 (1=python, >1=torchrun)
#   NUM_CASES  : 取前 N 个 case，0=全部
#   ONLINE     : online (test-time) training 开关 on/off
#   GEN_SCRIPT : 生成阶段运行的 python 文件
JOBS=(
    "infworld-online-cut21   3 6 on  generate_video_cut21.py"
    "infworld-offline-cut21  3 6 off generate_video_cut21.py"
)

# -----------------------------------------------------------------------------
# 断线续跑: 若当前不在 screen 会话内, 则在后台 detached screen 中重新启动自己,
# 这样远程 SSH 断开后任务依旧继续运行。会话名取第一个 job 的 MODEL。
#   实时查看: screen -r <MODEL>
#   分离会话: Ctrl-A 然后按 D
# -----------------------------------------------------------------------------
read -r SESSION _ <<< "${JOBS[0]}"   # 用第一个 job 的 MODEL 作为 screen 会话名
if [ -z "$STY" ]; then
    SCREEN_LOG="$LOG_DIR/batch_$(date +%Y%m%d_%H%M%S).log"
    echo "已在后台 screen 会话 '$SESSION' 中启动任务 (SSH 断开也会继续跑)"
    echo "  实时查看: screen -r $SESSION"
    echo "  分离会话: Ctrl-A 然后按 D"
    echo "  会话日志: $SCREEN_LOG"
    exec screen -dmS "$SESSION" -L -Logfile "$SCREEN_LOG" bash "$0" "$@"
fi

TOTAL=${#JOBS[@]}
i=0
for job in "${JOBS[@]}"; do
    i=$((i + 1))
    read -r MODEL NUM_GPUS NUM_CASES ONLINE GEN_SCRIPT <<< "$job"
    # 按本 job 的 GPU 数计算最优 OMP_NUM_THREADS = 可用核数 / 进程数(=NUM_GPUS)，至少为 1
    # torchrun 会启 NUM_GPUS 个进程，避免线程超额订阅 CPU 反而拖慢
    export OMP_NUM_THREADS=$(( CPUS / NUM_GPUS > 0 ? CPUS / NUM_GPUS : 1 ))
    echo "###################################################"
    echo "### [$i/$TOTAL] 开始: MODEL=$MODEL GPUS=$NUM_GPUS CASES=$NUM_CASES ONLINE=$ONLINE GEN=$GEN_SCRIPT OMP_NUM_THREADS=$OMP_NUM_THREADS"
    echo "###################################################"
    SECONDS=0
    bash run_eval.sh "$MODEL" "$NUM_GPUS" "$NUM_CASES" "$ONLINE" "$GEN_SCRIPT"
    STATUS=$?
    ELAPSED=$SECONDS
    DURATION=$(printf '%02d:%02d:%02d' $((ELAPSED / 3600)) $(((ELAPSED % 3600) / 60)) $((ELAPSED % 60)))
    echo "### [$i/$TOTAL] 完成: $MODEL (exit=$STATUS, 用时=$DURATION)"
    # 把本 job 的运行时间写入 run_eval.sh 为该 MODEL 生成的最新 log 文件
    JOB_LOG=$(ls -t "$LOG_DIR/${MODEL}_"*.log 2>/dev/null | head -n 1)
    if [ -n "$JOB_LOG" ]; then
        echo "" >> "$JOB_LOG"
        echo "### 本 job 运行时间: ${DURATION} (${ELAPSED}s) | exit=$STATUS" >> "$JOB_LOG"
    fi
    echo ""
done

echo "全部任务完成 ($TOTAL 个)"
