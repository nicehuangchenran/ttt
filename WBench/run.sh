#!/bin/bash
# 用法: bash run.sh <ID>
#   例: bash run.sh infer-2.2-wbench
set -o pipefail

ID="$1"
if [ -z "$ID" ]; then
    echo "用法: bash run.sh <ID>"
    exit 1
fi

cd /mnt/efs/chenran/ttt/WBench
source /mnt/efs/chenran/miniconda3/etc/profile.d/conda.sh
conda activate wbench-main
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

LOG="logs/$ID.logs"
NOTIFY=/mnt/efs/chenran/claude_notify.sh
mkdir -p logs

# 分阶段跑，避免 --phase all 重复触发 precompute
run() {  # run <描述> <tee参数> <main.py参数...>
    local desc="$1" tee_flag="$2"; shift 2
    echo "── $desc ──"
    if ! python main.py --model "$ID" "$@" 2>&1 | tee $tee_flag "$LOG"; then
        "$NOTIFY" "wbench $desc $ID" "失败"
        exit 1
    fi
}

run precompute ""        --phase precompute --skip_sam2 --skip_da3
run gpu-consistency "-a" --phase gpu --metrics consistency
run report "-a"          --phase report

"$NOTIFY" "wbench consistency 指标评测 $ID" "成功"
echo "完成: $LOG + work_dirs/$ID/evaluation/report.json"
