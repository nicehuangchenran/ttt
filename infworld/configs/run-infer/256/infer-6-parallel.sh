#!/bin/bash
# 三个 infer 并行：base 2 卡，sft / ttt 各 3 卡，共占满 8 卡

export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # 用可扩展显存段，回收每个 chunk 调 empty_cache 造成的碎片

CONFIG_BASE="configs/run-infer/256/infer-6-base-sekai.yaml"
CONFIG_SFT="configs/run-infer/256/infer-6-sft-sekai.yaml"
CONFIG_TTT="configs/run-infer/256/infer-6-ttt-sekai.yaml"
NOTIFY=/mnt/efs/chenran/bin/notify
TORCHRUN=/mnt/efs/chenran/miniconda3/envs/infworld/bin/torchrun  # 绝对路径：nohup 起的 shell 没有 conda 环境

# 一个任务：指定 GPU 列表 + 卡数 + 独立 master_port，成功/失败都发通知
run_infer() {
    local gpus=$1 nproc=$2 port=$3 config=$4
    local name=$(basename "${config}" .yaml)
    echo "推理: ${config}  (GPU ${gpus}, ${nproc} 卡)"
    if CUDA_VISIBLE_DEVICES="${gpus}" "${TORCHRUN}" --nproc_per_node="${nproc}" --master_port="${port}" \
        --local-ranks-filter=0 scripts/infer.py --config "${config}"; then
        "${NOTIFY}" "${name}" "成功"
    else
        "${NOTIFY}" "${name}" "失败"
    fi
}

echo "===== 开始并行推理 $(date) ====="

run_infer 0,1     2 29501 "${CONFIG_BASE}" &
run_infer 2,3,4   3 29502 "${CONFIG_SFT}"  &
run_infer 5,6,7   3 29503 "${CONFIG_TTT}"  &
wait

echo "===== 推理结束 $(date) ====="
