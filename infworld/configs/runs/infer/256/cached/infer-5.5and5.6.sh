#!/bin/bash

export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # 用可扩展显存段，回收每个 chunk 调 empty_cache 造成的碎片

CONFIG_1="configs/runs/infer/256/infer-5.5.yaml"
CONFIG_2="configs/runs/infer/256/infer-5.6.yaml"
NOTIFY=/mnt/efs/chenran/claude_notify.sh

# 一个任务：4 卡 torchrun，独立 master_port，成功/失败都发通知
run_infer() {
    local gpus=$1 port=$2 config=$3
    echo "推理: ${config}  (GPU ${gpus})"
    if CUDA_VISIBLE_DEVICES="${gpus}" torchrun --nproc_per_node=4 --master_port="${port}" \
        --local-ranks-filter=0 scripts/infer.py --config "${config}"; then
        "${NOTIFY}" 'aws上的实验' "${config} 完成"
    else
        "${NOTIFY}" 'aws上的实验' "${config} 失败"
    fi
}

echo "===== 开始并行推理 $(date) ====="

run_infer 0,1,2,3 29501 "${CONFIG_1}" &
run_infer 4,5,6,7 29502 "${CONFIG_2}" &
wait

echo "===== 推理结束 $(date) ====="

# # 启动 gpu_burn
# echo "启动 gpu_burn $(date)"
# "${NOTIFY}" 'aws上的实验' '推理完成，启动gpu_burn'
# while true; do
#     (cd /mnt/efs/chenran/gpu-burn && ./gpu_burn 3600)
#     echo "gpu_burn 继续 $(date)"
# done