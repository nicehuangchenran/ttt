#!/bin/bash
# real_hist（teacher forcing）推理：dl3dv2 上分别跑 sft / ttt 的 steps800 权重。
# wbench 两个 -tf 配置没有真值 latent，real_hist 跑不了，故不在此脚本内。

export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # 用可扩展显存段，回收每个 chunk 调 empty_cache 造成的碎片

CONFIG_1="configs/run-infer/256/infer-10-dl3dv2-sft-steps800-tf.yaml"
CONFIG_2="configs/run-infer/256/infer-10-dl3dv2-ttt-steps800-tf.yaml"
NOTIFY=notify
# 绝对路径：非交互 shell（脚本被直接 bash 调起）里 conda 环境不在 PATH 上
TORCHRUN=/mnt/efs/chenran/miniconda3/envs/infworld/bin/torchrun

# 一个任务：4 卡 torchrun，独立 master_port，成功/失败都发通知
run_infer() {
    local gpus=$1 port=$2 config=$3
    local name=$(basename "${config}" .yaml)
    echo "推理: ${config}  (GPU ${gpus})"
    if CUDA_VISIBLE_DEVICES="${gpus}" "${TORCHRUN}" --nproc_per_node=4 --master_port="${port}" \
        --local-ranks-filter=0 scripts/infer.py --config "${config}"; then
        "${NOTIFY}" "${name}" "成功"
    else
        "${NOTIFY}" "${name}" "失败"
    fi
}

echo "===== 开始并行推理 $(date) ====="

run_infer 0,1,2,3 29511 "${CONFIG_1}" &
run_infer 4,5,6,7 29512 "${CONFIG_2}" &
wait

echo "===== 推理结束 $(date) ====="
