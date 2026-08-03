#!/bin/bash

export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # 用可扩展显存段，回收每个 chunk 调 empty_cache 造成的碎片（OOM 报错里的 8.34GBreserved-but-unallocated

# 训练配置
TRAIN_CONFIG_1="configs/runs/train/256/train-5.8.yaml"
TRAIN_CONFIG_2="configs/runs/train/256/train-5.7.yaml"

echo "===== 开始顺序训练 $(date) ====="

# 任务 1
echo "训练: ${TRAIN_CONFIG_1}"
if torchrun --nproc_per_node=8 --master_port=29501  --local-ranks-filter=0 \
    scripts/train.py --config "${TRAIN_CONFIG_1}"; then
    /mnt/efs/chenran/claude_notify.sh "aws ${TRAIN_CONFIG_1}" "完成"
else
    /mnt/efs/chenran/claude_notify.sh "aws ${TRAIN_CONFIG_1}" "失败"
fi

# 任务 2
echo "训练: ${TRAIN_CONFIG_2}"
if torchrun --nproc_per_node=8 --master_port=29501  --local-ranks-filter=0\
    scripts/train.py --config "${TRAIN_CONFIG_2}"; then
    /mnt/efs/chenran/claude_notify.sh "aws ${TRAIN_CONFIG_2}" "完成"
else
    /mnt/efs/chenran/claude_notify.sh "aws ${TRAIN_CONFIG_2}" "失败"
fi


# 启动 gpu_burn
echo "启动 gpu_burn $(date)"
/mnt/efs/chenran/claude_notify.sh 'aws上的实验' '训练完成，启动gpu_burn'
while true; do
    (cd /mnt/efs/chenran/gpu-burn && ./gpu_burn 3600)
    echo "gpu_burn 继续 $(date)"
done
