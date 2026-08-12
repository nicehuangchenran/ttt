#!/bin/bash

export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # 用可扩展显存段，回收每个 chunk 调 empty_cache 造成的碎片
# 本机 nvme 挂载点是 /mnt/local_nvme，覆盖 storage.py 里 /mnt/nvme 的默认值
export INFWORLD_LOCAL_ROOT=/mnt/local_nvme/chenran/ttt/infworld

cd /mnt/efs/chenran/project/ttt/infworld

NOTIFY=/mnt/efs/chenran/bin/notify
# Blackwell(sm_120) 卡必须用 infworld-2（torch 2.7.0+cu128）
TORCHRUN=/mnt/efs/chenran/miniconda3/envs/infworld-2/bin/torchrun

CONFIG="configs/run-infer/256/infer-1-wbench-all.yaml"
NAME=$(basename "${CONFIG}" .yaml)

echo "===== 开始 ${NAME}  (GPU 0,1,2,3)  $(date) ====="
if CUDA_VISIBLE_DEVICES=0,1,2,3 "${TORCHRUN}" --nproc_per_node=4 --master_port=29541 \
    --local-ranks-filter=0 scripts/infer.py --config "${CONFIG}"; then
    echo "===== 完成 ${NAME}  $(date) ====="
    "${NOTIFY}" "${NAME}" "成功"
else
    echo "===== 失败 ${NAME}  $(date) ====="
    "${NOTIFY}" "${NAME}" "失败"
fi
