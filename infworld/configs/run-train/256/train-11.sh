#!/bin/bash
# train-11-sft：本机大文件根是 /mnt/local_nvme（不是默认的 /mnt/nvme）
set -u

export INFWORLD_LOCAL_ROOT=/mnt/local_nvme/chenran/ttt/infworld
export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /mnt/efs/chenran/project/ttt/infworld

CFG="${1:-configs/run-train/256/train-11-sft.yaml}"
PY=/mnt/efs/chenran/miniconda3/envs/infworld/bin/torchrun

echo "===== train $CFG  $(date) ====="
"$PY" --nproc_per_node=8 --master_port=29511 --local-ranks-filter=0 \
    scripts/train.py --config "$CFG"
echo "===== exit=$? $(date) ====="
