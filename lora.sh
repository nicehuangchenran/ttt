#!/bin/bash
set -eo pipefail
source /root/miniconda3/etc/profile.d/conda.sh

dir=test
conda activate /root/autodl-fs/conda_envs/infworld

MASTER_PORT=${MASTER_PORT:-29400}
NUM_GPUS=1
torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS \
    --rdzv_id=100 --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:$MASTER_PORT \
    infworld/scripts/main.py --output-dir /root/autodl-fs/ttt/WBench/work_dirs/$dir/videos --num 1 --online-training on --steps 1 --max-chunks 1

# torchrun --nproc_per_node=1 infworld/scripts/main.py --output-dir /root/autodl-tmp/ttt/WBench/work_dirs/$dir/videos --num 1 --online-training on

# wbench python自动使用多卡
conda activate /root/autodl-fs/conda_envs/wbench-main
python WBench/main.py --model "$dir" --phase precompute --skip_da3 --skip_sam2 
python WBench/main.py --model "$dir" --phase gpu --metrics consistency --skip_da3 --skip_sam2 --skip_megasam
python WBench/main.py --model "$dir" --phase report