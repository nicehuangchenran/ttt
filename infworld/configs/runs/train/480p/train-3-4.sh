#!/bin/bash

# 每个训练进程的 OMP 线程数。机器 192 逻辑核 / 8 进程 = 每进程 24 核，
# 取 16 既避免 torchrun 默认值 1 拖慢 CPU 端算子/数据处理，又留出余量不过订阅。
export OMP_NUM_THREADS=16

# 用可扩展显存段，回收每个 chunk 调 empty_cache 造成的碎片（OOM 报错里的 8.34GB
# reserved-but-unallocated），缓解 8 卡 DDP 主训练 backward 的显存峰值。
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 获取时间戳函数
timestamp() {
    date +%m%d_%H%M%S
}


echo "===== 开始顺序训练任务 $(date) ====="

# 任务 1: train-4.yaml
echo "开始训练: train-4.yaml"
if CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    torchrun --nproc_per_node=8 --master_port=29501 \
    scripts/train.py --config configs/runs/train/480p/train-4.yaml \
    > nohup_train-4.out 2>&1; then
    /mnt/efs/chenran/claude_notify.sh 'aws上的实验' 'train-4.yaml运行结束'
    echo "train-4.yaml 完成 $(date)"
else
    /mnt/efs/chenran/claude_notify.sh 'aws上的实验' 'train-4.yaml运行失败'
    echo "train-4.yaml 失败 $(date)"
fi

# 任务 2: train-3.yaml
echo "开始训练: train-3.yaml"
if CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    torchrun --nproc_per_node=8  --master_port=29501 \
    scripts/train.py --config configs/runs/train/480p/train-3.yaml \
    > nohup_train-3.out 2>&1; then
    /mnt/efs/chenran/claude_notify.sh 'aws上的实验' 'train-3.yaml运行结束'
    echo "train-3.yaml 完成 $(date)"
else
    /mnt/efs/chenran/claude_notify.sh 'aws上的实验' 'train-3.yaml运行失败'
    echo "train-3.yaml 失败 $(date)"
fi

# 无论训练成功还是失败，启动 gpu_burn 并一直运行
echo "启动 gpu_burn $(date)"
/mnt/efs/chenran/claude_notify.sh 'aws上的实验' '所有训练完成，启动gpu_burn'

# 无限循环运行 gpu_burn（每次 1 小时）
# gpu_burn 用相对路径加载 compare.fatbin，必须在它自己的目录下运行
while true; do
    (cd /mnt/efs/chenran/gpu-burn && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./gpu_burn 3600)
    echo "gpu_burn 循环继续 $(date)"
done
