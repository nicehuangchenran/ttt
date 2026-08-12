#!/bin/bash

export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # 用可扩展显存段，回收每个 chunk 调 empty_cache 造成的碎片
# 本机 nvme 挂载点是 /mnt/local_nvme，覆盖 storage.py 里 /mnt/nvme 的默认值
export INFWORLD_LOCAL_ROOT=/mnt/local_nvme/chenran/ttt/infworld

cd /mnt/efs/chenran/project/ttt/infworld

NOTIFY=/mnt/efs/chenran/bin/notify
# 绝对路径：非交互 shell 里 conda 环境不在 PATH 上；Blackwell 卡必须用 infworld-2
TORCHRUN=/mnt/efs/chenran/miniconda3/envs/infworld-2/bin/torchrun

run_infer() {
    local gpus=$1 port=$2 nproc=$3 config=$4
    local name=$(basename "${config}" .yaml)
    echo "===== 开始 ${name}  (GPU ${gpus}, ${nproc} 进程)  $(date) ====="
    if CUDA_VISIBLE_DEVICES="${gpus}" "${TORCHRUN}" --nproc_per_node="${nproc}" --master_port="${port}" \
        --local-ranks-filter=0 scripts/infer.py --config "${config}"; then
        echo "===== 完成 ${name}  $(date) ====="
        "${NOTIFY}" "${name}" "成功"
    else
        echo "===== 失败 ${name}  $(date) ====="
        "${NOTIFY}" "${name}" "失败"
    fi
}

echo "########## 开始并行推理（2 个配置，各 4 卡） $(date) ##########"
run_infer 0,1,2,3 29531 4 "configs/run-infer/256/infer-1-wbench-view.yaml" &
run_infer 4,5,6,7 29532 4 "configs/run-infer/256/infer-2-wbench-view.yaml" &
wait
echo "########## 全部推理结束 $(date) ##########"
"${NOTIFY}" "infer-1-2-wbench-view" "两个配置全部结束"
