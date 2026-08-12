#!/bin/bash

export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # 用可扩展显存段，回收每个 chunk 调 empty_cache 造成的碎片
# 本机 nvme 挂载点是 /mnt/local_nvme，覆盖 storage.py 里 /mnt/nvme 的默认值
export INFWORLD_LOCAL_ROOT=/mnt/local_nvme/chenran/ttt/infworld

cd /mnt/efs/chenran/project/ttt/infworld

NOTIFY=/mnt/efs/chenran/bin/notify
# 绝对路径：非交互 shell 里 conda 环境不在 PATH 上
TORCHRUN=/mnt/efs/chenran/miniconda3/envs/infworld/bin/torchrun

# 一个任务：独立 master_port，成功/失败都发通知
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

echo "########## 第一批：wbench-view（3 个并行） $(date) ##########"
run_infer 0,1     29511 2 "configs/run-infer/256/infer-11-wbench-view-base.yaml" &
run_infer 2,3,4   29512 3 "configs/run-infer/256/infer-11-wbench-view-sft-steps1100.yaml" &
run_infer 5,6,7   29513 3 "configs/run-infer/256/infer-11-wbench-view-ttt-steps1100.yaml" &
wait
echo "########## 第一批结束 $(date) ##########"
"${NOTIFY}" "infer-11-wbench-view" "三个配置全部结束"

echo "########## 第二批：context（3 个并行） $(date) ##########"
run_infer 0,1     29521 2 "configs/run-infer/256/infer-11-context-base.yaml" &
run_infer 2,3,4   29522 3 "configs/run-infer/256/infer-11-context-sft-steps1100.yaml" &
run_infer 5,6,7   29523 3 "configs/run-infer/256/infer-11-context-ttt-steps1100.yaml" &
wait
echo "########## 第二批结束 $(date) ##########"
"${NOTIFY}" "infer-11-context" "三个配置全部结束"

echo "########## 全部推理结束 $(date) ##########"
