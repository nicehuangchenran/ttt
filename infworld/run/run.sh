#!/bin/bash
# 自己写的，用于运行inference的main.py 脚本，多 GPU 端口冲突(EADDRINUSE)时: export MASTER_PORT=29500

# 准备 conda
source /root/miniconda3/etc/profile.d/conda.sh
conda activate infworld

# 在项目根目录运行（main.py 以 scripts.infworld_inference 形式 import，需要根目录在 cwd）
cd /root/autodl-tmp/ttt/infworld || exit 1
mkdir -p logs


# INPUT_DATASET : 从哪个文件夹下读取输入
# NUM_CASES : 取前 N 个 case，0 表示全部
# MAX_NUM_CHUNKS : 每个 case 最多生成的 chunk 数，<=0 表示不限制
# ONLINE    : online (test-time) training 开关 on/off
# OUTDIR    : 视频输出的文件夹名称，在 wbench/work_dir里面
# NUM_GPUS  
# INPUT_DATASET NUM_CASES MAX_NUM_CHUNKS ONLINE OUTDIR NUM_GPUS
JOBS=(
    "dataset/long_case 1 20 off online-long_${TS} 1"
    # "dataset/wbench 10 5 on infworld-on 8"
)
for job in "${JOBS[@]}"; do
    read -r INPUT_DATASET NUM_CASES MAX_NUM_CHUNKS ONLINE OUTDIR NUM_GPUS <<< "$job"

    # 统一的 Python CLI 参数（单/多 GPU 共用，torchrun 也会透传给每个进程）
    PY_ARGS="--input-dataset $INPUT_DATASET \
             --outdir $OUTDIR \
             --num-cases $NUM_CASES \
             --online-training $ONLINE \
             --max-num-chunks $MAX_NUM_CHUNKS"

    echo "=============================================="
    echo "Infinite World - Generate for WBench"
    echo "=============================================="
    echo "GPUs: $NUM_GPUS | num_cases: $NUM_CASES (0=all) | outdir: $OUTDIR | online: $ONLINE"

    TS=$(date +%m-%d_%H-%M)
    if [ "$NUM_GPUS" -eq 1 ]; then
        python run/main.py $PY_ARGS 2>&1 | tee "logs/${OUTDIR}_${TS}.log"
    else
        MASTER_PORT=${MASTER_PORT:-29400}
        echo "MASTER_PORT: $MASTER_PORT"
        torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS \
            --rdzv_id=100 --rdzv_backend=c10d \
            --rdzv_endpoint=localhost:$MASTER_PORT \
            run/main.py $PY_ARGS 2>&1 | tee "logs/${OUTDIR}_${TS}.log"
    fi
done
