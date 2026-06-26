#!/bin/bash

# 统一设置模型名称（只需修改这里）
MODEL="infworld-online-cut21"

cd /root/autodl-tmp/ttt/WBench
source /root/miniconda3/etc/profile.d/conda.sh
conda activate wbench-main

# 创建日志目录
mkdir -p log

# 记录输出到日志文件，同时显示在终端（第一次覆盖写入）
python main.py --model "$MODEL" --phase precompute --skip_da3 --skip_sam2 2>&1 | tee logs/"$MODEL".logs

echo "precompute 运行完成"

# 第二次追加写入
python main.py --model "$MODEL" --phase gpu --metrics consistency --skip_da3 --skip_sam2 --skip_megasam 2>&1 | tee -a logs/"$MODEL".logs

echo "gpu 运行完成"

python main.py --model "$MODEL" --phase report
echo "生成 report.json"