#!/bin/bash
# Robust resumable weights download loop (hf-mirror, auto-retry until complete)
cd /root/autodl-tmp/ttt/WBench
unset http_proxy https_proxy
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
HF=/root/autodl-tmp/conda_envs/wbench-main/bin/hf
n=0
while [ $n -lt 200 ]; do
  n=$((n+1))
  echo "=== attempt $n at $(date) ==="
  "$HF" download meituan-longcat/WBench-weights --local-dir weights/ && {
    echo "=== DOWNLOAD COMPLETE at $(date) ==="
    exit 0
  }
  echo "=== retry in 5s (exit code $?) ==="
  sleep 5
done
echo "=== gave up after $n attempts ==="
exit 1
