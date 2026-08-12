---
name: infer
description: 按 configs/run-infer/**/*.yaml 跑 scripts/infer.py 自回归推理（含 TTT）。当用户要求"跑 infer / 运行推理 / 用这个 yaml 推理 / 生成视频 / 跑 base 和 sft 对比 / 并行跑几个配置 / 后台跑推理"，或要写、改、重跑 infer-parallel.sh 这类多卡并行推理脚本时使用。覆盖 torchrun 起法、GPU 与 master_port 分配、conda 绝对路径、后台运行与日志、notify 通知、产物落在 videos/<run_name>/、yaml 里的大目录路径实际在 /mnt/nvme 与 S3 上（项目目录里看不到）、以及同名 mp4 已存在会直接报错退出等坑。
---

# 跑 infer

一份 yaml 一次推理，产物在 `videos/<run_name>/`，`<run_name>` 默认取 yaml 文件名（去扩展名）。写脚本时照抄 `configs/run-infer/256/infer-parallel.sh` 的结构。

## 先记住：yaml 里的大目录路径不在项目目录

yaml 里写的 `dataset_dir: dataset/wbench-view`、`checkpoint_path: weights/train-11-sft/steps1100.ckpt` 都是**相对路径**，由 `infworld/utils/storage.py` 映射到别处，不在 `/mnt/efs/chenran/project/ttt/infworld/` 下。`dataset` `preprocessed` `checkpoints` `weights` `videos` `outputs` 这六个顶层目录走本地 `/mnt/nvme/chenran/ttt/infworld/` + S3 权威副本，详见 `s3` 技能。

所以在项目目录里 `ls dataset/ weights/ checkpoints/ videos/` 一律 `No such file or directory` —— **这是正常的，不是数据被删了**。核实文件在不在要查这两个地方：

```bash
L=/mnt/nvme/chenran/ttt/infworld
B=s3://s3-us-west2-default/archives/chenran/ttt/infworld

ls -la "$L/weights/train-11-sft/steps1100.ckpt"   # 本地有没有
aws s3 ls "$B/weights/train-11-sft/"              # S3 有没有（本地缺就从这拉）
```

本地缺失时 infer.py 自己会拉（checkpoint、case 目录、noise cache 都走 `storage.ensure_local` / `ensure_files`），不用手动 sync；产物也会 `upload_async` 自动回传 S3。手动查证只是为了跑之前确认路径没写错。

## 单个配置

```bash
cd /mnt/efs/chenran/project/ttt/infworld
export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1,2,3 \
/mnt/efs/chenran/miniconda3/envs/infworld/bin/torchrun \
  --nproc_per_node=4 --master_port=29501 --local-ranks-filter=0 \
  scripts/infer.py --config configs/run-infer/256/infer-11-wbench-view-base.yaml
```

case 按 rank 分片，`--nproc_per_node` 就是用几张卡。`--local-ranks-filter=0` 只让 rank0 打日志。临时改字段用 `--set ExpConfig.max_chunks=1 ExpConfig.num=1`（冒烟测试常用）。

## 多配置并行

8 卡机器常见做法是 4+4 两个任务同时跑，每个任务独立 `master_port`：

```bash
#!/bin/bash
export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # 可扩展显存段，回收每 chunk empty_cache 的碎片

CONFIG_1="configs/run-infer/256/infer-11-wbench-view-base.yaml"
CONFIG_2="configs/run-infer/256/infer-11-wbench-view-sft-steps1100.yaml"
NOTIFY=notify
# 绝对路径：非交互 shell 里 conda 环境不在 PATH 上
TORCHRUN=/mnt/efs/chenran/miniconda3/envs/infworld/bin/torchrun

run_infer() {
    local gpus=$1 port=$2 config=$3
    local name=$(basename "${config}" .yaml)
    echo "推理: ${config}  (GPU ${gpus})"
    if CUDA_VISIBLE_DEVICES="${gpus}" "${TORCHRUN}" --nproc_per_node=4 --master_port="${port}" \
        --local-ranks-filter=0 scripts/infer.py --config "${config}"; then
        "${NOTIFY}" "${name}" "成功"
    else
        "${NOTIFY}" "${name}" "失败"
    fi
}

echo "===== 开始并行推理 $(date) ====="
run_infer 0,1,2,3 29501 "${CONFIG_1}" &
run_infer 4,5,6,7 29502 "${CONFIG_2}" &
wait
echo "===== 推理结束 $(date) ====="
```

超过两个配置就串起来：同一对 GPU 上顺序 `run_infer A; run_infer B` 放进一个后台子 shell，两组子 shell 再并行。

推理耗时长（20 chunks × 十几个 case 通常几小时），用后台跑并把输出留档：

```bash
nohup bash configs/run-infer/256/infer-parallel.sh > /tmp/infer-11.log 2>&1 &
tail -f /tmp/infer-11.log
```

## 跑之前先核实

1. **同名 mp4 已存在会 `FileExistsError` 直接退出**（S3 上有同名结果也算存在，防覆盖）。查 `$L/videos/<run_name>/` 和 `aws s3 ls "$B/videos/<run_name>/"`。要重跑就先改 yaml 文件名 / `ExpConfig.run_name`，或删掉旧产物（本地和 S3 都要删，删前跟用户确认）。
2. **checkpoint 在不在**。本地缺失会自动从 S3 拉（17 GB 级，第一次会慢），但路径写错要跑起来才报错。按上面那节查 `$L` 和 `$B`。训练还在跑时目标 step 的 ckpt 可能尚未落盘，`aws s3 ls` 看一眼最大 step 再定。
3. **数据集在不在**。同样查 `$L/dataset/<name>/`、`$L/preprocessed/<name>/`，别在项目目录里找。
4. **`ExpConfig.num` 与数据集 case 数**。`num: -1` 是全部，写死的数字不会随数据集变，容易只跑到前 N 个。
5. **`real_hist: true`（teacher forcing）只对 `preprocessed/` 格式有效**，原始 WBench 目录（只有 `image.jpg` + JSON）下设 true 会报错。
6. **GPU 空不空**。`nvidia-smi` 看一眼有没有训练在占卡（`ps -eo pid,etime,cmd | grep train.py`），8 卡被占满时 infer 起不来。

## 坑

- **`nproc_per_node` 必须 ≤ `CUDA_VISIBLE_DEVICES` 里的卡数**，多了会卡在 NCCL 初始化不报错。
- **两个任务用同一个 `master_port` 会抢端口失败**，并行时端口必须各不相同（29501/29502…）。
- **单进程内不能并发跑两个不同形状的前向** —— DiT 每次前向按当前 `image_cond` 长度重写 `num_c`。所以并行只在进程级做，别在一个 torchrun 里塞两个配置。
- **在项目目录 `ls` 不到 `dataset/` `weights/` `checkpoints/` `videos/` 不等于数据没了**（见开头那节）。别据此判断数据被删或路径写错，也别因此建议用户重新同步。
- `notify` 在 `/mnt/efs/chenran/bin/notify`；脚本里非交互 shell 若 PATH 没带上就写绝对路径。
- 日志除了 stdout 还落在 `<output_root>/<run_name>/infer_log/`（即 `$L/videos/<run_name>/infer_log/`），本次生效的完整配置在同目录 `infer_config.json`；长跑任务会 `start_periodic_sync` 周期上传日志到 S3。
- infer.py 跑完会自动往 yaml 末尾追加 `#finished at ...` 注释；新建 yaml 时不要手抄这一行。
