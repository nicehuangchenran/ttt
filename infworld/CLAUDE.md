# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目简介

Infinite-World（infworld）：交互式世界模型，从条件图像 + 文本 prompt + 逐帧动作序列自回归生成 1000+ 帧长视频。基于 1.3B Wan2.1 DiT，训练目标为 flow-matching（rectified flow）。本 fork 增加了离线 flow-matching 微调和 test-time training（TTT）路径。

## 环境

- Python：`/mnt/efs/chenran/miniconda3/envs/infworld/bin/python3`（conda 环境 `infworld`，Python 3.10，CUDA 12.4，torch 2.6.0）。默认 shell 是 `longlive` 环境，所以必须显式使用该解释器路径，不要直接用裸 `python`。
- 共享的 8×A100-80G 机器。启动任务前先 `nvidia-smi` 挑空闲卡，用 `CUDA_VISIBLE_DEVICES` 指定，并先用最小配置试跑（小 `--max-chunks`、少量 case）。
- 所有模型路径通过 `configs/infworld_config.yaml` 解析；预训练权重放在 `checkpoints/`（从 HuggingFace `MeiGen-AI/Infinite-World` 下载，见 README）。

## 三个核心脚本（`scripts/` 目录）

`scripts/` 下只有 `main.py`（推理 + 在线/test-time 训练）和 `train.py`（离线训练）是当前入口。`infworld_inference_origin.py`、`infworld_inference_with_ttt.py`、`train_old.py` 是遗留代码，不要在其基础上扩展。

端到端流水线：**预处理**（视频/文本 → 缓存张量）→ **训练**（离线微调 DiT）→ **main.py**（自回归推理，可选在线 TTT）。预处理与训练共享同一套磁盘数据契约，推理做验证时也读同一 `preprocessed/` 格式。

### 预处理：`preprocess/sekai_game_walking/2_preprocess_dataset.py`

把原始数据集（`case{n}/{video.mp4, move_view.json, prompts.json, image.jpg}`）编码为逐 case 的张量文件。只加载 VAE + T5（不加载 DiT）。torchrun 按 case 分片到多卡。

```bash
torchrun --nproc_per_node=8 preprocess/sekai_game_walking/2_preprocess_dataset.py \
  --bucket-config ASPECT_RATIO_256 \
  --dataset-dir dataset/sekai-game-walking-352_192_30fps \
  --output-dir preprocessed/sekai-game-walking-256px
```

每个 case 输出：`latent_full.pt` `[16, T_lat, h, w]`、`target_latent.pt` `[K, 16, 21, h, w]`、`text_emb.pt`、`actions.pt`、`meta.json`。`--verify-prefix` 参数可抽查下文的因果 VAE 前缀假设。

### 训练：`scripts/train.py`

离线 flow-matching 微调。**只**加载 DiT — VAE 和 T5 从不实例化（这正是预处理换来的收益）。一个数据集样本 = 一个完整视频；视频的 K 个 chunk 累积梯度后只做一次 `optimizer.step()`。

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29500 \
  scripts/train.py \
  --data-dir preprocessed/sekai-game-walking-352_192_30fps --shift 3 \
  --filter-location "East Maddon Park, London, United Kingdom" --filter-weather sunny \
  --epochs 20 --val-every-n-steps 16
```

恢复训练：`--resume weights/<run>/step100.ckpt`。筛选参数（`--filter-location/-scene/-crowd-density/-weather/-time-of-day`）按 `meta.json` 精确匹配，选出训练子集。

产物写入 `weights/<run_name>/step{N}.ckpt` 和 `logs/train_log/<run_name>/`（`train.log`、逐 rank 日志、`metrics.jsonl`、`config.json`、tensorboard）。`<run_name>` 自动编码数据集/shift/筛选条件/chunk 数/时间戳。完整设计见 `TRAIN_ARCHITECTURE.md`，日志布局见 `MULTI_GPU_LOGGING.md`。

### 推理：`scripts/main.py`

分 chunk 自回归生成。自动检测两种输入格式：原始 WBench（`image.jpg` + JSON，现场 VAE 编码）和 `preprocessed/`（读缓存 latent，支持 `--filter-*`）。单卡运行完全绕过 `torch.distributed`；多卡用 torchrun 按 case 分片（数据并行）。

```bash
# 用 preprocessed 格式验证训练出的 checkpoint
CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nnodes=1 --nproc_per_node=5 scripts/main.py \
  --dataset-dir preprocessed/sekai-game-walking-352_192_30fps --shift 3 \
  --filter-location "East Maddon Park, London, United Kingdom" --filter-weather sunny \
  --checkpoint weights/<run>/step500.ckpt --max-chunks 3 --num 4 \
  --output-dir videos/val/<run>
```

用 `--online-training on --train-steps 5` 可在 chunk 之间启用 test-time training。`cache.sh` 和 `infer_local.sh` 里有更多现成的调用命令（含 OSS 上传命令）。

## 架构要点（改动数据或 checkpoint 前必读）

- **每个 case 存两份 latent，各有原因。** `latent_full` 是整段视频一次编码；因为 Wan VAE 是*因果*的，任意前缀切片 `latent_full[:, :20k+1]` 与单独编码像素帧 `[0, 80k+1)` 精确相等 — 因此用作 `image_cond`。`target_latent[k]` 是像素帧 `[80k, 80k+81)` *独立*编码（帧 80k 被当作新的首帧），与推理时重新 encode buffer 尾部的行为一致 — 因此用作 `x_start`（GT）。两者不可互换。
- **chunk 数学（1 帧重叠）：** chunk `k` = 像素帧 `[80k, 80k+81)`；`K = (T_pix-1)//80`，可被 `--max-chunks` 截断。动作按 `[80k, 80k+81)` 切片，尾部零填充。这个重叠使训练与推理的自回归行为完全对齐。
- **`make_chunk_batch`（train.py）是唯一知道怎么把视频切成 chunk 的地方。** 要改条件策略（如"所有 chunk 只用第一帧作 cond"），只改那里的 `image_cond` 一行，别处不动。
- **损失只在生成帧上计算：** `pred[:, :, -21:]`，condition 区域切掉。与 `main.py::_online_train_step` 一致，并绕开 `RFlowScheduler.training_losses` 对非 None `x_ignore_mask` 的硬依赖。
- **`flow_matching_loss` 与在线训练 step 必须保持对齐。** 两者都用 `target = x_start - noise`，`use_reversed_velocity=True` 时取负（预训练约定）。`shift` 与分辨率配套：PX256→3、PX627→7、PX960→11，且 `--bucket-config-name` 与 `--shift` 要同步。
- **DiT 每次前向都会按当前 `image_cond` 长度重写 `block.self_attn.num_c`**，所以 chunk 间条件长度变化是安全的 — 但单进程内不能并发跑两个不同形状的前向。
- **checkpoint 契约：** train.py 写入 `{state_dict（已剔除 pos_embed*）, optimizer, lr_sched, global_step, config}`。`main.py` 通过 `state_dict` 键加载，并从 `checkpoint-(\d+)`/`step(\d+)` 文件名解析 step，因此训练出的 checkpoint 可直接用于推理。

## 硬性规则

- `checkpoints/` 只读（预训练权重）。所有训练产物写 `weights/<run>/` 和 `logs/train_log/<run>/`。绝不写 `checkpoints/`。
- `dataset/`、`videos/`、`checkpoints/`、`outputs/` 已 gitignore（大文件/生成物）；`preprocessed/` 目前只跟踪 `meta.json`。
- DiT 配置（`in_channels`、`dim=1536`、`num_layers=30` 等）在 `configs/infworld_config.yaml::model_cfg`；train.py 硬编码了少量常量（out_channels=16、caption_channels=4096、max_length=512），以免为了读两个常量而加载 10GB 文本编码器 — 若改模型形状，两处要保持一致。
