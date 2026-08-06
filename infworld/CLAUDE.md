# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目简介

Infinite-World（infworld）：交互式世界模型，从条件图像 + 文本 prompt + 逐帧动作序列自回归生成 1000+ 帧长视频。基于 1.3B Wan2.1 DiT，训练目标为 flow-matching（rectified flow）。本 fork 增加了离线 flow-matching 微调和 test-time training（TTT）路径。

## 环境

Python：`/mnt/efs/chenran/miniconda3/envs/infworld/bin/python3`（conda 环境 `infworld`，Python 3.10，CUDA 12.4，torch 2.6.0）。不要直接用裸 `python`。

## 运行流程

端到端流水线：**预处理**（视频/文本 → 缓存张量）→ **训练**（离线微调 DiT）→ **main.py**（自回归推理，可选在线 TTT）。

### 预处理：`preprocess/sekai_game_walking/2_preprocess_dataset.py`

把原始数据集（`case{n}/{video.mp4, move_view.json, prompts.json, image.jpg}`）编码为逐 case 的张量文件。只加载 VAE + T5（不加载 DiT）。torchrun 按 case 分片到多卡。

```bash
torchrun --nproc_per_node=8 preprocess/sekai_game_walking/2_preprocess_dataset.py \
  --bucket-config ASPECT_RATIO_256 \
  --dataset-dir dataset/sekai-game-walking-352_192_30fps \
  --output-dir preprocessed/sekai-game-walking-256px
```

每个 case 输出：`latent_full.pt` `[16, T_lat, h, w]`、`target_latent.pt` `[K, 16, 21, h, w]`、`text_emb.pt`、`actions.pt`、`meta.json`,上述文件放在 `<output-dir>/case{n}/`

### 训练：`scripts/train.py`

离线 flow-matching 微调。一次前向生成一个flow match 速度 v, 30 个 time steps 后生成一个视频 chunk, 一个视频的 K 个 chunk 累积梯度后只做一次 `optimizer.step()`。

运行参数全部来自 runs/train/[id}.yaml，命令行只指定用哪个 yaml：

```
torchrun --nproc_per_node=8 scripts/train.py --config configs/runs/train/london_sunny.yaml
```

配置分三层：`configs/infworld_config.yaml`（模型结构/VAE/T5/基础 ckpt）、`configs/train_default.yaml`（train.py 全部可配置项 + 默认值，作为字段全集文档）、`configs/runs/train/*.yaml`（单次实验，只写要改的字段，分 `train:`/`ttt:` 两段，对应 `TrainConfig`/`TTTConfig`）。写了不存在的 key 会报错而不是静默忽略；新增参数只需在 dataclass 加字段。临时覆盖用 `--set train.lr=2e-5`。

恢复训练：yaml 里设 `train.resume_from: weights/<run>/step100.ckpt`。筛选字段（`filter_location/scene/crowd_density/weather/time_of_day`）按 `meta.json` 精确匹配，选出训练子集。

一次运行的产物都在 `weights/<run_name>/` 下：`step{N}.ckpt` 和 `train_log/`（`train.log`、逐 rank 日志、`metrics.jsonl`、`config.json`、tensorboard）。`<run_name>` 默认取 yaml 文件名（去扩展名），也可在 yaml 里显式指定 `train.run_name` 或用 `--set train.run_name=my_exp` 临时覆盖。完整设计见 `TRAIN_ARCHITECTURE.md`，日志布局见 `MULTI_GPU_LOGGING.md`。

### 推理：`scripts/infer.py`

分 chunk 自回归生成。自动检测两种输入格式：原始 WBench（`image.jpg` + JSON，现场 VAE 编码）和 `preprocessed/`（读缓存 latent）。

与 train.py 一样，运行参数全部来自 run/infer/{id}.yaml，命令行只指定用哪个 yaml：

```bash
# 用 preprocessed 格式验证训练出的 checkpoint
CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nnodes=1 --nproc_per_node=5 \
  scripts/main.py --config configs/runs/infer/test.yaml
```

配置同样分三层：`configs/infworld_config.yaml`、`configs/infer_default.yaml`（main.py 全部可配置项 + 默认值，字段全集文档）、`configs/runs/infer/*.yaml`（单次推理，只写要改的字段，分 `ExpConfig:`/`OnlineTrainConfig:` 两段，段名就是 dataclass 名）。写了不存在的 key 会报错；临时覆盖用 `--set ExpConfig.max_chunks=1`。

一次运行的产物都在 `videos/test/<run_name>/` 下：`case_{n}_combined.mp4` 和 `infer_config.json`（本次生效的完整配置）。`<run_name>` 默认取 yaml 文件名（去扩展名），可用 `ExpConfig.run_name` / `ExpConfig.output_root` 改。目录里已存在同名 mp4 时立刻报错退出，不覆盖旧结果。

在 yaml 里设 `OnlineTrainConfig.open: true` 可在 chunk 之间启用 test-time training。`cache.sh` 和 `infer_local.sh` 里有更多现成的调用命令（含 OSS 上传命令）。

## 架构要点（改动数据或 checkpoint 前必读）

- **每个 case 存两份 latent，各有原因。** `latent_full` 是整段视频一次编码；因为 Wan VAE 是*因果*的，任意前缀切片 `latent_full[:, :20k+1]` 与单独编码像素帧 `[0, 80k+1)` 精确相等 — 因此用作 `image_cond`。`target_latent[k]` 是像素帧 `[80k, 80k+81)` *独立*编码（帧 80k 被当作新的首帧），与推理时重新 encode buffer 尾部的行为一致 — 因此用作 `x_start`（GT）。两者不可互换。
- **chunk 数学（1 帧重叠）：** chunk `k` = 像素帧 `[80k, 80k+81)`；`K = (T_pix-1)//80`，可被 `--max-chunks` 截断。动作按 `[80k, 80k+81)` 切片，尾部零填充。这个重叠使训练与推理的自回归行为完全对齐。
- **`make_chunk_batch`（train.py）是唯一知道怎么把视频切成 chunk 的地方。** 要改条件策略（如"所有 chunk 只用第一帧作 cond"），只改那里的 `image_cond` 一行，别处不动。
- **损失只在生成帧上计算：** `pred[:, :, -21:]`，condition 区域切掉。与 `main.py::_online_train_step` 一致，并绕开 `RFlowScheduler.training_losses` 对非 None `x_ignore_mask` 的硬依赖。
- **`flow_matching_loss` 与在线训练 step 必须保持对齐。** 两者都用 `target = x_start - noise`，`use_reversed_velocity=True` 时取负（预训练约定）。`shift` 与分辨率配套：PX256→3、PX627→7、PX960→11，且 `--bucket-config-name` 与 `--shift` 要同步。
- **DiT 每次前向都会按当前 `image_cond` 长度重写 `block.self_attn.num_c`**，所以 chunk 间条件长度变化是安全的 — 但单进程内不能并发跑两个不同形状的前向。
- **checkpoint 契约：** train.py 写入 `{state_dict（已剔除 pos_embed*）, optimizer, lr_sched, global_step, config}`。`main.py` 通过 `state_dict` 键加载，并从 `checkpoint-(\d+)`/`step(\d+)` 文件名解析 step，因此训练出的 checkpoint 可直接用于推理。

## 硬性规则

- `scripts/` 下只有 `infer.py`（推理 + 在线/test-time 训练）和 `train.py`（离线训练）是当前入口。`infworld_inference_origin.py`、`infworld_inference_with_ttt.py`、`train_old.py` 是遗留代码，不要在其基础上扩展。
- `checkpoints/` 只读（预训练权重）。所有训练产物写 `weights/<run>/`（ckpt 与 `train_log/` 同目录）。绝不写 `checkpoints/`。
- `dataset/`、`videos/`、`checkpoints/`、`outputs/` 已 gitignore（大文件/生成物）；`preprocessed/` 目前只跟踪 `meta.json`。
- DiT 配置（`in_channels`、`dim=1536`、`num_layers=30` 等）在 `configs/infworld_config.yaml::model_cfg`；train.py 硬编码了少量常量（out_channels=16、caption_channels=4096、max_length=512），以免为了读两个常量而加载 10GB 文本编码器 — 若改模型形状，两处要保持一致。
- 使用中文输出

## 代码生成原则

- 不要创建没有实际复用需求的辅助层、封装或配置。
- 保持实现简洁、清晰，避免不必要的抽象和过度设计。
- 简单的代码修改不要单独生成 md 文档说明
