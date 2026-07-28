# 分布式预处理指南

## 概述

`2_preprocess_dataset.py` 现已支持多GPU并行处理，可以显著加速大规模数据集的预处理。

## 工作原理

- 使用 `torch.distributed` 实现数据并行
- 每个 GPU 处理整个数据集的 1/N 子集（round-robin 分配）
- 例如：8张卡时，rank 0 处理 case0, case8, case16...；rank 1 处理 case1, case9, case17...
- 所有 rank 独立运行，无需通信（除了最后的汇总统计）

## 使用方法

### 方法1：使用便捷脚本（推荐）

```bash
# 默认使用8张卡
bash preprocess_distributed.sh

# 指定GPU数量
bash preprocess_distributed.sh 4

# 自定义配置（通过环境变量）
BUCKET_CONFIG=ASPECT_RATIO_627_F64 \
DATASET_DIR=dataset/my-dataset \
OUTPUT_DIR=preprocessed/my-dataset \
bash preprocess_distributed.sh 8
```

### 方法2：直接使用 torchrun

```bash
# 8 GPUs
torchrun --nproc_per_node=8 \
    prepare/sekai_game_walking/2_preprocess_dataset.py \
    --bucket-config ASPECT_RATIO_256_F64 \
    --dataset-dir dataset/sekai-game-walking-854_480_30fps \
    --output-dir preprocessed/sekai-game-walking-256px

# 如果端口冲突，指定其他端口
torchrun --nproc_per_node=8 --master_port=29501 \
    prepare/sekai_game_walking/2_preprocess_dataset.py \
    --bucket-config ASPECT_RATIO_256_F64 \
    --dataset-dir dataset/sekai-game-walking-854_480_30fps \
    --output-dir preprocessed/sekai-game-walking-256px
```

### 方法3：单GPU模式（向后兼容）

```bash
# 直接运行，无需 torchrun
python prepare/sekai_game_walking/2_preprocess_dataset.py \
    --bucket-config ASPECT_RATIO_256_F64 \
    --dataset-dir dataset/sekai-game-walking-854_480_30fps \
    --output-dir preprocessed/sekai-game-walking-256px
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset-dir` | 输入数据集目录 | `dataset/sekai-game-walking-854_480_30fps` |
| `--output-dir` | 输出目录 | `preprocessed/sekai-game-walking-854_480_30fps-256px` |
| `--bucket-config` | 分辨率配置（从 `infworld/configs/bucket_config.py`） | `ASPECT_RATIO_256_F64` |
| `--max-cases` | 最大处理 case 数量（-1=全部） | `-1` |
| `--skip-existing` | 跳过已处理的 case | `True` |
| `--verify` | 验证 VAE 因果性假设 | `False` |

## 性能对比

假设单 GPU 处理一个 case 需要 10 秒：

- **1 GPU**: 1000 cases × 10s = ~2.8 小时
- **8 GPUs**: 125 cases × 10s = ~21 分钟（理论加速比 8×）

实际加速比可能略低（因为 I/O 瓶颈和负载不均衡）。

## 注意事项

1. **GPU 选择**：在共享服务器上，先用 `nvidia-smi` 查看空闲的 GPU，使用 `CUDA_VISIBLE_DEVICES` 指定：
   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash preprocess_distributed.sh 8
   ```

2. **跳过已处理**：`--skip-existing` 默认开启，重新运行时会跳过已有 `meta.json` 的 case，支持断点续传。

3. **内存占用**：每个 GPU 会加载完整的 VAE + Text Encoder，确保显存充足（建议每卡至少 24GB）。

4. **端口冲突**：如果遇到 `Address already in use` 错误，修改脚本中的 `--master_port`。

## 验证

处理完成后，检查输出目录：

```bash
# 统计已处理的 case 数量
ls preprocessed/sekai-game-walking-256px/*/meta.json | wc -l

# 查看某个 case 的输出
ls -lh preprocessed/sekai-game-walking-256px/case0/
# 应该包含：latent_full.pt, target_latent.pt, text_emb.pt, actions.pt, meta.json
```

## 故障排除

### 问题：某些 case 处理失败

检查输出日志中的 `[SKIP]` 标记，常见原因：
- 缺少必需文件（video.mp4, move_view.json, prompts.json）
- 视频太短（< 81 帧）
- 视频加载失败

### 问题：分布式初始化失败

确保：
- 使用 `torchrun` 而不是 `python`
- `NCCL` 后端可用（`torch.distributed.is_nccl_available()`）
- GPU 间可通信（同一节点）

### 问题：进度条混乱

这是多进程输出叠加的正常现象，不影响处理结果。可以重定向输出：
```bash
bash preprocess_distributed.sh 8 2>&1 | tee preprocess.log
```
