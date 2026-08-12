---
name: yaml
description: 按现有 run yaml 的写法新建 configs/run-infer/**/*.yaml（推理配置，常见为 base / sft / ttt 一组三份）。当用户要求"仿照某个 yaml 再写几个""换数据集/换 ckpt/换 steps 写一份 infer yaml"时使用。
---

# 写 infer run yaml

`configs/run-infer/**/*.yaml` 是单次推理配置，只写要改的字段，未写的走 `configs/infer_default.yaml` 默认值（那里是字段全集，写了不存在的 key 直接报错）。段名固定为 `ExpConfig:` / `OnlineTrainConfig:`。

## 流程

1. **读参考 yaml**（用户给的那几个），照抄结构、字段顺序和注释，只改用户要求的字段。
2. **核实路径存在**，别凭命名规律猜（下节）。
3. **写文件**，一个变体一份。
4. 回复里说明：新文件路径、相对参考 yaml 改了哪些字段、核实到的事实（ckpt 大小/时间、dataset case 数）、以及沿用未改的可疑值（如 `num`）。

## 必须核实的三件事

```bash
B=s3://s3-us-west2-default/archives/chenran/ttt/infworld
L=/mnt/nvme/chenran/ttt/infworld

# ckpt 是否存在（本地缺失会在跑时自动从 S3 拉，所以查 S3 为准）
aws s3 ls "$B/weights/train-11-sft/" | grep steps1100
ls -d $L/weights/*train-11*                       # 顺带看本地有没有

# 权重目录名不要猜：先列出来
aws s3 ls "$B/weights/" | awk '{print $2}' | grep train-11

# 数据集存在性 + case 数（决定 num 合不合理）
aws s3 ls "$B/preprocessed/context-dataset-256px/" | awk '{print $2}' | wc -l
```

路径映射细节见 `s3` 技能。用 `awk '{print $2}'` 取目录名，别用 `head`（会 BrokenPipeError）。

## 三个变体的差异

同一组 base / sft / ttt 只有两处不同，其余字段（`dataset_dir` `shift` `bucket_config_name` `max_chunks` `num` `real_hist` `use_fixed_noise` `noise_cache_dir`）三份完全一致：

| 变体 | `checkpoint_path` | `OnlineTrainConfig` |
| --- | --- | --- |
| base | `checkpoints/infinite_world_model.ckpt` | `open: false` |
| sft | `weights/<run>-sft/steps<N>.ckpt` | `open: false` |
| ttt | `weights/<run>-ttt/steps<N>.ckpt` | `open: true` + 完整 TTT 段 |

TTT 段（`n_train_steps` `lr` `grad_clip_norm` `trainable_layers` `trainable_block_start/end` `reset_between_videos`）连同行内中文注释整段从参考 ttt yaml 抄过来，用户没要求就不要调参。

## 命名与配套字段

- 文件名即 `run_name`，产物落在 `videos/<run_name>/`；同名 mp4 已存在会直接报错退出。所以文件名要把区分维度都带上：`infer-<训练轮次>-<数据集>-<变体><-steps N><-tf>.yaml`，例 `infer-11-context-sft-steps1100.yaml`。
- `shift` 与 `bucket_config_name` 必须配套：PX256 → `shift: 3` + `ASPECT_RATIO_256`。
- `real_hist: true`（teacher forcing）只对 `preprocessed/` 格式有效，WBench 原始目录下设 true 会报错；文件名习惯加 `-tf` 后缀。用户说"关闭 tf"就是 `real_hist: false`。

## 坑

- **参考 yaml 末尾的 `#finished at ...` 注释是 infer.py 跑完自动追加的**，新文件不要抄这一行。
- **`num` 不会随数据集自动变**。参考 yaml 里是 `num: 24`，换数据集后仍是 24（只跑前 24 个 case）。照抄没错，但要在回复里点出数据集实际 case 数，让用户决定是否放大。
- **ckpt 目录名不总是成套**。可能只有 `train-11-ttt` 而没有 `train-11-sft`，或本地只有其中一个——列出来确认，不要按对称性假设。
