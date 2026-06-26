# Plan: 减小 `scripts/infworld_inference.py` 在线训练的显存占用

## Context
`infworld_inference.py` 在每个 chunk 生成后会做 N_TRAIN_STEPS=5 次 rectified-flow 在线微调（`online_train_step`）。当前实现以"够用即可"的方式写成，存在多处可在不改变行为的前提下大幅降低显存的浪费点。1.3B DiT 在 bf16 下，主要显存压力来自：

1. **AdamW 优化器状态**：每次 chunk 训练前 `optimizer = torch.optim.AdamW(trainable_params, ...)` 都会在 GPU 上新分配 fp32 的 m/v（~8GB），训练结束后没有显式释放，依赖 Python GC + `torch_gc()` 回收。
2. **`init_params` 整模型快照常驻 GPU**：`p.detach().clone()` 默认与 `p` 同设备，约 ~2GB bf16 常驻；只在切换 video 时使用。
3. **激活显存**：`set_grad_checkpoint` 已经开启，但作用粒度是"对所有 module 一刀切设 `grad_checkpointing=True`"，并不一定每个 block 都包了 checkpoint；同时训练时模型未切到 eval 之外的状态，dropout/norm 行为没问题，但 `image_cond` 在训练 forward 里仍走一遍 `latent_encoder`（冻结）会产生不必要的 activation。
4. **全参数可训练**：除少量 condition-entry 层冻结外，~1B 参数可训练，是 optimizer state 与 grad 显存的根源。
5. **训练 batch 与生成共享同一份 latent**：`x_start = samples.detach()`、`current_latent.detach()` 都仍在 GPU，本身不大但与 forward activation 叠加。

目标：在保持训练效果接近的前提下，降低峰值显存，使在线训练在更小显存的卡（如 24G/32G）上可跑。

## Recommended Approach（按"收益/改动量"排序，建议全部采用）

### 1. AdamW 改为 8-bit / fused，并在训练后显式释放
- 使用 `bitsandbytes.optim.AdamW8bit`（若可用）替换 `torch.optim.AdamW`，把 m/v 从 fp32 压到 int8，optimizer state 从 ~8GB → ~1GB。
- 不引入 bnb 依赖时的退路：`torch.optim.AdamW(..., fused=True)`，至少减少临时 buffer。
- 训练完成后显式 `optimizer.zero_grad(set_to_none=True)`；并 `del optimizer` + `torch_gc()`，避免新 optimizer 分配与旧 optimizer 短暂共存导致的峰值翻倍。

### 2. `init_params` 快照搬到 CPU（pinned）
当前位置：`infworld_inference.py:370-374`
```python
init_params = {n: p.detach().clone() for n, p in dit.named_parameters() if p.requires_grad}
```
改为 `.detach().to('cpu', copy=True)`（可加 `pin_memory()`）。恢复时（`infworld_inference.py:420-424`）改为 `p.data.copy_(init_params[n].to(p.device, non_blocking=True))`。预计省 ~2GB 常驻 GPU。

### 3. 收紧可训练参数范围（最大收益）
当前 `FREEZE_PREFIXES` 只冻结 condition entry 层，剩余 ~1B 参数全部可训练。两个方向（推荐二选一，默认 A）：

- **A. 仅训练 DiT block 的 attention/FFN 子集**：例如只解冻最后 K 个 block（`blocks.{i}` 中 i ≥ N-K），或只解冻每个 block 内的 `self_attn.o`/`ffn` 之类的少量投影矩阵。把 trainable 砍到 100–300M，optimizer state 与 grad 同步线性下降。
- **B. 加 LoRA（peft 风格）**：给 attention 的 q/k/v/o 加 rank=8/16 的 LoRA，只训 LoRA 参数（<10M）。当前仓库无 peft 依赖，需小段手写 LoRA wrapper（注入到 `WanModel` 的 attention 线性层）。改动较大但收益最高。

> 建议先做 A（改动 ≈ 修改 `FREEZE_PREFIXES` 与一段过滤逻辑），把基线降下来；后续若仍不足再上 B。

### 4. 调整 grad checkpoint 粒度并确认 backward 时 dtype
- 在 `set_grad_checkpoint` 处补一段只对 DiT 的 transformer block（`blocks.*`）开 checkpoint，避免对 patch_embed / encoder 这些"小且冻结"模块也包 checkpoint（包了反而占额外 stack）。
- 训练 forward 用 `torch.autocast(device_type='cuda', dtype=torch.bfloat16)` 显式包住，确保 activation 是 bf16 而不是 fp32。

### 5. `online_train_step` 内部的小优化
位置：`infworld_inference.py:244-279`
- `noise = torch.randn_like(x_start)` 之后用完即弃，无需改动；但 `target = x_start - noise` 可改 in-place：`target = x_start.sub(noise)`（注意不要污染 `x_start`，所以仍用新张量，但避免临时中间张量）。
- `pred = pred[:, :, -x_start.shape[2]:]` 之前的 `pred` 全长张量在切片后仍被引用：显式 `pred = pred[..., -T:].contiguous(); del full_pred` 帮助回收。
- backward 前后各加一次 `torch_gc()`（已在外层有，但内层多步训练时跨 step 也建议加）。

### 6. 训练前临时卸载 text encoder 到 CPU
T5-XXL（~5GB+ bf16）在 sample 阶段需要，但 online_train_step 里 y/y_mask 是 `cached_y` / `cached_y_mask`，**训练阶段并不需要 text encoder 本体**。可在进入训练循环前 `text_encoder.t5.model.to('cpu')`，训练完恢复回 GPU。需要确认 sample 调用是否每个 chunk 都触发 encoder（`scheduler.sample` 接受 `text_encoder` 参数），如果是则每 chunk 来回搬一次 PCIe 开销可接受（每 chunk ~秒级）。

## 关键修改文件
- `scripts/infworld_inference.py`
  - `FREEZE_PREFIXES` 列表与可训练参数收集（行 349-368）
  - `init_params` 快照（行 370-374）与恢复（行 420-425）
  - `online_train_step`（行 244-279）
  - 每 chunk optimizer 构造与销毁（行 479-494）
  - `set_grad_checkpoint` 调用（行 343-344），可能需要小改 `infworld/models/checkpoint.py` 以支持粒度参数
- 可选：`infworld/models/checkpoint.py` — 支持仅对 `blocks.*` 开启 checkpoint。
- 可选（方案 3B）：在 `infworld/models/dit_model.py` 的 attention 线性层注入 LoRA wrapper。

## Verification
1. 基线测量：现状下跑 `bash infer_local.sh 1` 并打开 `--online-training on`，用 `nvidia-smi --query-gpu=memory.used --format=csv -l 1` 或在 `online_train_step` 末尾打印 `torch.cuda.max_memory_allocated()/1e9`，记录峰值。
2. 每应用一项改动后重跑同一 prompt（`prompts/demo.yaml` 第一条），对比：
   - 峰值显存（`torch.cuda.max_memory_allocated`）。
   - 每个 chunk 的 5 步训练 loss 趋势（应与基线接近，不应发散）。
   - 最终视频质量目视检查（`outputs/...` 下的 mp4）。
3. 关闭 online training (`--online-training off`) 时显存峰值应基本不变（验证改动不影响纯推理路径）。
