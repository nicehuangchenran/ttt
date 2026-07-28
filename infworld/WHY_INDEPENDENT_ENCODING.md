# 为什么训练时 chunk 的 GT 必须独立编码？

## TL;DR

**推理时每个 chunk 会重新 encode buffer 尾部**（参见 `main.py:699`），这导致帧 80k（chunk k 的首帧）被当作**新的第一帧**编码。如果训练时用 `latent_full` 的中段切片作为 GT，其 latent 表示混合了前面帧的时序信息，**与推理时的 latent 不一致** → train-inference mismatch → 模型学到错误的分布。

---

## 1. Wan VAE 的因果性与非因果性

Wan VAE 基于 **3D 卷积 + 时序下采样**（`infworld/vae/` 中有 patch_size `(4,8,8)`），具有：

### 1.1 时序维度的**单向因果性**
- Latent 帧 t 只依赖**当前及之前**的像素帧 `[0, 4t+3]`（4 倍压缩）。
- **前缀切片性质**：`encode(video[:T1])` 的结果与 `encode(video[:T2])[:T1的latent帧数]` **精确相等**（只要 T1 是 4 的倍数+1）。
- 这就是为什么 `latent_full[:, :20k+1]` 可以作为 image_cond — 它等价于单独编码历史前缀。

### 1.2 空间维度的**非因果性**（边界效应）
- 3D 卷积的 padding 策略导致边界像素会"看到"邻近的像素。
- 但在完整分辨率下，中心区域的 latent 在空间上基本是局部的。
- **时序维度的影响更关键**，下面重点讨论它。

---

## 2. 推理时的 chunk 编码行为（关键事实）

查看 `scripts/main.py` 第 698-699 行：

```python
def _generate_chunk(...):
    with torch.no_grad():
        current_cond = video_buffer.to(device)         # video_buffer = 当前已生成的所有帧
        cond_latent = models.vae.encode(current_cond)  # ← 每次 chunk 都重新 encode buffer
```

以及 `generate_one_video` 的循环（第 770 行）：

```python
video_buffer = torch.cat([video_buffer, result.decoded[:, :, 1:]], dim=2)  # 拼接新生成的 80 帧
```

### 推理时的时序关系

| Chunk | video_buffer 像素帧范围 | VAE 输入 | cond_latent 形状 | 备注 |
|-------|----------------------|---------|----------------|------|
| 0 | `[0]` | 第一帧 | `[1,16,1,h,w]` | 首帧 |
| 1 | `[0, 1..81)` | 帧 0-80 | `[1,16,21,h,w]` | 帧 0 是"第一帧" |
| 2 | `[0, 1..81, 81..161)` | 帧 0-160 | `[1,16,41,h,w]` | 帧 0 仍是"第一帧"，但帧 80 **不是首帧** |
| k | `[0, ..., 80(k-1)+1..80k+1)` | 帧 0 到 80k | `[1,16,20k+1,h,w]` | 帧 80(k-1)+1 到 80k 是缓冲区的**尾部** |

**核心观察**：
- chunk 0 的 GT（帧 0-80）单独编码 → `latent_chunks[0]`，帧 0 被 VAE 看作**首帧**。
- chunk k (k>0) 的 GT（帧 80k 到 80k+80）在推理时**不是单独编码的**！而是：
  1. 推理先用已生成的 buffer `[0, ..., 80k]` 生成帧 80k+1 到 80k+80。
  2. 训练时，这 81 帧（80k 到 80k+80）的"真实 latent"应该是什么？

---

## 3. 训练时的两个可能方案与正确性分析

### 方案 A：用 `latent_full` 的中段切片（❌ 错误）

```python
# 假设这样做：
x_start = latent_full[:, 20k : 20k+21]   # latent_full 来自 encode(整段 1800 帧)
```

**问题**：
- `latent_full` 是一次性编码 1800 帧的结果。
- Latent 帧 `20k` 对应像素帧 `80k`，但 VAE 在编码时**知道这是视频的中段**（前面有 80k 帧的上下文）。
- VAE 的时序卷积（如 `TemporalDownsample`，stride=2 的 3D conv）在处理帧 80k 时，其感受野**包含了前面的帧**（卷积核跨越时间维度）。
- 结果：`latent_full[:, 20k]` 混合了帧 `[80k-3, 80k]`（假设 kernel_size=3）的信息 → **不等于单独编码帧 80k 作为首帧的结果**。

**与推理不一致**：
- 推理时 chunk k 会将像素帧 80k 到 80k+80 **独立编码**（作为新的 81 帧序列）。
- 训练时却用"帧 80k 在 1800 帧完整序列中的 latent 表示"作为 GT。
- 模型学到的 x_start 分布与推理时 VAE 实际产生的分布不匹配 → **train-inference mismatch**。

---

### 方案 B：独立编码每个 chunk（✅ 正确）

```python
# chunk k 的 GT：
chunk_pixels = video_frames[:, :, 80k : 80k+81]       # 提取这 81 帧
x_start = vae.encode(chunk_pixels)                     # 单独编码，帧 80k 被当作首帧
# 存储：latent_chunks[k] = x_start
```

**为什么正确**：
- 推理时 `_generate_chunk` 会重新 encode buffer → 帧 80k+1 到 80k+80 是**基于 buffer 尾部重新编码的 condition 生成的**。
- 训练时我们需要模拟"如果模型在 chunk k 生成了这 81 帧，VAE 编码后的 latent 是什么"。
- 独立编码 chunk k 的 81 帧，等价于：
  - 帧 80k 被 VAE 视为这个**新序列的首帧**（感受野内没有历史）。
  - 这与推理时"把 buffer 尾部当作独立序列"的行为完全一致。

**数学上的严格性**：
- 记 VAE 编码器为 `E`，像素帧序列为 `v[:]`。
- **因果性保证**：`E(v[:T])[:t] == E(v[:4t+3])`（前缀一致）。
- **独立性要求**：`E(v[80k:80k+81])` ≠ `E(v[:80k+81])[20k:20k+21]`
  - 前者：VAE 只"看到" 81 帧（帧 80k 是首帧）。
  - 后者：VAE"看到"完整 80k+81 帧（帧 80k 的 latent 混合了前面帧的信息）。

---

## 4. 实验验证方法（推理脚本已有的行为）

查看 `main.py:698-729` 的 `_generate_chunk` 函数：

```python
# 每个 chunk 都会：
current_cond = video_buffer.to(device)                # buffer = [0, ..., 80k]
cond_latent = models.vae.encode(current_cond)         # 重新 encode（不是缓存的 latent）

samples = models.scheduler.sample(
    ...,
    additional_args={"image_cond": cond_latent, ...}  # 用重新编码的 cond
)
decoded = models.vae.decode(samples).cpu()            # samples 是模型预测的 latent
```

**推理的真相**：
- 模型在 chunk k 时接收的 `image_cond` 是 `encode(buffer)`，其中 buffer 是**之前所有 chunk 拼接的像素帧**。
- 模型预测的 `samples` 会被 VAE decode 回像素。
- 如果训练时的 x_start 不等于"独立编码这 81 帧"的结果，模型学到的就是错误的目标。

---

## 5. 为什么 image_cond 可以用 `latent_full` 切片？

**因为 image_cond 是历史前缀**：
- chunk k 的 image_cond = `latent_full[:, :20k+1]` = 编码 `video[:80k+1]` 的结果。
- 因果性保证：`encode(video[:80k+1])` 的前 20k+1 个 latent 帧与 `encode(video[:整段])[:20k+1]` **完全相同**。
- 所以 image_cond 可以从 `latent_full` 切片，无需重复编码。

**但 x_start 是"未来的生成结果"**：
- x_start = chunk k 的 81 帧（80k 到 80k+80）编码后的 latent。
- 这 81 帧在推理时是**模型生成的**（不在 image_cond 里）。
- 推理时模型生成后，VAE 会把它们当作**新的独立序列**编码（下一个 chunk 的 cond）。
- 所以训练时 x_start 必须用"独立编码"的版本，才能与推理对齐。

---

## 6. 代码中的体现

### `preprocess_dataset.py` 中的处理（即将实现）

```python
def encode_latent_full(vae, frames):
    """整段编码 → 用于 image_cond（前缀切片）"""
    with torch.no_grad():
        return vae.encode(frames)  # [16, T_lat, h, w]

def encode_latent_chunks(vae, frames, K, frames_per_chunk, stride):
    """逐 chunk 独立编码 → 用于 x_start（GT）"""
    latent_chunks = []
    for k in range(K):
        start = k * stride                        # stride = 80（1 帧重叠）
        end = start + frames_per_chunk            # frames_per_chunk = 81
        chunk_frames = frames[:, :, start:end]    # 提取这 81 帧
        with torch.no_grad():
            chunk_latent = vae.encode(chunk_frames)  # 单独编码！
        latent_chunks.append(chunk_latent)
    return torch.stack(latent_chunks)             # [K, 16, 21, h, w]
```

### `train.py` 中的使用

```python
def make_chunk_batch(video: VideoSample, k: int, device) -> ChunkBatch:
    x_start    = video.latent_chunks[k]              # ← 独立编码的 GT
    image_cond = video.latent_full[:, : 20*k + 1]    # ← 前缀切片的 cond
    # ...
```

---

## 7. 总结

| 对象 | 是否可以用 latent_full 切片？ | 原因 |
|------|----------------------------|------|
| **image_cond** | ✅ **可以** | 是历史前缀，因果性保证切片一致 |
| **x_start** | ❌ **不可以** | 是"未来生成"，推理时会独立编码，训练必须对齐 |

**最终答案**：
- 推理时 chunk k 的 81 帧会在**下一个 chunk**被重新 encode（加入 buffer 后重新编码整个 buffer）。
- 训练时如果用 `latent_full` 切片作 GT，latent 表示会混合时序卷积感受野内的历史信息。
- 独立编码每个 chunk 确保：**训练时的 x_start 与推理时"模型生成 → VAE 编码"的结果分布一致**。

这是 **train-inference alignment** 的必然要求，否则模型优化的目标与推理时实际产生的 latent 不在同一分布上。
