# OOM 优化方案

## 问题诊断

根据代码分析，OOM 主要发生在：
1. **optimizer.step() 后**：AdamW 维护 fp32 动量状态，峰值显存 = 模型参数 × 3（梯度 + 一阶动量 + 二阶动量）
2. **第二个 step 开始时**：前一个 step 的中间激活、梯度、临时张量未完全释放
3. **TTT 阶段**：每个 chunk 创建新 AdamW optimizer，额外占用显存

## 优化策略

### 1. CPU Offload（最有效）
将不需要立即计算的数据放在 CPU：
- **视频数据**：latent_full、target_latent 在 Dataset 中保持在 CPU，按需移到 GPU
- **文本嵌入**：y、y_mask 在每个 chunk 开始时才移到 GPU
- **已处理的 chunk**：处理完立即移回 CPU 或删除

### 2. 更激进的显存清理
```python
# 在关键位置插入：
torch.cuda.synchronize()  # 等待所有 CUDA 操作完成
torch.cuda.empty_cache()  # 释放缓存池
```

关键位置：
- 每个 chunk backward 后
- optimizer.step() 前后
- 每个视频结束后

### 3. TTT 优化
- 使用 SGD 代替 AdamW（无状态开销）
- 或者降低 TTT 频率（每 N 个 chunk 做一次）

### 4. 梯度累积优化
- 及时释放不再需要的梯度
- 使用 `set_to_none=True` 而不是 `zero_grad()`

## 实施方案

### VideoSample 改为保持在 CPU
```python
@dataclass
class VideoSample:
    """所有张量保持在 CPU，按需移到 GPU"""
    name: str
    latent_full: torch.Tensor  # CPU
    target_latent: torch.Tensor  # CPU
    y: torch.Tensor  # CPU
    y_mask: torch.Tensor  # CPU
    move: torch.Tensor  # CPU
    view: torch.Tensor  # CPU
    num_chunks: int
```

### make_chunk_batch 改为按需加载
```python
def make_chunk_batch(video: VideoSample, k: int, device,
                     cond_chunk_num: int = -1) -> ChunkBatch:
    # 所有数据从 CPU 动态移到 GPU
    x_start = video.target_latent[k].unsqueeze(0).to(device)
    # ... 其他类似
```

### train_one_video 中插入清理点
```python
for k in range(K):
    chunk = make_chunk_batch(...)
    loss = flow_matching_loss(...)
    loss.backward()
    
    # 立即释放
    del loss, chunk
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    if ttt_cfg.open and k < K - 1:
        # TTT 后也清理
        ttt_losses.append(ttt_adapt_on_chunk(...))
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

# step 前清理
if is_step_boundary:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    grad_norm = torch.nn.utils.clip_grad_norm_(...)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    # step 后也清理
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
```

### TTT 使用 SGD
```python
# 在 ttt_adapt_on_chunk 中：
ttt_optimizer = torch.optim.SGD(params, lr=ttt_cfg.lr, momentum=0.9)
# 而不是 AdamW
```

## 预期效果

- **CPU Offload**：节省 ~40% 峰值显存（视频数据不常驻 GPU）
- **激进清理**：节省 ~10-15% 碎片显存
- **TTT 用 SGD**：节省 ~20% TTT 阶段显存（无动量状态）

总计可节省 **50-70% 峰值显存**。
