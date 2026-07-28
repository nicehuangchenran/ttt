# Infinite World — Flow-Matching 训练架构设计（最终版）

> 目标：对 `scripts/main.py` 中的 WanModel（1.3B DiT）做 flow-matching 微调。
> 训练数据：60s 视频（30fps ≈ 1800 帧），目录结构 `dataset-dir/case{n}/{video.mp4, move_view.json, prompts.json, image.jpg}`。
> 本版训练**不开启 TTT**，但保留接口。

---

## 0. 已确认的设计决策

| # | 决策项 | 结论 |
|---|--------|------|
| 1 | 分块策略 | 按 chunk（81 帧）训练；**同一视频所有 chunk 累积梯度后做一次 optimizer.step()** |
| 2 | chunk 重叠 | **1 帧重叠**（chunk k = 像素帧 `[80k, 80k+81)`），与推理自回归行为完全对齐 |
| 3 | image_cond | chunk 0 = 视频第一帧；chunk k = **GT 历史前缀** `[0, 80k+1)`（推理时 chunk 0 的 buffer 恰好就是第一帧，因此与"第一帧作为 condition"一致）。⚠️ 该逻辑隔离在 `make_chunk_batch()` 一个函数内，若要改为"所有 chunk 只用第一帧"只需改这一处 |
| 4 | VAE 编码 | **离线预处理**（单独脚本），训练时不加载 VAE |
| 5 | 文本编码 | **离线预处理**，训练时不加载 T5（省 ~10GB+ 显存） |
| 6 | 时序压缩 | 训练中**使用** TemporalLatentEncoder（与推理一致）；是否训练它由开关 `train_temporal_encoder` 控制 |
| 7 | 损失范围 | **仅对生成帧**（模型输出的最后 21 个 latent 帧）计算 MSE，condition 区域切掉 |
| 8 | TTT | 本版不开启；保留 `TTTConfig` + `ttt_adapt_on_chunk()` 接口，字段与 `main.py::OnlineTrainConfig` 对齐 |
| 9 | 日志 | 同时输出到命令行和 `train_log/<run_name>/` |
| 10 | 权重 | 保存到 `weights/<run_name>/checkpoint-{step}.ckpt`；**绝不写 `checkpoints/`**；命名兼容 `main.py::_extract_ckpt_step` 的 `checkpoint-(\d+).ckpt` 解析，可直接被推理脚本加载 |

---

## 1. 总体结构：两个脚本

```
scripts/preprocess_dataset.py   # 一次性：视频/文本 → latent/embedding，存盘
scripts/train.py                # 训练：只加载 DiT，读预处理结果
```

数据流总览：

```
dataset/sekai-.../case667/          preprocess_dataset.py           preprocessed/sekai-.../case667/
  video.mp4        ──VAE(整段)──────────────────────────────────►    latent_full.pt    [16, T_lat, h, w]
  video.mp4        ──VAE(逐chunk独立编码)──────────────────────►    target_latent.pt  [K, 16, 21, h, w]
  prompts.json     ──UMT5────────────────────────────────────►    text_emb.pt       {y, y_mask}
  move_view.json   ──ACTION_MAP──────────────────────────────►    actions.pt        {move, view}
                                                                   meta.json         {K, T_pix, prompt, ...}

preprocessed/case*/  ──Dataset──►  VideoSample  ──make_chunk_batch(k)──►  ChunkBatch
  ──flow_matching_loss──►  loss/K  ──backward(累积)──►  (K 个 chunk 后) optimizer.step()
  ──►  TrainLogger(命令行 + train_log/)  +  Checkpointer(weights/)
```

### 为什么存两份 latent（关键正确性设计）

- `latent_full`：整段视频一次编码。Wan VAE 是**因果**的，所以任意前缀切片
  `latent_full[:, :20k+1]` 与"单独编码像素帧 `[0, 80k+1)`"**精确相等** → 用作 image_cond。
- `target_latent[k]`：像素帧 `[80k, 80k+81)` **独立**编码（帧 80k 被当作新的首帧）。
  这与推理时"重新 encode buffer 尾部"的行为一致，而 `latent_full` 的中段切片做不到
  （中段 latent 帧混合了前 3 个像素帧的信息）→ 用作 x_start（GT）。

### chunk 数学（1 帧重叠）

| 量 | 公式 | 60s/1800 帧示例 |
|----|------|----------------|
| chunk k 像素帧 | `[80k, 80k+81)` | chunk 21 = `[1680, 1761)` |
| chunk 数 K | `(T_pix - 1) // 80`，可被 `max_chunks_per_video` 截断 | K = 22 |
| 动作切片 | `move/view[80k : 80k+81)`，尾部零填充（同推理 `_slice_move_view`） | |
| x_start latent | `latent_chunks[k]`，T=21 | |
| image_cond latent | `latent_full[:, :20k+1]` | chunk 21 → T_in=421，DiT 内部滑窗压缩到 20+1 |

---

## 2. `scripts/preprocess_dataset.py`

职责单一：把每个 case 变成训练可直接消费的张量文件。GPU 上只需 VAE + T5，不需要 DiT。

```python
# ============================================================
# §P1 配置
# ============================================================
@dataclass
class PreprocessConfig:
    dataset_dir: str = "dataset/sekai-game-walking-854_480_30fps"
    output_dir: str = "preprocessed/sekai-game-walking-854_480_30fps"
    model_config_path: str = "configs/infworld_config.yaml"   # 取 vae_cfg / text_encoder_cfg
    bucket_config_name: str = "ASPECT_RATIO_627_F64"          # 与推理同一套 resize+center-crop
    frames_per_chunk: int = 81
    chunk_stride: int = 80            # = frames_per_chunk - 1（1 帧重叠）
    save_dtype: str = "bfloat16"
    skip_existing: bool = True        # 断点续跑
    verify_prefix: bool = False       # 抽查因果性假设（见 §P4）

# ============================================================
# §P2 单视频编码
# ============================================================
def load_video_frames(video_path: str, bucket_config: dict) -> torch.Tensor:
    """读 mp4 全部帧 → resize+center-crop 到 bucket → 归一化 [-1,1]。
    截断到满足 T ≡ 1 (mod 4) 的最大帧数（VAE 时序压缩要求）。
    返回: [1, 3, T_pix, H, W]（CPU）。复用推理的 _resize_and_center_crop 逻辑。"""

def encode_latent_full(vae, frames: torch.Tensor) -> torch.Tensor:
    """整段编码。frames [1,3,T_pix,H,W] → [16, T_lat, h, w]，T_lat = (T_pix-1)//4 + 1。"""

def encode_target_latent(vae, frames: torch.Tensor, K: int,
                         frames_per_chunk: int, stride: int) -> torch.Tensor:
    """逐 chunk 独立编码：对 k in range(K) 编码 frames[:, :, 80k:80k+81]。
    返回: [K, 16, 21, h, w]。"""

# ============================================================
# §P3 单 case 处理与主循环
# ============================================================
def preprocess_case(case_dir: str, out_dir: str, vae, text_encoder,
                    bucket_config: dict, cfg: PreprocessConfig) -> dict:
    """一个 case 的全部预处理，写出 4 个 .pt + meta.json，返回 meta。
    输出契约（train.py 的唯一输入）:
      latent_full.pt    [16, T_lat, h, w]        (save_dtype)
      target_latent.pt  [K, 16, 21, h, w]        (save_dtype)
      text_emb.pt       {"y": [1,1,L,D], "y_mask": [1,L]}
      actions.pt        {"move": Long[T_pix], "view": Long[T_pix]}
      meta.json         {"name", "num_chunks", "num_frames", "height", "width", "prompt"}"""

def main():
    """扫描 case 目录（复用 main.py 的 _case_sort_key）→ 逐个 preprocess_case。
    支持 CUDA_VISIBLE_DEVICES 单卡运行；多卡时按 case 取模分片。"""

# ============================================================
# §P4 自检（可选）
# ============================================================
def verify_prefix_consistency(vae, frames: torch.Tensor, atol=1e-3) -> bool:
    """抽查: encode(frames[:, :, :81])  ≟  encode_latent_full(frames)[:, :21]。
    验证'因果 VAE ⇒ 前缀切片精确'这一 image_cond 的核心假设。"""
```

---

## 3. `scripts/train.py`

```
train.py
├── §1  TrainConfig / TTTConfig          配置（全部超参集中于此）
├── §2  parse_cli / build_train_config   CLI 覆盖
├── §3  setup_runtime                    单卡 / torchrun DDP（复用 main.py 模式）
├── §4  TrainLogger / Checkpointer       日志 + 权重
├── §5  PreprocessedVideoDataset         数据（一个样本 = 一个完整视频）
├── §6  build_dit / configure_trainable  模型加载 + 冻结策略
├── §7  make_chunk_batch                 视频 → 第 k 个 chunk 的训练输入（核心切分逻辑）
├── §8  flow_matching_loss               单 chunk 前向 + loss
├── §9  train_one_video                  chunk 循环 + 梯度累积 + 一次 step
├── §10 TTT 接口（预留，本版不实现）
├── §11 train_loop                       epoch/step 循环 + 日志 + 保存
└── §12 main                             纯编排
```

模块间只通过 3 个数据类传递：`VideoSample`（§5→§7/§9）、`ChunkBatch`（§7→§8/§10）、`TrainModels`（§6→§8/§9/§11）。

### §1 配置

```python
@dataclass
class TrainConfig:
    # --- 数据 ---
    data_dir: str = "preprocessed/sekai-game-walking-854_480_30fps"
    max_chunks_per_video: int = -1        # -1 = 全部（60s → 22）；调试时可设小
    num_workers: int = 2

    # --- 优化 ---
    lr: float = 1e-5
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.95)
    max_grad_norm: float = 1.0
    epochs: int = 10
    videos_per_step: int = 1              # 跨视频再累积（视频内必累积，见决策#1）
    lr_schedule: str = "constant"         # "constant" | "cosine"
    warmup_steps: int = 0

    # --- flow matching（与 configs/infworld_config.yaml 的 val_scheduler_cfg 对齐）---
    num_timesteps: int = 1000
    shift: float = 7.0                    # 与 627 分辨率配套（256:3, 960:11）
    use_reversed_velocity: bool = True    # target = -(x0 - noise)，同预训练
    timestep_sample: str = "uniform"

    # --- 模型 ---
    model_config_path: str = "configs/infworld_config.yaml"
    init_checkpoint: str = "checkpoints/infinite_world_model.ckpt"   # 只读，绝不写回
    train_temporal_encoder: bool = False  # 开关：latent_encoder（时序压缩）是否参与训练
    use_grad_checkpoint: bool = True
    master_dtype: str = "float32"         # fp32 主权重 + bf16 autocast（更稳）；"bfloat16" 省显存

    # --- 输出 ---
    run_name: str = ""                    # 空 → 自动 "ft-{YYYYmmdd-HHMMSS}"
    weights_dir: str = "weights"          # → weights/<run_name>/checkpoint-{step}.ckpt
    log_dir: str = "train_log"            # → train_log/<run_name>/{train.log, metrics.jsonl}
    save_every_n_steps: int = 200
    log_every_n_steps: int = 1
    resume_from: str = ""                 # weights/<run>/checkpoint-N.ckpt → 恢复训练

    seed: int = 42


@dataclass
class TTTConfig:
    """训练中 TTT 的接口预留。open=False 时整个训练路径与其无关。
    字段与 scripts/main.py::OnlineTrainConfig 对齐，便于日后合并。"""
    open: bool = False
    n_train_steps: int = 5
    lr: float = 1e-5
    trainable_layers: tuple = ("head", "blocks.modulation", "blocks.norm3")
    trainable_block_start: int = 18
    reset_between_videos: bool = True
```

### §3 运行时

```python
def setup_runtime(seed: int) -> RuntimeContext:
    """与 main.py 相同：有 RANK 环境变量 → DDP(nccl)；无 → 单卡直跑。
    设置 cp_util 全局 rank（context_parallel_size 固定为 1，训练走纯 DP）。"""
```

### §4 日志与权重

```python
class TrainLogger:
    """rank0 写文件，所有 rank 打印到命令行（带 [rank] 前缀）。"""
    def __init__(self, log_dir: str, run_name: str, rank: int): ...
    def info(self, msg: str) -> None:
        """命令行 + train_log/<run_name>/train.log（带时间戳）。"""
    def metrics(self, step: int, **kv) -> None:
        """结构化指标 → 命令行一行 + metrics.jsonl 追加一行 JSON。
        典型: loss, grad_norm, lr, video, num_chunks, sec_per_video。"""

class Checkpointer:
    def __init__(self, weights_dir: str, run_name: str, rank: int): ...
    def save(self, dit, optimizer, lr_sched, global_step: int, config) -> str:
        """rank0 写 weights/<run_name>/checkpoint-{global_step}.ckpt:
        {"state_dict": dit权重(去DDP前缀, 剔除 pos_embed*), 
         "optimizer": ..., "lr_sched": ..., "global_step": ..., "config": asdict}
        推理脚本可直接加载（它认 "state_dict" 键、按文件名解析 step）。"""
    def load_for_resume(self, path: str, dit, optimizer, lr_sched) -> int:
        """恢复权重+优化器状态，返回 global_step。"""
```

### §5 数据

```python
@dataclass
class VideoSample:
    """Dataset 的输出单位：一个完整视频（CPU 张量）。"""
    name: str                    # "case667"
    latent_full: torch.Tensor    # [16, T_lat, h, w]
    target_latent: torch.Tensor  # [K, 16, 21, h, w]
    y: torch.Tensor              # [1, 1, L, D]  预编码文本
    y_mask: torch.Tensor         # [1, L]
    move: torch.Tensor           # Long [T_pix]
    view: torch.Tensor           # Long [T_pix]
    num_chunks: int              # K（已按 max_chunks_per_video 截断）

class PreprocessedVideoDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, max_chunks_per_video: int = -1):
        """扫描含 meta.json 的 case 目录；缺文件的 case 跳过并告警。"""
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> VideoSample: ...

def build_dataloader(cfg: TrainConfig, runtime: RuntimeContext) -> DataLoader:
    """batch_size 固定 1（一个'样本'就是一个 22-chunk 的视频）。
    DDP 时用 DistributedSampler(shuffle=True)；collate_fn 直接透传 VideoSample。"""
```

### §6 模型

```python
@dataclass
class TrainModels:
    dit: nn.Module               # DDP 包装后（单卡则裸模型）
    dit_raw: nn.Module           # 未包装引用（保存/冻结用）
    trainable_params: list

def build_dit(cfg: TrainConfig, runtime: RuntimeContext) -> TrainModels:
    """1. 按 model_config_path 实例化 WanModel（不加载 VAE/T5！
          out_channels=16 / caption_channels=4096 / model_max_length=512 直接写死，
          与 yaml 一致，避免为读两个常量而加载 10GB 编码器）
       2. 加载 init_checkpoint（剔除 pos_embed*，strict=False，同 main.py）
       3. .to(master_dtype)，train_mode；use_grad_checkpoint → set_grad_checkpoint(dit)
       4. configure_trainable()  5. DDP 包装（find_unused_parameters 按需）"""

def configure_trainable(dit, train_temporal_encoder: bool) -> list:
    """DiT 主体全部可训练；'latent_encoder.*'（时序压缩）由开关决定 requires_grad。
    打印可训练/冻结参数量统计。返回可训练参数列表。"""
```

### §7 chunk 组装（切分逻辑唯一入口）

```python
@dataclass
class ChunkBatch:
    """一次 DiT 前向所需的全部输入（已在 device 上）。"""
    x_start: torch.Tensor      # [1, 16, 21, h, w]  GT latent
    image_cond: torch.Tensor   # [1, 16, T_in, h, w]  T_in = 20k+1
    move: torch.Tensor         # Long [1, 81]
    view: torch.Tensor         # Long [1, 81]
    y: torch.Tensor            # [1, 1, L, D]
    y_mask: torch.Tensor       # [1, L]

def make_chunk_batch(video: VideoSample, k: int, device) -> ChunkBatch:
    """第 k 个 chunk 的训练输入（本文件中唯一知道'怎么切'的函数）:
      x_start    = latent_chunks[k]                      # 独立编码，含重叠首帧
      image_cond = latent_full[:, : 20*k + 1]            # GT 历史前缀；k=0 → 第一帧
      move/view  = 像素帧 [80k, 80k+81) 切片，尾部零填充   # 同推理 _slice_move_view
    ⚠️ 若要改为'所有 chunk 只用第一帧作 cond'，仅改 image_cond 这一行。"""
```

### §8 flow-matching 损失（对齐 `main.py::_online_train_step` 的已验证路径）

```python
def sample_timestep(batch: int, cfg: TrainConfig, device) -> torch.Tensor:
    """t ~ U(0, num_timesteps)，再过 timestep_transform(shift=cfg.shift)。"""

def flow_matching_loss(models: TrainModels, chunk: ChunkBatch,
                       cfg: TrainConfig) -> torch.Tensor:
    """单 chunk 的 rectified-flow 损失（返回带梯度的标量）:
      t      = sample_timestep(...)
      noise  = randn_like(x_start)
      x_t    = (1 - t/N) * x_start + (t/N) * noise        # RFlowScheduler.add_noise
      target = x_start - noise; 若 use_reversed_velocity: target = -target
      pred   = dit(x_t, t, y, y_mask, x_ignore_mask=None,
                   image_cond=..., move=..., view=...)
      pred   = pred[:, :, -21:]                            # 只留生成帧（决策#7）
      loss   = mse(pred.float(), target.float())
    autocast(bf16) 包裹前向；loss 在 fp32 计算。"""
```

### §9 单视频训练（梯度累积核心）

```python
def train_one_video(models: TrainModels, video: VideoSample,
                    optimizer, cfg: TrainConfig, ttt_cfg: TTTConfig,
                    device, is_step_boundary: bool) -> dict:
    """一个视频 = K 次 前向+backward + （在 step 边界时）一次参数更新:

      K = video.num_chunks
      for k in range(K):
          chunk = make_chunk_batch(video, k, device)
          loss  = flow_matching_loss(models, chunk, cfg) / K     # K-chunk 均值
          with dit.no_sync() if (DDP 且非最后一次 backward):     # 只在最后同步梯度
              loss.backward()
          if ttt_cfg.open and k < K - 1:                          # ← TTT 挂载点（本版不触发）
              ttt_adapt_on_chunk(models, chunk, ttt_cfg)
          释放 chunk / _torch_gc()

      if is_step_boundary:                    # videos_per_step 个视频累积完
          grad_norm = clip_grad_norm_(trainable_params, cfg.max_grad_norm)
          optimizer.step(); optimizer.zero_grad(set_to_none=True)

      返回 {"loss": K个chunk损失均值, "per_chunk_loss": [...], "grad_norm": ...}"""
```

### §10 TTT 接口（预留）

```python
def ttt_adapt_on_chunk(models: TrainModels, chunk: ChunkBatch,
                       ttt_cfg: TTTConfig) -> list[float]:
    """预留：训练循环中、chunk 之间的 test-time 式适配。
    未来实现要点（与 main.py §8 同构）:
      - 用 _select_trainable_params 风格的 whitelist 建独立的快权重参数组
      - 每 chunk 新建 AdamW（Adam 动量不跨 chunk）
      - reset_between_videos: 视频开始前快权重回滚
    当前: raise NotImplementedError（open=False 时永远不会走到这里）。"""
```

### §11–§12 主循环与编排

```python
def train_loop(models, dataloader, optimizer, lr_sched, logger, checkpointer,
               cfg: TrainConfig, ttt_cfg: TTTConfig, runtime, start_step: int) -> None:
    """global_step = 每 videos_per_step 个视频 +1。
    for epoch: sampler.set_epoch(epoch)
      for video in dataloader:
          stats = train_one_video(..., is_step_boundary=...)
          按 log_every_n_steps → logger.metrics(step, loss=..., video=video.name, ...)
          按 save_every_n_steps → checkpointer.save(...)
    结束时保存 final checkpoint。"""

def main():
    """cli → TrainConfig/TTTConfig → setup_runtime → TrainLogger/Checkpointer
    → build_dataloader → build_dit → AdamW(trainable_params) + lr_sched
    → (可选 resume) → train_loop。main 内无任何业务逻辑。"""
```

---

## 4. 显存与正确性要点

- **训练进程不加载 VAE/T5**：离线预处理后，train.py 显存 = DiT(1.3B) + 优化器 + 激活。
  fp32 主权重 + AdamW ≈ 5.2 + 10.4 GB，bf16 autocast 前向 + grad checkpoint，A100-80G 单卡可跑；
  紧张时 `master_dtype="bfloat16"`（与 main.py 在线训练同款，精度略降）。
- **image_cond 随 k 增长**（最长 421 latent 帧过 latent_encoder 滑窗压缩）：这正是推理路径，
  TemporalLatentEncoder 自带 grad checkpoint；`train_temporal_encoder=False` 时它不产生参数梯度，
  但激活仍参与反传图（x 与 cond 拼接后进 DiT）——冻结用 requires_grad=False 即可，无需 no_grad 包裹。
- **DDP + 视频内累积**：K 次 backward 中前 K-1 次用 `no_sync()`，只在最后一次做 all-reduce。
- **损失只算生成帧**：`pred[:, :, -21:]`，绕开 `RFlowScheduler.training_losses` 对
  `x_ignore_mask` 非 None 的硬依赖（与 main.py `_online_train_step` 相同的处理，已被验证可跑通该模型）。
- **num_c 状态**：DiT forward 每次都会按当前 image_cond 长度重写 `block.self_attn.num_c`，
  chunk 间条件长度变化是安全的；但这意味着**单进程内不能并发跑两个不同形状的前向**（现有代码同款约束）。
- **权重文件安全**：`checkpoints/` 只读；所有产物落 `weights/<run_name>/`、`train_log/<run_name>/`。

---

## 5. 实现顺序

1. `preprocess_dataset.py`（§P1–§P3）+ 用 1 个 case 跑通，`verify_prefix=True` 抽查前缀一致性
2. `train.py` §1–§6（配置/数据/模型能加载、参数统计正确）
3. §7–§9（单卡、单视频、`max_chunks_per_video=2` 走通 loss.backward + step，看 loss 是否合理下降）
4. §11 完整循环 + 日志 + 保存；用推理脚本加载 `weights/.../checkpoint-N.ckpt` 验证端到端兼容
5. torchrun 多卡 DDP 验证
6. （下一阶段）实现 §10 TTT
