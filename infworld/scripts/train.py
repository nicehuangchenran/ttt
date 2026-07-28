"""
Infinite World - Flow-Matching Training Script
==============================================
训练 1.3B DiT 模型（rectified-flow），数据来自 preprocess_dataset.py 的输出。

Usage:
------
# 多 GPU 训练
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29500 \
    scripts/train.py \
    --data-dir preprocessed/sekai-game-walking-352_192_30fps --shift 3 \
    --filter-location "East Maddon Park, London, United Kingdom" --filter-weather sunny \
    --epochs 20 \
    --val-every-n-steps 16   
    
# 从 checkpoint 恢复训练
/mnt/efs/chenran/miniconda3/envs/infworld/bin/python3 scripts/train.py \
    --resume weights/my-run/step100.ckpt

参数说明:
  --data-dir: 预处理数据目录（必须包含 preprocess_dataset.py 的输出）
  --run-name: 运行名称（用于保存 checkpoint 和日志，默认自动生成）
  --max-chunks: 每个视频最多训练的 chunk 数（默认 3）
  --epochs: 训练轮数（默认 1）
  --lr: 学习率（默认 1e-5）
  --shift: 时间步转换的 shift 参数（默认 3.0，对应 256px 分辨率）
  --resume: 从指定 checkpoint 恢复训练
  --val-every-n-steps: 每隔 N 步做验证（0=关闭，默认 0） ,启用 timestep 分桶验证（观察不同噪声水平下的训练趋势）
  --filter-location: 按 location 筛选，如 "East Maddon Park, London, United Kingdom"
  --filter-scene: 按 scene 筛选，如 outdoor-urban
  --filter-crowd-density: 按 crowdDensity 筛选，如 empty
  --filter-weather: 按 weather 筛选，如 sunny
  --filter-time-of-day: 按 timeOfDay 筛选，如 day

输出:
  - Checkpoints: weights/<run-name>/step{step}.ckpt
      run-name 默认格式: [数据集名]-shift{shift}[-[filter]...]-chunks{max_chunks}-mm_dd-hh:mm:ss
      例: weights/[sekai-352_192]-shift3-[Castle_Rock_Beach]-[sunny]-chunks3-07_28-15:30:45/step100.ckpt
      ckpt 内容（5 个键）:
        state_dict  # DiT 权重，已剔除 pos_embed / pos_embed_temporal（加载时重算）
        optimizer   # AdamW 状态（动量等），用于 resume
        lr_sched    # 学习率调度器状态，未启用调度器时为 None
        global_step # 当前训练步数，resume 时接着计数
        config      # asdict(TrainConfig)，完整训练配置，供复现
      推理只用 state_dict

  - 日志文件（多卡训练优化，控制台只有 rank0 输出）:
      logs/train_log/<run-name>/tensorboard  
      logs/train_log/<run-name>/train.log           # rank0 主日志（控制台输出的镜像）
      logs/train_log/<run-name>/train_rank{N}.log   # 其他 rank 日志（调试用）
      logs/train_log/<run-name>/metrics.jsonl       # 训练指标（loss/lr/grad_norm）,只有 rank 0 记录
      logs/train_log/<run-name>/config.json         # 训练配置（可复现）,记录全部参数（TrainConfig/TTTConfig）、world_size、实际训练样本数

  - 调试技巧:
       tensorboard --logdir=logs/train_log/<run-name>/tensorboard  
      查看某个 rank 的日志: tail -f logs/train_log/<run-name>/train_rank3.log
      监控所有 rank 状态: grep -H "ERROR" logs/train_log/<run-name>/train_rank*.log

详细说明请查看: MULTI_GPU_LOGGING.md
"""

import sys
import os
import json
import time
import datetime
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass, asdict
from typing import Optional, Tuple
from omegaconf import OmegaConf
import numpy as np
import random
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from infworld.utils.prepare_dataloader import get_obj_from_str
from infworld.models.checkpoint import set_grad_checkpoint
from infworld.models.scheduler import timestep_transform


@dataclass
class TrainConfig:
    data_dir: str = "preprocessed/sekai-game-walking-854_480_30fps-256px"
    max_chunks_per_video: int = 3
    num_workers: int = 2
    lr: float = 1e-5
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.95)
    max_grad_norm: float = 1.0
    epochs: int = 1
    videos_per_step: int = 1
    lr_schedule: str = "constant"
    warmup_steps: int = 0
    num_timesteps: int = 1000
    shift: float = 3.0
    use_reversed_velocity: bool = True
    timestep_sample: str = "uniform"
    model_config_path: str = "configs/infworld_config.yaml"
    init_checkpoint: str = "checkpoints/infinite_world_model.ckpt"
    train_temporal_encoder: bool = False
    use_grad_checkpoint: bool = True
    master_dtype: str = "float32"
    run_name: str = ""
    weights_dir: str = "weights"
    log_dir: str = "logs/train_log"
    save_every_n_steps: int = 100
    log_every_n_steps: int = 1
    resume_from: str = ""
    seed: int = 42
    # timestep 分桶验证：每隔 N 步，在固定视频/固定 t/固定噪声上算 loss，
    # 剥离随机性以观察真实训练趋势。val_every_n_steps=0 表示关闭。
    val_every_n_steps: int = 0
    num_val_videos: int = 4
    num_val_buckets: int = 10
    # 数据筛选：根据 meta.json 的属性过滤样本，默认 None 表示不筛选
    filter_location: Optional[str] = None
    filter_scene: Optional[str] = None
    filter_crowd_density: Optional[str] = None
    filter_weather: Optional[str] = None
    filter_time_of_day: Optional[str] = None


@dataclass
class TTTConfig:
    open: bool = False
    n_train_steps: int = 5
    lr: float = 1e-5


def parse_cli():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--shift", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--val-every-n-steps", type=int, default=None)
    # 数据筛选参数
    parser.add_argument("--filter-location", type=str, default=None, help="Filter by location")
    parser.add_argument("--filter-scene", type=str, default=None, help="Filter by scene type")
    parser.add_argument("--filter-crowd-density", type=str, default=None, help="Filter by crowd density")
    parser.add_argument("--filter-weather", type=str, default=None, help="Filter by weather")
    parser.add_argument("--filter-time-of-day", type=str, default=None, help="Filter by time of day")
    return parser.parse_args()


def build_run_name(cfg: TrainConfig) -> str:
    """拼出默认 run 名：[数据集名]-shift{shift}[-[filter]...]-chunks{max_chunks}-mm_dd-hh:mm:ss。

    数据集名取 data_dir 的最后一段目录名，用方括号包住便于肉眼分段；
    shift 里的小数点换成 p，避免目录名出现多余的 '.'。
    生效的 filter 按固定顺序插在 shift 之后，每个 filter 也用方括号包裹，未指定的 filter 不出现。
    不显示 key（如 scene/weather），只显示值；location 只取逗号前的第一个地点。
    末尾加上当前时间 mm_dd-hh:mm:ss 作为唯一标识。
    """
    dataset = os.path.basename(cfg.data_dir.rstrip("/"))
    shift_str = f"{cfg.shift:g}".replace(".", "p")
    parts = [f"[{dataset}]", f"shift{shift_str}"]

    # 收集所有生效的 filter 值（不含 key）
    filter_values = []
    if cfg.filter_location:
        # location 只取逗号前的第一个部分
        loc_short = cfg.filter_location.split(",")[0].strip()
        filter_values.append(loc_short)
    if cfg.filter_scene:
        filter_values.append(cfg.filter_scene)
    if cfg.filter_crowd_density:
        filter_values.append(cfg.filter_crowd_density)
    if cfg.filter_weather:
        filter_values.append(cfg.filter_weather)
    if cfg.filter_time_of_day:
        filter_values.append(cfg.filter_time_of_day)

    # 每个 filter 值转为文件名安全的格式并用方括号包裹
    for value in filter_values:
        safe = "".join(c if c.isalnum() or c == "-" else "_" for c in str(value).strip())
        safe = "_".join(filter(None, safe.split("_")))
        parts.append(f"[{safe}]")

    parts.append(f"chunks{cfg.max_chunks_per_video}")
    timestamp = datetime.datetime.now().strftime("%m_%d-%H:%M:%S")
    parts.append(timestamp)
    return "-".join(parts)


def build_train_config(cli) -> Tuple[TrainConfig, TTTConfig]:
    cfg = TrainConfig()
    ttt_cfg = TTTConfig()
    if cli.data_dir: cfg.data_dir = cli.data_dir
    if cli.run_name: cfg.run_name = cli.run_name
    if cli.max_chunks: cfg.max_chunks_per_video = cli.max_chunks
    if cli.epochs: cfg.epochs = cli.epochs
    if cli.lr: cfg.lr = cli.lr
    if cli.shift: cfg.shift = cli.shift
    if cli.resume: cfg.resume_from = cli.resume
    if cli.val_every_n_steps is not None: cfg.val_every_n_steps = cli.val_every_n_steps
    # 数据筛选配置
    if cli.filter_location: cfg.filter_location = cli.filter_location
    if cli.filter_scene: cfg.filter_scene = cli.filter_scene
    if cli.filter_crowd_density: cfg.filter_crowd_density = cli.filter_crowd_density
    if cli.filter_weather: cfg.filter_weather = cli.filter_weather
    if cli.filter_time_of_day: cfg.filter_time_of_day = cli.filter_time_of_day
    if not cfg.run_name:
        cfg.run_name = build_run_name(cfg)
    return cfg, ttt_cfg


@dataclass
class RuntimeContext:
    local_rank: int
    global_rank: int
    world_size: int
    device: torch.device


def _setup_seed(seed: int, rank: int = 0):
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    torch.backends.cudnn.deterministic = True


def setup_runtime(seed: int) -> RuntimeContext:
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600))
        global_rank = dist.get_rank()
    else:
        local_rank = 0
        global_rank = 0
        world_size = 1
        torch.cuda.set_device(0)
    
    device = torch.device(f"cuda:{local_rank}")
    _setup_seed(seed, global_rank)
    
    import infworld.context_parallel.context_parallel_util as cp_util
    cp_util.dp_rank = global_rank
    cp_util.dp_size = world_size
    cp_util.cp_rank = 0
    cp_util.cp_size = 1
    
    return RuntimeContext(local_rank, global_rank, world_size, device)


class TrainLogger:
    def __init__(self, log_dir: str, run_name: str, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.log_dir = os.path.join(log_dir, run_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "train.log")
        self.metrics_file = os.path.join(self.log_dir, "metrics.jsonl")
        self.config_file = os.path.join(self.log_dir, "config.json")
        # 多卡时，非 rank0 的日志写到单独文件
        if self.rank != 0:
            self.log_file = os.path.join(self.log_dir, f"train_rank{self.rank}.log")

        # TensorBoard writer: 只有 rank0 创建
        self.writer = None
        if self.rank == 0:
            tensorboard_dir = os.path.join(self.log_dir, "tensorboard")
            self.writer = SummaryWriter(log_dir=tensorboard_dir)
            self.info(f"TensorBoard logging to: {tensorboard_dir}")
            self.info(f"Run: tensorboard --logdir={tensorboard_dir}")

    def dump_config(self, cfg: TrainConfig, ttt_cfg: TTTConfig,
                    runtime: RuntimeContext,
                    num_train_cases: Optional[int] = None):
        """把本次运行的全部参数写到 config.json，供事后复现和跑间对比。

        除 TrainConfig/TTTConfig 的字段外，还记录 world_size 和实际生效的数据量——
        这几项决定了 metrics.jsonl 该怎么解读。
        """
        if self.rank != 0:
            return
        record = {
            "run_name": cfg.run_name,
            "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "world_size": runtime.world_size,
            "num_train_cases": num_train_cases,
            "train_config": asdict(cfg),
            "ttt_config": asdict(ttt_cfg),
        }
        with open(self.config_file, 'w') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        self.info(f"Config saved: {self.config_file}")

    def info(self, msg: str, force_print: bool = False):
        """记录日志。多卡情况下只有 rank0 打印到控制台，避免混乱。

        Args:
            msg: 日志信息
            force_print: 强制所有 rank 打印（用于错误信息等关键场景）
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] [rank{self.rank}] {msg}"

        # 控制台输出：只有 rank0 或 force_print=True 时才打印
        if self.rank == 0 or force_print:
            print(line)

        # 文件输出：所有 rank 都写到各自的日志文件
        with open(self.log_file, 'a') as f:
            f.write(line + '\n')

    def metrics(self, step: int, **kv):
        """记录指标。只有 rank0 打印到控制台和写 metrics.jsonl。"""
        if self.rank != 0:
            return

        kv_str = ", ".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in kv.items())
        self.info(f"step {step}: {kv_str}")

        record = {"step": step, **kv}
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(record) + '\n')

        # TensorBoard logging
        if self.writer:
            for key, value in kv.items():
                # 只记录数值类型到 TensorBoard
                if isinstance(value, (int, float)):
                    # 将指标分类到不同的 group
                    if key.startswith("val_loss"):
                        # 验证 loss 按 timestep 分桶
                        self.writer.add_scalar(f"validation/{key}", value, step)
                    elif key == "loss":
                        self.writer.add_scalar("train/loss", value, step)
                    elif key == "grad_norm":
                        self.writer.add_scalar("train/grad_norm", value, step)
                    elif key == "lr":
                        self.writer.add_scalar("train/learning_rate", value, step)
                    elif key == "step_second":
                        self.writer.add_scalar("performance/step_time_seconds", value, step)
                    elif key == "num_chunks":
                        self.writer.add_scalar("data/num_chunks", value, step)
                    else:
                        # 其他未分类的数值指标
                        self.writer.add_scalar(f"other/{key}", value, step)
            self.writer.flush()

    def info_all_ranks(self, msg: str):
        """所有 rank 都打印的日志（例如错误信息、警告）。"""
        self.info(msg, force_print=True)

    def close(self):
        """关闭 TensorBoard writer"""
        if self.rank == 0 and self.writer:
            self.writer.close()
            self.info("TensorBoard writer closed")


class Checkpointer:
    def __init__(self, weights_dir: str, run_name: str, rank: int):
        self.rank = rank
        self.weights_dir = os.path.join(weights_dir, run_name)
        os.makedirs(self.weights_dir, exist_ok=True)

    def save(self, dit, optimizer, lr_sched, global_step: int, config):
        if self.rank != 0:
            return
        state_dict = dit.state_dict() if not isinstance(dit, nn.parallel.DistributedDataParallel) \
                     else dit.module.state_dict()
        state_dict.pop("pos_embed_temporal", None)
        state_dict.pop("pos_embed", None)
        ckpt = {
            "state_dict": state_dict,
            "optimizer": optimizer.state_dict(),
            "lr_sched": lr_sched.state_dict() if lr_sched else None,
            "global_step": global_step,
            "config": asdict(config),
        }
        path = os.path.join(self.weights_dir, f"step{global_step}.ckpt")
        torch.save(ckpt, path)
        print(f"[SAVE] Checkpoint saved: {path}")
    
    def load_for_resume(self, path: str, dit, optimizer, lr_sched) -> int:
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(dit, nn.parallel.DistributedDataParallel):
            dit.module.load_state_dict(ckpt["state_dict"], strict=False)
        else:
            dit.load_state_dict(ckpt["state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        if lr_sched and ckpt["lr_sched"]:
            lr_sched.load_state_dict(ckpt["lr_sched"])
        global_step = ckpt["global_step"]
        print(f"[RESUME] Loaded from {path}, step={global_step}")
        return global_step


@dataclass
class VideoSample:
    name: str
    latent_full: torch.Tensor
    target_latent: torch.Tensor
    y: torch.Tensor
    y_mask: torch.Tensor
    move: torch.Tensor
    view: torch.Tensor
    num_chunks: int


class PreprocessedVideoDataset(Dataset):
    def __init__(self, data_dir: str, max_chunks_per_video: int = -1,
                 filter_location: Optional[str] = None,
                 filter_scene: Optional[str] = None,
                 filter_crowd_density: Optional[str] = None,
                 filter_weather: Optional[str] = None,
                 filter_time_of_day: Optional[str] = None,
                 rank: int = 0):
        self.data_dir = data_dir
        self.max_chunks = max_chunks_per_video
        self.cases = []

        # 统计过滤信息
        total_cases = 0
        filtered_out = 0

        for case_name in sorted(os.listdir(data_dir)):
            case_path = os.path.join(data_dir, case_name)
            meta_path = os.path.join(case_path, "meta.json")
            if os.path.isdir(case_path) and os.path.exists(meta_path):
                total_cases += 1

                # 读取 meta.json 并检查筛选条件
                with open(meta_path, 'r') as f:
                    meta = json.load(f)

                # 应用筛选条件
                if filter_location and meta.get("location") != filter_location:
                    filtered_out += 1
                    continue
                if filter_scene and meta.get("scene") != filter_scene:
                    filtered_out += 1
                    continue
                if filter_crowd_density and meta.get("crowdDensity") != filter_crowd_density:
                    filtered_out += 1
                    continue
                if filter_weather and meta.get("weather") != filter_weather:
                    filtered_out += 1
                    continue
                if filter_time_of_day and meta.get("timeOfDay") != filter_time_of_day:
                    filtered_out += 1
                    continue

                self.cases.append(case_name)

        # 只有 rank0 打印数据集统计信息
        if rank == 0:
            print(f"[Dataset] Found {total_cases} total cases in {data_dir}")
            if filtered_out > 0:
                print(f"[Dataset] Filtered out {filtered_out} cases, using {len(self.cases)} cases")
                if filter_location:
                    print(f"  - location: {filter_location}")
                if filter_scene:
                    print(f"  - scene: {filter_scene}")
                if filter_crowd_density:
                    print(f"  - crowdDensity: {filter_crowd_density}")
                if filter_weather:
                    print(f"  - weather: {filter_weather}")
                if filter_time_of_day:
                    print(f"  - timeOfDay: {filter_time_of_day}")
            else:
                print(f"[Dataset] No filters applied, using all {len(self.cases)} cases")
    
    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx: int) -> VideoSample:
        case_name = self.cases[idx]
        case_dir = os.path.join(self.data_dir, case_name)
        latent_full = torch.load(os.path.join(case_dir, "latent_full.pt"))
        target_latent = torch.load(os.path.join(case_dir, "target_latent.pt"))
        text_emb = torch.load(os.path.join(case_dir, "text_emb.pt"))
        actions = torch.load(os.path.join(case_dir, "actions.pt"))
        K = target_latent.shape[0]
        if self.max_chunks > 0 and K > self.max_chunks:
            K = self.max_chunks
            target_latent = target_latent[:K]
        return VideoSample(
            name=case_name, latent_full=latent_full, target_latent=target_latent,
            y=text_emb["y"], y_mask=text_emb["y_mask"],
            move=actions["move"], view=actions["view"], num_chunks=K,
        )


def build_dataloader(cfg: TrainConfig, runtime: RuntimeContext) -> DataLoader:
    dataset = PreprocessedVideoDataset(
        cfg.data_dir, cfg.max_chunks_per_video,
        filter_location=cfg.filter_location,
        filter_scene=cfg.filter_scene,
        filter_crowd_density=cfg.filter_crowd_density,
        filter_weather=cfg.filter_weather,
        filter_time_of_day=cfg.filter_time_of_day,
        rank=runtime.global_rank,
    )
    sampler = DistributedSampler(dataset, shuffle=True) if runtime.world_size > 1 else None
    return DataLoader(
        dataset, batch_size=1, shuffle=(sampler is None), sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=True, collate_fn=lambda x: x[0],
    )


@dataclass
class TrainModels:
    dit: nn.Module
    dit_raw: nn.Module
    trainable_params: list


def _resolve_path(path):
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path.strip())


def build_dit(cfg: TrainConfig, runtime: RuntimeContext) -> TrainModels:
    config_path = _resolve_path(cfg.model_config_path)
    model_args = OmegaConf.load(config_path)
    from infworld.models.dit_model import WanModel
    master_dtype = getattr(torch, cfg.master_dtype)
    dit = WanModel(
        out_channels=16, caption_channels=4096, model_max_length=512,
        enable_context_parallel=False, **model_args.model_cfg
    ).to(master_dtype)
    checkpoint_path = _resolve_path(cfg.init_checkpoint)
    if runtime.global_rank == 0:
        print(f"[LOAD] DiT checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    state_dict.pop("pos_embed_temporal", None)
    state_dict.pop("pos_embed", None)
    missing, unexpected = dit.load_state_dict(state_dict, strict=False)
    if runtime.global_rank == 0:
        print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    dit.train()
    if cfg.use_grad_checkpoint:
        set_grad_checkpoint(dit)
    dit.to(runtime.device)
    trainable_params = configure_trainable(dit, cfg.train_temporal_encoder, runtime.global_rank)
    if runtime.world_size > 1:
        dit = nn.parallel.DistributedDataParallel(
            dit, device_ids=[runtime.local_rank], output_device=runtime.local_rank,
            find_unused_parameters=False,
        )
    dit_raw = dit.module if isinstance(dit, nn.parallel.DistributedDataParallel) else dit
    return TrainModels(dit=dit, dit_raw=dit_raw, trainable_params=trainable_params)


def configure_trainable(dit, train_temporal_encoder: bool, rank: int = 0) -> list:
    trainable = []
    frozen = []
    for name, param in dit.named_parameters():
        if name.startswith("latent_encoder."):
            param.requires_grad_(train_temporal_encoder)
            if train_temporal_encoder:
                trainable.append(param)
            else:
                frozen.append(param)
        else:
            param.requires_grad_(True)
            trainable.append(param)
    trainable_count = sum(p.numel() for p in trainable)
    frozen_count = sum(p.numel() for p in frozen)
    if rank == 0:
        print(f"[Model] Trainable: {trainable_count:,} params, Frozen: {frozen_count:,} params")
    return trainable


@dataclass
class ChunkBatch:
    x_start: torch.Tensor
    image_cond: torch.Tensor
    move: torch.Tensor
    view: torch.Tensor
    y: torch.Tensor
    y_mask: torch.Tensor


def make_chunk_batch(video: VideoSample, k: int, device) -> ChunkBatch:
    x_start = video.target_latent[k].unsqueeze(0)
    T_in = 20 * k + 1
    image_cond = video.latent_full[:, :T_in].unsqueeze(0)
    start_frame = 80 * k
    end_frame = start_frame + 81
    move_seg = video.move[start_frame:end_frame]
    view_seg = video.view[start_frame:end_frame]
    if len(move_seg) < 81:
        pad_len = 81 - len(move_seg)
        move_seg = torch.cat([move_seg, torch.zeros(pad_len, dtype=torch.long)])
        view_seg = torch.cat([view_seg, torch.zeros(pad_len, dtype=torch.long)])
    return ChunkBatch(
        x_start=x_start.to(device), image_cond=image_cond.to(device),
        move=move_seg.unsqueeze(0).to(device), view=view_seg.unsqueeze(0).to(device),
        y=video.y.to(device), y_mask=video.y_mask.to(device),
    )


def sample_timestep(batch_size: int, cfg: TrainConfig, device) -> torch.Tensor:
    t = torch.rand((batch_size,), device=device) * cfg.num_timesteps
    t = timestep_transform(t, shift=cfg.shift, num_timesteps=cfg.num_timesteps)
    return t


def flow_matching_loss(models: TrainModels, chunk: ChunkBatch, cfg: TrainConfig) -> torch.Tensor:
    x_start = chunk.x_start
    B = x_start.shape[0]
    device = x_start.device
    t = sample_timestep(B, cfg, device)
    noise = torch.randn_like(x_start)
    t_normalized = t / cfg.num_timesteps
    t_normalized = t_normalized.view(B, 1, 1, 1, 1)
    x_t = (1 - t_normalized) * x_start + t_normalized * noise
    target = x_start - noise
    if cfg.use_reversed_velocity:
        target = -target
    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=(cfg.master_dtype == "float32")):
        pred = models.dit(
            x_t, t, y=chunk.y, y_mask=chunk.y_mask, x_ignore_mask=None,
            image_cond=chunk.image_cond, move=chunk.move, view=chunk.view,
        )
    pred = pred[:, :, -21:]
    loss = ((pred.float() - target.float()) ** 2).mean()
    return loss


@torch.no_grad()
def loss_at_fixed_timestep(
    models: TrainModels, chunk: ChunkBatch, cfg: TrainConfig,
    t_value: float, noise: torch.Tensor,
) -> float:
    """在给定的 timestep 和给定噪声下计算 loss（无随机性，用于验证）。

    和 flow_matching_loss 的算法完全一致，唯一区别是 t 和 noise 由外部传入，
    因此结果可复现、可在不同训练步之间比较。
    """
    x_start = chunk.x_start
    B = x_start.shape[0]
    t = torch.full((B,), t_value, device=x_start.device)
    t_normalized = (t / cfg.num_timesteps).view(B, 1, 1, 1, 1)
    x_t = (1 - t_normalized) * x_start + t_normalized * noise
    target = x_start - noise
    if cfg.use_reversed_velocity:
        target = -target
    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=(cfg.master_dtype == "float32")):
        pred = models.dit(
            x_t, t, y=chunk.y, y_mask=chunk.y_mask, x_ignore_mask=None,
            image_cond=chunk.image_cond, move=chunk.move, view=chunk.view,
        )
    pred = pred[:, :, -21:]
    return ((pred.float() - target.float()) ** 2).mean().item()


@torch.no_grad()
def validate_by_timestep(
    models: TrainModels, val_videos: list, cfg: TrainConfig, device: torch.device,
) -> dict:
    """按 timestep 分桶验证：把 [0, num_timesteps] 均分成 num_val_buckets 个桶，
    每个桶取中心 t，在固定的验证视频上算 loss 并对视频求平均。

    返回 {"val_loss_t100": ..., ..., "val_loss_mean": ...}，
    t 值越大噪声越多、loss 天然越高，所以要按桶分别看趋势。
    """
    models.dit.eval()
    n_buckets = cfg.num_val_buckets
    bucket_width = cfg.num_timesteps / n_buckets
    # 每个桶取中心点作为固定 t，例如 10 桶 → t = 50, 150, ..., 950
    bucket_ts = [(i + 0.5) * bucket_width for i in range(n_buckets)]
    results = {}
    all_losses = []
    for t_value in bucket_ts:
        per_video = []
        for v_idx, video in enumerate(val_videos):
            chunk = make_chunk_batch(video, 0, device)  # 固定用第 0 个 chunk
            # 固定种子的噪声：种子由 (视频序号, t 桶) 决定，跨训练步完全一致
            gen = torch.Generator(device=device).manual_seed(
                cfg.seed + v_idx * 1000 + int(t_value))
            noise = torch.randn(chunk.x_start.shape, generator=gen, device=device)
            per_video.append(loss_at_fixed_timestep(models, chunk, cfg, t_value, noise))
            del chunk
        bucket_loss = sum(per_video) / len(per_video)
        results[f"val_loss_t{int(t_value)}"] = bucket_loss
        all_losses.append(bucket_loss)
    results["val_loss_mean"] = sum(all_losses) / len(all_losses)
    models.dit.train()
    return results


def train_one_video(
    models: TrainModels, video: VideoSample, optimizer: torch.optim.Optimizer,
    cfg: TrainConfig, ttt_cfg: TTTConfig, device: torch.device, is_step_boundary: bool,
) -> dict:
    K = video.num_chunks
    losses = []
    for k in range(K):
        chunk = make_chunk_batch(video, k, device)
        loss = flow_matching_loss(models, chunk, cfg) / K
        losses.append(loss.item())
        if isinstance(models.dit, nn.parallel.DistributedDataParallel):
            if k < K - 1:
                with models.dit.no_sync():
                    loss.backward()
            else:
                loss.backward()
        else:
            loss.backward()
        del chunk, loss
        torch.cuda.empty_cache()
    grad_norm = 0.0
    if is_step_boundary:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            models.trainable_params, cfg.max_grad_norm).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {"loss": sum(losses), "per_chunk_loss": losses, "grad_norm": grad_norm}


def train_loop(
    models: TrainModels, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
    lr_sched: Optional[torch.optim.lr_scheduler._LRScheduler],
    logger: TrainLogger, checkpointer: Checkpointer,
    cfg: TrainConfig, ttt_cfg: TTTConfig, runtime: RuntimeContext, start_step: int,
    val_videos: Optional[list] = None,
):
    global_step = start_step
    video_count = 0
    for epoch in range(cfg.epochs):
        logger.info(f"=== Epoch {epoch + 1}/{cfg.epochs} ===")
        if hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
        for video in tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=(runtime.global_rank != 0)):
            video_count += 1
            is_step_boundary = (video_count % cfg.videos_per_step == 0)
            t_start = time.time()
            stats = train_one_video(
                models, video, optimizer, cfg, ttt_cfg, runtime.device, is_step_boundary
            )
            t_video = time.time() - t_start
            if is_step_boundary:
                global_step += 1
                if global_step % cfg.log_every_n_steps == 0:
                    logger.metrics(
                        step=global_step, loss=stats["loss"], grad_norm=stats["grad_norm"],
                        lr=optimizer.param_groups[0]["lr"], video=video.name,
                        num_chunks=video.num_chunks, step_second=t_video,
                    )
                if cfg.val_every_n_steps > 0 and global_step % cfg.val_every_n_steps == 0 \
                        and val_videos and runtime.global_rank == 0:
                    val_stats = validate_by_timestep(models, val_videos, cfg, runtime.device)
                    logger.metrics(step=global_step, **val_stats)
                if global_step % cfg.save_every_n_steps == 0:
                    checkpointer.save(models.dit, optimizer, lr_sched, global_step, cfg)
                if lr_sched:
                    lr_sched.step()
    logger.info("Training complete, saving final checkpoint...")
    checkpointer.save(models.dit, optimizer, lr_sched, global_step, cfg)


def main():
    cli = parse_cli()
    cfg, ttt_cfg = build_train_config(cli)
    runtime = setup_runtime(cfg.seed)
    logger = TrainLogger(cfg.log_dir, cfg.run_name, runtime.global_rank, runtime.world_size)
    checkpointer = Checkpointer(cfg.weights_dir, cfg.run_name, runtime.global_rank)
    logger.info("=" * 70)
    logger.info(f"Infinite World - Training - {cfg.run_name}")
    logger.info("=" * 70)
    logger.info(f"Data: {cfg.data_dir}")
    logger.info(f"Device: {runtime.device}, World Size: {runtime.world_size}")
    logger.info("=" * 70)
    dataloader = build_dataloader(cfg, runtime)
    logger.dump_config(cfg, ttt_cfg, runtime, len(dataloader.dataset))
    models = build_dit(cfg, runtime)
    optimizer = torch.optim.AdamW(
        models.trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas,
    )
    lr_sched = None
    if cfg.lr_schedule == "cosine":
        total_steps = len(dataloader) * cfg.epochs // cfg.videos_per_step
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
    start_step = 0
    if cfg.resume_from:
        start_step = checkpointer.load_for_resume(cfg.resume_from, models.dit, optimizer, lr_sched)
    # 只有 rank0 做验证，取数据集前 num_val_videos 个作为固定验证集
    val_videos = None
    if cfg.val_every_n_steps > 0 and runtime.global_rank == 0:
        val_dataset = PreprocessedVideoDataset(
            cfg.data_dir, cfg.max_chunks_per_video,
            filter_location=cfg.filter_location,
            filter_scene=cfg.filter_scene,
            filter_crowd_density=cfg.filter_crowd_density,
            filter_weather=cfg.filter_weather,
            filter_time_of_day=cfg.filter_time_of_day,
        )
        n_val = min(cfg.num_val_videos, len(val_dataset))
        val_videos = [val_dataset[i] for i in range(n_val)]
        logger.info(f"Validation: {n_val} videos x {cfg.num_val_buckets} timestep buckets")
    logger.info("Starting training loop...")
    train_loop(models, dataloader, optimizer, lr_sched, logger, checkpointer,
              cfg, ttt_cfg, runtime, start_step, val_videos)
    logger.info("Training finished!")


if __name__ == "__main__":
    main()
