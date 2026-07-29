"""
Infinite World - 长视频监督微调（TTT 就绪）
==============================================================
用 rectified-flow 损失在真实长视频上训练 DiT 世界模型，结构上预留了在线 TTT
内循环（见 main.py），后续开启时无需改动训练主循环。

V1 默认：TTT 关闭、单 GPU、一个视频 = 一个训练样本。

核心思路（长视频）：
  一段 1 分钟的视频约等于 22 个 chunk 的分块自回归生成。真正重要的信号是
  整条轨迹的一致性，而不是单个 chunk 的精度。因此外层优化器累积一个视频里
  每个 chunk 的损失梯度，每个视频只走一步（outer_update_scope="video"），
  即优化整条 22-chunk 轨迹的平均损失。"chunk" 模式（逐 chunk 更新）保留给
  短片段 / 调试场景。

条件构造与 main._generate_chunk 完全一致（不截断历史）：
  第 k 个 chunk 监督像素帧 [80k, 80k+81)（步长 80，重叠 1 帧）；
  条件 latent 是 vae.encode(全部真实帧 [0, 80k]) —— 用完整增长历史做
  teacher forcing，DiT 看到的 image_cond 形状与推理时相同（其内部的
  滑窗压缩能处理 T>80）。

梯度卫生（TTT 开启后至关重要）：
  外层与内层参数集默认互不相交（DISJOINT）。二者的并集全程保持
  requires_grad=True。每个循环在 backward 前只清零自己的优化器，
  也只裁剪自己的参数。内层 TTT 循环运行在 `_only_trainable(inner_names)`
  上下文里，其余参数被临时冻结，因此内层 backward 绝不会污染外层的
  梯度累积。

章节：
  §1  Config        TrainConfig / OuterLoopConfig / TTTLoopConfig / ValidationConfig
  §2  CLI           parse_cli / build_configs
  §3  Run dir+log   prepare_run_dir / TrainLogger
  §4  Model+params  setup_models / select_params / apply_grad_flags / check_overlap / maybe_wrap_ddp
  §5  Dataset       VideoMeta / VideoFolderDataset
  §6  RF primitive  rf_loss / train_step
  §7  TTT inner     snapshot_params / _only_trainable / ttt_on_chunk / sample_chunk_latent
  §8  Per-video     train_on_video
  §9  Validation    run_validation
  §10 Checkpoint    save_checkpoint / save_training_state / load_resume
  §11 Loop          run_training
  §12 main
"""
import sys
import os
import json
import glob
import math
import time
import random
import datetime
import argparse
import numpy as np
import torch
from dataclasses import dataclass, field, asdict
from collections import OrderedDict
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import main  # scripts/main.py —— 推理入口；import 时无副作用
from main import (
    MOVE_ACTION_MAP, VIEW_ACTION_MAP,
    setup_runtime, load_models,
    _resolve_path, _torch_gc, _get_bucket_config,
    _resize_and_center_crop, _load_action_sequence, _load_input,
    _slice_move_view, _generate_chunk, _restore_init_params, _case_sort_key,
    ExpConfig,
)
from infworld.models.checkpoint import set_grad_checkpoint
from infworld.models.scheduler import timestep_transform
from infworld.utils.data_utils import save_silent_video


# ============================================================================
# §1  Config
# ============================================================================
@dataclass
class TrainConfig:
    """运行级开关。所有相对路径都相对 PROJECT_ROOT 解析。"""

    seed: int = 42
    model_config_path: str = "configs/infworld_config.yaml"
    dataset_dir: str = "dataset/train"       # 每个样本一个子目录：
                                             #   <name>/{video.mp4, move_view.json, prompts.json}
    output_dir: str = "outputs/train"        # run 目录的父目录
    run_name: Optional[str] = None           # None -> "run_<%m%d_%H%M%S>"
    resume: Optional[str] = None             # run 目录或 training_state-*.pt 路径

    num_frames: Optional[int] = None         # None -> 读 YAML validation_data.num_frames；必须是 81
                                             # （dit_model.py 硬编码了 move[:, -81:]）
    max_chunks_per_video: Optional[int] = None  # 每视频 chunk 数上限（None = 不限）
    bucket_config_name: str = "ASPECT_RATIO_627_F64"

    num_epochs: int = 1
    max_steps: Optional[int] = None          # 按外层优化器步数计；None = 只受 epoch 数约束
    use_grad_checkpoint: bool = True
    # 有意设计为 B=1：一个视频一个样本；视频内 chunk 必须顺序处理
    # （TTT 开启后是硬性要求），因此不做视频级 batch。

    log_every: int = 1                       # 按外层优化器步数计
    save_every: int = 50                     # 同上；实际在下一个视频边界触发
    keep_last_ckpts: int = 3
    val_every: int = 50                      # 同上；实际在下一个视频边界触发
    wandb: bool = True                       # wandb 不可用时自动退化为 no-op
    wandb_project: str = "infworld-train"


@dataclass
class OuterLoopConfig:
    """Supervised rectified-flow fine-tuning on real chunks."""

    lr: float = 1e-5
    weight_decay: float = 0.0
    grad_clip_norm: Optional[float] = 1.0
    # "video": accumulate every chunk's loss gradient across the whole video and
    #   take ONE optimizer step at the video end — optimizes the average loss of
    #   the full trajectory (the right objective for 1-minute clips).
    # "chunk": step after each chunk (short-clip / debug mode).
    update_scope: str = "video"
    # Trainable set = last K transformer blocks, minus exclude_layers.
    last_k_blocks: int = 6
    include_layers: tuple = ("blocks",)      # "blocks" = every param inside a block
    # Default excludes = the inner TTT whitelist, so the two sets are DISJOINT
    # and a future inner restore can never roll back outer progress.
    exclude_layers: tuple = ("head", "blocks.modulation", "blocks.norm3")
    on_overlap: str = "warn"                 # "warn" | "error" when sets intersect


@dataclass
class TTTLoopConfig:
    """Inner test-time-training loop between chunks. Fully inert when open=False.

    V1 ships with open=False; the loop, param selection, snapshot/restore and
    both data sources are implemented so flipping --ttt on is all it takes.
    """

    open: bool = False                       # master switch (v1 default: OFF)
    data_source: str = "real"                # "real": TTT on the real chunk latent
                                             # "generated": sample the chunk first, TTT on it
    n_steps: int = 5
    lr: float = 1e-5
    grad_clip_norm: Optional[float] = 1.0
    sampling_steps: int = 10                 # for data_source="generated"
    text_cfg_scale: float = 5.0              # ditto
    reset_between_videos: bool = True        # snapshot at video start, restore at end
    # Reptile-style soft restore: p <- alpha*snapshot + (1-alpha)*p.
    # 1.0 = hard reset (inference-aligned); <1.0 lets inner params meta-drift
    # across videos toward "a better TTT initialization".
    restore_alpha: float = 1.0
    fresh_optimizer_per_chunk: bool = True   # matches main.online_train_on_chunk
    trainable_layers: tuple = ("head", "blocks.modulation", "blocks.norm3")
    trainable_block_start: int = 18


@dataclass
class ValidationConfig:
    """Periodic generation on a fixed case to eyeball long-horizon drift."""

    case_dir: str = "dataset/wbench/case12"
    num_chunks: int = 2
    num_sampling_steps: int = 30
    shift: int = 7
    text_cfg_scale: float = 5.0
    negative_prompt: str = ExpConfig.negative_prompt
    fps: int = 30
    at_start: bool = False                   # baseline validation before training
# ============================================================================
# §2  CLI
# ============================================================================
def _parse_on_off(s):
    return s.strip().lower() in ("on", "true", "1", "yes")


def parse_cli():
    """Parse command-line overrides. Every flag defaults to None so build_configs
    knows which ones to apply over the dataclass defaults (same as main.py)."""
    parser = argparse.ArgumentParser(description="Infinite World long-video training")
    # run
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="run dir or training_state-*.pt to resume from")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="stop after this many outer optimizer steps")
    parser.add_argument("--max-chunks-per-video", type=int, default=None)
    parser.add_argument("--grad-checkpoint", type=_parse_on_off, default=None)
    # outer loop
    parser.add_argument("--outer-lr", type=float, default=None)
    parser.add_argument("--outer-last-k", type=int, default=None,
                        help="train the last K transformer blocks")
    parser.add_argument("--outer-scope", type=str, default=None,
                        choices=["video", "chunk"],
                        help="one optimizer step per video (grad accumulation) or per chunk")
    parser.add_argument("--grad-clip", type=float, default=None)
    # TTT inner loop (inert unless --ttt on)
    parser.add_argument("--ttt", type=_parse_on_off, default=None,
                        help="enable the TTT inner loop between chunks: on/off")
    parser.add_argument("--ttt-data-source", type=str, default=None,
                        choices=["real", "generated"])
    parser.add_argument("--ttt-steps", type=int, default=None)
    parser.add_argument("--ttt-lr", type=float, default=None)
    parser.add_argument("--ttt-sampling-steps", type=int, default=None)
    parser.add_argument("--ttt-reset", type=_parse_on_off, default=None)
    parser.add_argument("--ttt-restore-alpha", type=float, default=None,
                        help="1.0 = hard reset; <1.0 = Reptile soft restore")
    # cadence / logging
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--val-every", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--wandb", type=_parse_on_off, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    # validation
    parser.add_argument("--val-case", type=str, default=None)
    parser.add_argument("--val-chunks", type=int, default=None)
    parser.add_argument("--val-steps", type=int, default=None,
                        help="sampling steps for validation generation")
    parser.add_argument("--val-at-start", type=_parse_on_off, default=None)
    args, _ = parser.parse_known_args()
    return args


def build_configs(cli):
    """Build the four configs from dataclass defaults, overriding with any CLI
    flag that was actually provided (non-None)."""
    train_cfg = TrainConfig()
    outer_cfg = OuterLoopConfig()
    ttt_cfg = TTTLoopConfig()
    val_cfg = ValidationConfig()

    overrides = [
        (train_cfg, "dataset_dir", cli.dataset_dir),
        (train_cfg, "output_dir", cli.output_dir),
        (train_cfg, "run_name", cli.run_name),
        (train_cfg, "resume", cli.resume),
        (train_cfg, "seed", cli.seed),
        (train_cfg, "num_epochs", cli.epochs),
        (train_cfg, "max_steps", cli.max_steps),
        (train_cfg, "max_chunks_per_video", cli.max_chunks_per_video),
        (train_cfg, "use_grad_checkpoint", cli.grad_checkpoint),
        (train_cfg, "save_every", cli.save_every),
        (train_cfg, "val_every", cli.val_every),
        (train_cfg, "log_every", cli.log_every),
        (train_cfg, "wandb", cli.wandb),
        (train_cfg, "wandb_project", cli.wandb_project),
        (outer_cfg, "lr", cli.outer_lr),
        (outer_cfg, "last_k_blocks", cli.outer_last_k),
        (outer_cfg, "update_scope", cli.outer_scope),
        (outer_cfg, "grad_clip_norm", cli.grad_clip),
        (ttt_cfg, "open", cli.ttt),
        (ttt_cfg, "data_source", cli.ttt_data_source),
        (ttt_cfg, "n_steps", cli.ttt_steps),
        (ttt_cfg, "lr", cli.ttt_lr),
        (ttt_cfg, "sampling_steps", cli.ttt_sampling_steps),
        (ttt_cfg, "reset_between_videos", cli.ttt_reset),
        (ttt_cfg, "restore_alpha", cli.ttt_restore_alpha),
        (val_cfg, "case_dir", cli.val_case),
        (val_cfg, "num_chunks", cli.val_chunks),
        (val_cfg, "num_sampling_steps", cli.val_steps),
        (val_cfg, "at_start", cli.val_at_start),
    ]
    for cfg, name, value in overrides:
        if value is not None:
            setattr(cfg, name, value)

    return train_cfg, outer_cfg, ttt_cfg, val_cfg
# ============================================================================
# §3  Run dir & logging
# ============================================================================
def prepare_run_dir(train_cfg, outer_cfg, ttt_cfg, val_cfg):
    """Create outputs/train/<run_name>/{checkpoints,val_videos,tb} and dump the
    resolved configs to config.json for traceability. Returns the run dir."""
    if train_cfg.run_name is None:
        train_cfg.run_name = datetime.datetime.now().strftime("run_%m%d_%H%M%S")
    run_dir = os.path.join(_resolve_path(train_cfg.output_dir), train_cfg.run_name)
    for sub in ("checkpoints", "val_videos", "tb"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({
            "train": asdict(train_cfg), "outer": asdict(outer_cfg),
            "ttt": asdict(ttt_cfg), "validation": asdict(val_cfg),
        }, f, indent=2, default=str)
    print(f"[Train] Run dir: {run_dir}")
    return run_dir


class TrainLogger:
    """Writes metrics to tensorboard and (optionally) wandb. Rank-0 only; both
    sinks degrade to no-ops when unavailable. wandb is NOT in requirements.txt,
    so its import failure is expected and non-fatal."""

    def __init__(self, run_dir, use_wandb, project, run_name, config_dict,
                 is_main_rank=True, wandb_run_id=None):
        self.is_main_rank = is_main_rank
        self.tb = None
        self.wandb = None
        self.wandb_run_id = wandb_run_id
        if not is_main_rank:
            return
        from torch.utils.tensorboard import SummaryWriter
        self.tb = SummaryWriter(os.path.join(run_dir, "tb"))
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
                wandb.init(project=project, name=run_name, dir=run_dir,
                           config=config_dict, id=wandb_run_id, resume="allow")
                self.wandb_run_id = wandb.run.id
            except Exception as e:  # ImportError, network, auth — all non-fatal
                print(f"[Train] wandb unavailable ({e}); tensorboard only")
                self.wandb = None

    def log(self, metrics, step):
        if not self.is_main_rank:
            return
        for k, v in metrics.items():
            self.tb.add_scalar(k, v, step)
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)

    def close(self):
        if self.tb is not None:
            self.tb.close()
        if self.wandb is not None:
            self.wandb.finish()


# ============================================================================
# §4  Model loading & parameter selection
# ============================================================================
def setup_models(train_cfg, val_cfg, runtime):
    """Load VAE/text-encoder/scheduler/DiT via main.load_models with the online-
    training path disabled — train.py owns all requires_grad decisions.

    NOTE: after load_models every DiT param has requires_grad=True; call
    apply_grad_flags before building optimizers."""
    models = load_models(
        train_cfg.model_config_path, runtime.device, runtime.enable_context_parallel,
        num_sampling_steps=val_cfg.num_sampling_steps, shift=val_cfg.shift,
        online_train_open=False, use_grad_checkpoint=False,
        trainable_layers=(), trainable_block_start=0, reset_between_videos=False,
    )
    if train_cfg.use_grad_checkpoint:
        set_grad_checkpoint(models.dit)
    return models


def _canonical(name):
    """blocks.<i>.rest -> blocks.rest (drop the per-block index); others as-is."""
    parts = name.split(".")
    if parts[0] == "blocks" and len(parts) > 1 and parts[1].isdigit():
        parts = ["blocks"] + parts[2:]
    return ".".join(parts)


def _block_idx(name):
    parts = name.split(".")
    if parts[0] == "blocks" and len(parts) > 1 and parts[1].isdigit():
        return int(parts[1])
    return None


def select_params(dit, include_layers, block_start=0, exclude_layers=(), label=""):
    """Generalized version of main._select_trainable_params. PURE — does not
    touch requires_grad. Returns OrderedDict[name -> Parameter].

    A param is selected iff:
      - its canonical name (block index elided) matches an include entry
        ("blocks" alone matches every param inside any block; other entries
        match by exact-or-prefix, e.g. "blocks.norm3" -> blocks.<i>.norm3.*), AND
      - it is a non-block param, or its block index >= block_start, AND
      - it matches no exclude entry (same matching rule).
    """
    def matches(key, entries):
        # exact or prefix match on the canonical (block-index-elided) name;
        # "blocks" alone therefore matches every param inside any block.
        return any(key == e or key.startswith(e + ".") for e in entries)

    selected = OrderedDict()
    for n, p in dit.named_parameters():
        key = _canonical(n)
        idx = _block_idx(n)
        if not matches(key, include_layers):
            continue
        if idx is not None and idx < block_start:
            continue
        if exclude_layers and matches(key, exclude_layers):
            continue
        selected[n] = p
    n_params = sum(p.numel() for p in selected.values())
    print(f"[Select:{label}] {len(selected)} tensors, {n_params:,} params "
          f"(include={include_layers}, block_start={block_start}, exclude={exclude_layers})")
    if not selected:
        raise ValueError(f"[Select:{label}] empty parameter set — check include/exclude/block_start")
    return selected


def outer_param_set(dit, outer_cfg):
    """Outer supervised set: every param in the last K blocks minus excludes."""
    num_layers = len(dit.blocks)
    block_start = num_layers - outer_cfg.last_k_blocks
    return select_params(dit, outer_cfg.include_layers, block_start,
                         outer_cfg.exclude_layers, label="outer")


def inner_param_set(dit, ttt_cfg):
    """Inner TTT set: same whitelist as main.py's OnlineTrainConfig (Scheme A)."""
    return select_params(dit, ttt_cfg.trainable_layers, ttt_cfg.trainable_block_start,
                         label="inner")


def check_overlap(outer_names, inner_names, on_overlap="warn"):
    """The two sets should be DISJOINT: the inner restore rolls back ONLY inner
    params, so any overlap would silently roll back outer SFT progress too."""
    overlap = sorted(set(outer_names) & set(inner_names))
    if not overlap:
        print("[Select] outer/inner param sets are disjoint (as designed)")
        return
    msg = (f"[Select] WARNING: outer and inner param sets OVERLAP on "
           f"{len(overlap)} tensors, e.g. {overlap[:5]}. The inner restore will "
           f"roll back outer updates on these params every video.")
    if on_overlap == "error":
        raise ValueError(msg)
    print("!" * 80 + f"\n{msg}\n" + "!" * 80)


def apply_grad_flags(dit, union_names):
    """union(outer, inner) requires_grad=True, everything else frozen. Called
    ONCE; flags are never flipped afterwards (grad hygiene is handled by each
    loop zeroing only its own optimizer — see module docstring)."""
    frozen_n = 0
    for n, p in dit.named_parameters():
        if n in union_names:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
            frozen_n += p.numel()
    print(f"[Train] frozen params: {frozen_n:,}")


def maybe_wrap_ddp(dit, runtime):
    """DDP extension point — v1 returns the model unchanged.

    When enabling multi-GPU later (torchrun -> runtime.use_dist=True):
      - shard videos by `idx % dp_size == dp_rank` (VideoFolderDataset already does);
      - wrap: DDP(dit, device_ids=[runtime.local_rank], find_unused_parameters=True)
        (the outer set skips excluded layers, so some grads are absent per step);
      - static_graph must stay False (outer and inner backwards use different
        param subsets over the same graph);
      - the TTT inner loop must run inside `with ddp_model.no_sync():` — TTT is
        per-rank per-video local state and must NOT be all-reduced;
      - with reset_between_videos=True and restore_alpha=1.0, inner params are
        identical across ranks again at every video boundary.
    """
    return dit
# ============================================================================
# §5  Dataset (long-video folder scan + decord decode)
# ============================================================================
@dataclass
class VideoMeta:
    """One training sample's metadata (frames are decoded lazily)."""
    idx: int                       # global enumeration index (DP sharding + logging)
    name: str
    case_dir: str
    video_path: str
    prompt: str
    move: torch.Tensor             # LongTensor [N], full action sequence, CPU
    view: torch.Tensor             # LongTensor [N], CPU
    num_frames_avail: int          # frames readable from the video file
    num_chunks: int                # chunks this sample yields (see below)


class _LocalFolderDataset:
    """扫描 dataset_dir，寻找包含 {video.mp4, move_view.json, prompts.json} 的子目录。

        样本格式（目前固定，后续数据将导入）：结构与 wbench 示例相同，但使用 30fps 的视频替代 image.jpg；
        move_view.json 对每一帧包含一项 {"move","view"} 条目。

        第 k 个 chunk 覆盖像素帧区间 [80k, 80k+81) — 步长为 80，帧之间有 1 帧重叠，
        与推理时将 decoded[:, :, 1:] 追加到缓冲区的行为一致。
        num_chunks = min((F-1)//80, len(actions)//80, max_chunks_per_video)。

        格式错误的目录会被跳过并打印原因（容错风格与 main.load_inputs 相同）。视频按需解码（`load_video_u8`），
        根据第一帧的宽高比选择 bucket 进行缩放，并以 uint8 格式保存在 CPU 上（例如 1800 帧、627px 时约 1.2GB）—
        仅在 train_on_video 的每个 chunk 窗口内归一化为 [-1,1] 的浮点数。

        v1 使用简单的 Python 循环遍历（B=1；每个视频的 GPU 时间远大于解码时间）。
        如果解码成为瓶颈，可将 `load_video_u8` 的调用包装到 DataLoader(collate=identity, num_workers=1) 中以实现预取。
    """

    REQUIRED = ("video.mp4", "move_view.json", "prompts.json")

    def __init__(self, dataset_dir, bucket_config_name, num_frames=81,
                 max_chunks_per_video=None, dp_rank=0, dp_size=1):
        assert num_frames == 81, "dit_model.py hard-codes move[:, -81:]; num_frames must be 81"
        self.num_frames = num_frames
        self.bucket_config = _get_bucket_config(bucket_config_name)
        dataset_dir = _resolve_path(dataset_dir)

        names = sorted(
            (d for d in os.listdir(dataset_dir)
             if os.path.isdir(os.path.join(dataset_dir, d))),
            key=_case_sort_key,
        )
        print(f"[Data] Found {len(names)} candidate dirs under {dataset_dir}")

        self.metas = []
        for idx, name in enumerate(names):
            case_dir = os.path.join(dataset_dir, name)
            missing = [f for f in self.REQUIRED
                       if not os.path.exists(os.path.join(case_dir, f))]
            if missing:
                print(f"[Data] Skipping {name}: missing {missing}")
                continue
            if idx % dp_size != dp_rank:
                continue
            meta = self._load_meta(idx, name, case_dir, max_chunks_per_video)
            if meta is not None:
                self.metas.append(meta)
        print(f"[Data] rank {dp_rank}/{dp_size} has {len(self.metas)} videos, "
              f"{sum(m.num_chunks for m in self.metas)} chunks total")
        if not self.metas:
            raise ValueError(f"[Data] no usable videos under {dataset_dir}")

    def _load_meta(self, idx, name, case_dir, max_chunks_per_video):
        video_path = os.path.join(case_dir, "video.mp4")
        try:
            import decord
            vr = decord.VideoReader(video_path)
            num_frames_avail = len(vr)
            del vr
        except Exception as e:
            print(f"[Data] Skipping {name}: cannot open video ({e})")
            return None
        try:
            move_idx, view_idx = _load_action_sequence(
                os.path.join(case_dir, "move_view.json"))
            with open(os.path.join(case_dir, "prompts.json")) as f:
                prompt = json.load(f)["prompt"]
        except Exception as e:
            print(f"[Data] Skipping {name}: bad json ({e})")
            return None

        if num_frames_avail < self.num_frames:
            print(f"[Data] Skipping {name}: only {num_frames_avail} frames (<{self.num_frames})")
            return None
        if len(move_idx) < self.num_frames:
            print(f"[Data] Skipping {name}: only {len(move_idx)} actions (<{self.num_frames})")
            return None

        num_chunks = min((num_frames_avail - 1) // 80, len(move_idx) // 80)
        if max_chunks_per_video is not None:
            num_chunks = min(num_chunks, max_chunks_per_video)
        if num_chunks < 1:
            print(f"[Data] Skipping {name}: 0 chunks")
            return None

        return VideoMeta(
            idx=idx, name=name, case_dir=case_dir, video_path=video_path,
            prompt=prompt,
            move=torch.tensor(move_idx, dtype=torch.long),
            view=torch.tensor(view_idx, dtype=torch.long),
            num_frames_avail=num_frames_avail, num_chunks=num_chunks,
        )

    def __len__(self):
        return len(self.metas)

    def meta(self, i):
        return self.metas[i]

    def load_video_u8(self, i):
        """Decode frames [0, 80*num_chunks] of video i, bucket-resize, and
        return uint8 [1, 3, F, H, W] on CPU."""
        m = self.metas[i]
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(m.video_path)
        n_needed = 80 * m.num_chunks + 1
        frames = vr.get_batch(list(range(n_needed))).asnumpy()  # [F, H, W, 3] RGB uint8
        del vr

        ratio = frames.shape[1] / frames.shape[2]
        closest = sorted(self.bucket_config.keys(), key=lambda x: abs(float(x) - ratio))[0]
        target_h, target_w = self.bucket_config[closest][0]

        out = [_resize_and_center_crop(frame, (target_h, target_w))  # [1,C,1,h,w] uint8
               for frame in frames]
        video = torch.cat(out, dim=2)  # [1, 3, F, H, W] uint8
        print(f"[Data] {m.name}: {n_needed} frames -> bucket {closest} "
              f"({target_h}x{target_w}), {m.num_chunks} chunks")
        return video


def VideoFolderDataset(dataset_dir, bucket_config_name, num_frames=81,
                       max_chunks_per_video=None, dp_rank=0, dp_size=1):
    """数据集工厂（签名与原类一致，§12 调用点无需改动）。按目录布局分派：
    顶层含 *.npz 的平铺 mp4+npz 目录 -> prepare.sekai_game_walking.
    prepare_sekai_game_walking.build_sekai_dataset（Sekai 式：caption CSV +
    相机外参推导动作，SEKAI_* 环境变量可调）；
    否则 -> 原目录式 _LocalFolderDataset。

    两者都实现 train_on_video/run_training 消费的同一协议（完整契约见
    prepare_sekai_game_walking.py 模块 docstring，换新数据集时实现同样三个方法即可）：
      __len__() -> 本 DP rank 的视频数
      meta(i)   -> VideoMeta 同形对象（prompt、move/view LongTensor[N]、
                   num_frames_avail、num_chunks；chunk k 监督帧 [80k, 80k+81)）
      load_video_u8(i) -> uint8 [1, 3, F, H, W] RGB CPU，bucket 分辨率，
                   F >= 80*num_chunks+1
    """
    from trash.prepare_sekai_game_walking import (  # 惰性 import 防循环
        is_sekai_dir, build_sekai_dataset,
    )
    if is_sekai_dir(_resolve_path(dataset_dir)):
        return build_sekai_dataset(dataset_dir, bucket_config_name,
                                   num_frames=num_frames,
                                   max_chunks_per_video=max_chunks_per_video,
                                   dp_rank=dp_rank, dp_size=dp_size)
    return _LocalFolderDataset(dataset_dir, bucket_config_name,
                               num_frames=num_frames,
                               max_chunks_per_video=max_chunks_per_video,
                               dp_rank=dp_rank, dp_size=dp_size)


def normalize_pixels(u8):
    """uint8 [0,255] -> float32 [-1,1] (applied per chunk window, not globally)."""
    return (u8.float() / 255 - 0.5) * 2
# ============================================================================
# §6  Shared rectified-flow training primitive
# ============================================================================
def rf_loss(dit, scheduler, x_start, model_kwargs):
    """Build one rectified-flow loss on (x_start, model_kwargs) and return the
    scalar loss tensor (graph attached; caller decides how to backward).

    Mirrors main._online_train_step's math exactly — t sampling +
    timestep_transform, add_noise, reversed-velocity target, x_ignore_mask=None
    (scheduler.training_losses is unusable: it slices x_ignore_mask
    unconditionally), pred[:, :, -T:], fp32 MSE. No autocast, matching main.py
    (the DiT is bf16 wholesale; fp32 islands live inside the model)."""
    device = x_start.device
    B = x_start.shape[0]

    if scheduler.use_discrete_timesteps:
        t = torch.randint(0, scheduler.num_timesteps, (B,), device=device)
    elif scheduler.sample_method == "uniform":
        t = torch.rand((B,), device=device) * scheduler.num_timesteps
    else:  # logit-normal
        t = scheduler.sample_t(x_start) * scheduler.num_timesteps
    if scheduler.use_timestep_transform:
        t = timestep_transform(t, shift=scheduler.shift,
                               num_timesteps=scheduler.num_timesteps)

    noise = torch.randn_like(x_start)
    x_t = scheduler.add_noise(x_start, noise, t)
    target = x_start - noise
    if scheduler.use_reversed_velocity:
        target = -target

    pred = dit(x_t, t, x_ignore_mask=None, **model_kwargs)
    pred = pred[:, :, -x_start.shape[2]:]  # DiT input is [image_cond | x_t] on time
    return ((pred.float() - target.float()) ** 2).mean()


def train_step(dit, scheduler, x_start, model_kwargs, params, optimizer,
               grad_clip_norm):
    """One full optimizer step on the given param set (used by the TTT inner
    loop, and by the outer loop in update_scope="chunk").

    Grad hygiene: zero_grad(set_to_none=True) FIRST clears any stale gradient
    the other loop's backward deposited on this loop's params; clipping walks
    ONLY this loop's params (main._online_train_step clips dit.parameters(),
    which is wrong once two loops coexist). Returns (loss, grad_norm) floats."""
    optimizer.zero_grad(set_to_none=True)
    loss = rf_loss(dit, scheduler, x_start, model_kwargs)
    loss.backward()
    grad_norm = _clip_or_measure(params, grad_clip_norm)
    optimizer.step()
    return loss.item(), grad_norm


def _clip_or_measure(params, grad_clip_norm):
    """Clip (or just measure, when clipping is off) the grad norm of `params`."""
    tensors = list(params.values()) if isinstance(params, dict) else list(params)
    if grad_clip_norm is not None:
        return float(torch.nn.utils.clip_grad_norm_(tensors, max_norm=grad_clip_norm))
    norms = [p.grad.norm() for p in tensors if p.grad is not None]
    return float(torch.norm(torch.stack(norms))) if norms else 0.0


# ============================================================================
# §7  TTT inner loop (inert when ttt_cfg.open=False)
# ============================================================================
def snapshot_params(dit, names):
    """Clone the inner param set (~0.25M params, a few MB — kept on GPU)."""
    return {n: p.detach().clone() for n, p in dit.named_parameters() if n in names}


def restore_inner_params(dit, snapshot, alpha=1.0):
    """p <- alpha*snapshot + (1-alpha)*p.  alpha=1.0 is a hard reset (aligned
    with inference reset_between_videos); alpha<1.0 is a Reptile-style soft
    restore that lets inner params meta-drift across videos toward a better
    TTT initialization."""
    if alpha >= 1.0:
        _restore_init_params(dit, snapshot)  # main.py helper: p.data.copy_(snap)
        return
    with torch.no_grad():
        for n, p in dit.named_parameters():
            if n in snapshot:
                p.data.mul_(1.0 - alpha).add_(snapshot[n], alpha=alpha)


class _only_trainable:
    """Context manager: temporarily restrict requires_grad to `names` so a
    backward inside the block deposits gradients ONLY on the inner set.

    Why: the outer loop in update_scope="video" ACCUMULATES gradients across
    chunks and steps once per video. Without this guard, every inner-TTT
    backward would also add its (differently-scaled) gradient onto the outer
    params' .grad, silently corrupting the video-level accumulation. Restoring
    the previous flags on exit keeps §4's 'set once' policy intact elsewhere."""

    def __init__(self, dit, names):
        self.dit = dit
        self.names = names
        self.saved = None

    def __enter__(self):
        self.saved = {n: p.requires_grad for n, p in self.dit.named_parameters()}
        for n, p in self.dit.named_parameters():
            p.requires_grad_(n in self.names)
        return self

    def __exit__(self, *exc):
        for n, p in self.dit.named_parameters():
            p.requires_grad_(self.saved[n])
        return False


def sample_chunk_latent(models, prompt, negative_prompt, cfg_scale,
                        cond_latent, move, view, sampling_steps):
    """data_source="generated": sample this chunk's latent with the CURRENT
    weights (few steps), so the inner TTT trains on the model's own output —
    faithful to inference-time TTT. No vae.decode needed (TTT works in latent).

    scheduler.num_sampling_steps is a shared mutable attr — set/restore around
    the call so validation keeps its own step count."""
    scheduler = models.scheduler
    saved_steps = scheduler.num_sampling_steps
    scheduler.num_sampling_steps = sampling_steps
    try:
        z_size = torch.Size([1, models.vae.out_channels, 21,
                             cond_latent.shape[3], cond_latent.shape[4]])
        with torch.no_grad():
            samples = scheduler.sample(
                model=models.dit,
                text_encoder=models.text_encoder,
                null_embedder=models.dit.y_embedder,
                z_size=z_size,
                prompts=[prompt],
                guidance_scale=cfg_scale,
                negative_prompts=[negative_prompt],
                device=cond_latent.device,
                additional_args={"image_cond": cond_latent, "move": move, "view": view},
            )
    finally:
        scheduler.num_sampling_steps = saved_steps
    return samples


def ttt_on_chunk(models, ttt_cfg, x_start, model_kwargs, inner_params,
                 persistent_optimizer=None):
    """Run n_steps of TTT on one chunk's latent, updating ONLY the inner set
    (enforced by _only_trainable). A fresh AdamW per chunk by default (Adam
    moments do not accumulate across chunks — matches main.online_train_on_chunk).
    Returns the per-step losses."""
    if ttt_cfg.fresh_optimizer_per_chunk or persistent_optimizer is None:
        optimizer = torch.optim.AdamW(inner_params.values(), lr=ttt_cfg.lr)
    else:
        optimizer = persistent_optimizer

    x_start = x_start.detach()
    losses = []
    with _only_trainable(models.dit, set(inner_params.keys())):
        for _ in range(ttt_cfg.n_steps):
            loss, _ = train_step(models.dit, models.scheduler, x_start,
                                 model_kwargs, inner_params, optimizer,
                                 ttt_cfg.grad_clip_norm)
            losses.append(loss)
    del optimizer
    return losses
# ============================================================================
# §8  Per-video training (the core loop)
# ============================================================================
def train_on_video(models, device, train_cfg, outer_cfg, ttt_cfg, meta,
                   pixels_u8, outer_params, outer_opt, inner_params,
                   logger, global_step):
    """Train on one long video, chunk by chunk, in order.

    Conditioning is built EXACTLY like inference (main._generate_chunk): for
    chunk k the condition latent is vae.encode of ALL real pixel frames
    [0, 80k] — the full growing history, no truncation. The DiT's internal
    sliding-window compression handles latent T > 80 exactly as it does at
    inference time. Teacher forcing: real history instead of generated.

    Outer objective (update_scope="video"): grad of the AVERAGE chunk loss over
    the whole trajectory, one optimizer step per video — the model learns
    weights that carry a full 22-chunk (1-minute) rollout, not per-chunk myopia.

    TTT (when open): after chunk k's outer contribution, run the inner loop on
    chunk k's latent (real or freshly sampled) so it perturbs the weights that
    process chunk k+1 — the faithful analogue of inference-time TTT ordering.

    Returns the updated global_step (outer optimizer steps).
    """
    num_frames = train_cfg.num_frames
    with torch.no_grad():
        text_kwargs = models.text_encoder.encode([meta.prompt])
    y, y_mask = text_kwargs["y"], text_kwargs["y_mask"]

    ttt_active = ttt_cfg.open and meta.num_chunks > 1
    inner_snapshot = None
    if ttt_active and ttt_cfg.reset_between_videos:
        inner_snapshot = snapshot_params(models.dit, set(inner_params.keys()))

    video_scope = outer_cfg.update_scope == "video"
    if video_scope:
        outer_opt.zero_grad(set_to_none=True)

    chunk_losses = []
    for k in range(meta.num_chunks):
        torch.cuda.reset_peak_memory_stats()
        xs, xe = 80 * k, 80 * k + num_frames

        # ---- chunk tensors (all encoding under no_grad; VAE is frozen) ----
        with torch.no_grad():
            x_start = models.vae.encode(
                normalize_pixels(pixels_u8[:, :, xs:xe]).to(device))
            # Full real history [0, 80k] — same as inference's video_buffer tail
            # (curr_start = buffer.shape[2]-1 == 80k when the buffer is real).
            cond_latent = models.vae.encode(
                normalize_pixels(pixels_u8[:, :, :xs + 1]).to(device))
        move_k, view_k = _slice_move_view(meta.move, meta.view, xs, num_frames, device)
        model_kwargs = {
            "y": y, "y_mask": y_mask,
            "image_cond": cond_latent,
            "move": move_k.unsqueeze(0),
            "view": view_k.unsqueeze(0),
        }

        # ---- generated-mode TTT data: sample BEFORE the outer update so the
        #      latent reflects the weights that would generate chunk k at
        #      inference (i.e. after chunk <k's outer+TTT history) ----
        gen_latent = None
        if ttt_active and ttt_cfg.data_source == "generated" and k < meta.num_chunks - 1:
            gen_latent = sample_chunk_latent(
                models, meta.prompt, ExpConfig.negative_prompt,
                ttt_cfg.text_cfg_scale, cond_latent,
                model_kwargs["move"], model_kwargs["view"],
                ttt_cfg.sampling_steps)

        # ---- (1) outer supervised contribution on the REAL chunk ----
        if video_scope:
            # Accumulate: mean over chunks => scale each chunk's loss by 1/N.
            loss = rf_loss(models.dit, models.scheduler, x_start, model_kwargs)
            (loss / meta.num_chunks).backward()
            chunk_losses.append(loss.item())
            del loss
        else:  # "chunk" scope: step immediately (short-clip / debug mode)
            loss_val, gnorm = train_step(
                models.dit, models.scheduler, x_start, model_kwargs,
                outer_params, outer_opt, outer_cfg.grad_clip_norm)
            chunk_losses.append(loss_val)
            global_step += 1
            if global_step % train_cfg.log_every == 0:
                logger.log({"outer/loss": loss_val, "outer/grad_norm": gnorm,
                            "outer/lr": outer_cfg.lr,
                            "mem/max_alloc_gb":
                                torch.cuda.max_memory_allocated() / 1e9},
                           global_step)

        # ---- (2) TTT inner loop (not on the last chunk, like inference) ----
        if ttt_active and k < meta.num_chunks - 1:
            ttt_x = gen_latent if gen_latent is not None else x_start
            ttt_losses = ttt_on_chunk(models, ttt_cfg, ttt_x, model_kwargs,
                                      inner_params)
            print(f"[TTT] {meta.name} chunk {k}: "
                  f"loss {ttt_losses[0]:.5f} -> {ttt_losses[-1]:.5f}")
            logger.log({"ttt/loss_first": ttt_losses[0],
                        "ttt/loss_last": ttt_losses[-1],
                        "ttt/loss_mean": sum(ttt_losses) / len(ttt_losses)},
                       max(global_step, 1))

        print(f"[Train] {meta.name} chunk {k + 1}/{meta.num_chunks} "
              f"loss={chunk_losses[-1]:.5f} "
              f"cond_T={cond_latent.shape[2]} "
              f"mem={torch.cuda.max_memory_allocated() / 1e9:.1f}GB")
        del x_start, cond_latent, model_kwargs, gen_latent
        _torch_gc()

    # ---- video-scope: one outer step over the accumulated trajectory grad ----
    if video_scope:
        gnorm = _clip_or_measure(outer_params, outer_cfg.grad_clip_norm)
        outer_opt.step()
        outer_opt.zero_grad(set_to_none=True)
        global_step += 1
        video_loss = sum(chunk_losses) / len(chunk_losses)
        if global_step % train_cfg.log_every == 0:
            logger.log({"outer/loss": video_loss, "outer/grad_norm": gnorm,
                        "outer/lr": outer_cfg.lr,
                        "outer/num_chunks": meta.num_chunks,
                        "mem/max_alloc_gb":
                            torch.cuda.max_memory_allocated() / 1e9},
                       global_step)
        print(f"[Train] {meta.name}: video step {global_step} "
              f"avg_loss={video_loss:.5f} grad_norm={gnorm:.3f}")

    # ---- restore inner params (hard reset or Reptile soft restore) ----
    if inner_snapshot is not None:
        restore_inner_params(models.dit, inner_snapshot, ttt_cfg.restore_alpha)
        print(f"[TTT] {meta.name}: restored inner params "
              f"(alpha={ttt_cfg.restore_alpha})")

    return global_step


# ============================================================================
# §9  Validation (periodic generation on a fixed case)
# ============================================================================
def load_validation_input(val_cfg, bucket_config_name):
    """Load the fixed validation case once at startup (main.Input on CPU)."""
    case_dir = _resolve_path(val_cfg.case_dir)
    name = os.path.basename(case_dir.rstrip("/"))
    bucket_config = _get_bucket_config(bucket_config_name)
    return _load_input(0, name, case_dir, bucket_config)


def run_validation(models, device, val_cfg, num_frames, val_input, run_dir,
                   global_step, logger):
    """Generate num_chunks chunks on the fixed case and save an mp4.

    Called ONLY at video boundaries — inner params are already restored, so the
    weights are the clean "post-outer-SFT" model; no extra snapshot needed.
    scheduler.num_sampling_steps/shift are shared mutable attrs and
    scheduler.shift is ALSO read by rf_loss's timestep_transform, so both are
    saved and restored around validation. The output file gets a unique
    step-tagged name and is pre-deleted because save_silent_video APPENDS to an
    existing mp4."""
    scheduler = models.scheduler
    saved_steps, saved_shift = scheduler.num_sampling_steps, scheduler.shift
    scheduler.num_sampling_steps = val_cfg.num_sampling_steps
    scheduler.shift = val_cfg.shift

    video_buffer = val_input.input_image.clone().cpu()
    try:
        with torch.no_grad():
            cond_latent = models.vae.encode(val_input.input_image.to(device))
        latent_size = list(cond_latent.shape)
        latent_size[2] = 21
        latent_size = torch.Size(latent_size)
        del cond_latent

        for c in range(val_cfg.num_chunks):
            result = _generate_chunk(
                models, device, val_input.prompt, val_cfg.negative_prompt,
                val_cfg.text_cfg_scale, video_buffer, latent_size,
                val_input.move, val_input.view, num_frames)
            video_buffer = torch.cat([video_buffer, result.decoded[:, :, 1:]], dim=2)
            del result
            _torch_gc()

        save_path = os.path.join(run_dir, "val_videos",
                                 f"step{global_step:06d}_{val_input.name}")
        if os.path.exists(save_path + ".mp4"):
            os.remove(save_path + ".mp4")  # save_silent_video appends otherwise
        save_silent_video(video_buffer.to(device), save_path, fps=val_cfg.fps, quality=10)
        print(f"[Val] step {global_step}: saved {save_path}.mp4 "
              f"({video_buffer.shape[2]} frames)")
        logger.log({"val/frames": video_buffer.shape[2]}, global_step)
    finally:
        scheduler.num_sampling_steps, scheduler.shift = saved_steps, saved_shift
    del video_buffer
    _torch_gc()
# ============================================================================
# §10  Checkpoint & resume
# ============================================================================
def save_checkpoint(dit, ckpt_dir, step, keep_last=3):
    """Save the FULL DiT state_dict (bf16, ~2.8GB) as checkpoint-{step}.ckpt.

    Bare state_dict + this filename keeps it loadable by main.py inference
    as-is (main._load_dit_state_dict accepts bare dicts; _extract_ckpt_step
    parses r'checkpoint-(\\d+)\\.ckpt'). Rolls old checkpoints, keeping the
    newest `keep_last`."""
    path = os.path.join(ckpt_dir, f"checkpoint-{step}.ckpt")
    torch.save(dit.state_dict(), path)
    print(f"[Ckpt] saved {path}")

    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint-*.ckpt")),
                   key=lambda p: int(os.path.basename(p).split("-")[1].split(".")[0]))
    for old in ckpts[:-keep_last]:
        os.remove(old)
        print(f"[Ckpt] removed {old}")
    return path


def save_training_state(ckpt_dir, step, epoch, video_cursor, outer_opt,
                        configs_dict, wandb_run_id, keep_last=3):
    """Save resumable state next to the checkpoint. Inner-loop state is NOT
    saved: save points sit at video boundaries where inner params are already
    restored and inner Adam moments are per-chunk anyway."""
    state = {
        "step": step,
        "epoch": epoch,
        "video_cursor": video_cursor,      # videos finished within this epoch
        "outer_optimizer": outer_opt.state_dict(),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda_all": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
        "configs": configs_dict,
        "wandb_run_id": wandb_run_id,
    }
    path = os.path.join(ckpt_dir, f"training_state-{step}.pt")
    torch.save(state, path)

    states = sorted(glob.glob(os.path.join(ckpt_dir, "training_state-*.pt")),
                    key=lambda p: int(os.path.basename(p).split("-")[1].split(".")[0]))
    for old in states[:-keep_last]:
        os.remove(old)
    return path


def load_resume(resume, dit, outer_opt, device):
    """Resume from a run dir (picks the newest training_state) or a state file.

    Restores: DiT weights (strict), outer optimizer, RNG states, and the
    (step, epoch, video_cursor) cursor. Returns (step, epoch, video_cursor,
    wandb_run_id)."""
    resume = _resolve_path(resume)
    if os.path.isdir(resume):
        ckpt_dir = os.path.join(resume, "checkpoints")
        states = sorted(glob.glob(os.path.join(ckpt_dir, "training_state-*.pt")),
                        key=lambda p: int(os.path.basename(p).split("-")[1].split(".")[0]))
        if not states:
            raise FileNotFoundError(f"[Resume] no training_state-*.pt under {ckpt_dir}")
        state_path = states[-1]
    else:
        state_path = resume
        ckpt_dir = os.path.dirname(state_path)

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    step = state["step"]

    ckpt_path = os.path.join(ckpt_dir, f"checkpoint-{step}.ckpt")
    sd = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = dit.load_state_dict(sd, strict=True)
    del sd

    outer_opt.load_state_dict(state["outer_optimizer"])
    torch.set_rng_state(state["rng"]["torch"].cpu())
    try:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["rng"]["cuda_all"]])
    except Exception as e:
        print(f"[Resume] cuda RNG not restored ({e}); continuing")
    np.random.set_state(state["rng"]["numpy"])
    random.setstate(state["rng"]["python"])

    print(f"[Resume] restored step={step} epoch={state['epoch']} "
          f"video_cursor={state['video_cursor']} from {state_path}")
    return step, state["epoch"], state["video_cursor"], state.get("wandb_run_id")


# ============================================================================
# §11  Training loop (epoch/video scheduling, save/val cadence)
# ============================================================================
def run_training(models, device, dataset, train_cfg, outer_cfg, ttt_cfg, val_cfg,
                 outer_params, outer_opt, inner_params, val_input, run_dir,
                 logger, start_step=0, start_epoch=0, start_video=0):
    """Iterate epochs -> shuffled videos -> train_on_video. Save/val cadence is
    counted in outer optimizer steps but FIRES at video boundaries (inner
    params are restored there, so checkpoints and validations always see the
    clean post-SFT weights).

    Video order per epoch = random.Random(seed+epoch).shuffle — reproducible,
    so resume just skips the first `start_video` finished videos."""
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    configs_dict = {
        "train": asdict(train_cfg), "outer": asdict(outer_cfg),
        "ttt": asdict(ttt_cfg), "validation": asdict(val_cfg),
    }
    step = start_step
    next_save = step + train_cfg.save_every
    next_val = step + train_cfg.val_every

    def _save(epoch, video_cursor):
        save_checkpoint(models.dit, ckpt_dir, step, train_cfg.keep_last_ckpts)
        save_training_state(ckpt_dir, step, epoch, video_cursor, outer_opt,
                            configs_dict, logger.wandb_run_id,
                            train_cfg.keep_last_ckpts)

    if val_cfg.at_start and start_step == 0:
        run_validation(models, device, val_cfg, train_cfg.num_frames,
                       val_input, run_dir, step, logger)

    for epoch in range(start_epoch, train_cfg.num_epochs):
        order = list(range(len(dataset)))
        random.Random(train_cfg.seed + epoch).shuffle(order)
        for vi, idx in enumerate(order):
            if epoch == start_epoch and vi < start_video:
                continue  # resume fast-forward (skip without decoding)
            meta = dataset.meta(idx)
            print(f"[Train] epoch {epoch} video {vi + 1}/{len(order)}: "
                  f"{meta.name} ({meta.num_chunks} chunks) "
                  f"prompt={meta.prompt[:50]}...")
            t0 = time.time()
            pixels_u8 = dataset.load_video_u8(idx)
            step = train_on_video(models, device, train_cfg, outer_cfg, ttt_cfg,
                                  meta, pixels_u8, outer_params, outer_opt,
                                  inner_params, logger, step)
            del pixels_u8
            _torch_gc()
            print(f"[Train] {meta.name} done in {time.time() - t0:.1f}s "
                  f"(global step {step})")

            # ---- video boundary: cadence checks ----
            if step >= next_val:
                run_validation(models, device, val_cfg, train_cfg.num_frames,
                               val_input, run_dir, step, logger)
                next_val += train_cfg.val_every
            if step >= next_save:
                _save(epoch, vi + 1)
                next_save += train_cfg.save_every
            if train_cfg.max_steps is not None and step >= train_cfg.max_steps:
                print(f"[Train] reached max_steps={train_cfg.max_steps}")
                _save(epoch, vi + 1)
                return step
        start_video = 0

    _save(train_cfg.num_epochs - 1, len(dataset))
    print(f"[Train] finished {train_cfg.num_epochs} epoch(s) at step {step}")
    return step


# ============================================================================
# §12  main (orchestration only)
# ============================================================================
def main_train():
    _torch_gc()
    cli = parse_cli()
    train_cfg, outer_cfg, ttt_cfg, val_cfg = build_configs(cli)

    runtime = setup_runtime(train_cfg.seed)
    models = setup_models(train_cfg, val_cfg, runtime)

    if train_cfg.num_frames is None:
        train_cfg.num_frames = models.model_args.validation_data.num_frames
    assert train_cfg.num_frames == 81, \
        "dit_model.py hard-codes move[:, -81:]; num_frames must be 81"

    # ---- parameter sets: outer (SFT) and inner (TTT), disjoint by default ----
    outer_params = outer_param_set(models.dit, outer_cfg)
    inner_params = inner_param_set(models.dit, ttt_cfg)
    check_overlap(outer_params.keys(), inner_params.keys(), outer_cfg.on_overlap)
    union = set(outer_params.keys())
    if ttt_cfg.open:
        union |= set(inner_params.keys())
    apply_grad_flags(models.dit, union)
    models.dit = maybe_wrap_ddp(models.dit, runtime)

    outer_opt = torch.optim.AdamW(outer_params.values(), lr=outer_cfg.lr,
                                  weight_decay=outer_cfg.weight_decay)

    # ---- run dir, resume, logging ----
    # Resuming from a run dir continues IN that run dir (same checkpoints/tb)
    # unless --run-name explicitly asks for a fresh one.
    if (train_cfg.resume is not None and train_cfg.run_name is None
            and os.path.isdir(_resolve_path(train_cfg.resume))):
        resume_dir = _resolve_path(train_cfg.resume)
        train_cfg.output_dir = os.path.dirname(resume_dir)
        train_cfg.run_name = os.path.basename(resume_dir.rstrip("/"))
    run_dir = prepare_run_dir(train_cfg, outer_cfg, ttt_cfg, val_cfg)
    start_step = start_epoch = start_video = 0
    wandb_run_id = None
    if train_cfg.resume is not None:
        start_step, start_epoch, start_video, wandb_run_id = load_resume(
            train_cfg.resume, models.dit, outer_opt, runtime.device)
    logger = TrainLogger(
        run_dir, train_cfg.wandb, train_cfg.wandb_project, train_cfg.run_name,
        {"train": asdict(train_cfg), "outer": asdict(outer_cfg),
         "ttt": asdict(ttt_cfg), "validation": asdict(val_cfg)},
        is_main_rank=(runtime.global_rank == 0), wandb_run_id=wandb_run_id)

    # ---- data & validation case ----
    dataset = VideoFolderDataset(
        train_cfg.dataset_dir, train_cfg.bucket_config_name,
        num_frames=train_cfg.num_frames,
        max_chunks_per_video=train_cfg.max_chunks_per_video,
        dp_rank=runtime.dp_rank, dp_size=runtime.dp_size)
    val_input = load_validation_input(val_cfg, train_cfg.bucket_config_name)

    print(f"[Train] TTT {'ON (' + ttt_cfg.data_source + ')' if ttt_cfg.open else 'OFF'} | "
          f"outer scope={outer_cfg.update_scope} last_k={outer_cfg.last_k_blocks} "
          f"lr={outer_cfg.lr}")

    try:
        run_training(models, runtime.device, dataset, train_cfg, outer_cfg,
                     ttt_cfg, val_cfg, outer_params, outer_opt, inner_params,
                     val_input, run_dir, logger,
                     start_step, start_epoch, start_video)
    finally:
        logger.close()


if __name__ == "__main__":
    main_train()
