"""
Infinite World - Action-Conditioned Video Generation Inference Script
======================================================================
从条件图像 + 文本提示 + 动作序列生成长视频（1000+ 帧）。

支持两种数据格式:
------------------
1. WBench 格式（原始图像 + 动作 JSON）:
   dataset/
     case5/
       image.jpg
       move_view.json
       prompts.json

2. 预处理格式（与 train.py 共享，支持 filter）:
   preprocessed/sekai-game-walking-352_192_30fps/
     video_001/
       meta.json          # 包含 location/scene/weather 等元信息
       latent_full.pt     # VAE 编码后的完整 latent
       text_emb.pt        # 文本编码缓存
       actions.pt         # 动作序列

使用示例:
---------
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes=1 --nproc_per_node=4 --local-ranks-filter=0 \
  scripts/infer.py --config configs/runs/infer/256/test.yaml

  --set ExpConfig.max_chunks=1 ExpConfig.num=1

输入(相关文件):
  configs/infworld_config.yaml     模型自身配置（DiT 结构 / VAE / T5 / 基础 checkpoint）
  configs/infer_default.yaml       本脚本所有可配置项 + 默认值（字段全集，作为文档）
  configs/runs/infer/*.yaml        单次推理的输入，只写要改的字段，分 ExpConfig:/OnlineTrainConfig: 两段

输出文件:
  输出目录是<output_root>/<run_name>/
    videos/test/
      case_1_combined.mp4
      case_7_combined.mp4
      infer_config.json   # 本次运行的参数记录；只写文件，不打印到日志/控制台
      infer_log/
        infer.log         # rank0 日志（全部 log() 行，带时间戳）,同时输出在控制台
        infer_rank{N}.log # 其他 rank 各自的日志
  生成视频前如果目录中已存在同名 mp4，立刻停止整个程序，退出 torchrun。
  全部 case 正常跑完后，在用到的 run yaml 末尾写入"#finished at <时间> UTC+8, gpu num:<world_size>,run name:<yaml>"（中途报错不会标记）。

参数:
---------
  可配置项的全集、含义与默认值见 configs/infer_default.yaml（yaml 里写了 dataclass
  没声明的 key 会直接报错，而不是静默忽略）。两段分别对应本文件的 ExpConfig
  （采样 / 路径 / case 子集 / meta.json 筛选 / 保存）与 OnlineTrainConfig
  （chunk 之间的在线 test-time 训练）；新增参数只需在对应 dataclass 加字段。

代码结构（自上而下）:
  §1 Experiment parameters   - ExpConfig / OnlineTrainConfig dataclasses
  §2 Config                  - parse_cli / apply_section / build_configs
  §3 Runtime environment     - RuntimeContext / setup_runtime
  §4 Model loading           - Models / load_models
  §5 Input loading           - Input / load_inputs (自动检测格式)
  §6 Output paths            - prepare_output_dir / _prepare_output_file
  §7 Single-video generation - generate_one_video / _generate_chunk
  §8 Online training         - online_train_on_chunk
  §9 Batch loop              - run
  §10 main                   - orchestration only
"""

import sys
import os
import cv2
import math
import torch
import random
import json
import datetime
import argparse
import time
import numpy as np
from dataclasses import dataclass, asdict, fields
from typing import Optional
from omegaconf import OmegaConf
import torch.distributed as dist
import torchvision.transforms as transforms
import re

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from infworld.utils.prepare_dataloader import get_obj_from_str
from infworld.utils.data_utils import get_first_clip_from_video, save_silent_video
from infworld.utils.dataset_utils import is_vid, is_img
from infworld.models.checkpoint import set_grad_checkpoint
from infworld.models.scheduler import timestep_transform
from infworld.utils import storage

# ============================================================================
# Action Mapping Dictionaries
# ============================================================================
MOVE_ACTION_MAP = {
    'no-op': 0,
    'go forward': 1,
    'go back': 2,
    'go left': 3,
    'go right': 4,
    'go forward and go left': 5,
    'go forward and go right': 6,
    'go back and go left': 7,
    'go back and go right': 8,
    'uncertain': 9
}

VIEW_ACTION_MAP = {
    'no-op': 0,
    'turn up': 1,
    'turn down': 2,
    'turn left': 3,
    'turn right': 4,
    'turn up and turn left': 5,
    'turn up and turn right': 6,
    'turn down and turn left': 7,
    'turn down and turn right': 8,
    'uncertain': 9
}


# ============================================================================
# Logging helper (rank-aware, flushed)
# ============================================================================
_GLOBAL_RANK = 0    # set once by setup_runtime; used to prefix and gate logs
_LOG_FH = None      # this rank's log file handle, opened by setup_file_logging
_LOG_BUFFER = []    # lines emitted before the log file is known (runtime setup)


class _Tee:
    """Mirror a stream to the log file so *everything* printed lands on disk.

    Wrapping sys.stdout/sys.stderr (rather than only routing log()) is what makes
    the log file self-sufficient: library prints, tqdm sampling bars, warnings and
    uncaught tracebacks all go through these streams, and none of them know about
    log(). The terminal still gets the original stream unchanged.
    """

    def __init__(self, stream, fh):
        self.stream = stream
        self.fh = fh

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.fh.write(data)
        self.fh.flush()
        return len(data)

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def isatty(self):
        # tqdm asks this to decide between a live bar and plain lines; answer for
        # the real terminal so interactive behavior is unchanged.
        return self.stream.isatty()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def setup_file_logging(output_dir, log_subdir):
    """Open this rank's log file, replay the pre-open buffer, and tee the streams.

    Every rank writes its own file (rank0 -> infer.log, rank N -> infer_rank{N}.log),
    so per-rank progress is separable. Lines logged before this call (runtime /
    distributed setup, which happens before the output dir is known) are buffered
    and written here, so nothing is lost. After this returns, stdout and stderr
    are mirrored into the file — including tracebacks from a crash.
    """
    global _LOG_FH
    log_dir = os.path.join(output_dir, log_subdir)
    os.makedirs(log_dir, exist_ok=True)
    name = "infer.log" if _GLOBAL_RANK == 0 else f"infer_rank{_GLOBAL_RANK}.log"
    path = os.path.join(log_dir, name)
    _LOG_FH = open(path, "a", buffering=1)
    if _LOG_BUFFER:
        _LOG_FH.write("".join(_LOG_BUFFER))
        _LOG_BUFFER.clear()
    sys.stdout = _Tee(sys.stdout, _LOG_FH)
    sys.stderr = _Tee(sys.stderr, _LOG_FH)
    return log_dir
    return log_dir


def log(msg, *, tag="InfWorld", all_ranks=False):
    """Print a timestamped, rank-prefixed, flushed log line.

    By default only rank 0 prints (global info identical on every rank, e.g. model
    loading and dataset scan). Pass all_ranks=True for per-rank output (each rank
    runs a different DP shard, so its progress differs). flush=True so lines
    appear immediately when torchrun redirects stdout to a file.

    Printed lines reach the log file through the stdout tee; lines this rank does
    not print are written to the file directly, so each rank's file stays complete.
    Before the file is open, every line is buffered instead (no tee yet).
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{tag}][r{_GLOBAL_RANK}] {msg}"
    printed = all_ranks or _GLOBAL_RANK == 0
    if printed:
        print(line, flush=True)
    if _LOG_FH is None:
        _LOG_BUFFER.append(line + "\n")
    elif not printed:
        _LOG_FH.write(line + "\n")


# ============================================================================
# §1  Experiment parameters (defaults; overridden by the run yaml)
# ============================================================================
@dataclass
class ExpConfig:
    """Sampling knobs, paths, and save settings for one inference run.

    All relative paths are resolved against PROJECT_ROOT by _resolve_path.
    """

    # Sampling
    seed: int = 42
    text_cfg_scale: float = 5.0
    num_sampling_steps: int = 30
    shift: int = 7                       # PX256: 3, PX627: 7, PX960: 11
    max_chunks: int = 20                 # max autoregressive chunks allowed per video
    num_frames: Optional[int] = None     # None -> read model_args.validation_data.num_frames
    bucket_config_name: str = "ASPECT_RATIO_627_F64"
    log_steps: bool = False              # show tqdm sampling progress bar in logs
    negative_prompt: str = (
        "many cars, crowds, Vivid hues, overexposed, static, blurry details, "
        "subtitles, style, work, artwork, image, still, overall grayish, worst quality, "
        "low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, "
        "poorly drawn hands, poorly drawn face, deformed, disfigured, deformed limbs, "
        "fused fingers, motionless image, cluttered background, three legs, "
        "crowded background, walking backwards."
    )

    # Paths (relative -> resolved against PROJECT_ROOT)
    model_config_path: str = "configs/infworld_config.yaml"
    checkpoint_path: Optional[str] = None  # override checkpoint (e.g. weights/my-run/step500.ckpt)
    dataset_dir: str = "dataset/wbench"

    # Outputs go to <output_root>/<run_name>/; run_name defaults to the run yaml's
    # basename (test.yaml -> videos/test/test/).
    output_root: str = "videos"
    run_name: str = ""
    log_subdir: str = "infer_log"  # <output_root>/<run_name>/<log_subdir>/infer[_rank{N}].log

    # Case subset selection (by case number, e.g. case5 -> 5)
    begin_idx: int = -1                  # start at first case with number > begin_idx (-1 -> first case)
    num: int = -1                        # number of cases to run from that point (-1 -> all remaining)

    # Teacher forcing: condition每个 chunk 时用数据集真值 latent 前缀，而不是模型自己
    # 生成的历史帧（仅 preprocessed 格式可用）。
    real_hist: bool = False

    # Data filters (only for preprocessed format with meta.json)
    filter_location: Optional[str] = None
    filter_scene: Optional[str] = None
    filter_crowd_density: Optional[str] = None
    filter_weather: Optional[str] = None
    filter_time_of_day: Optional[str] = None

    # Save
    high_quality_save: bool = True
    fps: int = 30

    # Fixed noise for reproducibility
    use_fixed_noise: bool = False        # enable noise caching/loading
    noise_cache_dir: str = "noise_cache" # where to save/load noise tensors


@dataclass
class OnlineTrainConfig:
    """Online (test-time) training between chunks. Fully inert when open=False."""

    open: bool = False                   # master switch
    n_train_steps: int = 5
    lr: float = 1e-5
    grad_clip_norm: Optional[float] = 1.0
    use_grad_checkpoint: bool = True     # use_reentrant=False set in infworld/models/checkpoint.py
    reset_between_videos: bool = True    # restore init weights per video (train affects only current video)
    # Whitelist of layers to train; everything else is frozen. Entries are
    # parameter-name paths with the per-block index omitted ("blocks.norm3"
    # covers blocks.<i>.norm3 for every whitelisted block). Scheme A (~0.5M
    # params): per-block AdaLN modulation + norm3 affine + output head — adapts
    # global activation/output statistics (the usual long-horizon drift:
    # exposure/color/contrast) with minimal collapse risk from self-distilling
    # the model's own chunks.
    trainable_layers: tuple = (
        "head",              # output head (head.head linear + head.modulation)
        "blocks.modulation", # per-block AdaLN shift/scale (bare nn.Parameter)
        "blocks.norm3",      # per-block cross-attn LayerNorm affine
    )
    # Only blocks.<i>.* with i >= trainable_block_start train (0 -> all blocks;
    # non-block entries like "head" are unaffected). With 30 blocks, 18 keeps
    # the last 12. Backprop then stops at the first trainable block, so the
    # earlier blocks retain no activations and are skipped during backward.
    trainable_block_start: int = 18
    # 可训 block 上界（i <= 该值），-1 表示不设上界。
    trainable_block_end: int = -1


# ============================================================================
# §2  Config (run yaml -> dataclasses)
# ============================================================================
def parse_cli():
    """Only the run yaml and optional dotlist overrides; every run parameter
    itself lives in the yaml (configs/runs/infer/*.yaml)."""
    parser = argparse.ArgumentParser(
        description="Infinite World inference. All run parameters come from a "
                    "run yaml under configs/runs/infer/."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Run config yaml, e.g. configs/runs/infer/test.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="SECTION.KEY=VALUE",
                        help="Temporary overrides on top of the yaml, "
                             "e.g. --set ExpConfig.max_chunks=1")
    args, _ = parser.parse_known_args()
    return args


def apply_section(cfg, values, where):
    """Write one yaml section into a dataclass instance.

    Only fields the dataclass declares are accepted, so a typo'd key raises
    instead of being silently ignored. Values are cast to the default's type
    (bool/int/float/str/tuple); None passes through unchanged.
    """
    defaults = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
    for key, value in (values or {}).items():
        if key not in defaults:
            raise ValueError(f"{where}: unknown option '{key}'. "
                             f"Valid options: {', '.join(sorted(defaults))}")
        default = defaults[key]
        if value is not None and default is not None:
            if isinstance(default, bool):
                value = str(value).strip().lower() in ("1", "true", "yes", "on")
            elif isinstance(default, tuple):
                value = tuple(value)
            elif isinstance(default, (int, float, str)):
                value = type(default)(value)
        setattr(cfg, key, value)
    return cfg


def build_configs(cli):
    """Build (ExpConfig, OnlineTrainConfig) from the run yaml.

    The yaml has two sections named after the dataclasses: `ExpConfig:` and
    `OnlineTrainConfig:`. Fields left out keep their dataclass default (the full
    field set with defaults is documented in configs/infer_default.yaml).
    `--set ExpConfig.max_chunks=1` overrides on top. run_name defaults to the
    yaml's basename, which also names the output sub-directory.
    """
    path = _resolve_path(cli.config)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Run config not found: {path}")
    raw = OmegaConf.load(path)
    if cli.set:
        raw = OmegaConf.merge(raw, OmegaConf.from_dotlist(list(cli.set)))
    raw = OmegaConf.to_container(raw, resolve=True) or {}

    sections = {"ExpConfig": ExpConfig(), "OnlineTrainConfig": OnlineTrainConfig()}
    unknown = [k for k in raw if k not in sections]
    if unknown:
        raise ValueError(f"{path}: unknown section(s) {unknown}. "
                         f"Valid sections: {', '.join(sections)}")
    name = os.path.basename(path)
    exp_config = apply_section(sections["ExpConfig"], raw.get("ExpConfig"),
                               f"{name}:ExpConfig")
    online_train_config = apply_section(sections["OnlineTrainConfig"],
                                        raw.get("OnlineTrainConfig"),
                                        f"{name}:OnlineTrainConfig")
    if not exp_config.run_name:
        exp_config.run_name = os.path.splitext(name)[0]
    return exp_config, online_train_config


def dump_run_config(cli, exp_config, online_train_config, output_dir):
    """记录本次运行的全部参数，方便回顾:
      - config: 用的哪个 run yaml；overrides: --set 传入的临时覆盖
      - ExpConfig / OnlineTrainConfig: 最终生效的完整配置（默认值 + yaml + --set）
    只由 rank0 写 <output_dir>/infer_config.json，不打印到日志/控制台。
    """
    payload = {
        "config": cli.config,
        "overrides": list(cli.set),
        "ExpConfig": asdict(exp_config),
        "OnlineTrainConfig": asdict(online_train_config),
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False, default=str, sort_keys=True)
    if _GLOBAL_RANK == 0:
        path = os.path.join(output_dir, "infer_config.json")
        with open(path, "w") as f:
            f.write(text)
        storage.upload_async(path, is_dir=False)


def mark_config_finished(config_path, world_size, run_name=""):
    """在 run yaml 末尾追加一行完成标记，方便看出哪个配置已经跑完。

    只有 rank0 在 run() 正常返回后调用，所以中途报错的配置不会被标记。时间用东八区
    （机器时区不一定是 CST，所以显式指定 UTC+8，而不是用本地时间）。追加失败只警告
    不抛异常——视频已经生成好了，不该因为标记写不进去而让整个任务算失败。
    """
    path = _resolve_path(config_path)
    cst = datetime.timezone(datetime.timedelta(hours=8))
    stamp = datetime.datetime.now(cst).strftime("%Y-%m-%d %H:%M:%S")
    line = f"#finished at {stamp} UTC+8, gpu num:{world_size}, run_name:{run_name}\n"
    try:
        # yaml 末尾若没有换行，直接 append 会把标记粘到最后一个值上（num: 4 会变成
        # 字符串 "4#finished ..."），所以先补一个换行。
        with open(path) as f:
            existing = f.read()
        needs_newline = bool(existing) and not existing.endswith("\n")
        with open(path, "a") as f:
            if needs_newline:
                f.write("\n")
            f.write(line)
        log(f"Marked finished in {config_path}: {line.strip()}", tag="Config")
    except OSError as e:
        log(f"WARNING: could not mark {config_path} as finished: {e}", tag="Config")


# ============================================================================
# §3  Runtime environment (formerly module-level side effects)
# ============================================================================
@dataclass
class RuntimeContext:
    """Distributed / device state, produced once by setup_runtime."""
    local_rank: int
    global_rank: int
    world_size: int
    use_dist: bool
    dp_rank: int
    dp_size: int
    enable_context_parallel: bool
    device: torch.device


def _setup_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _torch_gc():
    """Clear GPU memory cache."""
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def _cap_gpu_memory(local_rank):
    """按 INFWORLD_GPU_MEM_FRACTION 限制本进程可用显存（占整卡总量的比例）。

    与别人的任务共卡时用：撞到上限时本进程自己 OOM，而不是把对方挤爆。
    未设置该环境变量则不做限制（默认行为不变）。
    """
    frac = os.environ.get('INFWORLD_GPU_MEM_FRACTION')
    if not frac:
        return
    frac = float(frac)
    torch.cuda.set_per_process_memory_fraction(frac, local_rank)
    total_gb = torch.cuda.get_device_properties(local_rank).total_memory / 1024 ** 3
    log(f"GPU 显存上限: fraction={frac} -> {frac * total_gb:.1f} GiB / {total_gb:.1f} GiB",
        all_ranks=True)


def _resolve_path(path, root=None):
    """Resolve a relative path.

    数据/产物目录（dataset/ preprocessed/ checkpoints/ weights/ videos/ ...）落到
    storage.LOCAL_ROOT 并与 S3 同步；代码与 yaml 仍按 PROJECT_ROOT 解析。
    显式传 root 时按该 root 解析（保留旧行为）。
    """
    if path is None:
        return path
    path = str(path).strip()
    if os.path.isabs(path):
        return path
    return os.path.join(root, path) if root is not None else storage.resolve(path)


def setup_runtime(seed, context_parallel_size=1):
    """Initialize distributed (or single-GPU) mode, context parallelism, and seeds.

    Single-GPU mode (no RANK env) bypasses torch.distributed entirely to avoid
    port conflicts, and sets cp_util globals directly so get_dp_rank/get_dp_size
    work without dist. Called exactly once from main() — this replaces the former
    import-time side effects.
    """
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        _cap_gpu_memory(local_rank)
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600 * 24))
        global_rank = dist.get_rank()
        num_processes = dist.get_world_size()
        use_dist = True
    else:
        local_rank = 0
        global_rank = 0
        num_processes = 1
        torch.cuda.set_device(local_rank)
        _cap_gpu_memory(local_rank)
        use_dist = False

    global _GLOBAL_RANK
    _GLOBAL_RANK = global_rank
    log(f"local_rank={local_rank} global_rank={global_rank} world_size={num_processes}",
        all_ranks=True)

    import infworld.context_parallel.context_parallel_util as cp_util
    if use_dist:
        from infworld.context_parallel.context_parallel_util import (
            init_context_parallel, get_dp_size, get_dp_rank,
        )
        init_context_parallel(
            context_parallel_size=context_parallel_size,
            global_rank=global_rank, world_size=num_processes,
        )
        dp_rank = get_dp_rank()
        dp_size = get_dp_size()
    else:
        cp_util.dp_rank = 0
        cp_util.dp_size = 1
        cp_util.cp_rank = 0
        cp_util.cp_size = 1
        dp_rank = 0
        dp_size = 1

    _setup_seed(seed + global_rank)

    return RuntimeContext(
        local_rank=local_rank,
        global_rank=global_rank,
        world_size=num_processes,
        use_dist=use_dist,
        dp_rank=dp_rank,
        dp_size=dp_size,
        enable_context_parallel=(context_parallel_size > 1),
        device=torch.device(local_rank),
    )


# ============================================================================
# §4  Model loading
# ============================================================================
@dataclass
class Models:
    """All loaded model components plus online-training bookkeeping."""
    vae: object
    text_encoder: object
    scheduler: object
    dit: object
    model_args: object                       # parsed OmegaConf config
    checkpoint_path: str
    trainable_params: list                   # online-training params; empty when frozen off
    init_params: Optional[dict]              # trainable-param snapshot for reset_between_videos


def _extract_ckpt_step(path):
    """Extract checkpoint step number from path (0 when the name has no step)."""
    match = re.search(r'checkpoint-(\d+)\.ckpt', path)
    return int(match.group(1)) if match else 0


def _load_dit_state_dict(checkpoint_path):
    """Load DiT state dict from .ckpt (torch) or .safetensors."""
    # 本地缺失则从 S3 拉（每个 rank 都要读，靠 storage 的临时文件+原子改名防竞态）
    checkpoint_path = storage.ensure_local(checkpoint_path, is_dir=False)
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    return state_dict


def _load_model_config(config_path):
    """Load the model YAML and resolve relative checkpoint/VAE/text-encoder paths."""
    args = OmegaConf.load(config_path)
    # VAE/T5 权重也在 checkpoints/ 下，本地缺失则从 S3 拉（tokenizer 是目录）
    if hasattr(args, "vae_cfg") and "vae_pth" in args.vae_cfg:
        args.vae_cfg.vae_pth = storage.ensure_local(args.vae_cfg.vae_pth, is_dir=False)
    if hasattr(args, "text_encoder_cfg"):
        if "checkpoint_path" in args.text_encoder_cfg:
            args.text_encoder_cfg.checkpoint_path = storage.ensure_local(
                args.text_encoder_cfg.checkpoint_path, is_dir=False)
        if "tokenizer_path" in args.text_encoder_cfg:
            args.text_encoder_cfg.tokenizer_path = storage.ensure_local(
                args.text_encoder_cfg.tokenizer_path, is_dir=True)
    return args


def _select_trainable_params(dit, trainable_layers, trainable_block_start,
                             trainable_block_end=-1):
    """Whitelist selection: params under the listed layers train, everything
    else is frozen. Layer entries omit the per-block index ("blocks.norm3"
    matches blocks.<i>.norm3.*); block params additionally require
    trainable_block_start <= index <= trainable_block_end (the latter -1 means
    no upper bound). Returns the trainable params."""
    def canonical(name):
        parts = name.split(".")
        if parts[0] == "blocks":
            parts = ["blocks"] + parts[2:]  # drop the block index
        return ".".join(parts)

    def block_idx(name):
        parts = name.split(".")
        return int(parts[1]) if parts[0] == "blocks" else None

    trainable_params = []
    frozen_n = 0
    for n, p in dit.named_parameters():
        key = canonical(n)
        idx = block_idx(n)
        whitelisted = any(key == l or key.startswith(l + ".") for l in trainable_layers)
        in_range = idx is None or (
            idx >= trainable_block_start
            and (trainable_block_end < 0 or idx <= trainable_block_end))
        if whitelisted and in_range:
            p.requires_grad_(True)
            trainable_params.append(p)
        else:
            p.requires_grad_(False)
            frozen_n += p.numel()
    upper = "inf" if trainable_block_end < 0 else str(trainable_block_end)
    log(f"frozen params: {frozen_n:,}; "
        f"trainable {trainable_layers} (blocks {trainable_block_start}..{upper}): "
        f"{sum(p.numel() for p in trainable_params):,}", tag="OnlineTrain")
    return trainable_params


def load_models(config_path, device, enable_context_parallel,
                num_sampling_steps, shift,
                online_train_open, use_grad_checkpoint,
                trainable_layers, trainable_block_start, trainable_block_end,
                reset_between_videos,
                checkpoint_override=None):
    """Load VAE, text encoder, scheduler, and DiT, then apply online-training setup.

    Only scalar members are passed in (not whole config/runtime objects). When
    checkpoint_override is given (e.g. a train.py checkpoint under weights/),
    it replaces the YAML's checkpoint_path. When online_train_open is True only
    params under the trainable_layers whitelist train (all else frozen); when
    reset_between_videos is also True a snapshot of the trainable params is
    kept in Models.init_params for per-video weight restoration.
    """
    config_path = _resolve_path(config_path)
    model_args = _load_model_config(config_path)
    checkpoint_path = _resolve_path(checkpoint_override or model_args.get(
        "checkpoint_path", "checkpoints/models/diffusion_pytorch_model.safetensors"))

    log("Loading models...")
    vae = get_obj_from_str(model_args.vae_target)(**model_args.vae_cfg).to(device)
    text_encoder = get_obj_from_str(model_args.text_encoder_target)(device=device, **model_args.text_encoder_cfg)
    text_encoder.t5.model.to(device)

    scheduler = get_obj_from_str(model_args.scheduler_target)(**model_args.val_scheduler_cfg)
    scheduler.num_sampling_steps = num_sampling_steps
    scheduler.shift = shift

    dtype = getattr(torch, model_args.amp_dtype)
    dit = get_obj_from_str(model_args.model_target)(
        out_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
        enable_context_parallel=enable_context_parallel,
        **model_args.model_cfg
    ).to(dtype)
    dit.eval()

    state_dict = _load_dit_state_dict(checkpoint_path)
    state_dict.pop("pos_embed_temporal", None)   # recomputed at load time
    state_dict.pop("pos_embed", None)
    missing, unexpected = dit.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        log(f"DiT state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
    dit.to(device)

    trainable_params = []
    init_params = None
    if online_train_open:
        if use_grad_checkpoint:
            set_grad_checkpoint(dit)
        trainable_params = _select_trainable_params(dit, trainable_layers,
                                                    trainable_block_start,
                                                    trainable_block_end)
        if reset_between_videos:
            # Only trainable params can drift, so only they need snapshotting.
            init_params = {n: p.detach().clone()
                           for n, p in dit.named_parameters() if p.requires_grad}

    return Models(
        vae=vae,
        text_encoder=text_encoder,
        scheduler=scheduler,
        dit=dit,
        model_args=model_args,
        checkpoint_path=checkpoint_path,
        trainable_params=trainable_params,
        init_params=init_params,
    )


# ============================================================================
# §5  Input loading (WBench case-directory scan)
# ============================================================================
@dataclass
class Input:
    """One case's fully-materialized inputs (tensors, not paths)."""
    dp_idx: int                    # global enumeration index (for DP sharding + logging)
    name: str                      # e.g. "case5", used for the output filename
    prompt: str                    # prompts.json["prompt"]
    input_image: torch.Tensor      # [1, C, 1, H, W], normalized to [-1, 1], on CPU
    move: torch.Tensor             # LongTensor [N], full action sequence, on CPU
    view: torch.Tensor             # LongTensor [N], on CPU
    case_dir: Optional[str] = None # preprocessed case dir; real_hist 按需读真值 latent


def _get_bucket_config(name):
    """Look up an aspect-ratio bucket table by name."""
    from infworld.configs import bucket_config as bucket_config_module
    return getattr(bucket_config_module, name)


def _resize_and_center_crop(image, target_size):
    """Resize image and center crop to target size -> [1, C, 1, H, W]."""
    orig_h, orig_w = image.shape[:2]
    target_h, target_w = target_size

    scale = max(target_h / orig_h, target_w / orig_w)
    final_h = math.ceil(scale * orig_h)
    final_w = math.ceil(scale * orig_w)

    resized = cv2.resize(image, (final_w, final_h), interpolation=cv2.INTER_AREA)
    resized = np.ascontiguousarray(resized)
    tensor = torch.from_numpy(resized)[None, ...].permute(0, 3, 1, 2).contiguous()
    cropped = transforms.functional.center_crop(tensor, target_size)
    return cropped[:, :, None, :, :]  # [1, C, 1, H, W]


def _load_condition_image(image_path, bucket_config):
    """Load an image/video, pick the closest aspect-ratio bucket, normalize to [-1, 1]."""
    if is_vid(image_path):
        frames = get_first_clip_from_video(image_path, clip_len=1)
    elif is_img(image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames = [image]
    else:
        raise ValueError(f'Unsupported file format: {image_path}')

    processed_frames = []
    for frame in frames:
        ratio = frame.shape[0] / frame.shape[1]
        closest_bucket = sorted(bucket_config.keys(), key=lambda x: abs(float(x) - ratio))[0]
        target_h, target_w = bucket_config[closest_bucket][0]

        tensor = _resize_and_center_crop(frame, (target_h, target_w))
        tensor = (tensor / 255 - 0.5) * 2  # Normalize to [-1, 1]
        processed_frames.append(tensor)

    return torch.cat(processed_frames, dim=2)


def _load_action_sequence(action_path):
    """Load a move/view action sequence from a move_view.json file."""
    with open(action_path, 'r') as f:
        actions = json.load(f)
    move_indices = [MOVE_ACTION_MAP[a['move']] for a in actions]
    view_indices = [VIEW_ACTION_MAP[a['view']] for a in actions]
    return move_indices, view_indices


def _case_sort_key(name):
    """Sort key so case5 < case12 < case100 (numeric, not lexicographic)."""
    match = re.search(r'(\d+)', name)
    return int(match.group(1)) if match else 0


def _load_input(dp_idx, name, case_dir, bucket_config):
    """Materialize one case directory into an Input (CPU tensors)."""
    case_dir = storage.ensure_files(
        case_dir, ("image.jpg", "move_view.json", "prompts.json"))
    image_path = os.path.join(case_dir, "image.jpg")
    action_path = os.path.join(case_dir, "move_view.json")
    prompts_path = os.path.join(case_dir, "prompts.json")

    input_image = _load_condition_image(image_path, bucket_config)
    move_indices, view_indices = _load_action_sequence(action_path)
    with open(prompts_path, 'r') as f:
        prompt = json.load(f)["prompt"]

    return Input(
        dp_idx=dp_idx,
        name=name,
        prompt=prompt,
        input_image=input_image,
        move=torch.tensor(move_indices, dtype=torch.long),
        view=torch.tensor(view_indices, dtype=torch.long),
    )


def _scan_case_names(dataset_dir, begin_idx, n):
    """List case{n} sub-dirs sorted numerically, then apply the subset selection:
    first n cases whose case number > begin_idx (begin_idx=-1 -> from the first;
    n=-1 -> all remaining). Selection happens before DP sharding, so n counts
    global cases, not per-rank."""
    case_names = sorted(
        (d for d in storage.list_dir(dataset_dir) if d.startswith("case")),
        key=_case_sort_key,
    )
    log(f"Found {len(case_names)} case dirs under {dataset_dir}")
    case_names = [name for name in case_names if _case_sort_key(name) > begin_idx]
    if n >= 0:
        case_names = case_names[:n]
    log(f"Selected {len(case_names)} cases (begin_idx={begin_idx}, n={n})")
    return case_names


def _is_preprocessed_dataset(raw_dataset_dir):
    """Dataset format is determined by the configured path, not by probing S3:
    `dataset/...` is always WBench (raw image + action json), `preprocessed/...`
    is always the preprocessed cache (latent + meta.json)."""
    top = raw_dataset_dir.replace(os.sep, "/").strip("/").split("/", 1)[0]
    if top == "preprocessed":
        return True
    if top == "dataset":
        return False
    raise ValueError(
        f"ExpConfig.dataset_dir 必须以 dataset/ 或 preprocessed/ 开头，得到: {raw_dataset_dir}")


def _meta_matches_filters(meta, exp_config):
    """Check one case's meta.json against the filter_* conditions (same
    semantics as train.py's PreprocessedVideoDataset: exact match, None -> skip)."""
    checks = (
        ("location", exp_config.filter_location),
        ("scene", exp_config.filter_scene),
        ("crowdDensity", exp_config.filter_crowd_density),
        ("weather", exp_config.filter_weather),
        ("timeOfDay", exp_config.filter_time_of_day),
    )
    return all(want is None or meta.get(key) == want for key, want in checks)


def _load_preprocessed_input(dp_idx, name, case_dir, meta, vae, device):
    """Materialize one preprocessed case into an Input.

    The condition image is recovered by VAE-decoding the first latent frame of
    latent_full (WanVAE is causal, so latent frame 0 decodes to pixel frame 0).
    Actions are already Long indices in actions.pt; the prompt comes from
    meta.json and is re-encoded by the text encoder during sampling (text_emb.pt
    is not used here because CFG also needs the negative prompt embedding).
    """
    # 只拉真正要读的两个张量：推理不需要 target_latent / text_emb
    case_dir = storage.ensure_files(case_dir, ("latent_full.pt", "actions.pt"))
    latent_full = torch.load(os.path.join(case_dir, "latent_full.pt"))  # [C, T_lat, h, w]
    actions = torch.load(os.path.join(case_dir, "actions.pt"))

    with torch.no_grad():
        first_latent = latent_full[:, :1].unsqueeze(0).to(device)  # [1, C, 1, h, w]
        input_image = vae.decode(first_latent).cpu()               # [1, 3, 1, H, W], in [-1, 1]

    return Input(
        dp_idx=dp_idx,
        name=name,
        prompt=meta["prompt"],
        input_image=input_image,
        move=actions["move"].long(),
        view=actions["view"].long(),
        case_dir=case_dir,
    )


def _load_inputs_preprocessed(dataset_dir, exp_config, dp_rank, dp_size, vae, device):
    """Load this rank's shard from a preprocessed dataset dir, applying the
    filter_* conditions on meta.json before subset selection and DP sharding
    (so filters shrink the candidate pool exactly like train.py)."""
    all_names = sorted(
        (d for d in storage.list_dir(dataset_dir) if d.startswith("case")),
        key=_case_sort_key,
    )
    log(f"Found {len(all_names)} case dirs under {dataset_dir}")

    # Filter by meta.json first (mirrors train.py), then apply begin_idx/n.
    metas = {}
    case_names = []
    for name in all_names:
        meta_path = os.path.join(dataset_dir, name, "meta.json")
        if not os.path.exists(meta_path):
            storage.ensure_local(meta_path, is_dir=False, check=False)
        if not os.path.exists(meta_path):
            log(f"Skipping {name}: missing meta.json")
            continue
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        if _meta_matches_filters(meta, exp_config):
            metas[name] = meta
            case_names.append(name)
    active = [(k, v) for k, v in (
        ("location", exp_config.filter_location),
        ("scene", exp_config.filter_scene),
        ("crowdDensity", exp_config.filter_crowd_density),
        ("weather", exp_config.filter_weather),
        ("timeOfDay", exp_config.filter_time_of_day)) if v is not None]
    if active:
        log(f"Filters {dict(active)} -> {len(case_names)} cases")

    case_names = [name for name in case_names if _case_sort_key(name) > exp_config.begin_idx]
    if exp_config.num >= 0:
        case_names = case_names[:exp_config.num]
    log(f"Selected {len(case_names)} cases "
        f"(begin_idx={exp_config.begin_idx}, num={exp_config.num})")

    inputs = []
    for dp_idx, name in enumerate(case_names):
        if dp_idx % dp_size != dp_rank:
            continue
        case_dir = os.path.join(dataset_dir, name)
        inputs.append(_load_preprocessed_input(dp_idx, name, case_dir, metas[name], vae, device))

    log(f"dp_rank {dp_rank}/{dp_size}: {len(inputs)} cases to run", all_ranks=True)
    return inputs


def _load_inputs_wbench(dataset_dir, bucket_config_name, dp_rank, dp_size, begin_idx, n):
    """Load this rank's shard from a WBench dataset dir. Directories missing any
    of image.jpg / move_view.json / prompts.json are skipped with a message.
    Only cases where dp_idx % dp_size == dp_rank are actually read; dp_idx is a
    contiguous index over the selected subset."""
    bucket_config = _get_bucket_config(bucket_config_name)
    required = ("image.jpg", "move_view.json", "prompts.json")
    case_names = _scan_case_names(dataset_dir, begin_idx, n)

    inputs = []
    for dp_idx, name in enumerate(case_names):
        # 先按 rank 分片再检查文件：本地 nvme 未缓存的 case 要先从 S3 拉下来才能
        # 判断"缺文件"，而每个 rank 只该为自己的分片付下载开销。
        # dp_idx 来自 enumerate(case_names)，跳过某个 case 不会让后面的编号错位。
        if dp_idx % dp_size != dp_rank:
            continue
        case_dir = storage.ensure_files(
            os.path.join(dataset_dir, name), required, check=False)
        if any(not os.path.exists(os.path.join(case_dir, f)) for f in required):
            log(f"Skipping {name}: missing one of {required}", all_ranks=True)
            continue
        inputs.append(_load_input(dp_idx, name, case_dir, bucket_config))

    log(f"dp_rank {dp_rank}/{dp_size}: {len(inputs)} cases to run", all_ranks=True)
    return inputs


def load_inputs(exp_config, dp_rank, dp_size, vae, device):
    """Dispatch to the matching loader based on exp_config.dataset_dir's top-level
    directory (dataset/ -> WBench, preprocessed/ -> preprocessed).

    Preprocessed format supports filter_*; WBench format keeps the original
    behavior. Filters on a WBench dir are a usage error, so fail fast instead
    of silently ignoring them.
    """
    dataset_dir = _resolve_path(exp_config.dataset_dir)
    if _is_preprocessed_dataset(exp_config.dataset_dir):
        log("Dataset format: preprocessed (path under preprocessed/)")
        return _load_inputs_preprocessed(dataset_dir, exp_config, dp_rank, dp_size, vae, device)

    if exp_config.real_hist:
        raise ValueError(
            f"real_hist=true 需要 preprocessed 数据集（含 meta.json / latent_full.pt），"
            f"但 {dataset_dir} 看起来是 WBench 目录")

    has_filters = any(v is not None for v in (
        exp_config.filter_location, exp_config.filter_scene,
        exp_config.filter_crowd_density, exp_config.filter_weather,
        exp_config.filter_time_of_day))
    if has_filters:
        raise ValueError(
            f"filter_* options require a preprocessed dataset (with meta.json), "
            f"but {dataset_dir} looks like a WBench dir")
    log("Dataset format: WBench (raw image + action json)")
    return _load_inputs_wbench(
        dataset_dir, exp_config.bucket_config_name, dp_rank, dp_size,
        exp_config.begin_idx, exp_config.num)


# ============================================================================
# §6  Output paths
# ============================================================================
def prepare_output_dir(output_root, run_name):
    """Resolve <output_root>/<run_name> (videos/ -> storage.LOCAL_ROOT) and mkdir it.

    run_name defaults to the run yaml's basename, so configs/runs/infer/test.yaml
    writes to videos/test/test/。
    """
    output_dir = os.path.join(_resolve_path(output_root), run_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _prepare_output_file(output_dir, name):
    """Build the (extension-less) save path; save_silent_video appends .mp4.

    WBench only picks up videos matching case_*_combined.mp4, so the dataset
    dir name 'case5' becomes 'case_5' here.

    e.g. name='case5' -> '<output_dir>/case_5_combined' -> case_5_combined.mp4
    """
    name = re.sub(r"^case(\d+)$", r"case_\1", name)
    return os.path.join(output_dir, f"{name}_combined")


# ============================================================================
# §6.5  Fixed noise loading
# ============================================================================
def _load_fixed_noise(noise_cache_dir, case_name, chunk_idx, z_size, device):
    """Load pre-generated noise from configs/noise/ directory.

    Args:
        noise_cache_dir: directory containing pre-generated noise (e.g. configs/noise/noise0)
        case_name: case name (ignored, noise_cache_dir already specifies which noise set)
        chunk_idx: chunk index
        z_size: expected noise shape
        device: torch device

    Returns:
        noise tensor of shape z_size

    固定噪声一旦开启就必须真的生效：缺文件或形状不符时直接报错，
    而不是退回随机噪声，否则实验结果不可复现却无从察觉。
    """
    noise_path = os.path.join(
        _resolve_path(noise_cache_dir),
        f"chunk_{chunk_idx}.pt"
    )

    if not os.path.exists(noise_path):
        raise FileNotFoundError(
            f"固定噪声文件不存在: {noise_path}（chunk {chunk_idx}）。"
            f"请用 configs/noise/generate_noise.py 生成足够数量的 chunk，"
            f"或关闭 ExpConfig.use_fixed_noise。"
        )

    noise = torch.load(noise_path, map_location=device)

    if noise.shape != torch.Size(z_size):
        raise ValueError(
            f"固定噪声形状不匹配: {noise_path} 是 {tuple(noise.shape)}，"
            f"当前需要 {tuple(z_size)}。请按当前分辨率重新生成噪声。"
        )

    return noise


# ============================================================================
# §7  Single-video generation (chunked autoregressive core)
# ============================================================================
@dataclass
class ChunkResult:
    """Everything one chunk produces, so generation and online training stay decoupled."""
    samples: torch.Tensor       # sampled latent (x_start for online training)
    decoded: torch.Tensor       # VAE-decoded pixel chunk, on CPU
    cond_latent: torch.Tensor   # this chunk's condition latent (reused by online training)
    move: torch.Tensor          # [1, num_frames]
    view: torch.Tensor          # [1, num_frames]
    gt_target: Optional[torch.Tensor] = None  # real_hist: 该 chunk 的真值 latent
                                              # [1, C, 21, h, w]，在线训练用它当 x_start


def _load_real_hist(case_dir, need_target):
    """读入 real_hist 需要的真值张量（CPU），逐视频加载、用完即弃。

    latent_full[:, :20k+1] 就是 train.py::make_chunk_batch 给 chunk k 的 image_cond：
    WanVAE 是因果的，前缀切片等价于单独编码像素帧 [0, 80k+1)，所以这里不需要解码成
    像素再 encode 一遍。need_target（在线训练开着）时再额外拉 target_latent.pt。
    """
    files = ("latent_full.pt", "target_latent.pt") if need_target else ("latent_full.pt",)
    case_dir = storage.ensure_files(case_dir, files)
    latent_full = torch.load(os.path.join(case_dir, "latent_full.pt"), map_location="cpu")
    target_latent = None
    if need_target:
        target_latent = torch.load(
            os.path.join(case_dir, "target_latent.pt"), map_location="cpu")
    return latent_full, target_latent


def _slice_move_view(move, view, start, num_frames, device):
    """Slice [start, start+num_frames) from the full action tensors, zero-pad the tail."""
    seg_move = move[start:start + num_frames].to(device)
    seg_view = view[start:start + num_frames].to(device)
    if seg_move.shape[0] < num_frames:
        pad_len = num_frames - seg_move.shape[0]
        pad = torch.zeros(pad_len, dtype=torch.long, device=device)
        seg_move = torch.cat([seg_move, pad])
        seg_view = torch.cat([seg_view, pad])
    return seg_move, seg_view


def _generate_chunk(models, device, prompt, negative_prompt, text_cfg_scale,
                    video_buffer, latent_size, move, view, num_frames, log_steps,
                    use_fixed_noise=False, noise_cache_dir=None, case_name=None, chunk_idx=None,
                    cond_latent_override=None):
    """Generate one chunk: encode buffer tail -> slice actions -> sample -> decode.

    Returns a ChunkResult holding the sampled latent, the decoded pixels (CPU),
    and the condition latent / action tensors that online training reuses.

    Args:
        use_fixed_noise: if True, load pre-generated noise
        noise_cache_dir: directory with pre-generated noise files
        case_name: case name for noise lookup
        chunk_idx: chunk index for noise lookup
        cond_latent_override: real_hist 模式下由调用方给出的真值条件 latent；
            给了就跳过对 video_buffer 的 VAE encode（那份 buffer 只用于存盘）。
    """
    if cond_latent_override is not None:
        cond_latent = cond_latent_override
    else:
        with torch.no_grad():
            current_cond = video_buffer.to(device)
            cond_latent = models.vae.encode(current_cond)

    curr_start = video_buffer.shape[2] - 1  # tail frame overlaps into the next chunk
    seg_move, seg_view = _slice_move_view(move, view, curr_start, num_frames, device)

    # Load fixed noise if enabled
    initial_noise = None
    if use_fixed_noise and noise_cache_dir:
        initial_noise = _load_fixed_noise(noise_cache_dir, case_name, chunk_idx, latent_size, device)

    additional_args = {
        "image_cond": cond_latent,       # DiT forward's formal arg name (kept as-is)
        "move": seg_move.unsqueeze(0),
        "view": seg_view.unsqueeze(0),
    }
    _torch_gc()
    with torch.no_grad():
        samples = models.scheduler.sample(
            model=models.dit,
            text_encoder=models.text_encoder,
            null_embedder=models.dit.y_embedder,
            z_size=latent_size,
            prompts=[prompt],
            guidance_scale=text_cfg_scale,
            negative_prompts=[negative_prompt],
            device=device,
            additional_args=additional_args,
            progress=log_steps,  # tqdm progress bar controlled by log_steps param
            initial_noise=initial_noise,
        )
        decoded = models.vae.decode(samples).cpu()

    return ChunkResult(
        samples=samples,
        decoded=decoded,
        cond_latent=cond_latent,
        move=seg_move.unsqueeze(0),
        view=seg_view.unsqueeze(0),
    )


def generate_one_video(models, device, exp_config, online_train_config, input):
    """Generate one full (1000+ frame) video by chunked autoregression.

    Seeds video_buffer with the condition image, then for each chunk conditions
    on the previously generated tail. When online training is on, the prompt is
    encoded once up front (see §8) and each chunk (except the last) fine-tunes
    the DiT before the next chunk is produced. Returns the full buffer on CPU.

    real_hist=True 时条件区换成数据集的真值 latent 前缀（teacher forcing）：生成帧
    不再回流进下一个 chunk 的条件，video_buffer 退化为纯粹的存盘缓冲。
    """
    num_frames = exp_config.num_frames
    video_buffer = input.input_image.clone().cpu()

    # Per-video chunk count: floor(move_view.json frame count / 80), capped by max_chunks.
    num_chunks = min(input.move.shape[0] // 80, exp_config.max_chunks)
    log(f"{input.name}: {input.move.shape[0]} action frames -> "
        f"num_chunks={num_chunks} (max_chunks={exp_config.max_chunks})", all_ranks=True)

    # real_hist：真值条件 latent（以及在线训练要的真值目标）。
    latent_full = target_latent = None
    if exp_config.real_hist:
        latent_full, target_latent = _load_real_hist(
            input.case_dir, need_target=online_train_config.open)
        # chunk k 需要 latent_full[:, :20k+1]，因此 k <= (T_lat-1)//20。
        cap = (latent_full.shape[1] - 1) // 20 + 1
        if target_latent is not None:
            cap = min(cap, target_latent.shape[0])
        log(f"{input.name}: real_hist 生效，条件区用真值 latent_full{list(latent_full.shape)}"
            + ("，在线训练目标用真值 target_latent" if target_latent is not None else ""),
            all_ranks=True)
        if cap < num_chunks:
            log(f"{input.name}: real_hist 真值只够 {cap} 个 chunk "
                f"(latent T={latent_full.shape[1]})，num_chunks {num_chunks} -> {cap}",
                all_ranks=True)
            num_chunks = cap

    with torch.no_grad():
        cond_latent = models.vae.encode(input.input_image.to(device))
    # Latent size is [B, C, T, H, W]; T=21 => 1 condition frame + 20 generated.
    latent_size = list(cond_latent.shape)
    latent_size[2] = 21
    latent_size = torch.Size(latent_size)

    # Cache the text embedding once per video for the online-training steps.
    cached_y = cached_y_mask = None
    if online_train_config.open:
        with torch.no_grad():
            text_kwargs = models.text_encoder.encode([input.prompt])
        cached_y, cached_y_mask = text_kwargs["y"], text_kwargs["y_mask"]

    for chunk_idx in range(num_chunks):
        chunk_start_time = time.time()
        # real_hist：条件区取真值前缀（与 train.py 的 image_cond 完全一致），
        # dtype 对齐现场 encode 的结果，避免 bf16 缓存与 fp32 VAE 输出混用。
        cond_override = None
        if latent_full is not None:
            cond_override = latent_full[:, :20 * chunk_idx + 1].unsqueeze(0).to(
                device=device, dtype=cond_latent.dtype)
        result = _generate_chunk(
            models, device, input.prompt, exp_config.negative_prompt,
            exp_config.text_cfg_scale, video_buffer, latent_size,
            input.move, input.view, num_frames, exp_config.log_steps,
            use_fixed_noise=exp_config.use_fixed_noise,
            noise_cache_dir=exp_config.noise_cache_dir,
            case_name=input.name,
            chunk_idx=chunk_idx,
            cond_latent_override=cond_override,
        )
        if target_latent is not None:
            result.gt_target = target_latent[chunk_idx].unsqueeze(0).to(
                device=device, dtype=cond_latent.dtype)
        chunk_elapsed = time.time() - chunk_start_time
        video_buffer = torch.cat([video_buffer, result.decoded[:, :, 1:]], dim=2)
        log(f"{input.name}: chunk {chunk_idx + 1}/{num_chunks} done, "
            f"total frames {video_buffer.shape[2]}, time {chunk_elapsed:.1f}s", all_ranks=True)
        _torch_gc()

        if online_train_config.open and chunk_idx < num_chunks - 1:
            train_start_time = time.time()
            losses = online_train_on_chunk(
                models, online_train_config, result, cached_y, cached_y_mask)
            train_elapsed = time.time() - train_start_time
            log(f"{input.name} chunk {chunk_idx}: losses "
                + " ".join(f"{v:.5f}" for v in losses)
                + f", time {train_elapsed:.1f}s", tag="OnlineTrain", all_ranks=True)
            _torch_gc()

    return video_buffer


# ============================================================================
# §8  Online training (test-time; weights affect only the current video)
# ============================================================================
def _online_train_step(dit, scheduler, x_start, model_kwargs, optimizer, grad_clip_norm):
    """One rectified-flow training step on a single (x_start, model_kwargs) pair.

    Mirrors RFlowScheduler.training_losses but calls the DiT with x_ignore_mask=None
    so we avoid the post-temporal-compression mask shape requirement. x_start is the
    just-sampled latent for the chunk (a video sample, not the sampling noise).
    """
    optimizer.zero_grad(set_to_none=True)
    device = x_start.device
    B = x_start.shape[0]

    if scheduler.use_discrete_timesteps:
        t = torch.randint(0, scheduler.num_timesteps, (B,), device=device)
    elif scheduler.sample_method == "uniform":
        t = torch.rand((B,), device=device) * scheduler.num_timesteps
    else:  # logit-normal
        t = scheduler.sample_t(x_start) * scheduler.num_timesteps
    if scheduler.use_timestep_transform:
        t = timestep_transform(t, shift=scheduler.shift, num_timesteps=scheduler.num_timesteps)

    noise = torch.randn_like(x_start)
    x_t = scheduler.add_noise(x_start, noise, t)
    target = x_start - noise
    if scheduler.use_reversed_velocity:
        target = -target

    pred = dit(x_t, t, x_ignore_mask=None, **model_kwargs)
    # DiT input is [image_cond | x_t] along time; keep the last T_x frames.
    pred = pred[:, :, -x_start.shape[2]:]

    loss = ((pred.float() - target.float()) ** 2).mean()
    loss.backward()
    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(dit.parameters(), max_norm=grad_clip_norm)
    optimizer.step()
    return loss.detach()


def online_train_on_chunk(models, online_train_config, chunk, cached_y, cached_y_mask):
    """Fine-tune the DiT on the chunk just generated, before producing the next one.

    A fresh AdamW is built per chunk so Adam moments do NOT accumulate across
    chunks. Returns the per-step loss values (floats).

    x_start 默认是刚采样出来的 latent（自蒸馏）；real_hist 模式下 chunk.gt_target
    非空，改用数据集真值 chunk，等价于在推理循环里做真监督的 flow-matching 微调。
    """
    optimizer = torch.optim.AdamW(models.trainable_params, lr=online_train_config.lr)
    x_start = (chunk.gt_target if chunk.gt_target is not None else chunk.samples).detach()
    train_kwargs = {
        "y": cached_y,
        "y_mask": cached_y_mask,
        "image_cond": chunk.cond_latent.detach(),
        "move": chunk.move.detach(),
        "view": chunk.view.detach(),
    }
    losses = []
    for _ in range(online_train_config.n_train_steps):
        loss_val = _online_train_step(
            models.dit, models.scheduler, x_start, train_kwargs,
            optimizer, online_train_config.grad_clip_norm)
        losses.append(loss_val.item())
    del x_start, train_kwargs
    return losses


def _restore_init_params(dit, init_params):
    """Copy the snapshotted initial trainable weights back into the DiT (no_grad)."""
    with torch.no_grad():
        for n, p in dit.named_parameters():
            if n in init_params:
                p.data.copy_(init_params[n])


# ============================================================================
# §9  Batch loop
# ============================================================================
def run(models, device, exp_config, online_train_config, inputs, output_dir):
    """Generate and save a video for each Input (already this rank's DP shard).

    When online training resets between videos, the initial weights are restored
    before every video after the first, so training only affects the current one.
    """
    # Pre-check: verify no output files exist before starting generation.
    # 本地 nvme 是新机器时也要防覆盖：S3 上已有同名结果同样算已存在。
    # 输出目录只列一次（几百个 case 逐个 s3 ls 会很慢）。
    existing = set(storage.list_dir(output_dir))
    for inp in inputs:
        save_path = _prepare_output_file(output_dir, inp.name)
        final_path = f"{save_path}.mp4"
        if os.path.exists(final_path) or os.path.basename(final_path) in existing:
            log(f"ERROR: Output file already exists: {final_path}", all_ranks=True)
            log(f"Stopping execution to avoid overwriting existing files.", all_ranks=True)
            log(f"Please remove existing files or use a different output directory.", all_ranks=True)
            raise FileExistsError(f"Output file already exists: {final_path}")

    for i, inp in enumerate(inputs):
        log(f"Task {inp.dp_idx} ({inp.name}): {inp.prompt[:50]}...", all_ranks=True)

        if online_train_config.open and online_train_config.reset_between_videos and i > 0:
            _restore_init_params(models.dit, models.init_params)
            log(f"{inp.name}: restored init params", tag="OnlineTrain", all_ranks=True)

        video = generate_one_video(models, device, exp_config, online_train_config, inp)

        save_path = _prepare_output_file(output_dir, inp.name)
        quality = 10 if exp_config.high_quality_save else 5
        save_silent_video(video.cpu(), save_path, fps=exp_config.fps, quality=quality)
        log(f"Saved: {save_path}.mp4", all_ranks=True)
        # 视频后台上传，下一个 case 的生成不等它
        storage.upload_async(f"{save_path}.mp4", is_dir=False)


# ============================================================================
# §10  main (orchestration only)
# ============================================================================
def main():
    _torch_gc()
    cli = parse_cli()
    exp_config, online_train_config = build_configs(cli)

    runtime = setup_runtime(exp_config.seed)

    # 先建输出目录、接上文件日志并记录参数（模型加载前落盘，崩溃也留有记录）。
    output_dir = prepare_output_dir(exp_config.output_root, exp_config.run_name)
    log_dir = setup_file_logging(output_dir, exp_config.log_subdir)
    log(storage.describe())
    dump_run_config(cli, exp_config, online_train_config, output_dir)
    # 推理日志每 5 分钟同步一次，长跑时也能在 S3 上看进度
    if runtime.global_rank == 0:
        storage.start_periodic_sync(log_dir, interval=300)

    models = load_models(
        exp_config.model_config_path, runtime.device, runtime.enable_context_parallel,
        exp_config.num_sampling_steps, exp_config.shift,
        online_train_config.open, online_train_config.use_grad_checkpoint,
        online_train_config.trainable_layers, online_train_config.trainable_block_start,
        online_train_config.trainable_block_end,
        online_train_config.reset_between_videos,
        checkpoint_override=exp_config.checkpoint_path,
    )

    # Fill num_frames from the model config when not overridden.
    if exp_config.num_frames is None:
        exp_config.num_frames = models.model_args.validation_data.num_frames

    inputs = load_inputs(
        exp_config, runtime.dp_rank, runtime.dp_size,
        models.vae, runtime.device,
    )

    run(models, runtime.device, exp_config, online_train_config, inputs, output_dir)

    # 全部 rank 跑完后才在 run yaml 上盖完成标记（任何 rank 报错都不会走到这里）。
    # 各 rank 先等自己的视频传完，再 barrier，保证盖完成标记时 S3 上结果是齐的
    storage.wait_uploads()
    if runtime.use_dist:
        dist.barrier()
    if runtime.global_rank == 0:
        mark_config_finished(cli.config, runtime.world_size, exp_config.run_name)
        storage.stop_periodic_sync(log_dir)


if __name__ == "__main__":
    main()
