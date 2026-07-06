"""
Infinite World - Action-Conditioned Video Generation Inference Script
======================================================================
A standalone inference script for generating long videos with action control.

Structure (top to bottom):
  §1 Experiment parameters   - ExpConfig / OnlineTrainConfig dataclasses (edit here)
  §2 CLI                     - parse_cli / build_configs
  §3 Runtime environment     - RuntimeContext / setup_runtime
  §4 Model loading           - Models / load_models
  §5 Input loading           - Input / load_inputs (WBench case scan)
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
import numpy as np
from dataclasses import dataclass, field
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
# §1  Experiment parameters (edit here or override via CLI)
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
    num_chunks: int = 20                 # number of autoregressive chunks per video
    num_frames: Optional[int] = None     # None -> read model_args.validation_data.num_frames
    bucket_config_name: str = "ASPECT_RATIO_627_F64"
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
    dataset_dir: str = "dataset/wbench"
    output_dir: str = "../WBench/work_dirs/test/videos"  # parent dir of PROJECT_ROOT

    # Case subset selection (by case number, e.g. case5 -> 5)
    begin_idx: int = -1                  # start at first case with number > begin_idx (-1 -> first case)
    n: int = -1                          # number of cases to run from that point (-1 -> all remaining)

    # Save
    high_quality_save: bool = True
    fps: int = 30


@dataclass
class OnlineTrainConfig:
    """Online (test-time) training between chunks. Fully inert when open=False."""

    open: bool = False                   # master switch
    n_train_steps: int = 5
    lr: float = 1e-5
    reset_between_videos: bool = True    # restore init weights per video (train affects only current video)
    use_grad_checkpoint: bool = True     # use_reentrant=False set in infworld/models/checkpoint.py
    grad_clip_norm: Optional[float] = 1.0
    # Whitelist of layers to train; everything else is frozen. Entries are
    # parameter-name paths with the per-block index omitted ("blocks.norm3"
    # covers blocks.0.norm3 ... blocks.29.norm3). Scheme A (~0.5M params):
    # per-block AdaLN modulation + norm3 affine + output head — adapts global
    # activation/output statistics (the usual long-horizon drift:
    # exposure/color/contrast) with minimal collapse risk from self-distilling
    # the model's own chunks.
    trainable_layers: tuple = (
        "head",              # output head (head.head linear + head.modulation)
        "blocks.modulation", # per-block AdaLN shift/scale (bare nn.Parameter)
        "blocks.norm3",      # per-block cross-attn LayerNorm affine
    )


# ============================================================================
# §2  CLI
# ============================================================================
def parse_cli():
    """Parse command-line overrides. Every flag defaults to None so build_configs
    knows which ones to apply over the dataclass defaults."""
    parser = argparse.ArgumentParser(description="Infinite World inference")
    parser.add_argument(
        "--online-training",
        type=lambda s: s.strip().lower() in ("on", "true", "1", "yes"),
        default=None,
        help="Enable online (test-time) training between chunks: on/off",
    )
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--num-chunks", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None, help="num_sampling_steps")
    parser.add_argument("--cfg-scale", type=float, default=None, help="text_cfg_scale")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--shift", type=int, default=None)
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--begin-idx", type=int, default=None,
                        help="Start at first case whose number > begin_idx (-1 -> first case)")
    parser.add_argument("--n", type=int, default=None,
                        help="Number of cases to run from begin_idx (-1 -> all remaining)")
    args, _ = parser.parse_known_args()
    return args


def build_configs(cli):
    """Build (ExpConfig, OnlineTrainConfig) from dataclass defaults, overriding
    with any CLI flag that was actually provided (non-None)."""
    exp_config = ExpConfig()
    online_train_config = OnlineTrainConfig()

    if cli.num_chunks is not None:
        exp_config.num_chunks = cli.num_chunks
    if cli.steps is not None:
        exp_config.num_sampling_steps = cli.steps
    if cli.cfg_scale is not None:
        exp_config.text_cfg_scale = cli.cfg_scale
    if cli.seed is not None:
        exp_config.seed = cli.seed
    if cli.shift is not None:
        exp_config.shift = cli.shift
    if cli.dataset_dir is not None:
        exp_config.dataset_dir = cli.dataset_dir
    if cli.output_dir is not None:
        exp_config.output_dir = cli.output_dir
    if cli.begin_idx is not None:
        exp_config.begin_idx = cli.begin_idx
    if cli.n is not None:
        exp_config.n = cli.n

    if cli.online_training is not None:
        online_train_config.open = cli.online_training
    if cli.train_steps is not None:
        online_train_config.n_train_steps=cli.train_steps

    return exp_config, online_train_config


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


def _resolve_path(path, root=PROJECT_ROOT):
    """Resolve path: if relative, join with project root."""
    if path is None:
        return path
    path = str(path).strip()
    if not os.path.isabs(path):
        path = os.path.join(root, path)
    return path


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
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600 * 24))
        global_rank = dist.get_rank()
        num_processes = dist.get_world_size()
        use_dist = True
    else:
        local_rank = 0
        global_rank = 0
        num_processes = 1
        torch.cuda.set_device(local_rank)
        use_dist = False

    print(f"[InfWorld] local_rank(在当前机器上的进程编号): {local_rank} | global_rank(在所有机器上): {global_rank} | world_size: {num_processes}")

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
    checkpoint_path = _resolve_path(checkpoint_path)
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
    if hasattr(args, "vae_cfg") and "vae_pth" in args.vae_cfg:
        args.vae_cfg.vae_pth = _resolve_path(args.vae_cfg.vae_pth)
    if hasattr(args, "text_encoder_cfg"):
        if "checkpoint_path" in args.text_encoder_cfg:
            args.text_encoder_cfg.checkpoint_path = _resolve_path(args.text_encoder_cfg.checkpoint_path)
        if "tokenizer_path" in args.text_encoder_cfg:
            args.text_encoder_cfg.tokenizer_path = _resolve_path(args.text_encoder_cfg.tokenizer_path)
    return args


def _select_trainable_params(dit, trainable_layers):
    """Whitelist selection: params under the listed layers train, everything
    else is frozen. Layer entries omit the per-block index ("blocks.norm3"
    matches blocks.<i>.norm3.*). Returns the trainable params."""
    def canonical(name):
        parts = name.split(".")
        if parts[0] == "blocks":
            parts = ["blocks"] + parts[2:]  # drop the block index
        return ".".join(parts)

    trainable_params = []
    frozen_n = 0
    for n, p in dit.named_parameters():
        key = canonical(n)
        if any(key == l or key.startswith(l + ".") for l in trainable_layers):
            p.requires_grad_(True)
            trainable_params.append(p)
        else:
            p.requires_grad_(False)
            frozen_n += p.numel()
    print(f"[OnlineTrain] frozen params: {frozen_n:,}; "
          f"trainable {trainable_layers}: "
          f"{sum(p.numel() for p in trainable_params):,}")
    return trainable_params


def load_models(config_path, device, enable_context_parallel,
                num_sampling_steps, shift,
                online_train_open, use_grad_checkpoint,
                trainable_layers, reset_between_videos):
    """Load VAE, text encoder, scheduler, and DiT, then apply online-training setup.

    Only scalar members are passed in (not whole config/runtime objects). When
    online_train_open is True only params under the trainable_layers whitelist
    train (all else frozen); when reset_between_videos is also True a snapshot
    of the trainable params is kept in Models.init_params for per-video weight
    restoration.
    """
    config_path = _resolve_path(config_path)
    model_args = _load_model_config(config_path)
    checkpoint_path = _resolve_path(model_args.get(
        "checkpoint_path", "checkpoints/models/diffusion_pytorch_model.safetensors"))

    print(f"[InfWorld] Loading checkpoint: {checkpoint_path}")
    print(f"[InfWorld] Config: {config_path}")

    print("[InfWorld] Loading VAE...")
    vae = get_obj_from_str(model_args.vae_target)(**model_args.vae_cfg).to(device)

    print("[InfWorld] Loading Text Encoder...")
    text_encoder = get_obj_from_str(model_args.text_encoder_target)(device=device, **model_args.text_encoder_cfg)
    text_encoder.t5.model.to(device)

    print("[InfWorld] Loading Scheduler...")
    scheduler = get_obj_from_str(model_args.scheduler_target)(**model_args.val_scheduler_cfg)
    scheduler.num_sampling_steps = num_sampling_steps
    scheduler.shift = shift

    print("[InfWorld] Loading DiT Model...")
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
    print(f"[InfWorld] Model loaded! Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    dit.to(device)

    trainable_params = []
    init_params = None
    if online_train_open:
        if use_grad_checkpoint:
            set_grad_checkpoint(dit)
        trainable_params = _select_trainable_params(dit, trainable_layers)
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


def load_inputs(dataset_dir, bucket_config_name, dp_rank, dp_size, begin_idx=-1, n=-1):
    """Scan a WBench dataset dir for case{n} sub-dirs and load this rank's shard.

    Cases are sorted numerically. The subset selection keeps the first n cases
    whose case number is > begin_idx (begin_idx=-1 -> start at the first case;
    n=-1 -> all remaining). Selection happens before DP sharding, so n counts
    global cases, not per-rank. Directories missing any of image.jpg /
    move_view.json / prompts.json are skipped with a message. Only cases where
    dp_idx % dp_size == dp_rank are actually read, so each rank loads just the
    cases it will run. dp_idx is a contiguous index over the selected subset.
    """
    dataset_dir = _resolve_path(dataset_dir)
    bucket_config = _get_bucket_config(bucket_config_name)

    required = ("image.jpg", "move_view.json", "prompts.json")
    case_names = sorted(
        (d for d in os.listdir(dataset_dir)
         if d.startswith("case") and os.path.isdir(os.path.join(dataset_dir, d))),
        key=_case_sort_key,
    )
    print(f"[InfWorld] Found {len(case_names)} case dirs under {dataset_dir}")

    # Subset: first n cases whose case number > begin_idx.
    case_names = [name for name in case_names if _case_sort_key(name) > begin_idx]
    if n >= 0:
        case_names = case_names[:n]
    print(f"[InfWorld] Selected {len(case_names)} cases (begin_idx={begin_idx}, n={n})")

    inputs = []
    for dp_idx, name in enumerate(case_names):
        case_dir = os.path.join(dataset_dir, name)
        if any(not os.path.exists(os.path.join(case_dir, f)) for f in required):
            print(f"[InfWorld] Skipping {name}: missing one of {required}")
            continue
        if dp_idx % dp_size != dp_rank:
            continue
        inputs.append(_load_input(dp_idx, name, case_dir, bucket_config))

    print(f"[InfWorld] rank {dp_rank}/{dp_size}(dp_rank/dp_size):total has {len(inputs)} cases to run")
    return inputs


# ============================================================================
# §6  Output paths
# ============================================================================
def prepare_output_dir(output_dir):
    """Resolve the output dir (default lives in PROJECT_ROOT's parent) and mkdir it."""
    output_dir = _resolve_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[InfWorld] Output directory: {output_dir}")
    return output_dir


def _prepare_output_file(output_dir, name):
    """Build the (extension-less) save path; save_silent_video appends .mp4.

    e.g. name='case5' -> '<output_dir>/case5_combined' -> case5_combined.mp4
    """
    return os.path.join(output_dir, f"{name}_combined")


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
                    video_buffer, latent_size, move, view, num_frames):
    """Generate one chunk: encode buffer tail -> slice actions -> sample -> decode.

    Returns a ChunkResult holding the sampled latent, the decoded pixels (CPU),
    and the condition latent / action tensors that online training reuses.
    """
    with torch.no_grad():
        current_cond = video_buffer.to(device)
        cond_latent = models.vae.encode(current_cond)

    curr_start = video_buffer.shape[2] - 1  # tail frame overlaps into the next chunk
    seg_move, seg_view = _slice_move_view(move, view, curr_start, num_frames, device)

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
    """
    num_frames = exp_config.num_frames
    video_buffer = input.input_image.clone().cpu()

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

    for chunk_idx in range(exp_config.num_chunks):
        print(f"[InfWorld] Generating chunk {chunk_idx + 1}/{exp_config.num_chunks}")
        result = _generate_chunk(
            models, device, input.prompt, exp_config.negative_prompt,
            exp_config.text_cfg_scale, video_buffer, latent_size,
            input.move, input.view, num_frames,
        )
        video_buffer = torch.cat([video_buffer, result.decoded[:, :, 1:]], dim=2)
        print(f"[InfWorld] Chunk {chunk_idx + 1} done. Total frames: {video_buffer.shape[2]}")
        _torch_gc()

        if online_train_config.open and chunk_idx < exp_config.num_chunks - 1:
            losses = online_train_on_chunk(
                models, online_train_config, result, cached_y, cached_y_mask)
            for step, loss_val in enumerate(losses):
                print(f"[OnlineTrain] {input.name} chunk {chunk_idx} "
                      f"step {step + 1}/{len(losses)} loss={loss_val:.5f}")
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
    """
    optimizer = torch.optim.AdamW(models.trainable_params, lr=online_train_config.lr)
    x_start = chunk.samples.detach()
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
    for i, inp in enumerate(inputs):
        print(f"[InfWorld] Task {inp.dp_idx} ({inp.name}): {inp.prompt[:50]}...")

        if online_train_config.open and online_train_config.reset_between_videos and i > 0:
            _restore_init_params(models.dit, models.init_params)
            print(f"[OnlineTrain] {inp.name}: restored init params")

        video = generate_one_video(models, device, exp_config, online_train_config, inp)

        save_path = _prepare_output_file(output_dir, inp.name)
        quality = 10 if exp_config.high_quality_save else 5
        save_silent_video(video.to(device), save_path, fps=exp_config.fps, quality=quality)
        print(f"[InfWorld] Saved: {save_path}.mp4")


# ============================================================================
# §10  main (orchestration only)
# ============================================================================
def main():
    _torch_gc()
    cli = parse_cli()
    exp_config, online_train_config = build_configs(cli)

    runtime = setup_runtime(exp_config.seed)

    models = load_models(
        exp_config.model_config_path, runtime.device, runtime.enable_context_parallel,
        exp_config.num_sampling_steps, exp_config.shift,
        online_train_config.open, online_train_config.use_grad_checkpoint,
        online_train_config.trainable_layers, online_train_config.reset_between_videos,
    )

    # Fill num_frames from the model config when not overridden.
    if exp_config.num_frames is None:
        exp_config.num_frames = models.model_args.validation_data.num_frames

    inputs = load_inputs(
        exp_config.dataset_dir, exp_config.bucket_config_name,
        runtime.dp_rank, runtime.dp_size,
        exp_config.begin_idx, exp_config.n,
    )
    output_dir = prepare_output_dir(exp_config.output_dir)

    run(models, runtime.device, exp_config, online_train_config, inputs, output_dir)


if __name__ == "__main__":
    main()
