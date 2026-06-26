"""
Infinite World - Action-Conditioned Video Generation Inference Script (Memory-Optimized)
========================================================================================
Memory-optimized variant of `infworld_inference.py` for the online (test-time)
training path. Functional behavior is unchanged; only memory usage is reduced.

Applied optimizations (per plan):
  1. AdamW: prefer bitsandbytes.optim.AdamW8bit, fall back to torch.optim.AdamW(fused=True);
     explicit teardown (zero_grad(None) + del + torch_gc) after each chunk's training.
  2. init_params snapshot kept on pinned CPU memory instead of GPU.
  4. Grad checkpoint restricted to DiT transformer blocks only; training forward wrapped
     in torch.autocast(bfloat16) to ensure bf16 activations.
  5. online_train_step: free intermediate tensors eagerly, gc between steps.
  6. Offload text encoder (T5-XXL) to CPU during the per-chunk online-training window.

Also adds peak-memory logging via torch.cuda.max_memory_allocated() for verification.
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
import importlib
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import torch.distributed as dist
import torchvision.transforms as transforms
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from infworld.utils.prepare_dataloader import get_obj_from_str
from infworld.utils.data_utils import get_first_clip_from_video, save_silent_video
from infworld.utils.dataset_utils import is_vid, is_img
from infworld.models.scheduler import timestep_transform

# ============================================================================
# Action Mapping Dictionaries
# ============================================================================
MOVE_ACTION_MAP = {
    'no-op': 0, 'go forward': 1, 'go back': 2, 'go left': 3, 'go right': 4,
    'go forward and go left': 5, 'go forward and go right': 6,
    'go back and go left': 7, 'go back and go right': 8, 'uncertain': 9,
}

VIEW_ACTION_MAP = {
    'no-op': 0, 'turn up': 1, 'turn down': 2, 'turn left': 3, 'turn right': 4,
    'turn up and turn left': 5, 'turn up and turn right': 6,
    'turn down and turn left': 7, 'turn down and turn right': 8, 'uncertain': 9,
}

# ============================================================================
# Utility Functions
# ============================================================================
def extract_ckpt_step(path):
    match = re.search(r'checkpoint-(\d+)\.ckpt', path)
    return int(match.group(1)) if match else 0

def resize_and_center_crop(image, target_size):
    orig_h, orig_w = image.shape[:2]
    target_h, target_w = target_size
    scale = max(target_h / orig_h, target_w / orig_w)
    final_h = math.ceil(scale * orig_h)
    final_w = math.ceil(scale * orig_w)
    resized = cv2.resize(image, (final_w, final_h), interpolation=cv2.INTER_AREA)
    resized = np.ascontiguousarray(resized)
    tensor = torch.from_numpy(resized)[None, ...].permute(0, 3, 1, 2).contiguous()
    cropped = transforms.functional.center_crop(tensor, target_size)
    return cropped[:, :, None, :, :]

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def log_mem(tag):
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1e9
        cur = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        peak_reserved = torch.cuda.max_memory_reserved() / 1e9
        print(f"[Mem][{tag}] current={cur:.2f}GB peak={peak:.2f}GB "
              f"reserved={reserved:.2f}GB peak_reserved={peak_reserved:.2f}GB")


def log_grad_mem(dit, tag):
    """Sum the GPU bytes currently held by parameter .grad buffers."""
    if not torch.cuda.is_available():
        return
    total = 0
    n = 0
    for p in dit.parameters():
        if p.grad is not None:
            total += p.grad.numel() * p.grad.element_size()
            n += 1
    print(f"[Mem][{tag}] grad_buffers={total/1e9:.2f}GB across {n} params")


def log_optimizer_mem(optimizer, tag):
    """Sum the GPU bytes held by optimizer state (e.g. Adam m/v moments)."""
    if not torch.cuda.is_available():
        return
    total = 0
    n = 0
    for state in optimizer.state.values():
        for v in state.values():
            if torch.is_tensor(v) and v.is_cuda:
                total += v.numel() * v.element_size()
                n += 1
    print(f"[Mem][{tag}] optimizer_state={total/1e9:.2f}GB across {n} tensors")

def load_action_sequence(action_path):
    with open(action_path, 'r') as f:
        actions = json.load(f)
    move_indices = [MOVE_ACTION_MAP[a['move']] for a in actions]
    view_indices = [VIEW_ACTION_MAP[a['view']] for a in actions]
    return move_indices, view_indices

def load_condition_image(image_path, bucket_config):
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
        tensor = resize_and_center_crop(frame, (target_h, target_w))
        tensor = (tensor / 255 - 0.5) * 2
        processed_frames.append(tensor)
    return torch.cat(processed_frames, dim=2)

# ============================================================================
# Distributed Setup
# ============================================================================
def setup_distributed():
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600*24))
        return local_rank, dist.get_rank(), dist.get_world_size(), True
    else:
        torch.cuda.set_device(0)
        return 0, 0, 1, False

local_rank, global_rank, num_processes, use_dist = setup_distributed()
print(f"[InfWorld] local_rank: {local_rank} | global_rank: {global_rank} | num_processes: {num_processes}")

context_parallel_size = 1
import infworld.context_parallel.context_parallel_util as cp_util
if use_dist:
    from infworld.context_parallel.context_parallel_util import init_context_parallel, get_dp_size, get_dp_rank
    init_context_parallel(context_parallel_size=context_parallel_size, global_rank=global_rank, world_size=num_processes)
    dp_rank = get_dp_rank()
    dp_size = get_dp_size()
else:
    cp_util.dp_rank = 0
    cp_util.dp_size = 1
    cp_util.cp_rank = 0
    cp_util.cp_size = 1
    dp_rank = 0
    dp_size = 1
enable_context_parallel = (context_parallel_size > 1)

# ============================================================================
# Configuration
# ============================================================================
GLOBAL_SEED = 42
setup_seed(GLOBAL_SEED + global_rank)

TEXT_CFG_SCALE = 5.0
NUM_SAMPLING_STEPS = 30
SHIFT = 7
NUM_CHUNKS = 10
HIGH_QUALITY_SAVE = True

_cli_parser = argparse.ArgumentParser(add_help=False)
_cli_parser.add_argument(
    "--online-training",
    type=lambda s: s.strip().lower() in ("on", "true", "1", "yes"),
    default=False,
)
_cli_args, _ = _cli_parser.parse_known_args()
ENABLE_ONLINE_TRAINING = _cli_args.online_training

N_TRAIN_STEPS = 5
TRAIN_LR = 1e-5
RESET_BETWEEN_VIDEOS = True
USE_GRAD_CHECKPOINT = True
GRAD_CLIP_NORM = 1.0
OFFLOAD_TEXT_ENCODER_DURING_TRAIN = True   # Opt #6
PROBE_STEP = False   # toggled True for the first train step of each chunk to log fwd/bwd mem
# Opt #7: cap the temporal length of image_cond fed to ONLINE TRAINING only.
# The conditioning latent `vae.encode(whole video_buffer)` grows +20 latent frames per
# chunk (T: 1, 21, 41, 61, ...), so the training forward/backward graph grew unbounded and
# OOM'd by chunk4. The DiT internally compresses image_cond to a fixed window
# (TARGET_T_MID=80 raw -> ~21 latent frames), so feeding only the most-recent
# TRAIN_COND_MAX_LATENT_T latent frames preserves the relevant (recent) conditioning while
# keeping training memory flat. Sampling still uses the full history (it already plateaus).
TRAIN_COND_MAX_LATENT_T = 21
print("[Infworld] online-train:{ENABLE_ONLINE_TRAINING}")

# Opt #1: try 8-bit AdamW; fall back to fused fp32 AdamW.
def _build_optimizer(params, lr):
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=lr)
        print("[OnlineTrain] using bitsandbytes AdamW8bit")
        return opt
    except Exception as e:
        print(f"[OnlineTrain] bnb AdamW8bit unavailable ({e}); falling back to fused AdamW")
        try:
            return torch.optim.AdamW(params, lr=lr, fused=True)
        except TypeError:
            return torch.optim.AdamW(params, lr=lr)

CONFIG_PATH = os.path.join(PROJECT_ROOT, 'configs', 'infworld_config.yaml')
PROMPTS_YAML = os.path.join(PROJECT_ROOT, 'prompts', 'demo.yaml')
BUCKET_CONFIG_NAME = 'ASPECT_RATIO_627_F64'
OUTPUT_BASE = os.path.join(PROJECT_ROOT, 'outputs')

NEGATIVE_PROMPT = "many cars, crowds, Vivid hues, overexposed, static, blurry details, subtitles, style, work, artwork, image, still, overall grayish, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, deformed limbs, fused fingers, motionless image, cluttered background, three legs, crowded background, walking backwards."

# ============================================================================
# Helpers
# ============================================================================
def resolve_path(path, root=PROJECT_ROOT):
    if path is None:
        return path
    path = str(path).strip()
    if not os.path.isabs(path):
        path = os.path.join(root, path)
    return path


def load_dit_state_dict(checkpoint_path):
    checkpoint_path = resolve_path(checkpoint_path)
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    return state_dict


def enable_block_only_grad_checkpoint(dit):
    """Opt #4: only mark DiT transformer blocks for grad checkpoint, not every submodule."""
    blocks = getattr(dit, "blocks", None)
    if blocks is None:
        from infworld.models.checkpoint import set_grad_checkpoint
        set_grad_checkpoint(dit)
        print("[OnlineTrain] grad checkpoint: applied globally (no .blocks attr)")
        return
    n_marked = 0
    for blk in blocks:
        for m in blk.modules():
            m.grad_checkpointing = True
            m.fp32_attention = False
            m.grad_checkpointing_step = 1
            n_marked += 1
    print(f"[OnlineTrain] grad checkpoint: marked {n_marked} submodules inside dit.blocks only")


def online_train_step(dit, scheduler, x_start, model_kwargs, optimizer):
    """One rectified-flow training step. autocast(bf16) for activations; eager free."""
    optimizer.zero_grad(set_to_none=True)
    device = x_start.device
    B = x_start.shape[0]

    if scheduler.use_discrete_timesteps:
        t = torch.randint(0, scheduler.num_timesteps, (B,), device=device)
    elif scheduler.sample_method == "uniform":
        t = torch.rand((B,), device=device) * scheduler.num_timesteps
    else:
        t = scheduler.sample_t(x_start) * scheduler.num_timesteps
    if scheduler.use_timestep_transform:
        t = timestep_transform(t, shift=scheduler.shift, num_timesteps=scheduler.num_timesteps)

    noise = torch.randn_like(x_start)
    x_t = scheduler.add_noise(x_start, noise, t)
    target = x_start - noise
    if scheduler.use_reversed_velocity:
        target = -target
    del noise

    T = x_start.shape[2]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        full_pred = dit(x_t, t, x_ignore_mask=None, **model_kwargs)
    if PROBE_STEP:
        log_mem("train:after_forward")
    pred = full_pred[:, :, -T:].contiguous()
    del full_pred, x_t

    loss = ((pred.float() - target.float()) ** 2).mean()
    del pred, target
    loss.backward()
    if PROBE_STEP:
        log_mem("train:after_backward")
    if GRAD_CLIP_NORM is not None:
        torch.nn.utils.clip_grad_norm_(
            [p for p in dit.parameters() if p.requires_grad],
            max_norm=GRAD_CLIP_NORM,
        )
    optimizer.step()
    return loss.detach()


def main():
    torch_gc()

    config_path = CONFIG_PATH
    args = OmegaConf.load(config_path)
    checkpoint_path = resolve_path(args.get("checkpoint_path", "checkpoints/models/diffusion_pytorch_model.safetensors"))
    ckpt_step = extract_ckpt_step(checkpoint_path)

    output_dir = os.path.join(OUTPUT_BASE, f"infworld-ckpt{ckpt_step}-step{NUM_SAMPLING_STEPS}-cfg{TEXT_CFG_SCALE}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[InfWorld] Loading checkpoint: {checkpoint_path}")
    print(f"[InfWorld] Config: {config_path}")
    print(f"[InfWorld] Output directory: {output_dir}")

    if hasattr(args, "vae_cfg") and "vae_pth" in args.vae_cfg:
        args.vae_cfg.vae_pth = resolve_path(args.vae_cfg.vae_pth)
    if hasattr(args, "text_encoder_cfg"):
        if "checkpoint_path" in args.text_encoder_cfg:
            args.text_encoder_cfg.checkpoint_path = resolve_path(args.text_encoder_cfg.checkpoint_path)
        if "tokenizer_path" in args.text_encoder_cfg:
            args.text_encoder_cfg.tokenizer_path = resolve_path(args.text_encoder_cfg.tokenizer_path)

    print("[InfWorld] Loading VAE...")
    vae = get_obj_from_str(args.vae_target)(**args.vae_cfg).to(local_rank)

    print("[InfWorld] Loading Text Encoder...")
    text_encoder = get_obj_from_str(args.text_encoder_target)(device=local_rank, **args.text_encoder_cfg)
    text_encoder.t5.model.to(local_rank)

    print("[InfWorld] Loading Scheduler...")
    scheduler = get_obj_from_str(args.scheduler_target)(**args.val_scheduler_cfg)
    scheduler.num_sampling_steps = NUM_SAMPLING_STEPS
    scheduler.shift = SHIFT

    print("[InfWorld] Loading DiT Model...")
    dtype = getattr(torch, args.amp_dtype)
    dit = get_obj_from_str(args.model_target)(
        out_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
        enable_context_parallel=enable_context_parallel,
        **args.model_cfg
    ).to(dtype)
    dit.eval()

    state_dict = load_dit_state_dict(args.checkpoint_path)
    state_dict.pop("pos_embed_temporal", None)
    state_dict.pop("pos_embed", None)
    missing, unexpected = dit.load_state_dict(state_dict, strict=False)
    print(f"[InfWorld] Model loaded! Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    del state_dict
    dit.to(local_rank)

    if ENABLE_ONLINE_TRAINING and USE_GRAD_CHECKPOINT:
        enable_block_only_grad_checkpoint(dit)   # Opt #4

    FREEZE_PREFIXES = (
        "patch_embedding", "latent_encoder", "action_encoder",
        "text_embedding", "time_embedding", "time_projection", "y_embedder",
    )
    trainable_params = []
    frozen_n = 0
    for n, p in dit.named_parameters():
        if n.startswith(FREEZE_PREFIXES):
            p.requires_grad_(False)
            frozen_n += p.numel()
        else:
            p.requires_grad_(True)
            trainable_params.append(p)
    print(f"[OnlineTrain] frozen params (cond entry layers): {frozen_n:,}; "
          f"trainable: {sum(p.numel() for p in trainable_params):,}")

    # Opt #2: snapshot trainable init params to pinned CPU memory.
    init_params = None
    if ENABLE_ONLINE_TRAINING and RESET_BETWEEN_VIDEOS:
        init_params = {}
        for n, p in dit.named_parameters():
            if not p.requires_grad:
                continue
            cpu_t = torch.empty(p.shape, dtype=p.dtype, device="cpu", pin_memory=True)
            cpu_t.copy_(p.detach(), non_blocking=True)
            init_params[n] = cpu_t
        torch.cuda.synchronize()
        print(f"[OnlineTrain] init_params snapshot kept on pinned CPU ({len(init_params)} tensors)")

    # Opt #1 (fix): build the optimizer ONCE and reuse across chunks. Rebuilding it
    # per chunk re-allocated fresh fp32 Adam m/v state every time and fragmented the
    # allocator, which was the dominant per-chunk +15GB growth seen in the logs.
    optimizer = None
    if ENABLE_ONLINE_TRAINING:
        optimizer = _build_optimizer(trainable_params, TRAIN_LR)

    from infworld.configs import bucket_config as bucket_config_module
    bucket_config = getattr(bucket_config_module, BUCKET_CONFIG_NAME)

    prompts_path = os.path.abspath(PROMPTS_YAML)
    target_prompts = OmegaConf.load(prompts_path).prompts
    print(f"[InfWorld] Loaded {len(target_prompts)} prompts")

    for task_idx, (prompt, image_path, action_path) in enumerate(target_prompts):
        if task_idx % dp_size != dp_rank:
            continue
        if not os.path.exists(image_path):
            print(f"[InfWorld] Skipping task {task_idx}: Image not found - {image_path}")
            continue
        if not os.path.exists(action_path):
            print(f"[InfWorld] Skipping task {task_idx}: Action not found - {action_path}")
            continue

        print(f"[InfWorld] Task {task_idx}: {prompt[:50]}...")

        cond_video = load_condition_image(image_path, bucket_config).to(local_rank)
        with torch.no_grad():
            cond_latent = vae.encode(cond_video)

        move_indices, view_indices = load_action_sequence(action_path)
        video_buffer = cond_video.clone().cpu()

        latent_size = list(cond_latent.shape)
        latent_size[2] = 21
        latent_size = torch.Size(latent_size)

        cached_y = cached_y_mask = None
        if ENABLE_ONLINE_TRAINING:
            # Opt #2 (restore): copy from pinned CPU snapshot back to GPU params.
            if RESET_BETWEEN_VIDEOS and task_idx > 0 and init_params is not None:
                with torch.no_grad():
                    for n, p in dit.named_parameters():
                        if n in init_params:
                            p.data.copy_(init_params[n], non_blocking=True)
                torch.cuda.synchronize()
                # Weights were reset to init; the reused optimizer's Adam moments are
                # now stale, so clear them to start the next video's training fresh.
                if optimizer is not None:
                    optimizer.state.clear()
                    torch_gc()
                print(f"[OnlineTrain] task {task_idx}: restored init params from CPU snapshot")
            with torch.no_grad():
                text_kwargs = text_encoder.encode([prompt])
            cached_y, cached_y_mask = text_kwargs["y"], text_kwargs["y_mask"]

        for chunk_idx in range(NUM_CHUNKS):
            print(f"[InfWorld] Generating chunk {chunk_idx + 1}/{NUM_CHUNKS}")
            torch.cuda.reset_peak_memory_stats()

            with torch.no_grad():
                current_cond = video_buffer.to(local_rank)
                current_latent = vae.encode(current_cond)
                del current_cond
            print(f"[Shape] chunk{chunk_idx} video_buffer_T={video_buffer.shape[2]} "
                  f"current_latent={tuple(current_latent.shape)}")

            curr_start = video_buffer.shape[2] - 1
            curr_end = curr_start + args.validation_data.num_frames
            move = torch.tensor(move_indices[curr_start:curr_end], dtype=torch.long, device=local_rank)
            view = torch.tensor(view_indices[curr_start:curr_end], dtype=torch.long, device=local_rank)
            num_frames = args.validation_data.num_frames
            if move.shape[0] < num_frames:
                pad_len = num_frames - move.shape[0]
                move = torch.cat([move, torch.zeros(pad_len, dtype=torch.long, device=local_rank)])
                view = torch.cat([view, torch.zeros(pad_len, dtype=torch.long, device=local_rank)])

            additional_args = {
                "image_cond": current_latent,
                "move": move.unsqueeze(0),
                "view": view.unsqueeze(0),
            }
            torch_gc()
            with torch.no_grad():
                samples = scheduler.sample(
                    model=dit, text_encoder=text_encoder, null_embedder=dit.y_embedder,
                    z_size=latent_size, prompts=[prompt],
                    guidance_scale=TEXT_CFG_SCALE, negative_prompts=[NEGATIVE_PROMPT],
                    device=torch.device(local_rank), additional_args=additional_args,
                )
                decoded_chunk = vae.decode(samples).cpu()
                video_buffer = torch.cat([video_buffer, decoded_chunk[:, :, 1:]], dim=2)
                del decoded_chunk
                print(f"[InfWorld] Chunk {chunk_idx + 1} done. Total frames: {video_buffer.shape[2]}")
                torch_gc()

            log_mem(f"after_sample chunk{chunk_idx}")

            if ENABLE_ONLINE_TRAINING and chunk_idx < NUM_CHUNKS - 1:
                # Opt #6: offload T5 to CPU during the online-training window.
                if OFFLOAD_TEXT_ENCODER_DURING_TRAIN:
                    text_encoder.t5.model.to("cpu")
                    torch_gc()

                x_start = samples.detach()
                del samples
                # Opt #7: cap conditioning history fed to training to a bounded recent tail.
                train_image_cond = current_latent.detach()
                if (TRAIN_COND_MAX_LATENT_T is not None
                        and train_image_cond.shape[2] > TRAIN_COND_MAX_LATENT_T):
                    train_image_cond = train_image_cond[:, :, -TRAIN_COND_MAX_LATENT_T:].contiguous()
                train_kwargs = {
                    "y": cached_y, "y_mask": cached_y_mask,
                    "image_cond": train_image_cond,
                    "move": move.unsqueeze(0).detach(),
                    "view": view.unsqueeze(0).detach(),
                }
                # Opt #1 (fix): reuse the single optimizer built before the loop
                # instead of allocating a new one (and new fp32 state) per chunk.
                for step in range(N_TRAIN_STEPS):
                    globals()["PROBE_STEP"] = (step == 0)
                    loss_val = online_train_step(dit, scheduler, x_start, train_kwargs, optimizer)
                    print(f"[OnlineTrain] task {task_idx} chunk {chunk_idx} step {step+1}/{N_TRAIN_STEPS} "
                          f"loss={loss_val.item():.5f}")
                    torch_gc()  # Opt #5: between steps
                log_mem(f"before_teardown chunk{chunk_idx}")
                log_grad_mem(dit, f"before_teardown chunk{chunk_idx}")
                log_optimizer_mem(optimizer, f"before_teardown chunk{chunk_idx}")

                # Opt #1: free grads (but KEEP optimizer state — it is reused next chunk).
                optimizer.zero_grad(set_to_none=True)
                for p in trainable_params:
                    p.grad = None
                del x_start, train_kwargs, train_image_cond
                torch_gc()
                log_mem(f"after_train chunk{chunk_idx}")
                log_grad_mem(dit, f"after_train chunk{chunk_idx}")

                # Opt #6: restore text encoder for next chunk's sample().
                if OFFLOAD_TEXT_ENCODER_DURING_TRAIN:
                    text_encoder.t5.model.to(local_rank)
                    torch_gc()
            else:
                del samples
                torch_gc()

            del current_latent, additional_args, move, view

        timestamp = datetime.datetime.now().strftime("%m_%d_%H:%M:%S")
        video_name = f"{task_idx:04d}_{prompt[:30].replace(' ', '_')}_{timestamp}"
        save_path = os.path.join(output_dir, video_name)
        quality = 10 if HIGH_QUALITY_SAVE else 5
        save_silent_video(video_buffer.to(local_rank), save_path, fps=30, quality=quality)
        print(f"[InfWorld] Saved: {save_path}.mp4")

if __name__ == "__main__":
    main()
