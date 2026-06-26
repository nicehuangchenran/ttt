"""
Infinite World - Action-Conditioned Video Generation Inference Script
======================================================================
A standalone inference script for generating long videos with action control.
"""

import sys
import os
import cv2
import glob
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
# Utility Functions
# ============================================================================
def extract_ckpt_step(path):
    """Extract checkpoint step number from path."""
    match = re.search(r'checkpoint-(\d+)\.ckpt', path)
    return int(match.group(1)) if match else 0

def resize_and_center_crop(image, target_size):
    """Resize image and center crop to target size."""
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

def setup_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def torch_gc():
    """Clear GPU memory cache."""
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def load_action_sequence(action_path):
    """Load action sequence from JSON file."""
    with open(action_path, 'r') as f:
        actions = json.load(f)
    
    move_indices = [MOVE_ACTION_MAP[a['move']] for a in actions]
    view_indices = [VIEW_ACTION_MAP[a['view']] for a in actions]
    return move_indices, view_indices

def get_input(dataset_dir=None, n=1):
    """从 dataset/caseXXX/ 构建 (prompt, image_path, action_path) 三元组。

    每个 case 目录包含：
      - image.jpg       : 条件图像
      - move_view.json  : 逐帧 [{"move","view"}, ...] 动作序列
      - prompts.json    : 对象，其 "prompt" 字段为文本提示词

    返回按 case 编号数值升序排列的三元组列表，
    与主循环解包 (prompt, image_path, action_path) 一致。

    n: 只取前 n 个 case（默认 1）；n=0 表示使用全部。
    """
    if dataset_dir is None:
        dataset_dir = DATASET_DIR

    def _case_id(path):
        m = re.search(r'case(\d+)', os.path.basename(path))
        return int(m.group(1)) if m else 0

    case_dirs = sorted(
        [d for d in glob.glob(os.path.join(dataset_dir, 'case*')) if os.path.isdir(d)],
        key=_case_id,
    )
    if n != 0:
        case_dirs = case_dirs[:n]

    triples = []
    for case_dir in case_dirs:
        image_path = os.path.join(case_dir, 'image.jpg')
        action_path = os.path.join(case_dir, 'move_view.json')
        prompts_json = os.path.join(case_dir, 'prompts.json')
        with open(prompts_json, 'r') as f:
            prompt = json.load(f)['prompt']
        triples.append((prompt, image_path, action_path))
    return triples

def load_condition_image(image_path, bucket_config):
    """Load and preprocess condition image."""
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
        tensor = (tensor / 255 - 0.5) * 2  # Normalize to [-1, 1]
        processed_frames.append(tensor)
    
    return torch.cat(processed_frames, dim=2)

# ============================================================================
# Distributed Setup (support single-GPU without torchrun to avoid port conflict)
# ============================================================================
def setup_distributed():
    """Setup distributed or single-GPU mode."""
    if 'RANK' in os.environ:
        # Launched by torchrun or similar
        rank = int(os.environ['RANK'])
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600*24))
        global_rank = dist.get_rank()
        num_processes = dist.get_world_size()
        return local_rank, global_rank, num_processes, True  # use_cp_init=True
    else:
        # Single process (no torchrun) - avoid port conflict, no dist init
        local_rank = 0
        global_rank = 0
        num_processes = 1
        torch.cuda.set_device(local_rank)
        return local_rank, global_rank, num_processes, False  # use_cp_init=False

local_rank, global_rank, num_processes, use_dist = setup_distributed()
print(f"[InfWorld] local_rank: {local_rank} | global_rank: {global_rank} | world_size: {num_processes}")

# Context parallel setup
context_parallel_size = 1
import infworld.context_parallel.context_parallel_util as cp_util
if use_dist:
    from infworld.context_parallel.context_parallel_util import init_context_parallel, get_dp_size, get_dp_rank
    init_context_parallel(context_parallel_size=context_parallel_size, global_rank=global_rank, world_size=num_processes)
    dp_rank = get_dp_rank()
    dp_size = get_dp_size()
else:
    # Single process: set globals so get_dp_rank/get_dp_size work without dist
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
# Inference settings
GLOBAL_SEED = 42
setup_seed(GLOBAL_SEED + global_rank)

TEXT_CFG_SCALE = 5.0
NUM_SAMPLING_STEPS = 30
SHIFT = 7  # PX256: 3, PX627: 7, PX960: 11
NUM_CHUNKS = 3  # Number of video chunks to generate
HIGH_QUALITY_SAVE = True

# Online (test-time) training between chunks (overridable via --online-training/--no-online-training)
_cli_parser = argparse.ArgumentParser(add_help=False)
_cli_parser.add_argument(
    "--online-training",
    type=lambda s: s.strip().lower() in ("on", "true", "1", "yes"),
    default=False,
    help="Enable online (test-time) training between chunks: on/off",
)
_cli_args, _ = _cli_parser.parse_known_args()
ENABLE_ONLINE_TRAINING = _cli_args.online_training

N_TRAIN_STEPS = 5
TRAIN_LR = 1e-5
RESET_BETWEEN_VIDEOS = True   # dynamic switch: restore pretrained weights between prompts
USE_GRAD_CHECKPOINT = True    # use_reentrant=False set in infworld/models/checkpoint.py
GRAD_CLIP_NORM = 1.0

# Paths - checkpoint_path is read from config (configs/infworld_config.yaml)
# Model config - use standalone config
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'configs', 'infworld_config.yaml')

PROMPTS_YAML = os.path.join(PROJECT_ROOT, 'prompts', 'demo.yaml')
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')
NUM_CASES = 1  # 只取前 N 个 case；0 表示全部
BUCKET_CONFIG_NAME = 'ASPECT_RATIO_627_F64'

# Output directory
OUTPUT_BASE = os.path.join(PROJECT_ROOT, 'outputs')

# Negative prompt for generation quality
NEGATIVE_PROMPT = "many cars, crowds, Vivid hues, overexposed, static, blurry details, subtitles, style, work, artwork, image, still, overall grayish, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, deformed limbs, fused fingers, motionless image, cluttered background, three legs, crowded background, walking backwards."

# ============================================================================
# Main Inference Loop
# ============================================================================
def resolve_path(path, root=PROJECT_ROOT):
    """Resolve path: if relative, join with project root."""
    if path is None:
        return path
    path = str(path).strip()
    if not os.path.isabs(path):
        path = os.path.join(root, path)
    return path


def load_dit_state_dict(checkpoint_path):
    """Load DiT state dict from .ckpt (torch) or .safetensors."""
    checkpoint_path = resolve_path(checkpoint_path)
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    return state_dict


def online_train_step(dit, scheduler, x_start, model_kwargs, optimizer):
    """One rectified-flow training step on a single (x_start, model_kwargs) pair.

    Mirrors RFlowScheduler.training_losses but calls the DiT with x_ignore_mask=None
    so we avoid the post-temporal-compression mask shape requirement.
    
    x_start:刚刚由 scheduler.sample(...) 采样出来、还没解码回像素的那段视频潜变量 (latent),是视频样本不是采样的噪声
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

    pred = dit(x_t, t, x_ignore_mask=None, **model_kwargs) # x_ignore_mask它是一个布尔/0-1 掩码，形状 [B, T, H, W]，标记潜变量里哪些时空位置应该被忽略——既不参与 loss 计算，也不让对应的 token 影响注意力/输出,主要用于batch 内长度/分辨率对齐的 padding：当不同样本的视频长度或时空形状不一致、需要 pad 到统一形状时，用 x_ignore_mask=1 标出 pad 区域，让模型和 loss 都跳过它们
    pred = pred[:, :, -x_start.shape[2]:] # DiT 的输入在时间维上是 [image_cond | x_t] 拼接的，所以输出 pred 时间长度 = T_cond + T_x，这里取后面T_x帧对应的输出作为 loss 计算对象

    loss = ((pred.float() - target.float()) ** 2).mean()
    loss.backward()
    if GRAD_CLIP_NORM is not None:
        torch.nn.utils.clip_grad_norm_(dit.parameters(), max_norm=GRAD_CLIP_NORM)
    optimizer.step()
    return loss.detach() # 返回一个和 loss 数值相同、但脱离计算图的新张量，没有反向传播的梯度，（requires_grad=False，没有 grad_fn）。

def main():
    torch_gc()
    
    config_path = CONFIG_PATH
    args = OmegaConf.load(config_path)
    checkpoint_path = resolve_path(args.get("checkpoint_path", "checkpoints/models/diffusion_pytorch_model.safetensors"))
    
    ckpt_step = extract_ckpt_step(checkpoint_path)
    
    # Create output directory
    output_dir = os.path.join(OUTPUT_BASE, f"infworld-ckpt{ckpt_step}-step{NUM_SAMPLING_STEPS}-cfg{TEXT_CFG_SCALE}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[InfWorld] Loading checkpoint: {checkpoint_path}")
    print(f"[InfWorld] Config: {config_path}")
    print(f"[InfWorld] Output directory: {output_dir}")
    
    # Resolve relative paths in config for models that load from disk
    if hasattr(args, "vae_cfg") and "vae_pth" in args.vae_cfg:
        args.vae_cfg.vae_pth = resolve_path(args.vae_cfg.vae_pth)
    if hasattr(args, "text_encoder_cfg"):
        if "checkpoint_path" in args.text_encoder_cfg:
            args.text_encoder_cfg.checkpoint_path = resolve_path(args.text_encoder_cfg.checkpoint_path)
        if "tokenizer_path" in args.text_encoder_cfg:
            args.text_encoder_cfg.tokenizer_path = resolve_path(args.text_encoder_cfg.tokenizer_path)
    
    # Initialize models
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
    
    # Load DiT checkpoint (from config)
    state_dict = load_dit_state_dict(args.checkpoint_path)
    
    # Remove position embeddings (will be recomputed)
    state_dict.pop("pos_embed_temporal", None)
    state_dict.pop("pos_embed", None)
    
    missing, unexpected = dit.load_state_dict(state_dict, strict=False)
    print(f"[InfWorld] Model loaded! Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
    dit.to(local_rank)

    if ENABLE_ONLINE_TRAINING and USE_GRAD_CHECKPOINT:
        set_grad_checkpoint(dit)

    # A: freeze condition-input layers so online training cannot drift the model's
    # interface to history (image_cond), actions, text, or timestep — these are what
    # carry chunk-to-chunk continuity.
    FREEZE_PREFIXES = (
        "patch_embedding",   # raw latent -> token conv
        "latent_encoder",    # image_cond (history) temporal encoder
        "action_encoder",    # move / view embeddings
        "text_embedding",    # T5 -> token projection
        "time_embedding",    # timestep MLP
        "time_projection",   # timestep -> per-block modulation
        "y_embedder",        # null caption embedding
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

    init_params = None
    if ENABLE_ONLINE_TRAINING and RESET_BETWEEN_VIDEOS:
        # Only snapshot trainable params — frozen ones can never drift.
        init_params = {n: p.detach().clone()
                       for n, p in dit.named_parameters() if p.requires_grad}

    # Load bucket config
    from infworld.configs import bucket_config as bucket_config_module
    bucket_config = getattr(bucket_config_module, BUCKET_CONFIG_NAME)
    
    # Load inputs from dataset/
    target_prompts = get_input(n=NUM_CASES)
    print(f"[InfWorld] Loaded {len(target_prompts)} cases from dataset")
    
    # Process each prompt
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
        
        # Load condition image
        cond_video = load_condition_image(image_path, bucket_config).to(local_rank)
        
        with torch.no_grad():
            cond_latent = vae.encode(cond_video)
        
        # Load action sequence
        move_indices, view_indices = load_action_sequence(action_path)
        
        # Initialize video buffer
        video_buffer = cond_video.clone().cpu()
        
        # 定义Latent size，size是 [B, C, T, H, W]，其中T是时间维度长度，改为21（1帧条件 + 20帧生成）以适配 DiT 的输出要求
        latent_size = list(cond_latent.shape)
        latent_size[2] = 21  # Output frames per chunk
        latent_size = torch.Size(latent_size)

        # Online training setup for this video (optimizer is created per-chunk; see C below)
        cached_y = cached_y_mask = None
        if ENABLE_ONLINE_TRAINING:
            if RESET_BETWEEN_VIDEOS and task_idx > 0 and init_params is not None:
                with torch.no_grad():
                    for n, p in dit.named_parameters():
                        if n in init_params:
                            p.data.copy_(init_params[n])
                print(f"[OnlineTrain] task {task_idx}: restored init params")
            with torch.no_grad():
                text_kwargs = text_encoder.encode([prompt])
            cached_y, cached_y_mask = text_kwargs["y"], text_kwargs["y_mask"]

        # Generate video chunks
        for chunk_idx in range(NUM_CHUNKS):
            print(f"[InfWorld] Generating chunk {chunk_idx + 1}/{NUM_CHUNKS}")
            
            with torch.no_grad():
                current_cond = video_buffer.to(local_rank) # 不是in-place操作，video_buffer 仍然在 CPU 上，current_cond 是它在 GPU 上的一个副本
                current_latent = vae.encode(current_cond)
            
            # 取出move和view序列，如果不够长就pad
            curr_start = video_buffer.shape[2] - 1  #从curr_start[int]帧开始，到curr_end这一帧结束
            curr_end = curr_start + args.validation_data.num_frames
            
            move = torch.tensor(move_indices[curr_start:curr_end], dtype=torch.long, device=local_rank)
            view = torch.tensor(view_indices[curr_start:curr_end], dtype=torch.long, device=local_rank)
            # Pad if needed
            num_frames = args.validation_data.num_frames
            if move.shape[0] < num_frames:
                pad_len = num_frames - move.shape[0]
                move = torch.cat([move, torch.zeros(pad_len, dtype=torch.long, device=local_rank)])
                view = torch.cat([view, torch.zeros(pad_len, dtype=torch.long, device=local_rank)])
            
            # 进行sample，生成此chunk视频
            additional_args = {
                "image_cond": current_latent,
                "move": move.unsqueeze(0),
                "view": view.unsqueeze(0),
            }
            torch_gc() # 自定义的，torch.cuda.empty_cache() 和 torch.cuda.ipc_collect()，用来回收被删除的张量的 CUDA 缓存和跨进程通信的缓存
            with torch.no_grad():
                samples = scheduler.sample(
                    model=dit,
                    text_encoder=text_encoder,
                    null_embedder=dit.y_embedder,
                    z_size=latent_size,
                    prompts=[prompt],
                    guidance_scale=TEXT_CFG_SCALE,
                    negative_prompts=[NEGATIVE_PROMPT],
                    device=torch.device(local_rank),
                    additional_args=additional_args,
                )
                    
                decoded_chunk = vae.decode(samples).cpu()
                video_buffer = torch.cat([video_buffer, decoded_chunk[:, :, 1:]], dim=2)

                print(f"[InfWorld] Chunk {chunk_idx + 1} done. Total frames: {video_buffer.shape[2]}")
                torch_gc()

            # Online training: fine-tune on the chunk we just generated before producing the next one.
            # C: optimizer is rebuilt per-chunk so Adam moments do NOT accumulate across chunks.
            if ENABLE_ONLINE_TRAINING and chunk_idx < NUM_CHUNKS - 1:
                optimizer = torch.optim.AdamW(trainable_params, lr=TRAIN_LR)
                x_start = samples.detach()
                train_kwargs = {
                    "y": cached_y,
                    "y_mask": cached_y_mask,
                    "image_cond": current_latent.detach(),
                    "move": move.unsqueeze(0).detach(),
                    "view": view.unsqueeze(0).detach(),
                }
                for step in range(N_TRAIN_STEPS):
                    loss_val = online_train_step(dit, scheduler, x_start, train_kwargs, optimizer)
                    print(f"[OnlineTrain] task {task_idx} chunk {chunk_idx} step {step+1}/{N_TRAIN_STEPS} "
                          f"loss={loss_val.item():.5f}")
                del x_start, train_kwargs
                torch_gc()

        # Save final video (append current timestamp mm_dd_HH:MM:SS)
        timestamp = datetime.datetime.now().strftime("%m_%d_%H:%M:%S")
        video_name = f"{task_idx:04d}_{prompt[:30].replace(' ', '_')}_{timestamp}"
        save_path = os.path.join(output_dir, video_name)
        
        quality = 10 if HIGH_QUALITY_SAVE else 5
        save_silent_video(video_buffer.to(local_rank), save_path, fps=30, quality=quality)
        print(f"[InfWorld] Saved: {save_path}.mp4")

if __name__ == "__main__":
    main()
