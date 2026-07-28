"""
Infinite World - Dataset Preprocessing Script
==============================================
将原始视频数据集离线编码为训练可直接加载的 latent/embedding 张量。

输入目录结构（来自 dataset_dir）:
  case{n}/
    ├── video.mp4
    ├── move_view.json
    ├── prompts.json
    └── image.jpg

输出目录结构（写入 output_dir）:
  case{n}/
    ├── latent_full.pt        [C, T_lat, h, w] # vae.encode 整个视频形成 latent, 作为 cond 输入,wanvae 是因果的,
    ├── target_latent.pt      [K, C, 21, h, w] # 从视频中单独切片 81 帧,vae encode生成的 latent, 作为生成目标
    ├── text_emb.pt           {"y": [1,1,L,D], "y_mask": [1,L]}
    ├── actions.pt            {"move": Long[T_pix], "view": Long[T_pix]}
    └── meta.json             {"name", "num_chunks", "num_frames", "height", "width", "prompt",
                               "location", "scene", "crowdDensity", "weather", "timeOfDay"}
会将 mp4裁剪为 bucket config中的宽高  --bucket-config ASPECT_RATIO_627_F64

使用示例：
  # 8 GPUs 并行处理
  torchrun --nproc_per_node=8 prepare/sekai_game_walking/2_preprocess_dataset.py \
     --bucket-config ASPECT_RATIO_256 \
     --dataset-dir dataset/sekai-game-walking-854_480_30fps \
     --output-dir preprocessed/sekai-game-walking-256px

nohup torchrun --nproc_per_node=8 prepare/sekai_game_walking/2_preprocess_dataset.py \
    --bucket-config ASPECT_RATIO_256 \
    --dataset-dir dataset/sekai-game-walking-352_192_30fps \
    --output-dir preprocessed/sekai-game-walking-256px \
    > preprocessed/sekai-game-walking-256px/preprocess_main.log 2>&1 &

     
  # 只处理前 N 个 case（调试用）
  python prepare/sekai_game_walking/2_preprocess_dataset.py --max-cases 5

  # 验证因果性假设（VAE 前缀一致性）
  python prepare/sekai_game_walking/2_preprocess_dataset.py --verify
"""

import sys
import os
import json
import math
import cv2
import torch
import torch.distributed as dist
import numpy as np
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm
import re
import logging
import time
from datetime import datetime

# 添加项目根目录到 path
# __file__ -> prepare/sekai_game_walking/2_preprocess_dataset.py
# 需要上溯到 infworld/ 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from omegaconf import OmegaConf
from decord import VideoReader
from infworld.utils.prepare_dataloader import get_obj_from_str
import torchvision.transforms as transforms

# decord 每个 VideoReader 的解码线程数。多进程并行时设小值避免 CPU 争抢。
DECORD_NUM_THREADS = int(os.environ.get("DECORD_NUM_THREADS", "2"))


# ============================================================================
# §P1 配置
# ============================================================================
@dataclass
class PreprocessConfig:
    dataset_dir: str = "dataset/sekai-game-walking-854_480_30fps"
    output_dir: str = "preprocessed/sekai-game-walking-854_480_30fps"
    model_config_path: str = "configs/infworld_config.yaml"
    bucket_config_name: str = "ASPECT_RATIO_256_F64"  # 256 像素训练测试
    frames_per_chunk: int = 81
    chunk_stride: int = 80  # 1 帧重叠
    save_dtype: str = "bfloat16"
    skip_existing: bool = True
    verify_prefix: bool = False  # 验证因果性假设
    max_cases: int = -1  # -1 = 全部；调试时可设小


# ============================================================================
# Action mapping (与 main.py 保持一致)
# ============================================================================
MOVE_ACTION_MAP = {
    'no-op': 0, 'go forward': 1, 'go back': 2, 'go left': 3, 'go right': 4,
    'go forward and go left': 5, 'go forward and go right': 6,
    'go back and go left': 7, 'go back and go right': 8, 'uncertain': 9
}

VIEW_ACTION_MAP = {
    'no-op': 0, 'turn up': 1, 'turn down': 2, 'turn left': 3, 'turn right': 4,
    'turn up and turn left': 5, 'turn up and turn right': 6,
    'turn down and turn left': 7, 'turn down and turn right': 8, 'uncertain': 9
}


# ============================================================================
# §P2 工具函数
# ============================================================================
def setup_distributed():
    """初始化分布式环境，返回 (rank, world_size, local_rank)"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank
    else:
        # 单机模式
        return 0, 1, 0


def setup_logging(output_dir: str, rank: int) -> logging.Logger:
    """配置日志：每个 rank 写独立文件，rank 0 额外输出到终端。

    日志文件写入 {output_dir}/logs/rank{rank}.log
    返回配置好的 logger。
    """
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(f"preprocess_rank{rank}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # 重复调用时清理旧 handler，避免日志重复
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt=f"%(asctime)s [rank{rank}] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 每个 rank 独立的日志文件
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"rank{rank}.log"), mode="a")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # 仅 rank 0 输出到终端，避免多进程刷屏
    if rank == 0:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

    return logger


def _resolve_path(path, root=PROJECT_ROOT):
    """解析路径：相对路径拼接项目根目录。"""
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(root, path.strip())


def _get_bucket_config(name: str) -> dict:
    """从 infworld/configs/bucket_config.py 加载宽高比配置。"""
    from infworld.configs import bucket_config as bucket_config_module
    return getattr(bucket_config_module, name)


def _case_sort_key(name: str) -> int:
    """提取 case 编号用于排序（case5 < case12）。"""
    match = re.search(r'(\d+)', name)
    return int(match.group(1)) if match else 0


def _load_model_config(config_path: str):
    """加载模型 YAML 并解析相对路径。"""
    args = OmegaConf.load(config_path)
    if hasattr(args, "vae_cfg") and "vae_pth" in args.vae_cfg:
        args.vae_cfg.vae_pth = _resolve_path(args.vae_cfg.vae_pth)
    if hasattr(args, "text_encoder_cfg"):
        if "checkpoint_path" in args.text_encoder_cfg:
            args.text_encoder_cfg.checkpoint_path = _resolve_path(
                args.text_encoder_cfg.checkpoint_path)
        if "tokenizer_path" in args.text_encoder_cfg:
            args.text_encoder_cfg.tokenizer_path = _resolve_path(
                args.text_encoder_cfg.tokenizer_path)
    return args


# ============================================================================
# §P3 视频加载与预处理
# ============================================================================
def load_video_frames(video_path: str, bucket_config: dict) -> torch.Tensor:
    """加载视频全部帧 → resize+center-crop 到 bucket → 归一化 [-1,1]。

    使用 decord 批量顺序解码（避免 cv2 逐帧 seek 的 O(N²) 开销）。

    返回: [1, 3, T_pix, H, W]（CPU），T_pix 截断到满足 T ≡ 1 (mod 4) 的最大值。
    """
    # decord 批量顺序解码：一次性拿到 [T, H, W, 3] uint8 numpy
    vr = VideoReader(video_path, num_threads=DECORD_NUM_THREADS)
    if len(vr) == 0:
        raise ValueError(f"Failed to load video: {video_path}")
    frames_np = vr.get_batch(range(len(vr))).asnumpy()  # [T, H, W, 3] RGB uint8

    # 选择最接近的宽高比 bucket
    h, w = frames_np.shape[1], frames_np.shape[2]
    ratio = h / w
    closest_bucket = sorted(bucket_config.keys(),
                          key=lambda x: abs(float(x) - ratio))[0]
    target_h, target_w = bucket_config[closest_bucket][0]

    # Resize + center crop（整段一次性搬到 tensor，减少 Python 循环开销）
    scale = max(target_h / h, target_w / w)
    new_h = math.ceil(scale * h)
    new_w = math.ceil(scale * w)

    # [T, H, W, 3] uint8 → [T, 3, H, W] float tensor
    video = torch.from_numpy(frames_np).permute(0, 3, 1, 2).contiguous().float()
    # 批量 resize（antialias 近似 cv2 INTER_AREA 的降采样质量）
    video = transforms.functional.resize(
        video, (new_h, new_w),
        interpolation=transforms.InterpolationMode.BILINEAR,
        antialias=True,
    )
    video = transforms.functional.center_crop(video, (target_h, target_w))  # [T, 3, H, W]

    # [T, 3, H, W] → [1, 3, T, H, W] + 归一化 [-1, 1]
    video = video.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 3, T, H, W]
    video = (video / 255.0 - 0.5) * 2.0

    # 截断到 T ≡ 1 (mod 4)（VAE 时序压缩要求）
    T = video.shape[2]
    T_valid = ((T - 1) // 4) * 4 + 1
    if T_valid < T:
        video = video[:, :, :T_valid]

    return video


def load_actions(action_path: str, T_pix: int) -> tuple:
    """加载 move_view.json → Long tensor，长度匹配视频帧数。

    返回: (move, view)，每个是 Long[T_pix]
    """
    with open(action_path, 'r') as f:
        actions = json.load(f)

    move = [MOVE_ACTION_MAP[a['move']] for a in actions]
    view = [VIEW_ACTION_MAP[a['view']] for a in actions]

    # 截断或零填充到 T_pix
    if len(move) > T_pix:
        move = move[:T_pix]
        view = view[:T_pix]
    elif len(move) < T_pix:
        pad_len = T_pix - len(move)
        move.extend([0] * pad_len)
        view.extend([0] * pad_len)

    return torch.tensor(move, dtype=torch.long), torch.tensor(view, dtype=torch.long)


# ============================================================================
# §P4 VAE 编码
# ============================================================================
def encode_latent_full(vae, frames: torch.Tensor, device) -> torch.Tensor:
    """整段编码（用于 image_cond 前缀切片）。

    输入: frames [1, 3, T_pix, H, W]
    输出: [C, T_lat, h, w]，T_lat = (T_pix - 1) // 4 + 1
    """
    with torch.no_grad():
        frames = frames.to(device)
        latent = vae.encode(frames)  # [1, C, T_lat, h, w]
    return latent.squeeze(0).cpu()  # [C, T_lat, h, w]


def encode_target_latent(vae, frames: torch.Tensor, K: int,
                        frames_per_chunk: int, stride: int, device) -> torch.Tensor:
    """逐 chunk 独立编码（用于 x_start GT）。

    输入: frames [1, 3, T_pix, H, W]
    输出: [K, C, 21, h, w]
    """
    target_latent = []
    for k in range(K):
        start = k * stride
        end = start + frames_per_chunk
        chunk_frames = frames[:, :, start:end].to(device)  # [1, 3, 81, H, W]

        with torch.no_grad():
            chunk_latent = vae.encode(chunk_frames)  # [1, C, 21, h, w]
        target_latent.append(chunk_latent.squeeze(0).cpu())  # [C, 21, h, w]

    return torch.stack(target_latent)  # [K, C, 21, h, w]


# ============================================================================
# §P5 单 case 处理
# ============================================================================
def preprocess_case(case_name: str, case_dir: str, out_dir: str,
                   vae, text_encoder, bucket_config: dict,
                   cfg: PreprocessConfig, device, logger) -> dict:
    """预处理一个 case，写出 4 个 .pt + meta.json。

    返回: meta dict（或 None 如果跳过/失败）
    """
    t_start = time.time()

    required_files = ["video.mp4", "move_view.json", "prompts.json"]
    for fname in required_files:
        if not os.path.exists(os.path.join(case_dir, fname)):
            logger.warning(f"[SKIP] {case_name}: missing {fname}")
            return None

    # 检查输出目录
    os.makedirs(out_dir, exist_ok=True)
    if cfg.skip_existing and os.path.exists(os.path.join(out_dir, "meta.json")):
        logger.info(f"[SKIP] {case_name}: already preprocessed")
        return None

    logger.info(f"[PROCESS] {case_name}")

    # 1. 加载视频
    video_path = os.path.join(case_dir, "video.mp4")
    frames = load_video_frames(video_path, bucket_config)  # [1, 3, T_pix, H, W]
    T_pix = frames.shape[2]
    H, W = frames.shape[3], frames.shape[4]
    logger.info(f"  {case_name}: {T_pix} frames, {H}x{W}")

    # 2. 计算 chunk 数量
    K = (T_pix - 1) // cfg.chunk_stride
    if K == 0:
        logger.warning(f"[SKIP] {case_name}: video too short ({T_pix} frames < 81)")
        return None

    # 3. VAE 编码
    latent_full = encode_latent_full(vae, frames, device)  # [C, T_lat, h, w]
    target_latent = encode_target_latent(vae, frames, K,
                                         cfg.frames_per_chunk,
                                         cfg.chunk_stride, device)  # [K, C, 21, h, w]

    # 4. 文本编码
    prompts_path = os.path.join(case_dir, "prompts.json")
    with open(prompts_path, 'r') as f:
        prompts_data = json.load(f)
        prompt = prompts_data["prompt"]

    with torch.no_grad():
        text_kwargs = text_encoder.encode([prompt])
    y = text_kwargs["y"].cpu()              # [1, 1, L, D]
    y_mask = text_kwargs["y_mask"].cpu()    # [1, L]

    # 5. 动作序列
    action_path = os.path.join(case_dir, "move_view.json")
    move, view = load_actions(action_path, T_pix)  # Long[T_pix]

    # 6. 转换为保存精度
    save_dtype = getattr(torch, cfg.save_dtype)
    latent_full = latent_full.to(save_dtype)
    target_latent = target_latent.to(save_dtype)
    y = y.to(save_dtype)

    # 7. 保存
    torch.save(latent_full, os.path.join(out_dir, "latent_full.pt"))
    torch.save(target_latent, os.path.join(out_dir, "target_latent.pt"))
    torch.save({"y": y, "y_mask": y_mask}, os.path.join(out_dir, "text_emb.pt"))
    torch.save({"move": move, "view": view}, os.path.join(out_dir, "actions.pt"))

    # 8. 元数据
    meta = {
        "name": case_name,
        "num_chunks": K,
        "num_frames": T_pix,
        "height": H,
        "width": W,
        "prompt": prompt,
        "latent_shape": list(latent_full.shape),
        "save_dtype": cfg.save_dtype,
    }

    # 添加 prompts.json 中的额外属性（如果存在）
    extra_fields = ["location", "scene", "crowdDensity", "weather", "timeOfDay"]
    for field in extra_fields:
        if field in prompts_data:
            meta[field] = prompts_data[field]
    with open(os.path.join(out_dir, "meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t_start
    logger.info(f"  ✓ {case_name}: {K} chunks saved in {elapsed:.1f}s")
    return meta


# ============================================================================
# §P6 自检（验证因果性假设）
# ============================================================================
def verify_prefix_consistency(vae, frames: torch.Tensor, device, atol=1e-3) -> bool:
    """验证: encode(frames[:, :, :81]) ≈ encode(frames)[:, :21]

    证明 VAE 的因果性 → latent_full 的前缀切片可用作 image_cond。
    """
    print("\n[VERIFY] Testing prefix consistency...")

    with torch.no_grad():
        # 完整编码
        full_latent = vae.encode(frames.to(device))  # [1, C, T_lat, h, w]

        # 前 81 帧独立编码
        prefix_frames = frames[:, :, :81].to(device)
        prefix_latent = vae.encode(prefix_frames)  # [1, C, 21, h, w]

        # 比较前 21 个 latent 帧
        full_prefix = full_latent[:, :, :21]
        diff = (full_prefix - prefix_latent).abs().max().item()

        print(f"  Max diff: {diff:.6f} (threshold: {atol})")
        if diff < atol:
            print("  ✓ Prefix consistency verified")
            return True
        else:
            print(f"  ✗ Prefix inconsistency detected (diff={diff:.6f})")
            return False


# ============================================================================
# §P7 主循环
# ============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str,
                       default="dataset/sekai-game-walking-854_480_30fps")
    parser.add_argument("--output-dir", type=str,
                       default="preprocessed/sekai-game-walking-854_480_30fps-256px")
    parser.add_argument("--bucket-config", type=str, default="ASPECT_RATIO_256_F64")
    parser.add_argument("--max-cases", type=int, default=-1)
    parser.add_argument("--verify", action="store_true", help="验证因果性")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    # 初始化分布式环境
    rank, world_size, local_rank = setup_distributed()

    cfg = PreprocessConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        bucket_config_name=args.bucket_config,
        max_cases=args.max_cases,
        verify_prefix=args.verify,
        skip_existing=args.skip_existing,
    )

    # 解析路径
    cfg.dataset_dir = _resolve_path(cfg.dataset_dir)
    cfg.output_dir = _resolve_path(cfg.output_dir)
    cfg.model_config_path = _resolve_path(cfg.model_config_path)

    # 初始化日志（每个 rank 独立文件，rank 0 同时输出终端）
    logger = setup_logging(cfg.output_dir, rank)

    if rank == 0:
        logger.info("=" * 70)
        logger.info("Infinite World - Dataset Preprocessing (Distributed)")
        logger.info("=" * 70)
        logger.info(f"Dataset: {cfg.dataset_dir}")
        logger.info(f"Output:  {cfg.output_dir}")
        logger.info(f"Bucket:  {cfg.bucket_config_name}")
        logger.info(f"World size: {world_size} GPUs")
        logger.info(f"Logs dir: {os.path.join(cfg.output_dir, 'logs')}")
        logger.info("=" * 70)

    # 设备
    device = torch.device(f"cuda:{local_rank}")
    logger.info(f"Device: cuda:{local_rank} (rank {rank})")

    # 加载模型配置
    model_args = _load_model_config(cfg.model_config_path)
    bucket_config = _get_bucket_config(cfg.bucket_config_name)

    # 加载 VAE
    logger.info("[LOAD] VAE...")
    vae = get_obj_from_str(model_args.vae_target)(**model_args.vae_cfg).to(device)
    vae.eval()

    # 加载 Text Encoder
    logger.info("[LOAD] Text Encoder...")
    text_encoder = get_obj_from_str(model_args.text_encoder_target)(
        device=device, **model_args.text_encoder_cfg)
    text_encoder.t5.model.to(device)
    text_encoder.t5.model.eval()

    # 扫描 case 目录（所有 rank 都扫描以保持一致）
    case_names = sorted(
        [d for d in os.listdir(cfg.dataset_dir)
         if d.startswith("case") and os.path.isdir(os.path.join(cfg.dataset_dir, d))],
        key=_case_sort_key
    )

    if rank == 0:
        logger.info(f"[SCAN] Found {len(case_names)} cases in {cfg.dataset_dir}")

    if cfg.max_cases > 0:
        case_names = case_names[:cfg.max_cases]
        if rank == 0:
            logger.info(f"Processing first {cfg.max_cases} cases")

    # 分配 cases 给各个 rank（round-robin）
    my_cases = [case_names[i] for i in range(len(case_names)) if i % world_size == rank]
    logger.info(f"[DISTRIBUTE] Rank {rank} assigned {len(my_cases)}/{len(case_names)} cases")

    # 处理分配给当前 rank 的 cases
    success_count = 0
    fail_count = 0
    t_loop_start = time.time()
    desc = f"Rank {rank}"
    for case_name in tqdm(my_cases, desc=desc, position=rank):
        case_dir = os.path.join(cfg.dataset_dir, case_name)
        out_dir = os.path.join(cfg.output_dir, case_name)

        try:
            meta = preprocess_case(case_name, case_dir, out_dir,
                                 vae, text_encoder, bucket_config, cfg, device, logger)
            if meta:
                success_count += 1
        except Exception:
            fail_count += 1
            logger.exception(f"[FAIL] {case_name}: unhandled exception")

    elapsed_loop = time.time() - t_loop_start
    logger.info(f"[DONE] Rank {rank}: {success_count} success, {fail_count} failed "
                f"in {elapsed_loop/60:.1f} min")

    # 同步所有进程
    if world_size > 1:
        dist.barrier()

        # 汇总成功/失败数量
        stat_tensor = torch.tensor([success_count, fail_count],
                                   dtype=torch.long, device=device)
        dist.all_reduce(stat_tensor, op=dist.ReduceOp.SUM)
        total_success, total_fail = stat_tensor[0].item(), stat_tensor[1].item()
    else:
        total_success, total_fail = success_count, fail_count

    if rank == 0:
        logger.info("=" * 70)
        logger.info(f"Preprocessing complete: {total_success} success, "
                    f"{total_fail} failed / {len(case_names)} total cases")
        logger.info("=" * 70)

    # 可选：验证因果性（仅 rank 0 执行）
    if cfg.verify_prefix and rank == 0 and len(case_names) > 0:
        logger.info("=" * 70)
        test_case = case_names[0]
        test_video = os.path.join(cfg.dataset_dir, test_case, "video.mp4")
        test_frames = load_video_frames(test_video, bucket_config)
        verify_prefix_consistency(vae, test_frames, device)
        logger.info("=" * 70)

    # 清理分布式环境
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
