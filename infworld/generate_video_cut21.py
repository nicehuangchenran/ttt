#!/usr/bin/env python3
"""为 WBench 评测产出视频：复用 scripts/infworld_inference.py 的全部逻辑，
仅替换数据读取（infworld dataset）与输出命名（WBench work_dirs 布局），支持多 GPU。

可选 online (test-time) training：完全复用 infworld_inference 的实现
（inf.online_train_step / inf.set_grad_checkpoint 与同名训练超参），
由环境变量 ONLINE_TRAINING=on 开启（默认 off）。

运行（在 Infinite-World 目录下）：
    单 GPU： python generate_video.py
    多 GPU： torchrun --nproc_per_node=8 generate_video.py
或经 generate.sh：
    bash generate.sh 1                  # 单 GPU，全部 case
    bash generate.sh 8                  # 8 GPU
    bash generate.sh 8 10 infworld on   # 8 GPU，前 10 个 case，开启 online training

可用环境变量覆盖：INPUT_DATASET / OUTPUT_MODEL_NAME / NUM_CASES / ONLINE_TRAINING。
"""

import os
import json
import math
import glob
import torch

# 导入即触发其模块级 setup_distributed()/CP 初始化（单 GPU 或 torchrun 多 GPU 均可），
# 之后直接复用 inf.local_rank / inf.dp_rank / inf.dp_size 等全局量与全部工具函数/常量。
import scripts.infworld_inference as inf

from omegaconf import OmegaConf
from infworld.utils.prepare_dataloader import get_obj_from_str
from infworld.configs import bucket_config as bucket_config_module

# ============================================================================
# 可编辑常量
# ============================================================================
WBENCH_WORK_DIRS = "/root/autodl-tmp/ttt/WBench/work_dirs"

# 入口默认值（可被环境变量覆盖）
INPUT_DATASET = os.environ.get("INPUT_DATASET", os.path.join(inf.PROJECT_ROOT, "dataset"))
OUTPUT_MODEL_NAME = os.environ.get("OUTPUT_MODEL_NAME", "infworld")
NUM_CASES = int(os.environ.get("NUM_CASES", "0"))  # <=0 表示全部

# online (test-time) training 开关（复用 infworld_inference 的实现）；off / on
ENABLE_ONLINE_TRAINING = os.environ.get("ONLINE_TRAINING", "off").strip().lower() in ("on", "true", "1", "yes")

# 冻结的条件输入层前缀（与 infworld_inference.main 完全一致）：online training 时不漂移这些
# 承载历史(image_cond)/动作/文本/时间步接口的层，保证 chunk 间连续性。
FREEZE_PREFIXES = (
    "patch_embedding",   # raw latent -> token conv
    "latent_encoder",    # image_cond (history) temporal encoder
    "action_encoder",    # move / view embeddings
    "text_embedding",    # T5 -> token projection
    "time_embedding",    # timestep MLP
    "time_projection",   # timestep -> per-block modulation
    "y_embedder",        # null caption embedding
)


# ============================================================================
# 数据读取（仅此部分替换原脚本的 prompts/demo.yaml）
# ============================================================================
def list_cases(input_dataset_path, n):
    """列出 case<id> 目录，按 int(id) 升序，取前 n 个（n<=0 取全部）。

    返回 [(prompt, image_path, action_path, case_id), ...]
    """
    case_dirs = []
    for d in glob.glob(os.path.join(input_dataset_path, "case*")):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        cid = name[len("case"):]
        if not cid.isdigit():
            continue
        case_dirs.append((int(cid), cid, d))

    case_dirs.sort(key=lambda x: x[0])
    if n and n > 0:
        case_dirs = case_dirs[:n]

    cases = []
    for _, cid, d in case_dirs:
        with open(os.path.join(d, "prompts.json"), "r", encoding="utf-8") as f:
            prompt = json.load(f)["prompt"]
        image_path = os.path.join(d, "image.jpg")
        action_path = os.path.join(d, "move_view.json")
        cases.append((prompt, image_path, action_path, cid))
    return cases


# ============================================================================
# 模型加载（照搬 infworld_inference.main() 的建模流程，复用 inf.* 常量/函数）
# 含 online training 的冻结/快照设置（与原脚本一致）
# ============================================================================
def load_models():
    inf.torch_gc()

    args = OmegaConf.load(inf.CONFIG_PATH)

    # 解析配置中模型权重的相对路径
    if hasattr(args, "vae_cfg") and "vae_pth" in args.vae_cfg:
        args.vae_cfg.vae_pth = inf.resolve_path(args.vae_cfg.vae_pth)
    if hasattr(args, "text_encoder_cfg"):
        if "checkpoint_path" in args.text_encoder_cfg:
            args.text_encoder_cfg.checkpoint_path = inf.resolve_path(args.text_encoder_cfg.checkpoint_path)
        if "tokenizer_path" in args.text_encoder_cfg:
            args.text_encoder_cfg.tokenizer_path = inf.resolve_path(args.text_encoder_cfg.tokenizer_path)

    print("[GenVideo] Loading VAE...")
    vae = get_obj_from_str(args.vae_target)(**args.vae_cfg).to(inf.local_rank)

    print("[GenVideo] Loading Text Encoder...")
    text_encoder = get_obj_from_str(args.text_encoder_target)(device=inf.local_rank, **args.text_encoder_cfg)
    text_encoder.t5.model.to(inf.local_rank)

    print("[GenVideo] Loading Scheduler...")
    scheduler = get_obj_from_str(args.scheduler_target)(**args.val_scheduler_cfg)
    scheduler.num_sampling_steps = inf.NUM_SAMPLING_STEPS
    scheduler.shift = inf.SHIFT

    print("[GenVideo] Loading DiT Model...")
    dtype = getattr(torch, args.amp_dtype)
    dit = get_obj_from_str(args.model_target)(
        out_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
        enable_context_parallel=inf.enable_context_parallel,
        **args.model_cfg
    ).to(dtype)
    dit.eval()

    state_dict = inf.load_dit_state_dict(args.checkpoint_path)
    state_dict.pop("pos_embed_temporal", None)
    state_dict.pop("pos_embed", None)
    missing, unexpected = dit.load_state_dict(state_dict, strict=False)
    print(f"[GenVideo] Model loaded! Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    dit.to(inf.local_rank)

    # ---- online training 设置（与 infworld_inference.main 完全一致）----
    trainable_params = None
    init_params = None
    if ENABLE_ONLINE_TRAINING:
        if inf.USE_GRAD_CHECKPOINT:
            inf.set_grad_checkpoint(dit)
        # 冻结条件输入层，收集可训练参数
        trainable_params = []
        frozen_n = 0
        for nm, p in dit.named_parameters():
            if nm.startswith(FREEZE_PREFIXES):
                p.requires_grad_(False)
                frozen_n += p.numel()
            else:
                p.requires_grad_(True)
                trainable_params.append(p)
        print(f"[OnlineTrain] frozen params (cond entry layers): {frozen_n:,}; "
              f"trainable: {sum(p.numel() for p in trainable_params):,}")
        # 快照初始可训练参数，用于每个 video 之间恢复
        if inf.RESET_BETWEEN_VIDEOS:
            init_params = {nm: p.detach().clone()
                           for nm, p in dit.named_parameters() if p.requires_grad}

    bucket_config = getattr(bucket_config_module, inf.BUCKET_CONFIG_NAME)
    return args, vae, text_encoder, scheduler, dit, bucket_config, trainable_params, init_params


# ============================================================================
# 单 case 生成（照搬分块自回归生成核心；chunk 数自动覆盖完整 action 序列）
# 含 chunk 间 online training（与原脚本一致）
# ============================================================================
def _generate_one(prompt, image_path, action_path, models, task_idx):
    args, vae, text_encoder, scheduler, dit, bucket_config, trainable_params, init_params = models

    cond_video = inf.load_condition_image(image_path, bucket_config).to(inf.local_rank)
    with torch.no_grad():
        cond_latent = vae.encode(cond_video)

    move_indices, view_indices = inf.load_action_sequence(action_path)

    video_buffer = cond_video.clone().cpu()

    # latent size: [B, C, T, H, W]，T=21（1 帧条件 + 20 帧生成，解码为 81 像素帧）
    latent_size = list(cond_latent.shape)
    latent_size[2] = 21
    latent_size = torch.Size(latent_size)

    num_frames = args.validation_data.num_frames  # 81
    step = num_frames - 1                          # 每 chunk 推进 80 像素帧
    num_chunks = max(1, math.ceil(len(move_indices) / step))
    print(f"[GenVideo] action frames={len(move_indices)} -> num_chunks={num_chunks}")

    # ---- 本 video 的 online training 准备（与 infworld_inference.main 一致）----
    cached_y = cached_y_mask = None
    if ENABLE_ONLINE_TRAINING:
        if inf.RESET_BETWEEN_VIDEOS and task_idx > 0 and init_params is not None:
            with torch.no_grad():
                for nm, p in dit.named_parameters():
                    if nm in init_params:
                        p.data.copy_(init_params[nm])
            print(f"[OnlineTrain] task {task_idx}: restored init params")
        with torch.no_grad():
            text_kwargs = text_encoder.encode([prompt])
        cached_y, cached_y_mask = text_kwargs["y"], text_kwargs["y_mask"]

    for chunk_idx in range(num_chunks):
        print(f"[GenVideo] chunk {chunk_idx + 1}/{num_chunks}")
        with torch.no_grad():
            current_cond = video_buffer.to(inf.local_rank)
            current_latent = vae.encode(current_cond)

        curr_start = video_buffer.shape[2] - 1
        curr_end = curr_start + num_frames

        move = torch.tensor(move_indices[curr_start:curr_end], dtype=torch.long, device=inf.local_rank)
        view = torch.tensor(view_indices[curr_start:curr_end], dtype=torch.long, device=inf.local_rank)
        if move.shape[0] < num_frames:
            pad_len = num_frames - move.shape[0]
            move = torch.cat([move, torch.zeros(pad_len, dtype=torch.long, device=inf.local_rank)])
            view = torch.cat([view, torch.zeros(pad_len, dtype=torch.long, device=inf.local_rank)])

        additional_args = {
            "image_cond": current_latent,
            "move": move.unsqueeze(0),
            "view": view.unsqueeze(0),
        }
        inf.torch_gc()
        with torch.no_grad():
            samples = scheduler.sample(
                model=dit,
                text_encoder=text_encoder,
                null_embedder=dit.y_embedder,
                z_size=latent_size,
                prompts=[prompt],
                guidance_scale=inf.TEXT_CFG_SCALE,
                negative_prompts=[inf.NEGATIVE_PROMPT],
                device=torch.device(inf.local_rank),
                additional_args=additional_args,
            )
            decoded_chunk = vae.decode(samples).cpu()
            video_buffer = torch.cat([video_buffer, decoded_chunk[:, :, 1:]], dim=2)
            print(f"[GenVideo] chunk {chunk_idx + 1} done. total frames: {video_buffer.shape[2]}")
            inf.torch_gc()

        # ---- chunk 间 online training：在刚生成的 chunk 上微调，再生成下一个 ----
        # optimizer 每 chunk 重建，Adam 动量不跨 chunk 累积（与原脚本一致）
        if ENABLE_ONLINE_TRAINING and chunk_idx < num_chunks - 1:
            optimizer = torch.optim.AdamW(trainable_params, lr=inf.TRAIN_LR)
            x_start = samples.detach()
            train_kwargs = {
                "y": cached_y,
                "y_mask": cached_y_mask,
                "image_cond": current_latent.detach(),
                "move": move.unsqueeze(0).detach(),
                "view": view.unsqueeze(0).detach(),
            }
            for tstep in range(inf.N_TRAIN_STEPS):
                loss_val = inf.online_train_step(dit, scheduler, x_start, train_kwargs, optimizer)
                print(f"[OnlineTrain] task {task_idx} chunk {chunk_idx} step {tstep + 1}/{inf.N_TRAIN_STEPS} "
                      f"loss={loss_val.item():.5f}")
            del x_start, train_kwargs
            inf.torch_gc()

    return video_buffer


# ============================================================================
# 主函数
# ============================================================================
def generate_video(input_dataset_path, output_video_path, n):
    """为 infworld dataset 前 n 个 case（按 case id 升序）生成视频，按 WBench 布局落盘：
        <WBENCH_WORK_DIRS>/<output_video_path>/videos/case_{id}_combined.mp4
    支持多 GPU：每个 rank 仅处理 task_idx % dp_size == dp_rank 的 case。
    """
    cases = list_cases(input_dataset_path, n)
    print(f"[GenVideo] rank {inf.dp_rank}/{inf.dp_size} | total cases: {len(cases)} | "
          f"online_training={ENABLE_ONLINE_TRAINING}")

    models = load_models()

    save_dir = os.path.join(WBENCH_WORK_DIRS, output_video_path, "videos")
    os.makedirs(save_dir, exist_ok=True)

    written = []
    quality = 10 if inf.HIGH_QUALITY_SAVE else 5
    for task_idx, (prompt, image_path, action_path, case_id) in enumerate(cases):
        if task_idx % inf.dp_size != inf.dp_rank:
            continue
        if not (os.path.exists(image_path) and os.path.exists(action_path)):
            print(f"[GenVideo] skip case {case_id}: missing image/action")
            continue

        print(f"[GenVideo] [rank {inf.dp_rank}] case {case_id}: {prompt[:50]}...")
        video_buffer = _generate_one(prompt, image_path, action_path, models, task_idx)

        save_path = os.path.join(save_dir, f"case_{case_id}_combined")  # 不含扩展名
        final_path = f"{save_path}.mp4"
        # save_silent_video 对已存在文件会追加帧，重跑前先删除避免重复拼接
        if os.path.exists(final_path):
            os.remove(final_path)

        inf.save_silent_video(video_buffer.to(inf.local_rank), save_path, fps=30, quality=quality)
        print(f"[GenVideo] saved: {final_path} ({video_buffer.shape[2]} frames)")
        written.append(final_path)

    print(f"[GenVideo] rank {inf.dp_rank} done, wrote {len(written)} videos")
    return written


if __name__ == "__main__":
    generate_video(INPUT_DATASET, OUTPUT_MODEL_NAME, NUM_CASES)
