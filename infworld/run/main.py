#!/usr/bin/env python3
"""为 WBench 评测产出视频：复用 scripts/infworld_inference.py 的全部逻辑，
仅替换数据读取（infworld dataset）与输出命名（WBench work_dirs 布局），支持多 GPU。

可选 online (test-time) training：完全复用 infworld_inference 的实现
（inf.online_train_step / inf.set_grad_checkpoint 与同名训练超参），
由 CLI 参数 --online-training on 开启（默认 off）。

运行（需在项目根目录 /root/autodl-tmp/ttt/infworld 下，使 scripts.infworld_inference 可导入）：
    单 GPU： python run/main.py [args]
    多 GPU： torchrun --nproc_per_node=8 run/main.py [args]
或经 run/run.sh（在 JOBS 数组里配置每个 job，自动选择 python / torchrun）：
    bash run/run.sh

CLI 参数：--input-dataset / --outdir / --num-cases / --online-training / --max-num-chunks
（详见 parse_args；torchrun 会把这些参数透传给每个进程）。
"""

import os
import json
import math
import glob
import argparse
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

# 入口参数默认值（由 __main__ 中的 CLI 覆盖）
INPUT_DATASET = os.path.join(inf.PROJECT_ROOT, "dataset")
OUTPUT_MODEL_NAME = "infworld"
NUM_CASES = 0  # <=0 表示全部

# online (test-time) training 开关（复用 infworld_inference 的实现）；由 --online-training 覆盖
ENABLE_ONLINE_TRAINING = False

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
    """列出 dataset 下的 case<id> 目录，按 int(id) 升序，取前 n 个。

    每个 case 目录需含 prompts.json（{"prompt": ...}）、image.jpg、move_view.json。

    Args:
        input_dataset_path (str): dataset 根目录，内含若干 case<id>/ 子目录。
        n (int): 取前 N 个 case；<=0 表示全部。

    Returns:
        list[tuple[str, str, str, str]]: [(prompt, image_path, action_path, case_id), ...]，
            其中 case_id 为字符串形式的原始 id。
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
    """加载并初始化整条推理所需的全部模型组件（照搬 infworld_inference.main 的建模流程）。

    依次完成：解析配置中权重的相对路径 -> 加载 VAE / Text Encoder / Scheduler / DiT，
    并从 checkpoint 载入 DiT 权重（剔除 pos_embed*，运行时重算）。若开启 online
    training，则冻结条件输入层（FREEZE_PREFIXES）、收集可训练参数，并按需快照初始权重
    用于 video 之间恢复。

    依赖模块级全局 ENABLE_ONLINE_TRAINING 及 inf.* 的常量/工具函数。

    Returns:
        tuple: (args, vae, text_encoder, scheduler, dit, bucket_config,
                trainable_params, init_params)
            - args (OmegaConf): 解析后的配置。
            - bucket_config: 由 inf.BUCKET_CONFIG_NAME 选出的分桶表。
            - trainable_params (list[Tensor] | None): 仅 online training 时非 None。
            - init_params (dict[str, Tensor] | None): 初始可训练参数快照，
              仅 online training 且 RESET_BETWEEN_VIDEOS 时非 None。
    """
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
def _generate_one(prompt, image_path, action_path, models, task_idx, max_num_chunks=0):
    """对单个 case 做分块自回归生成，返回完整视频帧缓冲。

    流程：编码条件图 -> 按 action 长度算出 num_chunks（可被 max_num_chunks 封顶）->
    逐 chunk 重编码已生成尾帧作 image_cond、切片 move/view 动作（不足则补零）、
    调 scheduler.sample 采样并 VAE 解码，追加 decoded_chunk[:, :, 1:]（丢弃重叠帧）到
    video_buffer。若开启 online training，则在每个 chunk 生成后于刚生成的 chunk 上微调，
    再生成下一个；video 起始时按 RESET_BETWEEN_VIDEOS 恢复初始权重。

    Args:
        prompt (str): 文本提示。
        image_path (str): 条件图路径。
        action_path (str): move/view 动作序列 JSON 路径。
        models (tuple): load_models() 的返回值。
        task_idx (int): case 在全局列表中的序号（用于 online training 的 video 间恢复）。
        max_num_chunks (int): 生成 chunk 数上限；<=0 表示不限制（跑完整 action 序列）。

    Returns:
        torch.Tensor: 视频帧缓冲，形状 [B, C, T, H, W]（CPU），T 为累积像素帧数。
    """
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
    # max_num_chunks <= 0 表示不限制；否则封顶到 max_num_chunks
    if max_num_chunks and max_num_chunks > 0 and num_chunks > max_num_chunks:
        print(f"[GenVideo] num_chunks {num_chunks} > max_num_chunks {max_num_chunks}, capped to {max_num_chunks}")
        num_chunks = max_num_chunks
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
def generate_video(input_dataset_path, output_video_path, n, max_num_chunks=0):
    """为 dataset 前 n 个 case 生成视频并按 WBench 布局落盘。

    输出路径：<WBENCH_WORK_DIRS>/<output_video_path>/videos/case_{id}_combined.mp4。
    支持多 GPU：每个 rank 仅处理 task_idx % dp_size == dp_rank 的 case；缺失
    image/action 的 case 跳过；已存在的输出文件会先删除以避免重复拼接。

    Args:
        input_dataset_path (str): dataset 根目录（见 list_cases）。
        output_video_path (str): 输出子目录名（落在 WBENCH_WORK_DIRS 下）。
        n (int): 取前 N 个 case；<=0 表示全部。
        max_num_chunks (int): 每个 case 的生成 chunk 数上限；<=0 表示不限制。

    Returns:
        list[str]: 本 rank 实际写出的 mp4 文件路径列表。
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
        video_buffer = _generate_one(prompt, image_path, action_path, models, task_idx, max_num_chunks)

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


def parse_args():
    """解析命令行参数（单/多 GPU 共用；torchrun 会把这些参数透传给每个进程）。

    Returns:
        argparse.Namespace: 含 input_dataset / outdir / num_cases /
            online_training / max_num_chunks 五个字段。
    """
    p = argparse.ArgumentParser(
        description="为 WBench 评测产出 Infinite-World 视频（单/多 GPU）"
    )
    p.add_argument("--input-dataset", default=os.path.join(inf.PROJECT_ROOT, "dataset"),
                   help="infworld dataset 根目录（含 case<id>/ 子目录）")
    p.add_argument("--outdir", default="infworld",
                   help="输出模型名，落盘到 WBench/work_dirs/<outdir>/videos/")
    p.add_argument("--num-cases", type=int, default=0,
                   help="取前 N 个 case（按 case id 升序），<=0 表示全部")
    p.add_argument("--online-training",
                   type=lambda s: s.strip().lower() in ("on", "true", "1", "yes"),
                   default=False, metavar="on/off",
                   help="chunk 间 online（test-time）training 开关")
    p.add_argument("--max-num-chunks", type=int, default=0,
                   help="每个 case 最多生成的 chunk 数，<=0 表示不限制")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # 在模块顶层作用域重新绑定全局，供 load_models/_generate_one/generate_video 引用
    ENABLE_ONLINE_TRAINING = args.online_training
    generate_video(args.input_dataset, args.outdir, args.num_cases, args.max_num_chunks)
