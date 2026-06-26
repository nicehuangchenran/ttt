# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Infinite-World is an interactive (action-conditioned) world model that generates long-horizon videos (1000+ frames) from a single condition image, a text prompt, and a per-frame action sequence. The current public code path is **inference only**; there is no training entrypoint in this tree.

## Common Commands

### Environment setup
```bash
conda create -n infworld python=3.10 && conda activate infworld
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```
PyTorch must be installed before `requirements.txt` (which pins `flash_attn==2.7.4.post1` and expects a matching torch). Do not pin `triton` — it ships with torch.

### Run inference
```bash
bash infer_local.sh 1     # single GPU, runs python directly (no torchrun, no port)
bash infer_local.sh 8     # 8 GPUs via torchrun
MASTER_PORT=29500 bash infer_local.sh 8   # override on EADDRINUSE
```

Single-GPU mode bypasses `torch.distributed` entirely (see `setup_distributed()` and the `use_dist` flag) — `dp_rank`/`cp_rank` are set as module-level globals in `infworld.context_parallel.context_parallel_util`. Multi-GPU mode goes through `init_context_parallel`. Any code that relies on context-parallel ranks must work in both modes.

### Checkpoints
All four required artifacts go under `checkpoints/` (paths configured in `configs/infworld_config.yaml`):
- `infinite_world_model.ckpt` — DiT weights (`checkpoint_path` at config root)
- `models/Wan2.1_VAE.pth` — VAE (`vae_cfg.vae_pth`)
- `models/models_t5_umt5-xxl-enc-bf16.pth` — UMT5 encoder (`text_encoder_cfg.checkpoint_path`)
- `models/google/umt5-xxl/` — UMT5 tokenizer (`text_encoder_cfg.tokenizer_path`)

Relative paths in the YAML are resolved against `PROJECT_ROOT` by `resolve_path()` in [scripts/infworld_inference.py](scripts/infworld_inference.py).

There is no test suite, lint config, or build step in the repo.

## Architecture

### Inference pipeline
[scripts/infworld_inference.py](scripts/infworld_inference.py) is the single entrypoint. It is not a CLI — all knobs (`NUM_SAMPLING_STEPS`, `TEXT_CFG_SCALE`, `SHIFT`, `NUM_CHUNKS`, `BUCKET_CONFIG_NAME`, `PROMPTS_YAML`, etc.) are module-level constants. To change behavior, edit the script directly.

The pipeline is **chunked autoregressive generation**:
1. Load condition image, resize+center-crop to the closest aspect-ratio bucket from [infworld/configs/bucket_config.py](infworld/configs/bucket_config.py) (default bucket: `ASPECT_RATIO_627_F64`).
2. VAE-encode it to a latent and seed `video_buffer`.
3. For each of `NUM_CHUNKS` chunks:
   - Re-encode the current `video_buffer` tail as `image_cond`.
   - Slice `move`/`view` action indices for the next `validation_data.num_frames` frames (default 81), pad with zeros if the action JSON is shorter.
   - Call `scheduler.sample(...)` with `additional_args={image_cond, move, view}`.
   - VAE-decode and append `decoded_chunk[:, :, 1:]` (drops the overlapping frame) to `video_buffer`.

This is what enables the 1000+ frame horizon: each chunk conditions on the previously generated tail, so memory propagates via the VAE-encoded `image_cond` rather than full-history attention.

### Action vocabulary
Two parallel 10-class action streams, with fixed integer encodings in [scripts/infworld_inference.py:34-58](scripts/infworld_inference.py#L34-L58):
- `MOVE_ACTION_MAP`: no-op, go forward/back/left/right, diagonal combinations, uncertain
- `VIEW_ACTION_MAP`: no-op, turn up/down/left/right, diagonal combinations, uncertain

Action JSONs (see `assets/example_case/*.json`) are arrays of `{"move": "...", "view": "..."}` objects, one per frame. The `'uncertain'` class is the uncertainty-aware labeling mechanism from the paper.

### Model components
Wired via `target` + `cfg` pairs in [configs/infworld_config.yaml](configs/infworld_config.yaml), instantiated by `get_obj_from_str()` from [infworld/utils/prepare_dataloader.py](infworld/utils/prepare_dataloader.py).

- **DiT** ([infworld/models/dit_model.py](infworld/models/dit_model.py), `WanModel`) — 1.3B-parameter diffusion transformer derived from the Wan2.1 architecture (`in_channels=20`, `dim=1536`, `num_layers=30`). Consumes the text embedding, VAE latent conditioning, and the move/view action streams. Tries flash-attn 3, then flash-attn 2, then transformer-engine `DotProductAttention` — all imports are guarded so the model still loads if one is missing. Position embeddings (`pos_embed`, `pos_embed_temporal`) are stripped from the checkpoint and recomputed at load time.
- **VAE** ([infworld/vae/](infworld/vae/), `WanVAEModelWrapper`) — wraps Wan2.1 VAE with `patch_size=(4, 8, 8)` and `out_channels=16`. The `(4, 8, 8)` factor governs the latent shape used everywhere downstream (e.g., `latent_size[2] = 21` for an 81-frame chunk).
- **Text encoder** ([infworld/models/umt5.py](infworld/models/umt5.py), `T5EncoderModel`) — UMT5-XXL, `model_max_length=512`. `t5.py` exists alongside but the config wires umt5.
- **Scheduler** ([infworld/models/scheduler.py](infworld/models/scheduler.py), `RFlowScheduler`) — rectified-flow sampler. `shift` is resolution-dependent (3 for 256px, 7 for 627px, 11 for 960px); the script default of 7 is paired with `ASPECT_RATIO_627_F64`.

### Context parallelism
[infworld/context_parallel/context_parallel_util.py](infworld/context_parallel/context_parallel_util.py) defines a 2D parallelism (DP × CP) over the world. `context_parallel_size` is hard-coded to 1 in the inference script, so multi-GPU runs currently do pure data parallelism — each rank handles `task_idx % dp_size == dp_rank` from `prompts/demo.yaml`. The DiT itself accepts `enable_context_parallel=(context_parallel_size > 1)` for when CP > 1 is enabled.

### Bucket configs
[infworld/configs/bucket_config.py](infworld/configs/bucket_config.py) defines aspect-ratio → `(H, W)` lookup tables at multiple resolutions. The inference script picks the bucket whose key (a ratio as a string) is closest to the input image's `H/W` ratio. Changing `BUCKET_CONFIG_NAME` changes the output resolution — keep `SHIFT` in sync (see config comment).

### Prompt format
[prompts/demo.yaml](prompts/demo.yaml) — list of `[text_prompt, condition_image_path, action_json_path]` triples. Paths in the YAML are interpreted as-is (relative to `cwd`, not project root) by the script's existence checks; the demo file uses `./assets/example_case/...` and assumes you run from the project root.

### Outputs
Written to `outputs/infworld-ckpt{step}-step{NUM_SAMPLING_STEPS}-cfg{TEXT_CFG_SCALE}/` as `{task_idx:04d}_{prompt_prefix}.mp4` at 30 fps. The checkpoint step is parsed from filenames matching `checkpoint-(\d+).ckpt`; non-matching names (e.g. `infinite_world_model.ckpt`) yield step=0.
