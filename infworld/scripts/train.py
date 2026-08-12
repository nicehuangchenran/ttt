"""
Infinite World - Flow-Matching 离线训练
========================================
训练 1.3B DiT 模型（rectified-flow），数据来自预处理输出，所有参数写在 run yaml。

用法:
----
torchrun --nproc_per_node=8 --master_port=29501 scripts/train.py --config configs/runs/infer/256/train-.yaml

临时覆盖参数：
--set train.lr=2e-5 train.epochs=2

配置文件:
  configs/infworld_config.yaml     模型结构、预训练权重路径
  configs/train_default.yaml       所有可配置项 + 默认值（字段全集）
  configs/runs/train/*.yaml        单次实验配置（只写要改的字段）

输出:
  weights/<run-name>/
    ├── steps{N}.ckpt              训练快照
    └── train_log/
        ├── train.log              rank0 主日志（**包含所有错误堆栈**）
        ├── train_rank{N}.log      其他 rank 日志
        ├── metrics.jsonl          训练指标（loss/lr/grad_norm）
        ├── config.json            完整配置快照
        └── tensorboard/           TensorBoard 事件

  目录已存在时报错退出，不覆盖旧结果。
  训练正常结束后在 run yaml 末尾追加 "#finished at <时间> UTC+8, gpu num:<world_size>,run_name:<yaml>" 标记。

查看日志:
  tail -f weights/<run-name>/train_log/train.log              # 实时查看
  grep "FATAL ERROR" weights/<run-name>/train_log/train*.log  # 查找错误
  tensorboard --logdir=weights/train-5.6/train_log/tensorboard
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
from dataclasses import dataclass, asdict, fields, field
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
from infworld.utils import storage


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
    # 日志与 ckpt 同放在 weights/<run_name>/ 下：日志进 <run_name>/<log_subdir>/
    log_subdir: str = "train_log"
    save_every_n_steps: int = 100
    # 按 epoch 间隔存 ckpt，可为小数（如 0.5 表示每半个 epoch 存一次）。设 >0 时
    # 覆盖 save_every_n_steps：按本 rank 每 epoch 的 step 数换算成 step 间隔
    # （steps_per_epoch = len(dataloader) // videos_per_step），至少 1 步。0 表示关闭。
    ckpt_every_n_epoch: float = 0.0
    log_every_n_steps: int = 1
    # 是否输出 chunk 级日志（每个 chunk 生成后打一行 epoch/step/video/chunk/loss）。
    # 默认关闭，只在 rank0 输出，避免刷屏。
    log_chunk: bool = False
    resume_from: str = ""
    seed: int = 42
    # timestep 分桶验证：每隔 N 步，在固定视频/固定 t/固定噪声上算 loss，
    # 剥离随机性以观察真实训练趋势。val_every_n_steps=0 表示关闭。
    val_every_n_steps: int = 0
    # 按 epoch 间隔验证，可为小数（如 0.5 表示每半个 epoch 验证一次）。设 >0 时
    # 覆盖 val_every_n_steps：按本 rank 每 epoch 的 step 数换算成 step 间隔
    # （steps_per_epoch = len(dataloader) // videos_per_step），至少 1 步。0 表示关闭。
    val_every_n_epoch: float = 0.0
    num_val_videos: int = 4
    num_val_buckets: int = 10
    # 数据筛选：根据 meta.json 的属性过滤样本，默认 None 表示不筛选
    filter_location: Optional[str] = None
    filter_scene: Optional[str] = None
    filter_crowd_density: Optional[str] = None
    filter_weather: Optional[str] = None
    filter_time_of_day: Optional[str] = None
    # 筛选后最多使用几个 case 训练（按 case 名排序取前 N 个），-1 表示用全部
    max_train_cases_num: int = -1
    # image_cond 最多包含最近几个 chunk（1 个 chunk = 20 latent 帧 + 首帧共 21 帧）。
    # -1 表示不限制（默认，条件区随 chunk 线性增长直至整段视频）；设为 N 后每个
    # chunk 的条件区固定为最近 20N+1 latent 帧，显存不再随 chunk 序号增长。
    # 注意：截断后的窗口不再是 latent_full 的前缀，与因果 VAE 的前缀等价性略有出入。
    cond_chunk_num: int = -1


@dataclass
class TTTConfig:
    """训练循环内的 test-time training，移植自 main.py::OnlineTrainConfig。

    open=True 时：每个 chunk 的主损失 backward 之后（最后一个 chunk 除外），在该
    chunk 的 GT latent 上用独立的 AdamW 对白名单参数做 n_train_steps 次 flow-matching
    更新，让后续 chunk 的主训练在"TTT 适应过"的权重上进行，与推理时的在线 TTT 对齐。
    推理时 x_start 是刚采样出的 latent；train.py 不做采样，GT 是最近的对应物。
    """
    open: bool = False
    n_train_steps: int = 5
    lr: float = 1e-5
    grad_clip_norm: Optional[float] = 1.0  # 只裁剪白名单参数的梯度；null 表示不裁剪
    # 白名单：只有这些层参与 TTT 更新。条目省略 block 序号（"blocks.norm3" 匹配
    # blocks.<i>.norm3.*）。默认与 main.py 的 Scheme A（~0.5M 参数）一致。
    trainable_layers: tuple = (
        "head",              # 输出头（head.head linear + head.modulation）
        "blocks.modulation", # 每个 block 的 AdaLN shift/scale（裸 nn.Parameter）
        "blocks.norm3",      # 每个 block 的 cross-attn LayerNorm affine
    )
    # 只训 blocks.<i>.*（i >= 该值）；TTT 反传在第一个可训 block 停住，省显存。
    trainable_block_start: int = 18
    # 可训 block 的上界（i <= 该值），-1 表示不设上界。用于只训中层 block：
    # 浅层管局部纹理（改动引入闪烁），中层管空间结构与物体身份（一致性所在），
    # 最后几层偏输出细化（只影响画质）。
    trainable_block_end: int = -1
    # 每个视频结束后（optimizer.step() 之前）把白名单参数恢复到视频开始时的快照。
    # 多卡下必须为 True：TTT 更新是各 rank 本地行为，不恢复会让 DDP 各副本权重发散。
    reset_between_videos: bool = True
    # TTT optimizer 类型：'adamw' 或 'sgd'。SGD 无状态，可节省 ~20% TTT 显存。
    optimizer_type: str = "adamw"
    # SGD 动量（仅当 optimizer_type='sgd' 时生效）
    sgd_momentum: float = 0.9


def parse_cli():
    import argparse
    parser = argparse.ArgumentParser(
        description="Infinite World - offline training. All run parameters come from a "
                    "run yaml under configs/runs/train/."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Run config yaml, e.g. configs/runs/train/london_sunny.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="SECTION.KEY=VALUE",
                        help="Temporary overrides on top of the yaml, e.g. --set train.lr=2e-5")
    args, _ = parser.parse_known_args()
    return args


def apply_section(cfg, values, where: str):
    """把 yaml 里一段配置写进 dataclass 实例。

    只认 dataclass 已声明的字段，拼错 key 直接报错，而不是静默用默认值；
    值按字段默认值的类型转换（bool/int/float/str/tuple），None 原样保留。
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


def build_train_config(cli) -> Tuple[TrainConfig, TTTConfig]:
    """从 run yaml 构造配置。yaml 分两段：`train:`（TrainConfig）、`ttt:`（TTTConfig）。

    未写的字段沿用 dataclass 默认值；`--set train.lr=2e-5` 可临时覆盖。
    """
    path = _resolve_path(cli.config)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Run config not found: {path}")
    raw = OmegaConf.load(path)
    if cli.set:
        raw = OmegaConf.merge(raw, OmegaConf.from_dotlist(list(cli.set)))
    raw = OmegaConf.to_container(raw, resolve=True) or {}

    sections = {"train": TrainConfig(), "ttt": TTTConfig()}
    unknown = [k for k in raw if k not in sections]
    if unknown:
        raise ValueError(f"{path}: unknown section(s) {unknown}. "
                         f"Valid sections: {', '.join(sections)}")
    name = os.path.basename(path)
    cfg = apply_section(sections["train"], raw.get("train"), f"{name}:train")
    ttt_cfg = apply_section(sections["ttt"], raw.get("ttt"), f"{name}:ttt")
    if not cfg.run_name:
        cfg.run_name = os.path.splitext(name)[0]
    return cfg, ttt_cfg


def mark_config_finished(config_path: str, world_size: int, run_name: str = "", logger=None):
    """在 run yaml 末尾追加一行完成标记，方便看出哪个配置已经跑完。

    只有 rank0 在 train_loop 正常返回后调用，所以中途报错的配置不会被标记。时间用
    东八区（机器时区不一定是 CST，所以显式指定 UTC+8，而不是用本地时间）。追加失败
    只警告不抛异常——checkpoint 已经存好了，不该因为标记写不进去而让整个任务算失败。
    """
    path = _resolve_path(config_path)
    cst = datetime.timezone(datetime.timedelta(hours=8))
    stamp = datetime.datetime.now(cst).strftime("%Y-%m-%d %H:%M:%S")
    line = f"#finished at {stamp} UTC+8, gpu num:{world_size}, run_name:{run_name}\n"
    say = logger.info if logger else print
    try:
        # yaml 末尾若没有换行，直接 append 会把标记粘到最后一个值上
        # （epochs: 1 会变成字符串 "1#finished ..."），所以先补一个换行。
        with open(path) as f:
            existing = f.read()
        needs_newline = bool(existing) and not existing.endswith("\n")
        with open(path, "a") as f:
            if needs_newline:
                f.write("\n")
            f.write(line)
        say(f"Marked finished in {config_path}: {line.strip()}")
    except OSError as e:
        say(f"WARNING: could not mark {config_path} as finished: {e}")


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


class StderrTee:
    """将 stderr 同时输出到原 stderr 和文件。

    用于捕获 NCCL/PyTorch C++ 层的错误（这些错误不经过 Python logging）。
    """
    def __init__(self, file_path: str, original_stderr):
        self.file = open(file_path, 'a', buffering=1)  # 行缓冲
        self.original_stderr = original_stderr
        self.file_path = file_path

    def write(self, message):
        self.original_stderr.write(message)
        self.file.write(message)
        self.file.flush()  # 立即刷新到磁盘

    def flush(self):
        self.original_stderr.flush()
        self.file.flush()

    def close(self):
        if self.file and not self.file.closed:
            self.file.close()


class TrainLogger:
    def __init__(self, log_dir: str, rank: int, world_size: int):
        """log_dir: 本次运行的日志目录，即 weights/<run_name>/<log_subdir>/。"""
        self.rank = rank
        self.world_size = world_size
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "train.log")
        self.metrics_file = os.path.join(self.log_dir, "metrics.jsonl")
        self.config_file = os.path.join(self.log_dir, "config.json")
        # 多卡时，非 rank0 的日志写到单独文件
        if self.rank != 0:
            self.log_file = os.path.join(self.log_dir, f"train_rank{self.rank}.log")

        # ⭐ 重定向 stderr 到日志文件（捕获 NCCL/C++ 错误）
        # stderr 文件与 Python 日志同名，这样所有输出都在一个文件里
        self.stderr_tee = None
        self.original_stderr = sys.stderr
        try:
            self.stderr_tee = StderrTee(self.log_file, sys.stderr)
            sys.stderr = self.stderr_tee
        except Exception as e:
            print(f"WARNING: Failed to redirect stderr: {e}", file=sys.stderr)

        # TensorBoard writer: 只有 rank0 创建
        self.writer = None
        if self.rank == 0:
            tensorboard_dir = os.path.join(self.log_dir, "tensorboard")
            self.writer = SummaryWriter(log_dir=tensorboard_dir)
            self.info(f"TensorBoard logging to: {tensorboard_dir}")
            self.info(f"Run: tensorboard --logdir={tensorboard_dir}")

    def dump_config(self, cfg: TrainConfig, ttt_cfg: TTTConfig,
                    runtime: RuntimeContext,
                    num_train_cases: Optional[int] = None,
                    run_config_path: Optional[str] = None):
        """把本次运行的全部参数写到 config.json，供事后复现和跑间对比。

        除 TrainConfig/TTTConfig 的字段外，还记录 world_size 和实际生效的数据量——
        这几项决定了 metrics.jsonl 该怎么解读。
        """
        if self.rank != 0:
            return
        record = {
            "run_name": cfg.run_name,
            "run_config_path": run_config_path,
            "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "world_size": runtime.world_size,
            "num_train_cases": num_train_cases,
            "train_config": asdict(cfg),
            "ttt_config": asdict(ttt_cfg),
        }
        with open(self.config_file, 'w') as f:
            json.dump(record, f, indent=2, ensure_ascii=False, sort_keys=True)
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
                    elif key.startswith("ttt_"):
                        self.writer.add_scalar(f"ttt/{key}", value, step)
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
        self.weights_dir = os.path.join(_resolve_path(weights_dir), run_name)
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
        path = os.path.join(self.weights_dir, f"steps{global_step}.ckpt")
        torch.save(ckpt, path)
        print(f"[SAVE] Checkpoint saved: {path}")
        # 3.5G 的 ckpt 交给后台进程传 S3，训练不等它；退出前 wait_uploads 兜底
        storage.upload_async(path, is_dir=False)

    def load_for_resume(self, path: str, dit, optimizer, lr_sched) -> int:
        path = storage.ensure_local(path, is_dir=False)
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
                 max_train_cases_num: int = -1,
                 rank: int = 0):
        self.data_dir = _resolve_path(data_dir)
        data_dir = self.data_dir
        self.max_chunks = max_chunks_per_video
        self.cases = []

        # 统计过滤信息
        total_cases = 0
        filtered_out = 0

        # 目录清单取「本地 ∪ S3」，本地只缓存了部分 case 时也能看到全量。
        # 筛选只需要 meta.json（几 KB），所以扫描阶段只按需拉 meta.json，
        # 真正的 .pt（几百 MB/case）留到 __getitem__ 里再拉。
        for case_name in storage.list_dir(data_dir):
            case_path = os.path.join(data_dir, case_name)
            # list_dir 返回目录 ∪ 文件，数据目录里还混着 preprocess.log、logs/ 这类
            # 预处理产物。只认 case 前缀：否则会对着普通文件拼出 <file>/meta.json，
            # ensure_local 再 makedirs 就撞 FileExistsError。
            if not case_name.startswith("case"):
                continue
            meta_path = os.path.join(case_path, "meta.json")
            if not os.path.exists(meta_path):
                storage.ensure_local(meta_path, is_dir=False, check=False)
            if os.path.exists(meta_path):
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

        # 截断到最多 max_train_cases_num 个 case（在筛选之后）
        if max_train_cases_num > 0 and len(self.cases) > max_train_cases_num:
            self.cases = self.cases[:max_train_cases_num]
            if rank == 0:
                print(f"[Dataset] Truncated to first {len(self.cases)} cases "
                      f"(max_train_cases_num={max_train_cases_num})")
    
    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx: int) -> VideoSample:
        case_name = self.cases[idx]
        # 本地缺哪个 .pt 就从 S3 拉哪个（dataloader worker 里同步下载，随后即读）
        case_dir = storage.ensure_files(
            os.path.join(self.data_dir, case_name),
            ("latent_full.pt", "target_latent.pt", "text_emb.pt", "actions.pt"),
        )
        # 所有张量保持在 CPU，在 make_chunk_batch 中按需移到 GPU
        latent_full = torch.load(os.path.join(case_dir, "latent_full.pt"), map_location='cpu')
        target_latent = torch.load(os.path.join(case_dir, "target_latent.pt"), map_location='cpu')
        text_emb = torch.load(os.path.join(case_dir, "text_emb.pt"), map_location='cpu')
        actions = torch.load(os.path.join(case_dir, "actions.pt"), map_location='cpu')
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
        max_train_cases_num=cfg.max_train_cases_num,
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
    # TTT 白名单参数（name -> Parameter），ttt.open=False 时为空 dict
    ttt_params: dict = field(default_factory=dict)


def _resolve_path(path):
    """相对路径解析：数据/产物目录（dataset/ preprocessed/ checkpoints/ weights/ ...）
    落到 storage.LOCAL_ROOT 并与 S3 同步，代码与 yaml 仍按 PROJECT_ROOT 解析。"""
    return storage.resolve(path)


def build_dit(cfg: TrainConfig, ttt_cfg: TTTConfig, runtime: RuntimeContext) -> TrainModels:
    config_path = _resolve_path(cfg.model_config_path)
    model_args = OmegaConf.load(config_path)
    from infworld.models.dit_model import WanModel
    master_dtype = getattr(torch, cfg.master_dtype)
    dit = WanModel(
        out_channels=16, caption_channels=4096, model_max_length=512,
        enable_context_parallel=False, **model_args.model_cfg
    ).to(master_dtype)
    # 如果设置了 resume_from，跳过预训练权重加载（稍后会被 resume checkpoint 覆盖）
    if not cfg.resume_from:
        # 只让 rank0 下载：8 个 rank 同时 aws s3 cp 到同一路径会互相覆盖到半截文件。
        # 其余 rank 在 barrier 处等下载完成后再读本地文件。
        checkpoint_path = _resolve_path(cfg.init_checkpoint)
        if runtime.global_rank == 0:
            storage.ensure_local(checkpoint_path, is_dir=False)
            print(f"[LOAD] DiT checkpoint: {checkpoint_path}")
        if runtime.world_size > 1:
            dist.barrier()
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

    # TTT 白名单参数：只在 ttt.open=True 时选择（否则留空 dict，避免误用）
    ttt_params = {}
    if ttt_cfg.open:
        ttt_params = select_ttt_params(dit, ttt_cfg.trainable_layers,
                                       ttt_cfg.trainable_block_start,
                                       ttt_cfg.trainable_block_end, runtime.global_rank)

    if runtime.world_size > 1:
        dit = nn.parallel.DistributedDataParallel(
            dit, device_ids=[runtime.local_rank], output_device=runtime.local_rank,
            find_unused_parameters=False,
        )
    dit_raw = dit.module if isinstance(dit, nn.parallel.DistributedDataParallel) else dit
    return TrainModels(dit=dit, dit_raw=dit_raw, trainable_params=trainable_params,
                       ttt_params=ttt_params)


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


def select_ttt_params(dit, trainable_layers: tuple, trainable_block_start: int,
                      trainable_block_end: int = -1, rank: int = 0) -> dict:
    """选择 TTT 白名单参数（name -> Parameter dict），与 main.py 的逻辑完全一致。

    白名单条目省略 block 序号（"blocks.norm3" 匹配 blocks.<i>.norm3.*）；blocks.<i>.*
    额外要求 trainable_block_start <= i <= trainable_block_end（后者为 -1 时不设上界）。
    与 main.py 不同：这里不改 requires_grad —— 主训练仍训练全部参数，TTT 期间的临时
    冻结在 ttt_adapt_on_chunk 内完成。
    """
    def canonical(name):
        parts = name.split(".")
        if parts[0] == "blocks":
            parts = ["blocks"] + parts[2:]  # 去掉 block 序号
        return ".".join(parts)

    def block_idx(name):
        parts = name.split(".")
        return int(parts[1]) if parts[0] == "blocks" else None

    ttt_params = {}
    for n, p in dit.named_parameters():
        key = canonical(n)
        idx = block_idx(n)
        whitelisted = any(key == layer or key.startswith(layer + ".") for layer in trainable_layers)
        in_range = idx is None or (
            idx >= trainable_block_start
            and (trainable_block_end < 0 or idx <= trainable_block_end))
        if whitelisted and in_range:
            ttt_params[n] = p

    if rank == 0:
        count = sum(p.numel() for p in ttt_params.values())
        upper = "inf" if trainable_block_end < 0 else str(trainable_block_end)
        print(f"[TTT] Selected {len(ttt_params)} params ({count:,} elements) from "
              f"layers {trainable_layers}, blocks {trainable_block_start}..{upper}")
    return ttt_params


@dataclass
class ChunkBatch:
    x_start: torch.Tensor
    image_cond: torch.Tensor
    move: torch.Tensor
    view: torch.Tensor
    y: torch.Tensor
    y_mask: torch.Tensor


def make_chunk_batch(video: VideoSample, k: int, device,
                     cond_chunk_num: int = -1) -> ChunkBatch:
    """从 CPU 上的 VideoSample 切片并移到 GPU，按需加载以节省显存。"""
    # video 的所有张量都在 CPU 上，这里按需切片并移到 GPU
    x_start = video.target_latent[k].unsqueeze(0).to(device)
    T_in = 20 * k + 1
    # cond_chunk_num=N 时条件区只保留最近 N 个 chunk（20N+1 latent 帧，含窗口首帧），
    # 取 latent_full[:, :T_in] 的尾部窗口；-1 表示不限制，条件区为完整前缀。
    if cond_chunk_num > 0:
        T_cond = min(T_in, 20 * cond_chunk_num + 1)
        image_cond = video.latent_full[:, T_in - T_cond:T_in].unsqueeze(0).to(device)
    else:
        image_cond = video.latent_full[:, :T_in].unsqueeze(0).to(device)
    start_frame = 80 * k
    end_frame = start_frame + 81
    move_seg = video.move[start_frame:end_frame]
    view_seg = video.view[start_frame:end_frame]
    if len(move_seg) < 81:
        pad_len = 81 - len(move_seg)
        move_seg = torch.cat([move_seg, torch.zeros(pad_len, dtype=torch.long)])
        view_seg = torch.cat([view_seg, torch.zeros(pad_len, dtype=torch.long)])
    return ChunkBatch(
        x_start=x_start, image_cond=image_cond,
        move=move_seg.unsqueeze(0).to(device), view=view_seg.unsqueeze(0).to(device),
        y=video.y.to(device), y_mask=video.y_mask.to(device),
    )


def sample_timestep(batch_size: int, cfg: TrainConfig, device) -> torch.Tensor:
    t = torch.rand((batch_size,), device=device) * cfg.num_timesteps
    t = timestep_transform(t, shift=cfg.shift, num_timesteps=cfg.num_timesteps)
    return t


def flow_matching_loss(dit: nn.Module, chunk: ChunkBatch, cfg: TrainConfig) -> torch.Tensor:
    """一次 flow-matching 损失前向。dit 传 models.dit（主训练，走 DDP 同步）或
    models.dit_raw（TTT，绕过 DDP —— TTT 是各 rank 本地行为，不做梯度同步）。"""
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
        pred = dit(
            x_t, t, y=chunk.y, y_mask=chunk.y_mask, x_ignore_mask=None,
            image_cond=chunk.image_cond, move=chunk.move, view=chunk.view,
        )
    pred = pred[:, :, -21:]
    loss = ((pred.float() - target.float()) ** 2).mean()
    return loss


def snapshot_ttt_params(ttt_params: dict) -> dict:
    """快照白名单参数当前值（视频开始时各 rank 权重一致，视频结束后恢复到这里）。"""
    return {n: p.detach().clone() for n, p in ttt_params.items()}


def restore_ttt_params(ttt_params: dict, snapshot: dict):
    with torch.no_grad():
        for n, p in ttt_params.items():
            p.data.copy_(snapshot[n])


def ttt_adapt_on_chunk(models: TrainModels, chunk: ChunkBatch,
                       cfg: TrainConfig, ttt_cfg: TTTConfig) -> list:
    """在刚算完主损失的 chunk 上做 n_train_steps 次 TTT 更新，
    对齐 main.py::online_train_on_chunk / _online_train_step。

    x_start 用该 chunk 的 GT latent（chunk.x_start）：推理时是刚采样出的 latent，
    离线训练不做采样，GT 是最近的对应物。

    与主训练的梯度累积互不污染：
      - 非白名单参数临时 requires_grad_(False)，TTT backward 不写它们的 .grad，
        反传在第一个可训 block 就停住（省显存省算力）；
      - 白名单参数上已累积的主梯度先摘下（保留张量引用），TTT 结束后原样放回；
      - 前向走 dit_raw，绕过 DDP —— 各 rank 在自己的视频上本地适应，不同步。
    每个 chunk 新建 optimizer，Adam/SGD 动量不跨 chunk 累积（与 main.py 一致）。
    """
    dit_raw = models.dit_raw
    # 临时冻结非白名单参数（记录原 flag，latent_encoder 可能本来就是 False）
    saved_flags = {}
    for n, p in dit_raw.named_parameters():
        if n not in models.ttt_params:
            saved_flags[n] = p.requires_grad
            p.requires_grad_(False)
    # 摘下白名单参数已累积的主梯度，TTT 期间它们的 .grad 只属于 TTT
    saved_grads = {n: p.grad for n, p in models.ttt_params.items()}
    for p in models.ttt_params.values():
        p.grad = None

    params = list(models.ttt_params.values())
    # 根据配置选择 optimizer：SGD 无状态，节省显存
    if ttt_cfg.optimizer_type == "sgd":
        ttt_optimizer = torch.optim.SGD(params, lr=ttt_cfg.lr, momentum=ttt_cfg.sgd_momentum)
    else:
        ttt_optimizer = torch.optim.AdamW(params, lr=ttt_cfg.lr)
    losses = []
    try:
        for _ in range(ttt_cfg.n_train_steps):
            ttt_optimizer.zero_grad(set_to_none=True)
            loss = flow_matching_loss(dit_raw, chunk, cfg)
            loss.backward()
            if ttt_cfg.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(params, max_norm=ttt_cfg.grad_clip_norm)
            ttt_optimizer.step()
            losses.append(loss.item())
            del loss
    finally:
        # 恢复所有参数的原始 requires_grad 状态
        for n, p in dit_raw.named_parameters():
            if n in saved_flags:
                p.requires_grad_(saved_flags[n])
        # 恢复白名单参数的主梯度
        for n, p in models.ttt_params.items():
            p.grad = saved_grads[n]
    return losses


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
            chunk = make_chunk_batch(video, 0, device, cfg.cond_chunk_num)  # 固定用第 0 个 chunk
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
    logger: Optional[TrainLogger] = None, epoch: int = 0, cur_step: int = 0,
) -> dict:
    K = video.num_chunks
    losses = []
    ttt_losses = []
    # TTT：视频开始时快照白名单参数（此时各 rank 权重一致），视频结束后恢复
    ttt_snapshot = None
    if ttt_cfg.open and ttt_cfg.reset_between_videos:
        ttt_snapshot = snapshot_ttt_params(models.ttt_params)
    for k in range(K):
        t_chunk = time.time()
        chunk = make_chunk_batch(video, k, device, cfg.cond_chunk_num)
        loss = flow_matching_loss(models.dit, chunk, cfg) / K
        losses.append(loss.item())
        chunk_loss = loss.item() * K  # 已被 /K，乘回还原为单 chunk loss
        if isinstance(models.dit, nn.parallel.DistributedDataParallel):
            if k < K - 1:
                with models.dit.no_sync():
                    loss.backward()
            else:
                loss.backward()
        else:
            loss.backward()
        del loss
        # 对齐推理（main.py：chunk_idx < num_chunks - 1）：最后一个 chunk 后不做 TTT。
        # 顺序也与推理一致：chunk k 的主损失（≈生成 chunk k）之后、chunk k+1 之前适应。
        if ttt_cfg.open and k < K - 1:
            ttt_losses.append(ttt_adapt_on_chunk(models, chunk, cfg, ttt_cfg))
            # TTT 后立即清理显存
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        del chunk
        # 每个 chunk 处理完立即清理显存（backward 后的中间激活已不需要）
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if cfg.log_chunk and logger is not None:
            # is_step_boundary 时本视频结束后 global_step 才 +1，所以此处显示的 step
            # 是这些 chunk 梯度将并入的那一步。耗时含前向 + 反向 + 该 chunk 的 TTT。
            shown_step = cur_step + 1 if is_step_boundary else cur_step
            logger.info(
                f"[Epoch {epoch + 1}/{cfg.epochs}] [Step {shown_step}] "
                f"[Video {video.name}] Chunk {k + 1}/{K} loss={chunk_loss:.6f} "
                f"time={time.time() - t_chunk:.2f}s"
            )
    # 必须在 optimizer.step() 之前恢复：主 step 要作用在各 rank 一致的原始权重上，
    # TTT 的效果只体现在"主梯度是在适应过的权重上算出来的"这一点（一阶近似）。
    if ttt_snapshot is not None:
        restore_ttt_params(models.ttt_params, ttt_snapshot)
    grad_norm = 0.0
    if is_step_boundary:
        # optimizer.step() 前清理显存，降低峰值（AdamW 会分配动量缓冲）
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            models.trainable_params, cfg.max_grad_norm).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        # optimizer.step() 后也清理，释放动量更新产生的临时张量
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    stats = {"loss": sum(losses), "per_chunk_loss": losses, "grad_norm": grad_norm}
    if ttt_losses:
        flat = [v for per_chunk in ttt_losses for v in per_chunk]
        stats["ttt_loss"] = sum(flat) / len(flat)
        stats["ttt_loss_first"] = ttt_losses[0][0]
        stats["ttt_loss_last"] = ttt_losses[-1][-1]
    return stats


def train_loop(
    models: TrainModels, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
    lr_sched: Optional[torch.optim.lr_scheduler._LRScheduler],
    logger: TrainLogger, checkpointer: Checkpointer,
    cfg: TrainConfig, ttt_cfg: TTTConfig, runtime: RuntimeContext, start_step: int,
    val_videos: Optional[list] = None,
):
    global_step = start_step
    video_count = 0
    # ckpt 间隔：ckpt_every_n_epoch>0 时按 epoch 换算成 step 间隔（可为小数 epoch），
    # 覆盖 save_every_n_steps；否则沿用 save_every_n_steps。
    save_interval = cfg.save_every_n_steps
    if cfg.ckpt_every_n_epoch > 0:
        steps_per_epoch = max(1, len(dataloader) // cfg.videos_per_step)
        save_interval = max(1, round(steps_per_epoch * cfg.ckpt_every_n_epoch))
        logger.info(f"Checkpoint interval: every {cfg.ckpt_every_n_epoch} epoch(s) = "
                    f"{save_interval} step(s) (steps_per_epoch={steps_per_epoch})")
    # 验证间隔：val_every_n_epoch>0 时按 epoch 换算成 step 间隔（可为小数 epoch），
    # 覆盖 val_every_n_steps；否则沿用 val_every_n_steps。
    val_interval = cfg.val_every_n_steps
    if cfg.val_every_n_epoch > 0:
        steps_per_epoch = max(1, len(dataloader) // cfg.videos_per_step)
        val_interval = max(1, round(steps_per_epoch * cfg.val_every_n_epoch))
        logger.info(f"Validation interval: every {cfg.val_every_n_epoch} epoch(s) = "
                    f"{val_interval} step(s) (steps_per_epoch={steps_per_epoch})")
    for epoch in range(cfg.epochs):
        logger.info(f"=== Epoch {epoch + 1}/{cfg.epochs} ===")
        if hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
        for video in tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=(runtime.global_rank != 0)):
            video_count += 1
            is_step_boundary = (video_count % cfg.videos_per_step == 0)
            t_start = time.time()
            stats = train_one_video(
                models, video, optimizer, cfg, ttt_cfg, runtime.device, is_step_boundary,
                logger=logger, epoch=epoch, cur_step=global_step,
            )
            t_video = time.time() - t_start
            if is_step_boundary:
                global_step += 1
                if global_step % cfg.log_every_n_steps == 0:
                    ttt_kv = {k: v for k, v in stats.items() if k.startswith("ttt_")}
                    logger.metrics(
                        step=global_step, loss=stats["loss"], grad_norm=stats["grad_norm"],
                        lr=optimizer.param_groups[0]["lr"], video=video.name,
                        num_chunks=video.num_chunks, step_second=t_video, **ttt_kv,
                    )
                if val_interval > 0 and global_step % val_interval == 0 \
                        and val_videos and runtime.global_rank == 0:
                    val_stats = validate_by_timestep(models, val_videos, cfg, runtime.device)
                    logger.metrics(step=global_step, **val_stats)
                if global_step % save_interval == 0:
                    checkpointer.save(models.dit, optimizer, lr_sched, global_step, cfg)
                if lr_sched:
                    lr_sched.step()
    logger.info("Training complete, saving final checkpoint...")
    checkpointer.save(models.dit, optimizer, lr_sched, global_step, cfg)


def main():
    import traceback

    cli = parse_cli()
    cfg, ttt_cfg = build_train_config(cli)
    runtime = setup_runtime(cfg.seed)

    # 早期错误（logger 创建前）需要降级到 print + 抛出
    logger = None
    log_dir = None
    try:
        # ckpt 和日志同放一处：weights/<run_name>/{step*.ckpt, <log_subdir>/}
        # 目标目录已存在就报错退出，绝不覆盖/追加到旧结果里（对齐 main.py 的防覆盖行为）。
        # 例外：设了 resume_from 是恢复训练，目录本就该存在，放行。
        # 只让 rank0 判定目录是否已存在，再广播给所有 rank，大家基于同一结果一起退出或一起继续。
        # 若各 rank 自己判定，rank0 会领先跑到 Checkpointer 把目录 makedirs 出来，其余 rank 随后
        # 命中 FileExistsError —— 即便启动前目录不存在，也会因这个竞态而误报"目录已存在"。
        # 本地 nvme 可能是新机器（空目录），所以 S3 上已有同名 run 也算"已存在"，
        # 否则两台机器用同一 run_name 会在 S3 上互相覆盖 ckpt。
        run_dir = os.path.join(_resolve_path(cfg.weights_dir), cfg.run_name)
        if runtime.global_rank == 0:
            exists = (os.path.exists(run_dir) or storage.s3_exists(run_dir)) and not cfg.resume_from
        else:
            exists = False
        if runtime.world_size > 1:
            flag = [exists]
            dist.broadcast_object_list(flag, src=0)
            exists = flag[0]
        if exists:
            raise FileExistsError(
                f"Output directory already exists: {run_dir}\n"
                f"重复运行会覆盖 checkpoint、把日志追加到旧文件里，造成混乱。请改用新的 "
                f"run_name（--set train.run_name=xxx），或删除该目录后重试；"
                f"若是要恢复训练，请设置 train.resume_from。"
            )
        checkpointer = Checkpointer(cfg.weights_dir, cfg.run_name, runtime.global_rank)
        log_dir = os.path.join(checkpointer.weights_dir, cfg.log_subdir)
        logger = TrainLogger(log_dir, runtime.global_rank, runtime.world_size)
        # 日志/metrics/tensorboard 每 5 分钟同步一次，训练中途也能在 S3 上看进度
        if runtime.global_rank == 0:
            storage.start_periodic_sync(log_dir, interval=300)
        logger.info(storage.describe())
        logger.info("=" * 70)
        logger.info(f"Infinite World - Training - {cfg.run_name}")
        logger.info("=" * 70)
        logger.info(f"Run config: {cli.config}")
        logger.info(f"Data: {cfg.data_dir}")
        logger.info(f"Device: {runtime.device}, World Size: {runtime.world_size}")
        logger.info("=" * 70)
        # TTT 前置校验：多卡下 TTT 更新是各 rank 本地行为，不恢复快照会让 DDP 各副本权重发散
        if ttt_cfg.open:
            if runtime.world_size > 1 and not ttt_cfg.reset_between_videos:
                raise ValueError(
                    "ttt.reset_between_videos=false is unsupported with world_size > 1: "
                    "TTT updates are rank-local; without per-video restore the DDP "
                    "replicas' weights diverge.")
            logger.info(f"TTT enabled: n_train_steps={ttt_cfg.n_train_steps}, lr={ttt_cfg.lr}, "
                        f"layers={ttt_cfg.trainable_layers}, "
                        f"blocks={ttt_cfg.trainable_block_start}.."
                        f"{'inf' if ttt_cfg.trainable_block_end < 0 else ttt_cfg.trainable_block_end}, "
                        f"reset_between_videos={ttt_cfg.reset_between_videos}")
        dataloader = build_dataloader(cfg, runtime)
        logger.dump_config(cfg, ttt_cfg, runtime, len(dataloader.dataset), cli.config)
        models = build_dit(cfg, ttt_cfg, runtime)
        if ttt_cfg.open and not models.ttt_params:
            raise ValueError(f"ttt.trainable_layers={ttt_cfg.trainable_layers} matched no "
                             f"parameters (blocks {ttt_cfg.trainable_block_start}.."
                             f"{ttt_cfg.trainable_block_end})")
        # fused=True: 逐参数原地更新，避免默认 foreach 路径在 step 内部分配
        # 全尺寸临时张量（~一份 fp32 参数大小的瞬时峰值，1.4B 全参训练时会 OOM）
        optimizer = torch.optim.AdamW(
            models.trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas,
            fused=True,
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
        if (cfg.val_every_n_steps > 0 or cfg.val_every_n_epoch > 0) and runtime.global_rank == 0:
            val_dataset = PreprocessedVideoDataset(
                cfg.data_dir, cfg.max_chunks_per_video,
                filter_location=cfg.filter_location,
                filter_scene=cfg.filter_scene,
                filter_crowd_density=cfg.filter_crowd_density,
                filter_weather=cfg.filter_weather,
                filter_time_of_day=cfg.filter_time_of_day,
                max_train_cases_num=cfg.max_train_cases_num,
            )
            n_val = min(cfg.num_val_videos, len(val_dataset))
            val_videos = [val_dataset[i] for i in range(n_val)]
            logger.info(f"Validation: {n_val} videos x {cfg.num_val_buckets} timestep buckets")
        logger.info("Starting training loop...")
        train_loop(models, dataloader, optimizer, lr_sched, logger, checkpointer,
                  cfg, ttt_cfg, runtime, start_step, val_videos)
        logger.info("Training finished!")
        # 全部 rank 跑完后才在 run yaml 上盖完成标记（任何 rank 报错都不会走到这里）。
        if runtime.world_size > 1:
            dist.barrier()
        if runtime.global_rank == 0:
            mark_config_finished(cli.config, runtime.world_size, cfg.run_name, logger)
        logger.close()
        # 等最后一个 ckpt 传完再退出，并把日志目录做最终同步
        if runtime.global_rank == 0:
            storage.stop_periodic_sync(log_dir)
        storage.wait_uploads()

    except Exception as e:
        # 捕获所有异常，写入日志文件（如果 logger 已创建）并重新抛出
        error_msg = f"FATAL ERROR in rank {runtime.global_rank}:\n{traceback.format_exc()}"
        if logger is not None:
            # info_all_ranks 会强制所有 rank 打印并写入各自的日志文件
            logger.info_all_ranks("=" * 70)
            logger.info_all_ranks(error_msg)
            logger.info_all_ranks("=" * 70)
            logger.close()
            # 崩溃时也把日志推上 S3，否则报错原因只留在本机 nvme 上
            try:
                if log_dir is not None and runtime.global_rank == 0:
                    storage.stop_periodic_sync(log_dir)
            except Exception:
                pass
        else:
            # logger 还没创建（极早期错误），降级到 print
            print(error_msg, file=sys.stderr)
        # 重新抛出，让进程以非零状态退出（nohup/torchrun 能检测到失败）
        raise


if __name__ == "__main__":
    main()
