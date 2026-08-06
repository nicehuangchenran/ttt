"""
生成固定的采样噪声文件，用于推理时的可复现性对比实验。

使用方法:
---------
python configs/noise/generate_noise.py --seed 42 --max-chunks 64 \
    --latent-shape 1 16 21 24 44 --output-dir configs/noise/352_192/noise0

参数说明:
  --seed: 随机种子
  --max-chunks: 生成多少个 chunk 的噪声，必须 >= 推理时的 max_chunks（缺文件会直接报错）
  --latent-shape: latent 的形状 [B, C, T, H, W]，H/W 是像素分辨率的 1/8
                  352x192 -> 1 16 21 24 44；627x352 -> 1 16 21 40 78
  --output-dir: 输出目录

输出结构:
  <output-dir>/
    chunk_0.pt
    chunk_1.pt
    ...
    metadata.json

在 infer.py 中使用:
  ExpConfig:
    use_fixed_noise: true
    noise_cache_dir: "configs/noise/352_192/noise0"
"""

import os
import json
import torch
import argparse
import numpy as np
import random


def setup_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def generate_noise_files(seed, max_chunks, latent_shape, output_dir):
    """生成固定噪声文件"""
    setup_seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    print(f"生成噪声文件:")
    print(f"  种子: {seed}")
    print(f"  chunk 数: {max_chunks}")
    print(f"  latent 形状: {latent_shape}")
    print(f"  输出目录: {output_dir}")
    print()

    # 直接在输出目录下生成噪声文件
    for chunk_idx in range(max_chunks):
        noise = torch.randn(*latent_shape)
        noise_path = os.path.join(output_dir, f"chunk_{chunk_idx}.pt")
        torch.save(noise, noise_path)

    print(f"✓ 生成 {max_chunks} 个 chunk 的噪声")

    # 保存元数据
    metadata = {
        "seed": seed,
        "max_chunks": max_chunks,
        "latent_shape": latent_shape,
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, indent=2, fp=f)

    print(f"\n完成！元数据已保存到: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="生成固定的采样噪声文件")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--max-chunks", type=int, default=30, help="生成多少个 chunk 的噪声")
    parser.add_argument(
        "--latent-shape",
        type=int,
        nargs=5,
        default=[1, 16, 21, 24, 48],
        help="latent 形状 [B, C, T, H, W]，PX256: 1 16 21 24 48, PX627: 1 16 21 40 78",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="输出目录",
    )

    args = parser.parse_args()

    generate_noise_files(
        seed=args.seed,
        max_chunks=args.max_chunks,
        latent_shape=args.latent_shape,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
