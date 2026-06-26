import os
from huggingface_hub import snapshot_download

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_XET_HIGH_PERFORMANCE"] = Trued

# 定义你想存放数据集的本地相对路径或绝对路径
local_dir = "./dataset"

print("正在从云端下载 meituan-longcat/WBench 数据集...")

# 使用 snapshot_download 下载整个数据集仓库
path = snapshot_download(
    repo_id="meituan-longcat/WBench",
    repo_type="dataset",           # 必须指定为 dataset
    local_dir=local_dir,
)
print(f"✅ 数据集下载成功！已存放在: {path}")s