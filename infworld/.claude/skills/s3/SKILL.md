---
name: s3
description: infworld 项目的大文件路径映射（本地 nvme 与 S3 一一对应，挂载点是 /mnt/nvme 或 /mnt/local_nvme）。当需要在 S3 与本地之间上传/下载/同步 dataset、preprocessed、checkpoints、weights、videos、outputs，或需要拼 S3 路径、排查 aws s3 报错时使用。
---

# 大文件路径映射

代码与 yaml（`configs/`、`scripts/`、`infworld/`）留在项目目录走 git。大文件不放项目目录，而是本地实体与 S3 权威副本用**同名相对路径**一一对应：

| 角色 | 根路径 |
| --- | --- |
| 本地 | `/mnt/nvme/chenran/ttt/infworld/` |
| S3 | `s3://s3-us-west2-default/archives/chenran/ttt/infworld/` |

走这套映射的顶层目录（`storage.py::DATA_PREFIXES`）：
`dataset` `preprocessed` `checkpoints` `weights` `videos` `outputs`

其余相对路径按项目根（`storage.py::PROJECT_ROOT`，即仓库所在目录，因机器而异）解析。

### 本地根：因机器而异，storage.py 会自动探测

`storage.py` 的 `LOCAL_ROOT` 默认值按机器自动选（`INFWORLD_LOCAL_ROOT` 优先）：`/mnt/nvme` 挂载存在 → `/mnt/nvme/chenran/ttt/infworld`；否则 `/mnt/local_nvme` 存在 → `/mnt/local_nvme/chenran/ttt/infworld`；两个 nvme 都没有 → 退回项目根 `PROJECT_ROOT`（仓库所在目录，如新机器的 `/data/chenran/project/ttt/infworld`）。所以代码侧一般不用管，也别改默认值；要放到别的盘再 `export INFWORLD_LOCAL_ROOT=...` 覆盖。

**手写 aws 命令时同样先确认本地根**（下面惯用命令里的 `$L`）：

```bash
if [ -d /mnt/nvme ]; then       L=/mnt/nvme/chenran/ttt/infworld
elif [ -d /mnt/local_nvme ]; then L=/mnt/local_nvme/chenran/ttt/infworld
else                            L=<仓库根>   # 项目就在本地盘上（如 /data/.../infworld）时用它
fi
```

⚠️ 退回项目根只在**仓库位于本地盘**时才合理。老机器上仓库在 EFS（`/mnt/efs/.../infworld/`，慢），别把大文件放那；此时应有 nvme 挂载或显式设 `INFWORLD_LOCAL_ROOT`。当 `LOCAL_ROOT == PROJECT_ROOT` 时大文件落在仓库工作树内，但六个 data 目录（`dataset/preprocessed/checkpoints/weights/videos/outputs`）都已 gitignore，`git status` 不受影响。

## 换算

相对路径直接拼在两个根后面，中间不加别的层级：

```
videos/gt-dl3dv-2/case_000000_combined.mp4
  本地  /mnt/nvme/chenran/ttt/infworld/videos/gt-dl3dv-2/case_000000_combined.mp4
  S3    s3://s3-us-west2-default/archives/chenran/ttt/infworld/videos/gt-dl3dv-2/case_000000_combined.mp4
```

## 凭证：default 失败就加 `--profile chenran`

裸 `aws s3 ...`（default profile）在有些机器上没权限，报 `AccessDenied`（`ListObjectsV2`/`GetObject` 等）。先试 default，一旦 AccessDenied 就给命令加 `--profile chenran`（`~/.aws/config` 里已配好该 profile，region us-west-2）。代码侧走 storage.py 时用 `export AWS_PROFILE=chenran` 让整条链路都带上（别用 `INFWORLD_AWS_CLI="aws --profile chenran"`——storage.py 把 `_AWS` 当作 argv 单个元素，带空格会被当成一个可执行名报 FileNotFoundError）。

## 惯用命令

```bash
L=/mnt/nvme/chenran/ttt/infworld        # /mnt/nvme 不存在时换成 /mnt/local_nvme，见上一节
B=s3://s3-us-west2-default/archives/chenran/ttt/infworld
# 下面命令 AccessDenied 时统一追加 --profile chenran

# 下载（只传有差异的文件，可反复重跑）
aws s3 sync "$B/videos/gt-dl3dv-2/" "$L/videos/gt-dl3dv-2/" --only-show-errors

# 上传
aws s3 sync "$L/weights/train-10-sft/" "$B/weights/train-10-sft/" --only-show-errors

# S3 内部复制大文件：必须加 --copy-props none
aws s3 cp "$B/dataset/dl3dv-2/case000000/video.mp4" \
          "$B/videos/gt-dl3dv-2/case_000000_combined.mp4" --copy-props none
```

## 坑

- **S3→S3 复制超过 multipart 阈值（8MB）的文件要加 `--copy-props none`。** 否则 CLI 会去读源对象的 tag 复制属性，当前 role 没有 `s3:GetObjectTagging`，报 `AccessDenied`。小文件走单次 copy 不触发，所以同一批里小文件成功、大文件失败是这个原因，不是权限配错了。
- **`aws s3 ls | head` 会打 `BrokenPipeError`**，无害，但会污染输出；要计数用 `| wc -l`。
- **批量 `aws s3 cp` 后台并发时，`wait` 只返回最后一个 job 的退出码**，不能用它判断整批成功。可靠做法是拷完用 `aws s3 ls` 对比文件数和字节大小。
- 校验文件名里的数字不要用 `gsub(/[^0-9]/,"",f)`——`case_000004_combined.mp4` 的 `mp4` 里那个 `4` 会被算进去。用 `sed -E 's/case_0*([0-9]+)_combined\.mp4/\1/'`。

## 代码里怎么用

`infworld/utils/storage.py` 已封装这套映射，不要在业务代码里手写路径拼接：

- `resolve(path)` — 相对路径映射到 `LOCAL_ROOT`（非 `DATA_PREFIXES` 的按项目根解析）
- `s3_uri(local_path)` / `rel_to_local_root(abs_path)` — 反向换算
- `ensure_local(path)` / `ensure_files(dir, names)` — 本地缺失时从 S3 拉
- `list_dir(dir)` / `s3_exists(local_path)` — 不下载就查 S3
- `upload_async(path)` — 写完本地后台上传，不阻塞训练/推理（单 worker 串行，避免同一 key 并发写出半新半旧的对象）；退出前用 `wait_uploads()` 收尾
- `start_periodic_sync(dir, interval=300)` / `stop_periodic_sync()` — 长跑任务周期上传

环境变量：`INFWORLD_LOCAL_ROOT`、`INFWORLD_S3_ROOT` 覆盖两个根；`INFWORLD_S3_DISABLE=1` 退化成纯本地模式；`INFWORLD_AWS_CLI` 指定 aws 可执行文件。
