"""大文件的本地缓存 + S3 同步。

大文件目录（DATA_PREFIXES）不再放在项目目录下，而是：
  - 本地实体在 LOCAL_ROOT（/mnt/nvme/...）下的同名相对路径；
  - 权威副本在 S3_ROOT（s3://.../infworld/）下的同名相对路径。

读：resolve() 把相对路径映射到 LOCAL_ROOT，ensure_local() 在本地缺失时从 S3 拉。
写：照常写本地，写完调 upload_async() 让后台进程传 S3，不阻塞训练/推理。

代码与 yaml（configs/、scripts/、infworld/）仍留在项目目录，走 git，不受影响。
环境变量 INFWORLD_LOCAL_ROOT / INFWORLD_S3_ROOT 可覆盖两个根；
INFWORLD_S3_DISABLE=1 则完全退化成纯本地模式（不下载、不上传）。
"""

import os
import queue
import subprocess
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOCAL_ROOT = os.environ.get("INFWORLD_LOCAL_ROOT", "/mnt/nvme/chenran/ttt/infworld").rstrip("/")
S3_ROOT = os.environ.get(
    "INFWORLD_S3_ROOT", "s3://s3-us-west2-default/archives/chenran/ttt/infworld"
).rstrip("/")

# 走 LOCAL_ROOT + S3 的顶层目录；其余相对路径仍按 PROJECT_ROOT 解析。
DATA_PREFIXES = ("dataset", "preprocessed", "checkpoints", "weights", "videos", "outputs")

S3_DISABLED = os.environ.get("INFWORLD_S3_DISABLE", "") == "1"

_AWS = os.environ.get("INFWORLD_AWS_CLI", "aws")

# 已 ensure_local 过的路径，避免同一进程内反复调 aws
_ensured = set()
_ensured_lock = threading.Lock()

# 上传队列 + 单个后台 worker。用队列而不是直接起 Popen，是为了让同一路径的多次上传
# 严格串行：train.py 会把同一个 step 存两次（周期存盘 + 结束存盘），两个 aws s3 cp
# 并发写同一个 S3 key，而本地文件又正被 torch.save 重写，S3 上可能留下半新半旧的对象。
_upload_q = None
_upload_worker = None
_upload_lock = threading.Lock()


def _log(msg):
    print(f"[storage] {msg}", flush=True)


def is_data_path(path) -> bool:
    """相对路径的第一段是否属于 DATA_PREFIXES（即该走 nvme + S3）。"""
    if path is None:
        return False
    path = str(path).strip()
    if not path or os.path.isabs(path):
        return False
    return path.replace("\\", "/").split("/", 1)[0] in DATA_PREFIXES


def resolve(path):
    """相对路径 -> 绝对路径：数据目录落到 LOCAL_ROOT，其余落到 PROJECT_ROOT。

    绝对路径原样返回（用户显式指定的路径优先，不做重定向）。
    """
    if path is None:
        return path
    path = str(path).strip()
    if os.path.isabs(path):
        return path
    root = LOCAL_ROOT if is_data_path(path) else PROJECT_ROOT
    return os.path.join(root, path)


def rel_to_local_root(abs_path):
    """LOCAL_ROOT 下的绝对路径 -> 相对 LOCAL_ROOT 的路径；不在其下则返回 None。"""
    if abs_path is None:
        return None
    abs_path = os.path.abspath(str(abs_path))
    if abs_path == LOCAL_ROOT:
        return ""
    prefix = LOCAL_ROOT + os.sep
    return abs_path[len(prefix):] if abs_path.startswith(prefix) else None


def s3_uri(local_path):
    """LOCAL_ROOT 下的本地路径 -> 对应 S3 URI；不在 LOCAL_ROOT 下则返回 None。"""
    rel = rel_to_local_root(local_path)
    if rel is None:
        return None
    return f"{S3_ROOT}/{rel.replace(os.sep, '/')}" if rel else S3_ROOT


def _run(cmd, check, what, quiet=False):
    """跑一条 aws 命令，返回 (是否成功, 输出)。

    quiet=True 用于"预期可能失败"的探测（如 s3 ls 判断对象是否存在），
    这类失败不该打 WARNING，否则正常流程里满屏噪声。
    """
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except FileNotFoundError:
        raise RuntimeError(f"找不到 aws CLI（{_AWS}），无法访问 S3。设 INFWORLD_S3_DISABLE=1 可跳过 S3。")
    if proc.returncode != 0:
        msg = f"{what} 失败（exit {proc.returncode}）：{' '.join(cmd)}\n{proc.stdout}"
        if check:
            raise RuntimeError(msg)
        if not quiet:
            _log(f"WARNING: {msg}")
    return proc.returncode == 0, proc.stdout


def s3_exists(local_path) -> bool:
    """S3 上是否存在该路径（文件或前缀）。S3 关闭时恒为 False。"""
    uri = s3_uri(local_path)
    if S3_DISABLED or uri is None:
        return False
    ok, out = _run([_AWS, "s3", "ls", uri], check=False, what="s3 ls", quiet=True)
    return ok and bool(out.strip())


def ensure_local(path, is_dir=None, check=True):
    """确保 path 在本地可读：缺失就从对应 S3 位置拉到本地。

    is_dir=None 时按路径有无扩展名猜测（.pt/.json/.ckpt 等视为文件）。
    目录用 `aws s3 sync`（增量，已有文件不重传）；文件用 `aws s3 cp`。
    返回本地绝对路径，方便 `torch.load(ensure_local(p))` 这样连写。
    """
    local = resolve(path)
    if local is None:
        return local
    uri = s3_uri(local)
    if S3_DISABLED or uri is None:
        return local

    if is_dir is None:
        is_dir = not os.path.splitext(local)[1]

    with _ensured_lock:
        if local in _ensured:
            return local

    if is_dir:
        # 目录：不存在或为空才 sync；已有内容认为已就绪（增量补齐见 ensure_case）
        need = not os.path.isdir(local) or not os.listdir(local)
        if need:
            os.makedirs(local, exist_ok=True)
            _log(f"下载目录 {uri} -> {local}")
            _run([_AWS, "s3", "sync", uri + "/", local], check=check, what="s3 sync")
    else:
        if not os.path.exists(local):
            os.makedirs(os.path.dirname(local), exist_ok=True)
            _log(f"下载文件 {uri} -> {local}")
            # 先下到进程私有的临时文件再原子改名：多 rank / 多 dataloader worker
            # 同时拉同一个文件时，读到的永远是完整文件而不是别人写了一半的内容。
            tmp = f"{local}.part.{os.getpid()}.{threading.get_ident()}"
            ok, _ = _run([_AWS, "s3", "cp", uri, tmp], check=check, what="s3 cp")
            if ok:
                os.replace(tmp, local)
            elif os.path.exists(tmp):
                os.remove(tmp)

    with _ensured_lock:
        _ensured.add(local)
    return local


def ensure_files(dir_path, filenames, check=True):
    """确保某目录下的指定文件都在本地（缺哪个拉哪个）。返回本地目录绝对路径。

    用于 case 目录：只拉真正要读的 .pt / meta.json，不整目录 sync。
    """
    local_dir = resolve(dir_path)
    for name in filenames:
        ensure_local(os.path.join(local_dir, name), is_dir=False, check=check)
    return local_dir


def list_dir(dir_path, check=True):
    """列目录内容（本地 ∪ S3 顶层）。

    本地只缓存了部分 case 时，仅靠 os.listdir 会漏掉 S3 上的其余 case，
    所以两边取并集。返回名字列表（目录名不带斜杠）。
    """
    local_dir = resolve(dir_path)
    names = set(os.listdir(local_dir)) if os.path.isdir(local_dir) else set()
    uri = s3_uri(local_dir)
    if not S3_DISABLED and uri is not None:
        ok, out = _run([_AWS, "s3", "ls", uri + "/"], check=False, what="s3 ls", quiet=True)
        if ok:
            for line in out.splitlines():
                line = line.strip()
                if line.startswith("PRE "):
                    names.add(line[4:].rstrip("/"))
                else:
                    parts = line.split(None, 3)
                    if len(parts) == 4 and parts[3]:
                        names.add(parts[3])
        elif not names:
            raise RuntimeError(f"目录既不在本地也读不到 S3: {local_dir} / {uri}")
    if not names and not os.path.isdir(local_dir):
        raise FileNotFoundError(f"目录不存在: {local_dir}")
    names.discard("")
    return sorted(names)


def _upload_loop():
    while True:
        item = _upload_q.get()
        if item is None:            # 关闭信号
            _upload_q.task_done()
            return
        cmd, what = item
        try:
            _log(f"上传 {what}")
            ok, out = _run(cmd, check=False, what="上传")
            if ok:
                _log(f"上传完成 {what}")
        except Exception as e:      # worker 绝不能死，否则后续上传全部堆积
            _log(f"WARNING: 上传异常 {what}: {e}")
        finally:
            _upload_q.task_done()


def upload_async(path, is_dir=None):
    """把本地文件/目录排进后台上传队列，立刻返回，不阻塞调用方。

    队列由单个 worker 串行消费，所以同一路径的多次上传不会并发写同一个 S3 key。
    退出前必须调 wait_uploads()，否则队列里的 ckpt 可能还没传完。
    """
    global _upload_q, _upload_worker
    local = resolve(path)
    uri = s3_uri(local)
    if S3_DISABLED or uri is None or not os.path.exists(local):
        return
    if is_dir is None:
        is_dir = os.path.isdir(local)
    cmd = ([_AWS, "s3", "sync", local, uri + "/"] if is_dir
           else [_AWS, "s3", "cp", local, uri])
    with _upload_lock:
        if _upload_q is None:
            _upload_q = queue.Queue()
            _upload_worker = threading.Thread(target=_upload_loop, daemon=True,
                                              name="s3-upload")
            _upload_worker.start()
        _upload_q.put((cmd, f"{local} -> {uri}"))


def wait_uploads(timeout=None):
    """等待上传队列排空（进程退出前调用，否则 ckpt 可能没传完）。"""
    with _upload_lock:
        q = _upload_q
    if q is None or q.unfinished_tasks == 0:
        return
    _log(f"等待 {q.unfinished_tasks} 个上传任务完成…")
    if timeout is None:
        q.join()
    else:
        # Queue.join() 不支持超时，只能轮询 unfinished_tasks
        deadline = time.time() + timeout
        while q.unfinished_tasks > 0 and time.time() < deadline:
            time.sleep(0.5)
        if q.unfinished_tasks > 0:
            _log(f"WARNING: 上传超时，仍有 {q.unfinished_tasks} 个任务未完成")
            return
    _log("上传队列已排空")


_sync_thread = None
_sync_stop = threading.Event()


def start_periodic_sync(dir_path, interval=300):
    """起一个守护线程，每 interval 秒把目录（日志等）同步到 S3。

    只该由 rank0 调用一次。stop_periodic_sync() 会停线程并做最后一次同步。
    """
    global _sync_thread
    local = resolve(dir_path)
    if S3_DISABLED or s3_uri(local) is None or _sync_thread is not None:
        return

    def loop():
        # 只入队，不等待：等待交给统一的上传 worker，避免和主线程抢 wait_uploads
        while not _sync_stop.wait(interval):
            upload_async(local, is_dir=True)

    _sync_stop.clear()
    _sync_thread = threading.Thread(target=loop, daemon=True, name="s3-log-sync")
    _sync_thread.start()
    _log(f"日志定时同步已启动（每 {interval}s）: {local}")


def stop_periodic_sync(dir_path=None):
    """停止定时同步线程，并对 dir_path 做最后一次同步（等待完成）。"""
    global _sync_thread
    _sync_stop.set()
    if _sync_thread is not None:
        _sync_thread.join(timeout=10)
        _sync_thread = None
    if dir_path is not None:
        upload_async(dir_path, is_dir=True)
    wait_uploads(timeout=600)


def describe() -> str:
    if S3_DISABLED:
        return f"storage: 本地模式（S3 已禁用），data root={LOCAL_ROOT}"
    return f"storage: local={LOCAL_ROOT}  s3={S3_ROOT}  data dirs={', '.join(DATA_PREFIXES)}"
