"""
Sekai-game-walking -> dataset/sekai-game-walking/case{n}/ 离线物化转换
=======================================================================
把源目录（成对的 <name>.mp4 + <name>.npz，字幕在 CSV 中）转换为逐 case
的训练数据目录：

    <out-dir>/                  # 默认 dataset/sekai-game-walking-{w}_{h}_{fps}fps
      case{n}/
        video.mp4               # 转码后的视频（默认 854x480 @ 30fps，中心裁剪）
        video_path              # 文本文件：video.mp4 的绝对路径
        move_view.json          # 每帧 {"move": <name>, "view": <name>}，与视频帧一一对应
        prompts.json            # {"prompt": caption, 以及 CSV 中其余描述性字段}

case 序号 n 按字幕 CSV 的行顺序从 0 开始（CSV 是样本列表的唯一权威来源，
顺序确定即可复现）。动作标签复用 prepare_sekai_game_walking.py 中已在全量
数据上验证过的 camera_actions_from_extrinsics（npz 为 camera-to-world 外参，
OpenCV 坐标系）。

用法：
  # 先处理 2 个 case 预览
  python prepare/sekai_game_walking/convert_to_cases.py --num 2

  # 全量，8 进程，自定义分辨率
  python prepare/sekai_game_walking/convert_to_cases.py --width 1280 --height 720 --workers 8

已完成的 case（三个文件 + mp4 都存在且 move_view 长度与视频帧数一致）默认
跳过，可用 --overwrite 强制重做，因此中断后直接重跑即可断点续传。
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from trash.prepare_sekai_game_walking import (  # noqa: E402
    ACTION_CFG, _MOVE_NAMES, _VIEW_NAMES, _rle,
    camera_actions_from_extrinsics,
)

DEFAULT_SOURCE = ("/mnt/s3files/s3-us-west2-default/dataprocessing/raw/"
                  "opendata/Sekai-game/sekai-game-walking")
DEFAULT_CSV = ("/mnt/s3files/s3-us-west2-default/dataprocessing/raw/"
               "opendata/Sekai-game/train/sekai-game-walking.csv")


def default_out_dir(width, height, fps):
    return os.path.join(PROJECT_ROOT, "dataset",
                        f"sekai-game-walking-{width}_{height}_{fps:g}fps")

# CSV 中除 videoFile/cameraFile/caption 外的描述性字段，一并写入 prompts.json
EXTRA_CAPTION_KEYS = ("location", "scene", "crowdDensity", "weather", "timeOfDay")


def _ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def read_csv_rows(csv_path):
    """按 CSV 文件行顺序返回样本（该顺序定义 case 序号）；重复行只保留首个。"""
    rows, seen = [], set()
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["videoFile"] in seen:
                print(f"[convert] duplicate CSV row skipped: {r['videoFile']}")
                continue
            seen.add(r["videoFile"])
            rows.append(r)
    return rows


def probe_video(video_path):
    """返回 (帧数, fps)。decord 打开失败即视为坏样本，由调用方捕获。"""
    import decord
    vr = decord.VideoReader(video_path)
    n, fps = len(vr), float(vr.get_avg_fps()) or 30.0
    del vr
    return n, fps


def count_frames(video_path):
    import decord
    vr = decord.VideoReader(video_path)
    n = len(vr)
    del vr
    return n


def transcode(src, dst, width, height, fps, n_frames, crf, preset):
    """ffmpeg：重采样到 fps -> 等比缩放覆盖目标 -> 中心裁剪 -> x264 编码。
    写临时文件后原子改名，避免半成品被断点续传误判为已完成。"""
    vf = (f"fps={fps},"
          f"scale={width}:{height}:force_original_aspect_ratio=increase:"
          f"force_divisible_by=2,crop={width}:{height}")
    tmp = dst + ".tmp.mp4"
    cmd = [_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", src,
           "-vf", vf, "-frames:v", str(n_frames),
           "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
           "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", tmp]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    os.replace(tmp, dst)


def case_is_done(case_dir, video_dst):
    try:
        with open(os.path.join(case_dir, "move_view.json")) as fh:
            n_labels = len(json.load(fh))
        with open(os.path.join(case_dir, "prompts.json")) as fh:
            json.load(fh)
        with open(os.path.join(case_dir, "video_path")) as fh:
            vp = fh.read().strip()
        return vp == video_dst and os.path.exists(video_dst) \
            and count_frames(video_dst) == n_labels
    except Exception:
        return False


def process_case(n, row, args):
    """转换一个 case；返回 (n, 状态字符串, 摘要 dict 或 None)。"""
    case_dir = os.path.join(args.out_dir, f"case{n}")
    video_dst = os.path.abspath(os.path.join(case_dir, "video.mp4"))
    if not args.overwrite and case_is_done(case_dir, video_dst):
        return n, "skip (done)", None

    src_video = os.path.join(args.source_dir, row["videoFile"])
    src_npz = os.path.join(args.source_dir, row["cameraFile"])
    if not (os.path.exists(src_video) and os.path.exists(src_npz)):
        return n, "skip (missing src)", None

    n_avail, src_fps = probe_video(src_video)
    ext = np.load(src_npz)["extrinsic"]
    if ext.ndim != 3 or ext.shape[1:] != (4, 4):
        raise ValueError(f"bad extrinsic shape {tuple(ext.shape)}")
    usable = min(n_avail, len(ext))
    os.makedirs(case_dir, exist_ok=True)

    # 与标注同一套帧索引：源 fps -> 目标 fps 的重采样
    step = src_fps / args.fps
    if abs(step - 1.0) < 1e-3:
        step = 1.0
    idxs = np.round(np.arange(int((usable - 1) / step) + 1) * step).astype(np.int64)
    idxs = idxs[idxs < usable]
    move, view = camera_actions_from_extrinsics(ext, idxs, ACTION_CFG)

    transcode(src_video, video_dst, args.width, args.height, args.fps,
              len(idxs), args.crf, args.preset)

    # ffmpeg fps 滤镜按时间戳重采样，帧数可能与 len(idxs) 差 1-2 帧；
    # 以实际输出为准，标签截断或重复末帧对齐
    n_out = count_frames(video_dst)
    if abs(n_out - len(idxs)) > 3:
        raise RuntimeError(f"frame count mismatch: video {n_out} vs labels {len(idxs)}")
    move = (move + [move[-1]] * (n_out - len(move)))[:n_out]
    view = (view + [view[-1]] * (n_out - len(view)))[:n_out]

    move_view = [{"move": _MOVE_NAMES[m], "view": _VIEW_NAMES[v]}
                 for m, v in zip(move, view)]
    with open(os.path.join(case_dir, "move_view.json"), "w") as fh:
        json.dump(move_view, fh, indent=2)

    prompts = {"prompt": row["caption"]}
    for k in EXTRA_CAPTION_KEYS:
        if row.get(k):
            prompts[k] = row[k]
    prompts["src_video"] = row["videoFile"]
    prompts["src_camera"] = row["cameraFile"]
    with open(os.path.join(case_dir, "prompts.json"), "w") as fh:
        json.dump(prompts, fh, indent=2, ensure_ascii=False)

    with open(os.path.join(case_dir, "video_path"), "w") as fh:
        fh.write(video_dst + "\n")

    summary = {
        "case": n, "src": row["videoFile"], "frames": n_out,
        "size_mb": round(os.path.getsize(video_dst) / 1e6, 1),
        "move_rle": _rle(np.array(move), _MOVE_NAMES, max_chars=200),
        "view_rle": _rle(np.array(view), _VIEW_NAMES, max_chars=200),
    }
    return n, "ok", summary


def main():
    ap = argparse.ArgumentParser(
        description="Sekai-game-walking -> per-case training folders")
    ap.add_argument("--source-dir", default=DEFAULT_SOURCE)
    ap.add_argument("--caption-csv", default=DEFAULT_CSV)
    ap.add_argument("--out-dir", default=None,
                    help="默认 dataset/sekai-game-walking-{w}_{h}_{fps}fps")
    ap.add_argument("--width", type=int, default=854)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--start", type=int, default=0, help="起始 case 序号")
    ap.add_argument("--num", type=int, default=None, help="处理多少个 case（默认全部）")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = default_out_dir(args.width, args.height, args.fps)

    rows = read_csv_rows(args.caption_csv)
    end = len(rows) if args.num is None else min(args.start + args.num, len(rows))
    todo = list(range(args.start, end))
    print(f"[convert] {len(rows)} rows in CSV; processing cases "
          f"{args.start}..{end - 1} -> {args.out_dir} "
          f"({args.width}x{args.height} @ {args.fps:g} fps)")
    os.makedirs(args.out_dir, exist_ok=True)

    t0, done, failed = time.time(), 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process_case, n, rows[n], args): n for n in todo}
        for fut in as_completed(futs):
            n = futs[fut]
            try:
                n, status, summary = fut.result()
                done += 1
                print(f"[convert] case{n}: {status}  ({done}/{len(todo)}, "
                      f"{time.time() - t0:.0f}s elapsed)")
                if summary:
                    print(f"          {summary['frames']} frames, "
                          f"{summary['size_mb']} MB, src={summary['src']}")
                    print(f"          move: {summary['move_rle']}")
                    print(f"          view: {summary['view_rle']}")
            except Exception as e:
                failed.append((n, repr(e)))
                print(f"[convert] case{n}: FAILED {e!r}")
                traceback.print_exc()

    print(f"\n[convert] finished: {done - len(failed)} ok, {len(failed)} failed, "
          f"{time.time() - t0:.0f}s total")
    for n, err in failed:
        print(f"  case{n}: {err}")


if __name__ == "__main__":
    main()
