"""
要更新 dataset/, preprocessed/, videos/sekai-game-walking-prompts

用修正后的阈值重算动作标签，原地更新 move_view.json 与 actions.pt
=======================================================================
原阈值（1_convert_to_cases.py 中的 ACTION_CFG）平移用 ±0.05、旋转用 ±0.05，
与数据实际尺度差 1~2 个数量级，导致标签退化：
    |forward| 中位数 0.0008  → 阈值 0.05 时 move 恒为 no-op
    |right|   中位数 0.000007
    |yaw|     中位数 0.00008 rad → 阈值 0.05 rad 时 turn left/right 触发 0.0%

本脚本按各轴实际分布分别设阈值（见 ACTION_CFG），从源 npz 重算：
    dataset/<ds>/case{n}/move_view.json      （全部 case）
    preprocessed/<ds>/case{n}/actions.pt     （仅已预处理的 case）

actions.pt 是 move_view.json 的派生物，故先写 json 再按 2_preprocess_dataset.py
的 load_actions 逻辑（按 num_frames 截断 / 补 0）重算 pt，保持两者一致。

用法：
  # 先 dry-run 看标签分布变化，不落盘
  python preprocess/sekai_game_walking/update_actions.py --num 20 --dry-run
  # 全量更新
  python preprocess/sekai_game_walking/update_actions.py
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))

DEFAULT_SOURCE = ("/mnt/s3files/s3-us-west2-default/dataprocessing/raw/"
                  "opendata/Sekai-game/sekai-game-walking")

# 各轴阈值按实测分布分别设定（40 case / 7.2 万帧对采样）：
#   前后 Z：中位 0.0008，0.0003 处切分 → forward ~90%（步行数据本就持续前进）
#   左右 X：中位 0.000007，比 Z 小两个数量级，2e-4 → 左右各约 6%
#   pitch：中位 0.0004 rad，双阈值 [0.003, 0.015] → 明确上下约 13%，uncertain 约 20%
#   yaw：  中位 0.00008 rad，双阈值 [0.001, 0.005] → 明确左右约 7%，uncertain 约 19%
#
# 视角使用双阈值机制：
#   |angle| > high_thresh  → 明确方向 (turn up/down/left/right)
#   |angle| < low_thresh   → no-op
#   low <= |angle| <= high → uncertain（微弱运动，不明确）
ACTION_CFG = {
    'move_forward_thresh': 0.0003,
    'move_backward_thresh': -0.0003,
    'move_right_thresh': 0.0002,
    'move_left_thresh': -0.0002,
    # pitch 双阈值：[low, high]，注意当前实现中 pitch 符号与判断都反向（但抵消后正确）
    'view_pitch_low_thresh': 0.003,   # |pitch| < 0.003 (0.17°) → no-op
    'view_pitch_high_thresh': 0.015,  # |pitch| > 0.015 (0.86°) → 明确 turn up/down
    # yaw 双阈值
    'view_yaw_low_thresh': 0.001,     # |yaw| < 0.001 (0.06°) → no-op
    'view_yaw_high_thresh': 0.005,    # |yaw| > 0.005 (0.29°) → 明确 turn left/right
}

_MOVE_NAMES = [
    'no-op', 'go forward', 'go back', 'go left', 'go right',
    'go forward and go left', 'go forward and go right',
    'go back and go left', 'go back and go right', 'uncertain',
]
_VIEW_NAMES = [
    'no-op', 'turn up', 'turn down', 'turn left', 'turn right',
    'turn up and turn left', 'turn up and turn right',
    'turn down and turn left', 'turn down and turn right', 'uncertain',
]
_MOVE_MAP = {n: i for i, n in enumerate(_MOVE_NAMES)}
_VIEW_MAP = {n: i for i, n in enumerate(_VIEW_NAMES)}


def camera_actions_from_extrinsics(extrinsics, cfg):
    """外参序列 -> (move, view) 动作索引数组。向量化实现，语义与逐帧版一致。

    extrinsics: (N, 4, 4) camera-to-world。第 0 帧无前序参考，固定 no-op。
    """
    n = len(extrinsics)
    move = np.zeros(n, dtype=np.int64)
    view = np.zeros(n, dtype=np.int64)
    if n < 2:
        return move, view

    R, t = extrinsics[:, :3, :3], extrinsics[:, :3, 3]
    prev_R = R[:-1]

    # 世界位移转到前一帧相机系：delta_t_cam = prev_R^T @ (t_k - t_{k-1})
    delta_t_cam = np.einsum('nij,nj->ni', np.transpose(prev_R, (0, 2, 1)),
                            t[1:] - t[:-1])
    # 相对旋转 delta_R = R_k @ R_{k-1}^T
    delta_R = np.einsum('nij,nkj->nik', R[1:], prev_R)

    # 注意：pitch 公式中的负号与判断逻辑都是反向的，但两者抵消后结果正确
    # （相机抬头 → pitch < 0 → 判断为 turn up，符合实际）
    pitch = np.arctan2(-delta_R[:, 2, 0],
                       np.sqrt(delta_R[:, 0, 0] ** 2 + delta_R[:, 1, 0] ** 2))
    yaw = np.arctan2(delta_R[:, 1, 0], delta_R[:, 0, 0])

    forward, right = delta_t_cam[:, 2], delta_t_cam[:, 0]
    mf = forward > cfg['move_forward_thresh']
    mb = forward < cfg['move_backward_thresh']
    mr = right > cfg['move_right_thresh']
    ml = right < cfg['move_left_thresh']

    # 视角双阈值：|angle| < low → no-op, low <= |angle| <= high → uncertain, |angle| > high → 明确方向
    pitch_abs = np.abs(pitch)
    yaw_abs = np.abs(yaw)

    pitch_low = cfg['view_pitch_low_thresh']
    pitch_high = cfg['view_pitch_high_thresh']
    yaw_low = cfg['view_yaw_low_thresh']
    yaw_high = cfg['view_yaw_high_thresh']

    # 明确方向（配合 pitch 的反向约定：pitch < 0 表示向上）
    vu_clear = (pitch < -pitch_high)  # 明确 turn up
    vd_clear = (pitch > pitch_high)   # 明确 turn down
    vl_clear = (yaw < -yaw_high)      # 明确 turn left
    vr_clear = (yaw > yaw_high)       # 明确 turn right

    # uncertain 区间
    vu_uncertain = (pitch < -pitch_low) & ~vu_clear
    vd_uncertain = (pitch > pitch_low) & ~vd_clear
    vl_uncertain = (yaw < -yaw_low) & ~vl_clear
    vr_uncertain = (yaw > yaw_low) & ~vr_clear

    # 任一轴 uncertain → 整体 uncertain
    view_uncertain = vu_uncertain | vd_uncertain | vl_uncertain | vr_uncertain

    # 赋值顺序对应原逐帧实现的 if/elif 优先级：组合动作覆盖单方向
    m = np.zeros(n - 1, dtype=np.int64)
    m[mf] = 1
    m[mb] = 2
    m[ml & ~mf & ~mb] = 3
    m[mr & ~mf & ~mb] = 4
    m[mf & ml] = 5
    m[mf & mr] = 6
    m[mb & ml] = 7
    m[mb & mr] = 8

    v = np.zeros(n - 1, dtype=np.int64)
    # 先处理明确方向
    v[vu_clear] = 1
    v[vd_clear] = 2
    v[vl_clear & ~vu_clear & ~vd_clear] = 3
    v[vr_clear & ~vu_clear & ~vd_clear] = 4
    v[vu_clear & vl_clear] = 5
    v[vu_clear & vr_clear] = 6
    v[vd_clear & vl_clear] = 7
    v[vd_clear & vr_clear] = 8
    # uncertain 覆盖所有非明确方向（包括 no-op 和组合）
    v[view_uncertain & (v == 0)] = 9  # 只对 no-op 区间标记 uncertain

    move[1:], view[1:] = m, v
    return move, view


def _fit_length(seq, target):
    """与 2_preprocess_dataset.py::load_actions 一致：超长截断，不足补 0(no-op)。"""
    seq = list(seq)
    if len(seq) > target:
        return seq[:target]
    return seq + [0] * (target - len(seq))


def process_case(case_name, args):
    """重算一个 case。返回 (case_name, status, stats)。"""
    ds_dir = os.path.join(args.dataset_dir, case_name)
    mv_path = os.path.join(ds_dir, "move_view.json")
    prompts_path = os.path.join(ds_dir, "prompts.json")
    if not os.path.exists(prompts_path):
        return case_name, "skip (no prompts.json)", None

    with open(prompts_path) as fh:
        src_camera = json.load(fh).get("src_camera")
    if not src_camera:
        return case_name, "skip (no src_camera)", None
    npz_path = os.path.join(args.source_dir, src_camera)
    if not os.path.exists(npz_path):
        return case_name, "skip (missing src npz)", None

    ext = np.load(npz_path)["extrinsic"]
    if ext.ndim != 3 or ext.shape[1:] != (4, 4):
        raise ValueError(f"bad extrinsic shape {tuple(ext.shape)}")

    move, view = camera_actions_from_extrinsics(ext, ACTION_CFG)

    # 旧标签（用于统计改动量）；move_view.json 的长度即视频帧数，是权威长度
    old_move = None
    n_label = len(move)
    if os.path.exists(mv_path):
        with open(mv_path) as fh:
            old = json.load(fh)
        n_label = len(old)
        old_move = [_MOVE_MAP[a["move"]] for a in old]
        old_view = [_VIEW_MAP[a["view"]] for a in old]

    move = _fit_length(move, n_label)
    view = _fit_length(view, n_label)

    changed_m = changed_v = None
    if old_move is not None:
        changed_m = sum(a != b for a, b in zip(move, old_move))
        changed_v = sum(a != b for a, b in zip(view, old_view))

    # 落盘 move_view.json
    if not args.dry_run:
        tmp = mv_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump([{"move": _MOVE_NAMES[m], "view": _VIEW_NAMES[v]}
                       for m, v in zip(move, view)], fh, indent=2)
        os.replace(tmp, mv_path)

    # 同步 actions.pt（只处理已预处理的 case，长度对齐 meta.num_frames）
    pt_status = "no preprocessed"
    pre_dir = os.path.join(args.preprocessed_dir, case_name)
    pt_path = os.path.join(pre_dir, "actions.pt")
    meta_path = os.path.join(pre_dir, "meta.json")
    if os.path.exists(pt_path) and os.path.exists(meta_path):
        with open(meta_path) as fh:
            t_pix = json.load(fh)["num_frames"]
        m_pt = torch.tensor(_fit_length(move, t_pix), dtype=torch.long)
        v_pt = torch.tensor(_fit_length(view, t_pix), dtype=torch.long)
        if not args.dry_run:
            tmp = pt_path + ".tmp"
            torch.save({"move": m_pt, "view": v_pt}, tmp)
            os.replace(tmp, pt_path)
        pt_status = f"actions.pt[{t_pix}]"

    return case_name, "ok", {
        "n_label": n_label,
        "changed_move": changed_m,
        "changed_view": changed_v,
        "pt": pt_status,
        "move_counts": Counter(move),
        "view_counts": Counter(view),
    }


def main():
    ap = argparse.ArgumentParser(description="按修正阈值重算 move_view.json / actions.pt")
    ap.add_argument("--dataset-dir",
                    default=os.path.join(PROJECT_ROOT, "dataset",
                                         "sekai-game-walking-352_192_30fps"))
    ap.add_argument("--preprocessed-dir",
                    default=os.path.join(PROJECT_ROOT, "preprocessed",
                                         "sekai-game-walking-352_192_30fps"))
    ap.add_argument("--source-dir", default=DEFAULT_SOURCE,
                    help="原始 <name>.npz 所在目录")
    ap.add_argument("--num", type=int, default=None, help="只处理前 N 个 case")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="只统计不写盘")
    args = ap.parse_args()

    cases = sorted(
        (d for d in os.listdir(args.dataset_dir) if d.startswith("case")
         and os.path.isdir(os.path.join(args.dataset_dir, d))),
        key=lambda s: int(s[4:]))
    if args.num is not None:
        cases = cases[:args.num]

    print(f"[update] dataset      : {args.dataset_dir}")
    print(f"[update] preprocessed : {args.preprocessed_dir}")
    print(f"[update] source npz   : {args.source_dir}")
    print(f"[update] {len(cases)} cases"
          f"{'  (DRY RUN, 不写盘)' if args.dry_run else ''}")
    print("[update] 阈值: " + ", ".join(f"{k}={v}" for k, v in ACTION_CFG.items()))
    print()

    t0 = time.time()
    ok = pt_done = 0
    failed, skipped = [], []
    tot_m, tot_v = Counter(), Counter()
    chg_m = chg_v = tot_frames = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process_case, c, args): c for c in cases}
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                c, status, st = fut.result()
                if status != "ok":
                    skipped.append((c, status))
                    continue
                ok += 1
                tot_m.update(st["move_counts"])
                tot_v.update(st["view_counts"])
                tot_frames += st["n_label"]
                if st["changed_move"] is not None:
                    chg_m += st["changed_move"]
                    chg_v += st["changed_view"]
                if st["pt"].startswith("actions.pt"):
                    pt_done += 1
                if ok % 200 == 0:
                    print(f"[update] {ok}/{len(cases)}  {time.time() - t0:.0f}s")
            except Exception as e:
                failed.append((c, repr(e)))
                print(f"[update] {c}: FAILED {e!r}")

    print(f"\n[update] done in {time.time() - t0:.0f}s: "
          f"{ok} cases 重算, {pt_done} 个 actions.pt 同步, "
          f"{len(skipped)} skipped, {len(failed)} failed")
    if tot_frames:
        print(f"[update] 标签改动: move {chg_m}/{tot_frames} 帧 "
              f"({100 * chg_m / tot_frames:.1f}%), "
              f"view {chg_v}/{tot_frames} 帧 ({100 * chg_v / tot_frames:.1f}%)")
        print("\n[update] 新 move 分布:")
        for k, n in sorted(tot_m.items(), key=lambda x: -x[1]):
            print(f"    {_MOVE_NAMES[k]:<26} {100 * n / tot_frames:5.1f}%")
        print("[update] 新 view 分布:")
        for k, n in sorted(tot_v.items(), key=lambda x: -x[1]):
            print(f"    {_VIEW_NAMES[k]:<26} {100 * n / tot_frames:5.1f}%")
    for c, s in skipped[:10]:
        print(f"  skip {c}: {s}")
    for c, e in failed[:10]:
        print(f"  fail {c}: {e}")


if __name__ == "__main__":
    main()
