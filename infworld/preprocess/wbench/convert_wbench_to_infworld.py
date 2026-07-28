#!/usr/bin/env python3
"""将 WBench 第一人称（且全部 turn 为 navigation）的 case 转换为 Infinite-World 数据集格式。

输出布局：
    OUT_DIR/case<WBench id>/
        prompts.json   # case.json，3 个 prompt 合并为单个 "prompt" 键，其余键不变
        image.jpg      # WBench images/case_<id>.jpg 的拷贝
        move_view.json # 逐帧 [{"move","view"}, ...]（与 0001.json 格式相同）

仅使用标准库。可编辑的常量见下方。
"""

import glob
import json
import os
import shutil

# ---------------------------------------------------------------------------
# 可编辑常量
# ---------------------------------------------------------------------------
WBENCH_DIR = "/root/autodl-tmp/ttt/WBench/dataset"
OUT_DIR = "/root/autodl-tmp/ttt/Infinite-World/dataset"
FRAMES_PER_TURN = 120  # 每个 navigation turn 展开为多少帧


# ---------------------------------------------------------------------------
# 动作分解：复用 WBench/examples/hy_worldplay/convert_cases_to_jsonl.py 的逻辑
# move: [forward, right]; yaw: left(-)/right(+); pitch: down(-)/up(+)
# ---------------------------------------------------------------------------
SINGLE_KEY_NAV = {
    "W":     {"move": [1, 0],  "yaw": 0,  "pitch": 0},
    "S":     {"move": [-1, 0], "yaw": 0,  "pitch": 0},
    "A":     {"move": [0, -1], "yaw": 0,  "pitch": 0},
    "D":     {"move": [0, 1],  "yaw": 0,  "pitch": 0},
    "→":     {"move": [0, 0],  "yaw": 1,  "pitch": 0},
    "right": {"move": [0, 0],  "yaw": 1,  "pitch": 0},
    "←":     {"move": [0, 0],  "yaw": -1, "pitch": 0},
    "left":  {"move": [0, 0],  "yaw": -1, "pitch": 0},
    "↑":     {"move": [0, 0],  "yaw": 0,  "pitch": 1},
    "up":    {"move": [0, 0],  "yaw": 0,  "pitch": 1},
    "↓":     {"move": [0, 0],  "yaw": 0,  "pitch": -1},
    "down":  {"move": [0, 0],  "yaw": 0,  "pitch": -1},
    "stop":  {"move": [0, 0],  "yaw": 0,  "pitch": 0},
}
NAV_KEYS = set(SINGLE_KEY_NAV.keys())


def action_to_navigation(action_str):
    """将 action 字符串转换为 navigation 信号。支持单键与 '+' 组合键。
    返回 None 表示非导航动作。"""
    parts = [p.strip() for p in action_str.split("+")]
    if not all(p in NAV_KEYS for p in parts):
        return None
    fwd = right = yaw = pitch = 0
    for p in parts:
        nav = SINGLE_KEY_NAV[p]
        fwd += nav["move"][0]
        right += nav["move"][1]
        yaw += nav["yaw"]
        pitch += nav["pitch"]
    return {"fwd": fwd, "right": right, "yaw": yaw, "pitch": pitch}


def nav_to_move_label(fwd, right):
    """根据 (fwd, right) 返回 Infinite-World 的 move 标签。"""
    if fwd > 0 and right < 0:
        return "go forward and go left"
    if fwd > 0 and right > 0:
        return "go forward and go right"
    if fwd < 0 and right < 0:
        return "go back and go left"
    if fwd < 0 and right > 0:
        return "go back and go right"
    if fwd > 0:
        return "go forward"
    if fwd < 0:
        return "go back"
    if right < 0:
        return "go left"
    if right > 0:
        return "go right"
    return "no-op"


def nav_to_view_label(yaw, pitch):
    """根据 (yaw, pitch) 返回 Infinite-World 的 view 标签。
    yaw>0 -> turn right, yaw<0 -> turn left; pitch>0 -> turn up, pitch<0 -> turn down。"""
    if pitch > 0 and yaw < 0:
        return "turn up and turn left"
    if pitch > 0 and yaw > 0:
        return "turn up and turn right"
    if pitch < 0 and yaw < 0:
        return "turn down and turn left"
    if pitch < 0 and yaw > 0:
        return "turn down and turn right"
    if pitch > 0:
        return "turn up"
    if pitch < 0:
        return "turn down"
    if yaw < 0:
        return "turn left"
    if yaw > 0:
        return "turn right"
    return "no-op"


# ---------------------------------------------------------------------------
# 筛选 / 转换
# ---------------------------------------------------------------------------
def is_first_person_nav_only(case):
    """仅保留：第一人称，且所有 interaction 的 type == 'navigation'。"""
    if case.get("settings", {}).get("perspective") != "first_person":
        return False
    interactions = case.get("interactions", [])
    if not interactions:
        return False
    return all(it.get("type") == "navigation" for it in interactions)


def merge_prompts(case):
    """返回新字典：将 3 个 prompt 合并为单个 "prompt"，其余键不变，"prompt" 放最前。"""
    merged = " ".join(
        p for p in (
            case.get("environment_prompt", ""),
            case.get("character_prompt", ""),
            case.get("perspective_prompt", ""),
        ) if p
    )
    out = {"prompt": merged}
    for k, v in case.items():
        if k in ("environment_prompt", "character_prompt", "perspective_prompt"):
            continue
        out[k] = v
    return out


def build_move_view(case):
    """按 turn 顺序，将每个 navigation turn 展开为 FRAMES_PER_TURN 帧，拼接成扁平列表。"""
    interactions = sorted(
        case.get("interactions", []), key=lambda it: it.get("turn", 0)
    )
    frames = []
    for it in interactions:
        nav = action_to_navigation(it.get("action", "stop"))
        if nav is None:
            move_label, view_label = "no-op", "no-op"
        else:
            move_label = nav_to_move_label(nav["fwd"], nav["right"])
            view_label = nav_to_view_label(nav["yaw"], nav["pitch"])
        frame = {"move": move_label, "view": view_label}
        frames.extend(frame.copy() for _ in range(FRAMES_PER_TURN))
    return frames


def main():
    pattern = os.path.join(WBENCH_DIR, "cases", "case_*.json")
    files = sorted(glob.glob(pattern))
    os.makedirs(OUT_DIR, exist_ok=True)

    kept_ids = []
    total = 0
    for fp in files:
        total += 1
        with open(fp, "r", encoding="utf-8") as fh:
            case = json.load(fh)

        if not is_first_person_nav_only(case):
            continue

        case_id = case["id"]
        case_dir = os.path.join(OUT_DIR, f"case{case_id}")
        os.makedirs(case_dir, exist_ok=True)

        # prompts.json
        with open(os.path.join(case_dir, "prompts.json"), "w", encoding="utf-8") as fh:
            json.dump(merge_prompts(case), fh, ensure_ascii=False, indent=2)

        # image.jpg
        src_img = os.path.join(WBENCH_DIR, "images", f"case_{case_id}.jpg")
        shutil.copyfile(src_img, os.path.join(case_dir, "image.jpg"))

        # move_view.json
        with open(os.path.join(case_dir, "move_view.json"), "w", encoding="utf-8") as fh:
            json.dump(build_move_view(case), fh, ensure_ascii=False, indent=2)

        kept_ids.append(case_id)

    print(f"保留 {len(kept_ids)} / {total} 个 case，输出到 {OUT_DIR}")
    print("保留的 case id:", ", ".join(kept_ids))


if __name__ == "__main__":
    main()
