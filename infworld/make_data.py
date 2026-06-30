#!/usr/bin/env python3
"""按 dataset/wbench/case<id>/ 的两个 JSON 文件格式，批量生成数据集作为推理输入。

仅以 case5/{move_view.json, prompts.json} 为输出模板（本脚本不读取它们）：
  - move_view.json：逐帧动作数组 [{"move": ..., "view": ...}, ...]
  - prompts.json  ：只保留 prompt 字段 -> {"prompt": <text>}
不生成 image.jpg（由用户自行放入对应 case 目录）。

动作以「分段」形式给出，每段为 (move, view, n)，表示该 {move, view} 重复 n 帧，
避免逐帧手写大量重复元素。直接编辑 __main__ 中的 CASES 即可造新数据。
"""

import os
import json
from typing import List, Tuple

# 合法动作键（与 scripts/infworld_inference.py 的 MOVE_ACTION_MAP / VIEW_ACTION_MAP 一致；
# 这里内联，避免 import 推理模块触发 distributed 初始化）
VALID_MOVE = {
    'no-op', 'go forward', 'go back', 'go left', 'go right',
    'go forward and go left', 'go forward and go right',
    'go back and go left', 'go back and go right', 'uncertain',
}
VALID_VIEW = {
    'no-op', 'turn up', 'turn down', 'turn left', 'turn right',
    'turn up and turn left', 'turn up and turn right',
    'turn down and turn left', 'turn down and turn right', 'uncertain',
}


def expand_segments(segments: List[Tuple[str, str, int]]) -> List[dict]:
    """将分段动作展开为逐帧动作列表。

    Args:
        segments (List[Tuple[str, str, int]]): 分段动作列表，每段为
            ``(move, view, n)``，表示动作 ``{"move": move, "view": view}``
            连续重复 ``n`` 帧。``move`` 须属于 :data:`VALID_MOVE`，
            ``view`` 须属于 :data:`VALID_VIEW`，``n`` 为正整数。

    Returns:
        List[dict]: 逐帧动作列表 ``[{"move": move, "view": view}, ...]``，
            总长度为各段 ``n`` 之和。

    Raises:
        ValueError: 当某段不是三元组、``move``/``view`` 非法，或 ``n`` 不是
            正整数时抛出。
    """
    frames = []
    for i, seg in enumerate(segments):
        if len(seg) != 3:
            raise ValueError(f"segment[{i}] 应为 (move, view, n)，得到: {seg!r}")
        move, view, n = seg
        if move not in VALID_MOVE:
            raise ValueError(f"segment[{i}] 非法 move={move!r}，合法值: {sorted(VALID_MOVE)}")
        if view not in VALID_VIEW:
            raise ValueError(f"segment[{i}] 非法 view={view!r}，合法值: {sorted(VALID_VIEW)}")
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"segment[{i}] 帧数 n 必须为正整数，得到: {n!r}")
        frames.extend({"move": move, "view": view} for _ in range(n))
    return frames


def make_case(
    case_dir: str,
    prompt: str,
    segments: List[Tuple[str, str, int]],
) -> Tuple[str, int]:
    """生成单个 case 目录下的两个 JSON 文件。

    写入内容：
        ``<case_dir>/prompts.json``   -> ``{"prompt": prompt}``（仅保留 prompt 字段）
        ``<case_dir>/move_view.json`` -> :func:`expand_segments` 展开后的逐帧动作

    Args:
        case_dir (str): 目标 case 目录路径，不存在时会自动创建。
        prompt (str): 文本提示词，写入 ``prompts.json`` 的 ``prompt`` 字段。
        segments (List[Tuple[str, str, int]]): 分段动作列表，格式见
            :func:`expand_segments`。

    Returns:
        Tuple[str, int]: ``(case_dir, n_frames)``，即写入的目录路径与逐帧动作总帧数。

    Raises:
        ValueError: 当 ``segments`` 非法时（由 :func:`expand_segments` 抛出）。
    """
    os.makedirs(case_dir, exist_ok=True)

    frames = expand_segments(segments)

    with open(os.path.join(case_dir, "prompts.json"), "w", encoding="utf-8") as f:
        json.dump({"prompt": prompt}, f, indent=2, ensure_ascii=False)

    with open(os.path.join(case_dir, "move_view.json"), "w", encoding="utf-8") as f:
        json.dump(frames, f, indent=2, ensure_ascii=False)

    return case_dir, len(frames)


def make_dataset(
    cases: List[Tuple[str, List[Tuple[str, str, int]]]],
    output_dir: str,
    start_id: int = 1,
) -> List[str]:
    """批量生成数据集（``len(cases)`` 可为 1）。

    第 ``i`` 个 case 写入 ``<output_dir>/case<start_id + i>/``，每个 case 仅生成
    ``prompts.json`` 与 ``move_view.json``，不生成 ``image.jpg``。

    Args:
        cases (List[Tuple[str, List[Tuple[str, str, int]]]]): case 列表，每个元素
            为 ``(prompt, segments)``；``segments`` 为分段动作列表，格式见
            :func:`expand_segments`。
        output_dir (str): 数据集输出根目录。
        start_id (int): 起始 case 编号，第 ``i`` 个 case 的目录名为
            ``case<start_id + i>``。默认 ``1``。

    Returns:
        List[str]: 实际写出的 case 目录路径列表。

    Raises:
        ValueError: 当任一 case 的 ``segments`` 非法时（由 :func:`expand_segments` 抛出）。
    """
    written = []
    for offset, (prompt, segments) in enumerate(cases):
        case_dir = os.path.join(output_dir, f"case{start_id + offset}")
        case_dir, n_frames = make_case(case_dir, prompt, segments)
        print(f"[make_data] wrote {case_dir} ({n_frames} frames)")
        written.append(case_dir)
    return written


if __name__ == "__main__":
    # 在此编辑要生成的 case：每个元素为 (prompt, segments)，
    # segments 中每段 (move, view, n) 表示该动作重复 n 帧，一个 chunk 是 80 帧
    CASES = [
        (
            "A grand neoclassical museum gallery interior, first-person eye-level view "
            "facing down the length of the hall. Polished marble floor, tall grey marble "
            "columns, large oil paintings in gold frames, white marble sculptures on "
            "pedestals, an arched glass skylight flooding the hall with warm light.",
            # 往返：先左转 120 帧，再右转 120 帧（参考 case5 的 turn left -> turn right 结构）
            [
                ("go forward","no-op",80*10),
                ("no-op", "turn left", 120),
                ("no-op", "turn right", 120),
                ("go back","no-op",80*10)
            ],
        ),
    ]

    make_dataset(CASES, output_dir="dataset/long_case", start_id=1)
