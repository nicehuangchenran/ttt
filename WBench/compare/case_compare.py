#!/usr/bin/env python3
"""对比两个 work_dir 下、每个 case 各指标的评测 score，生成 HTML 页面。

用法:
    python case_compare.py <dir_a> <dir_b> -m <metric> [<metric> ...] [-o out.html]

评测分数已由评测流程写在:
    <dir>/evaluation/<metric>/case_<id>.json    # 内含 "score": <float>

本脚本只读取这些 score，逐 case 对比两个目录（两种方法/配置）下各指标的分数，
输出一张 HTML 表格（默认写到 compare/case/ 下），A / B / Δ 三列，Δ 着色，缺失显示 —。
"""
import argparse
import json
import os
import re
from html import escape

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # WBench 根
CASES_DIR = os.path.join(ROOT, "dataset", "cases")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "case")

CASE_RE = re.compile(r"case_(\d+)\.json$")


def resolve_dir(path):
    """把用户输入（相对 WBench 根 / 绝对 / 带 ./）解析成绝对路径。"""
    p = path if os.path.isabs(path) else os.path.join(ROOT, path)
    return os.path.normpath(p)


def dir_name(path):
    """解析后目录的 basename，用于表头和文件名。"""
    return os.path.basename(resolve_dir(path).rstrip(os.sep))


def load_scores(dir_path, metric):
    """返回 {case_id: score}，扫描 <dir>/evaluation/<metric>/case_<id>.json。

    score 缺失/None 则跳过；坏 JSON 容错跳过。
    """
    mdir = os.path.join(resolve_dir(dir_path), "evaluation", metric)
    out = {}
    if not os.path.isdir(mdir):
        return out
    for fn in os.listdir(mdir):
        m = CASE_RE.search(fn)
        if not m:
            continue
        try:
            with open(os.path.join(mdir, fn), encoding="utf-8") as f:
                score = json.load(f).get("score")
        except (OSError, ValueError):
            continue
        if isinstance(score, (int, float)):
            out[m.group(1)] = float(score)
    return out


def load_prompt(case_id):
    """返回该 case 的 environment_prompt，取不到则空串。"""
    path = os.path.join(CASES_DIR, f"case_{case_id}.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("environment_prompt", "")
    except (OSError, ValueError):
        return ""


def fmt(v):
    return "—" if v is None else f"{v:.4f}"


def delta_cell(a, b):
    """Δ = b - a 的单元格 html，b>a 绿、b<a 红。"""
    if a is None or b is None:
        return '<td class="num miss">—</td>'
    d = b - a
    color = "#5cb85c" if d > 0 else ("#d9534f" if d < 0 else "#999")
    sign = "+" if d > 0 else ""
    return f'<td class="num" style="color:{color}">{sign}{d:.4f}</td>'


def build_html(dir_a, dir_b, metrics, out_path):
    name_a, name_b = dir_name(dir_a), dir_name(dir_b)
    # scores[metric] = ({cid: score_a}, {cid: score_b})
    scores = {m: (load_scores(dir_a, m), load_scores(dir_b, m)) for m in metrics}

    case_ids = set()
    for a, b in scores.values():
        case_ids |= set(a) | set(b)
    case_ids = sorted(case_ids, key=lambda c: int(c))

    # 表头：每个指标下 3 个子列
    metric_head = "".join(
        f'<th colspan="3" class="metric">{escape(m)}</th>' for m in metrics
    )
    sub_head = "".join('<th class="num">A</th><th class="num">B</th><th class="num">Δ</th>'
                       for _ in metrics)

    rows = []
    for cid in case_ids:
        cells = []
        for m in metrics:
            a = scores[m][0].get(cid)
            b = scores[m][1].get(cid)
            cells.append(
                f'<td class="num{" miss" if a is None else ""}">{fmt(a)}</td>'
                f'<td class="num{" miss" if b is None else ""}">{fmt(b)}</td>'
                f'{delta_cell(a, b)}'
            )
        prompt = escape(load_prompt(cid))
        rows.append(
            f'<tr><td class="case"><div class="cid">case {escape(cid)}</div>'
            f'<div class="prompt">{prompt}</div></td>{"".join(cells)}</tr>'
        )

    # 汇总行：每个 (指标) 的 A/B 均值与均 Δ，仅统计两侧都有的 case
    summary_cells = []
    common_counts = []
    for m in metrics:
        a, b = scores[m]
        common = sorted(set(a) & set(b), key=lambda c: int(c))
        common_counts.append((m, len(common)))
        if common:
            ma = sum(a[c] for c in common) / len(common)
            mb = sum(b[c] for c in common) / len(common)
            summary_cells.append(
                f'<td class="num">{ma:.4f}</td><td class="num">{mb:.4f}</td>'
                f'{delta_cell(ma, mb)}'
            )
        else:
            summary_cells.append(
                '<td class="num miss">—</td><td class="num miss">—</td>'
                '<td class="num miss">—</td>'
            )
    summary_row = (
        f'<tr class="summary"><td class="case">均值（两侧公共 case）</td>'
        f'{"".join(summary_cells)}</tr>'
    )

    meta = " &middot; ".join(f"{escape(m)}: {n} 公共" for m, n in common_counts)
    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>case compare: {escape(name_a)} vs {escape(name_b)}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; background: #111; color: #eee; }}
  header {{ position: sticky; top: 0; background: #1c1c1c; padding: 12px 20px;
           border-bottom: 1px solid #333; z-index: 10; }}
  header h1 {{ font-size: 16px; margin: 0 0 4px; }}
  header .meta {{ font-size: 12px; color: #999; }}
  header .ab {{ font-size: 12px; color: #bbb; margin-top: 4px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th {{ position: sticky; background: #1c1c1c; padding: 6px 10px; font-size: 13px;
        border-bottom: 1px solid #333; z-index: 5; }}
  thead tr:first-child th {{ top: 72px; }}
  thead tr:nth-child(2) th {{ top: 104px; }}
  th.metric {{ border-left: 1px solid #333; }}
  td {{ vertical-align: top; padding: 8px 10px; border-bottom: 1px solid #2a2a2a;
        font-size: 13px; }}
  td.case {{ width: 300px; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
  td.miss {{ color: #666; }}
  .cid {{ font-weight: 600; margin-bottom: 4px; }}
  .prompt {{ font-size: 12px; color: #aaa; line-height: 1.5; max-height: 160px;
            overflow: auto; }}
  tr.summary td {{ background: #181818; font-weight: 600; border-top: 2px solid #333; }}
</style>
</head>
<body>
<header>
  <h1>{escape(name_a)} &nbsp;vs&nbsp; {escape(name_b)}</h1>
  <div class="ab">A = {escape(name_a)} &nbsp;|&nbsp; B = {escape(name_b)}</div>
  <div class="meta">{len(case_ids)} cases &middot; {escape(meta)}</div>
</header>
<table>
  <thead>
    <tr><th rowspan="2" class="case">case</th>{metric_head}</tr>
    <tr>{sub_head}</tr>
  </thead>
  <tbody>
{os.linesep.join(rows)}
    {summary_row}
  </tbody>
</table>
</body>
</html>"""

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path, len(case_ids)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir_a", help="第一个 work_dir 路径（相对 WBench 根或绝对）")
    ap.add_argument("dir_b", help="第二个 work_dir 路径（相对 WBench 根或绝对）")
    ap.add_argument("-m", "--metrics", nargs="+", required=True,
                    help="一个或多个指标名，如 background_consistency segment_continuity")
    ap.add_argument("-o", "--output", default=None,
                    help="输出 html 路径（默认 compare/case/case_compare_<a>_vs_<b>.html）")
    args = ap.parse_args()

    out = args.output or os.path.join(
        OUT_DIR, f"case_compare_{dir_name(args.dir_a)}_vs_{dir_name(args.dir_b)}.html")
    path, n = build_html(args.dir_a, args.dir_b, args.metrics, out)
    print(f"Wrote {path} — {n} cases, metrics: {', '.join(args.metrics)}")


if __name__ == "__main__":
    main()
