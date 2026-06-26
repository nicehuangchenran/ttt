#!/usr/bin/env python3
"""Generate an HTML page comparing generated videos from two work_dirs.

Usage:
    python compare.py <name_a> <name_b> [-o output.html]

Reads work_dirs/<name>/videos/case_*_combined.mp4 for each name and builds a
single self-contained HTML page placing the two folders' videos side by side,
one row per case, so different model/config outputs can be compared visually.
"""
import argparse
import json
import os
import re
from html import escape

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK_DIRS = os.path.join(ROOT, "work_dirs")
CASES_DIR = os.path.join(ROOT, "dataset", "cases")

CASE_RE = re.compile(r"case_(\d+)_combined\.mp4$")


def list_videos(name):
    """Return {case_id: abspath} for videos under work_dirs/<name>/videos."""
    vdir = os.path.join(WORK_DIRS, name, "videos")
    if not os.path.isdir(vdir):
        raise SystemExit(f"Not a directory: {vdir}")
    out = {}
    for fn in os.listdir(vdir):
        m = CASE_RE.search(fn)
        if m:
            out[m.group(1)] = os.path.join(vdir, fn)
    return out


def load_prompt(case_id):
    """Return the environment_prompt for a case, or '' if unavailable."""
    path = os.path.join(CASES_DIR, f"case_{case_id}.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("environment_prompt", "")
    except (OSError, ValueError):
        return ""


def rel(path, start):
    """Relative URL (forward slashes) from the output file to a video."""
    return os.path.relpath(path, start).replace(os.sep, "/")


def build_html(name_a, name_b, out_path):
    vids_a = list_videos(name_a)
    vids_b = list_videos(name_b)
    case_ids = sorted(set(vids_a) | set(vids_b), key=lambda c: int(c))

    out_dir = os.path.dirname(os.path.abspath(out_path))
    rows = []
    for cid in case_ids:
        prompt = escape(load_prompt(cid))

        def cell(vids):
            p = vids.get(cid)
            if not p:
                return '<div class="missing">— missing —</div>'
            src = escape(rel(p, out_dir))
            return (
                f'<video src="{src}" controls muted loop preload="metadata"></video>'
            )

        rows.append(
            f"""<tr>
  <td class="case">
    <div class="cid">case {escape(cid)}</div>
    <div class="prompt">{prompt}</div>
  </td>
  <td>{cell(vids_a)}</td>
  <td>{cell(vids_b)}</td>
</tr>"""
        )

    n_common = len(set(vids_a) & set(vids_b))
    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>compare: {escape(name_a)} vs {escape(name_b)}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; background: #111; color: #eee; }}
  header {{ position: sticky; top: 0; background: #1c1c1c; padding: 12px 20px;
           border-bottom: 1px solid #333; z-index: 10; }}
  header h1 {{ font-size: 16px; margin: 0 0 4px; }}
  header .meta {{ font-size: 12px; color: #999; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th {{ position: sticky; top: 56px; background: #1c1c1c; padding: 8px;
        font-size: 14px; border-bottom: 1px solid #333; z-index: 5; }}
  td {{ vertical-align: top; padding: 10px; border-bottom: 1px solid #2a2a2a; }}
  td.case {{ width: 320px; }}
  .cid {{ font-weight: 600; margin-bottom: 6px; }}
  .prompt {{ font-size: 12px; color: #aaa; line-height: 1.5; max-height: 240px;
            overflow: auto; }}
  video {{ width: 100%; max-width: 480px; border-radius: 4px; background: #000;
          display: block; }}
  .missing {{ color: #c66; font-style: italic; padding: 40px 0; text-align: center; }}
</style>
</head>
<body>
<header>
  <h1>{escape(name_a)} &nbsp;vs&nbsp; {escape(name_b)}</h1>
  <div class="meta">{len(case_ids)} cases &middot; {n_common} in common
    &middot; {len(vids_a)} in {escape(name_a)} &middot; {len(vids_b)} in {escape(name_b)}</div>
</header>
<table>
  <thead>
    <tr>
      <th>case</th>
      <th>{escape(name_a)}</th>
      <th>{escape(name_b)}</th>
    </tr>
  </thead>
  <tbody>
{os.linesep.join(rows)}
  </tbody>
</table>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path, len(case_ids), n_common


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name_a", help="first work_dirs folder name")
    ap.add_argument("name_b", help="second work_dirs folder name")
    ap.add_argument("-o", "--output", default=None,
                    help="output html path (default: compare_<a>_vs_<b>.html)")
    args = ap.parse_args()

    out = args.output or f"compare_{args.name_a}_vs_{args.name_b}.html"
    path, n, common = build_html(args.name_a, args.name_b, out)
    print(f"Wrote {path} — {n} cases ({common} in common)")


if __name__ == "__main__":
    main()
