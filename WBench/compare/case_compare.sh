#!/usr/bin/env bash
set -euo pipefail

# ===== 在这里修改要比较的两个文件夹名称 =====
# 对应 work_dirs/<名称>/evaluation/<指标>/case_*.json
DIR_A="infworld-offline-cut21_2026-06-27_01-49-33"
DIR_B="infworld-online-cut21_2026-06-26_23-32-16"

# ===== 在这里修改要对比的指标（一个或多个，空格分隔）=====
METRICS="background_consistency segment_continuity spatial_consistency"
# ==========================================

cd "$(dirname "$0")"

python3 case_compare.py \
  "work_dirs/${DIR_A}" \
  "work_dirs/${DIR_B}" \
  -m ${METRICS}

# 如需在浏览器查看，可取消下面两行注释：
# cd case
# python3 -m http.server 8000
