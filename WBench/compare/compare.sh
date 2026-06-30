#!/usr/bin/env bash
set -euo pipefail
cd 
# ===== 在这里修改要比较的两个文件夹名称 =====
# 对应 work_dirs/<名称>/videos/case_*_combined.mp4
NAME_A="infworld-offline-cut21_2026-06-27_01-49-33"
NAME_B="infworld-online-cut21_2026-06-26_23-32-16"
# 输出 html 文件名（生成在 compare/web/ 下）
OUTPUT="web/compare_${NAME_A}_vs_${NAME_B}.html"
# ==========================================

cd "$(dirname "$0")"
mkdir -p web
python3 compare.py "$NAME_A" "$NAME_B" -o "$OUTPUT"

cd compare/web
python3 -m http.server 8000