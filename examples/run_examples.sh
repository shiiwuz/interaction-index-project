#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   cp .env.example .env && edit it
#   bash examples/run_examples.sh

set -a
# shellcheck disable=SC1091
. ./.env
set +a

PY=".venv/bin/python"
PRED="skills/interaction-index-predictor/scripts/predict.py"
MODEL_REPORT="outputs/pref_xgb_fe_report.json"
SCALE10="outputs/pref_rating_scale_10pt.json"

uv run -p "$PY" python3 "$PRED" \
  --model-report "$MODEL_REPORT" \
  --rating-scale "$SCALE10" \
  --tme-url "https://t.me/zaihuapd/39376"

echo

uv run -p "$PY" python3 "$PRED" \
  --model-report "$MODEL_REPORT" \
  --rating-scale "$SCALE10" \
  --ts "2026-02-05T14:00:00+08:00" \
  --text $'\
\U0001F4F1 2026年B站与央视春晚再度合作，直播开放真弹幕互动\n\
2月5日，哔哩哔哩（bilibili）宣布与中央广播电视总台（CCTV）的《2026年春节联欢晚会》再度达成合作，成为马年春晚独家弹幕视频平台。\n\
https://www.jiemian.com/article/13980081.html\
'
