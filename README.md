# Interaction Index Predictor (OpenClaw skill)

This repo contains an OpenClaw skill that predicts an interaction index (expected emoji/reaction count) from:
- a text snippet (title + summary + links), or
- a Telegram public post link (t.me/.../123).

## Quick start

1) Copy `.env.example` -> `.env` and fill in your embeddings settings.
2) Create a venv with uv and install deps:

```bash
uv venv .venv
uv pip install -p .venv/bin/python -r requirements.txt
```

3) Run a prediction:

```bash
set -a && . ./.env && set +a
uv run -p .venv/bin/python python3 skills/interaction-index-predictor/scripts/predict.py \
  --model-report outputs/pref_xgb_fe_report.json \
  --rating-scale outputs/pref_rating_scale_10pt.json \
  --tme-url https://t.me/<channel>/<id>
```
