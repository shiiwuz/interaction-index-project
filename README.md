# Interaction Index Predictor (OpenClaw skill)

Predict an interaction index (expected emoji/reaction count) from:
- a text snippet (title + summary + links), or
- a Telegram public post link (`t.me/.../123`).

## What's included

- Trained XGBoost model weights: `outputs/pref_xgb_fe_report.xgb.json`
- Model report (best iteration + metrics): `outputs/pref_xgb_fe_report.json`
- Rating scales:
  - 10-point deciles: `outputs/pref_rating_scale_10pt.json`
  - Legacy 5-star (mu/sigma): `outputs/pref_rating_scale.json`
- OpenClaw skill folder: `skills/interaction-index-predictor/`

## What's NOT included

- The embeddings model weights (we call an OpenAI-compatible embeddings API at runtime).
- Your `.env` secrets.
- Training caches/checkpoints (they can be large and may contain derived data).

## Quick start

1) Copy `.env.example` -> `.env` and fill in your embeddings settings.
2) Create a venv with uv and install deps:

```bash
uv venv .venv
uv pip install -p .venv/bin/python -r requirements.txt
```

3) Run a prediction:

Telegram link (uses post time):

```bash
set -a && . ./.env && set +a
uv run -p .venv/bin/python python3 skills/interaction-index-predictor/scripts/predict.py \
  --model-report outputs/pref_xgb_fe_report.json \
  --rating-scale outputs/pref_rating_scale_10pt.json \
  --tme-url https://t.me/<channel>/<id>
```

Text snippet (optional time override):

```bash
set -a && . ./.env && set +a
uv run -p .venv/bin/python python3 skills/interaction-index-predictor/scripts/predict.py \
  --model-report outputs/pref_xgb_fe_report.json \
  --rating-scale outputs/pref_rating_scale_10pt.json \
  --ts 2026-02-05T14:00:00+08:00 \
  --text $'标题\n正文...\nhttps://example.com'
```

More examples: `examples/`

## API (FastAPI + Docker)

Build and run locally:

```bash
cp .env.example .env
# fill in EMBEDDINGS_BASE_URL + EMBEDDINGS_API_KEY

docker compose up --build
curl -s http://localhost:8000/health
```

Predict with a text snippet:

```bash
curl -s http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"text":"标题\n正文...\nhttps://example.com"}'
```

Predict from a Telegram public post link:

```bash
curl -s http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"tme_url":"https://t.me/<channel>/<id>"}'
```

GitHub Actions will build + push to GHCR on every `main` push:
- `ghcr.io/<owner>/interaction-index-api:latest`
