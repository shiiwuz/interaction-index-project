---
name: interaction-index-predictor
description: Predict an "interaction index" (expected emoji/reaction count) for short news/posts using an embeddings API plus a trained XGBoost model. Use when the user asks to predict互动指数/互动分/互动分数 for a text snippet or a Telegram post link (t.me/.../123). Also use to retrain the model from a Telegram HTML export JSONL with feature engineering (weekday/hour/title_len/summary_len + embeddings for title/summary/source_domain).
---

# Interaction Index Predictor

## Inputs

- **Text**: a news blurb (title + summary + links)
- **Telegram post**: `https://t.me/<channel>/<id>` (public channels only)

## Predict

- Run `scripts/predict.py` with either `--text` or `--tme-url`.
- Provide embeddings settings via env vars:
  - `EMBEDDINGS_BASE_URL`
  - `EMBEDDINGS_MODEL`
  - `EMBEDDINGS_API_KEY`

Example:

```bash
uv run -p .venv/bin/python python3 skills/interaction-index-predictor/scripts/predict.py \
  --model-report outputs/pref_xgb_fe_report.json \
  --tme-url https://t.me/zaihuapd/39376
```

## Retrain (Feature-Engineered XGBoost)

- Use `scripts/train_xgb_fe.py` on the cleaned Telegram JSONL.
- Label: `y = log1p(reactions_total)`.
- Features: `weekday`, `hour`, `title_len`, `summary_len`, and embeddings of:
  - `title` (first line or heuristic split)
  - `summary` (rest)
  - `source_domain` (first external URL domain)

Example:

```bash
uv run -p .venv/bin/python python3 skills/interaction-index-predictor/scripts/train_xgb_fe.py \
  --data outputs/chat_clean_v3.jsonl \
  --cache outputs/pref_xgb_fe_cache_seed42.npz \
  --out-model outputs/pref_xgb_fe_report.json \
  --seed 42
```

## Notes

- For consistent weekday/hour, prefer using a `t.me` link so the predictor can extract the post time.
- Do not commit secrets; store keys in a local `.env` and export them before running.
