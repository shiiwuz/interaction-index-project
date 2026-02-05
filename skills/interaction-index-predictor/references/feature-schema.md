# Feature Schema

Model input vector X is a concatenation of:

- `emb_title`: embedding(title)
- `emb_summary`: embedding(summary)
- `emb_domain`: embedding(source_domain)
- `weekday`: 0=Mon .. 6=Sun
- `hour`: 0..23
- `title_len`: char length
- `summary_len`: char length

Where:
- `title`: first non-empty line if present; otherwise heuristic split (date marker like "2 月 5 日" or first "。")
- `summary`: remaining text; fallback to title if empty
- `source_domain`: from first URL found in text (strip leading `www.`); fallback "(none)"

Label:
- `reactions_total`: emoji/reaction total
- `y = log1p(max(reactions_total, 0))`
- Prediction returns `yhat` and `expm1(yhat)` as the expected reactions_total.
