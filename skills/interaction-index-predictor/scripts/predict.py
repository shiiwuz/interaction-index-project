#!/usr/bin/env python3
"""Predict interaction index from text or a Telegram public post.

Outputs:
- yhat_log1p_reactions: predicted y = log1p(reactions_total)
- pred_reactions_total: exp(y)-1

Feature schema matches the training script:
  X = concat([emb_title, emb_summary, emb_domain, weekday, hour, title_len, summary_len])

Embeddings:
- Uses an OpenAI-compatible embeddings endpoint: POST {base_url}/v1/embeddings

Telegram scraping:
- Uses https://t.me/s/<channel>/<id> HTML and extracts time + message text.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import numpy as np
import urllib.request

import lxml.html
import xgboost as xgb


URL_RE = re.compile(r"https?://[^\s)\]>\"']+", re.IGNORECASE)
DATE_RE = re.compile(r"\b\d{1,2}\s*月\s*\d{1,2}\s*日\b")


@dataclass
class Parsed:
    text: str
    ts_utc: Optional[datetime]


def _to_s_url(tme_url: str) -> str:
    u = urlparse(tme_url)
    if u.netloc.lower() not in {"t.me", "telegram.me"}:
        raise ValueError("Not a t.me URL")

    # /<channel>/<id>
    parts = [p for p in u.path.split("/") if p]
    if len(parts) < 2:
        raise ValueError("Unexpected t.me path")
    channel, msg_id = parts[0], parts[1]
    return f"https://t.me/s/{channel}/{msg_id}"


def fetch_telegram_post(tme_url: str) -> Parsed:
    url = _to_s_url(tme_url)
    html = urllib.request.urlopen(url, timeout=30).read().decode("utf-8", "ignore")
    doc = lxml.html.fromstring(html)

    ts = None
    t = doc.xpath("//time[@datetime]/@datetime")
    if t:
        try:
            ts = datetime.fromisoformat(t[0])
        except Exception:
            ts = None

    blocks = doc.xpath("//div[contains(@class,'tgme_widget_message_text')]")
    if not blocks:
        raise RuntimeError("Could not find tgme_widget_message_text in page")
    b = blocks[0]

    # Preserve line breaks roughly; Telegram HTML uses <br>.
    text = "\n".join([s.strip() for s in b.itertext() if s and s.strip()])

    # Append external links from anchors to make domain extraction stable.
    hrefs = [a.get("href") for a in b.xpath(".//a[@href]")]
    ext = []
    for h in hrefs:
        if not h:
            continue
        try:
            netloc = urlparse(h).netloc.lower()
        except Exception:
            netloc = ""
        if netloc and not netloc.endswith("t.me") and not netloc.endswith("telegram.me"):
            ext.append(h)
    if ext:
        text = text + "\n" + "\n".join(ext)

    return Parsed(text=text, ts_utc=ts)


def extract_domain(text: str) -> str:
    m = URL_RE.search(text or "")
    if not m:
        return "(none)"
    netloc = urlparse(m.group(0)).netloc.lower().strip()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc or "(none)"


def extract_title_summary(text: str) -> tuple[str, str]:
    t = (text or "").strip()
    if not t:
        return "", ""

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    first = lines[0] if lines else t

    # If the first line looks like a real title, use it.
    if len(lines) > 1 and len(first) <= 80:
        title = first
        summary = "\n".join(lines[1:]).strip() or title
        return title, summary

    # Otherwise heuristically cut at date marker, else first Chinese period.
    m = DATE_RE.search(t)
    cut = m.start() if m else None
    if cut is None:
        p = t.find("。")
        cut = p + 1 if p != -1 else None

    if cut is None or cut <= 0:
        title = first[:80]
        summary = t
    else:
        title = t[:cut].strip(" ，,;；。")
        summary = t[cut:].strip() or title

    return title, summary


def fetch_embeddings(base_url: str, api_key: str, model: str, texts: list[str]) -> np.ndarray:
    url = base_url.rstrip("/") + "/v1/embeddings"
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        obj = json.loads(r.read())
    data = sorted(obj.get("data") or [], key=lambda d: d.get("index", 0))
    return np.asarray([d["embedding"] for d in data], dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="Path to XGBoost JSON model (overrides --model-report)")
    ap.add_argument("--model-report", default=None, help="JSON report file produced by training")
    ap.add_argument("--rating-scale", default=None, help="Rating scale JSON (supports 5-star mu/sigma or 10-point deciles)")
    ap.add_argument("--tme-url", default=None)
    ap.add_argument("--text", default=None)
    ap.add_argument("--ts", default=None, help="Override timestamp (ISO8601). If absent, uses t.me time or now.")
    ap.add_argument("--tz", default="Asia/Shanghai")
    args = ap.parse_args()

    if not args.tme_url and not args.text:
        raise SystemExit("Provide --tme-url or --text")

    base_url = os.environ.get("EMBEDDINGS_BASE_URL")
    model_name = os.environ.get("EMBEDDINGS_MODEL", "bge-m3")
    api_key = os.environ.get("EMBEDDINGS_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not base_url or not api_key:
        raise SystemExit("Missing EMBEDDINGS_BASE_URL and/or EMBEDDINGS_API_KEY")

    # Parse input
    ts_utc = None
    text = ""
    if args.tme_url:
        p = fetch_telegram_post(args.tme_url)
        text = p.text
        ts_utc = p.ts_utc
    else:
        text = args.text or ""

    if args.ts:
        ts_utc = datetime.fromisoformat(args.ts)

    if ts_utc is None:
        ts_utc = datetime.now(timezone.utc)

    # Convert to Shanghai for features (weekday/hour). Keep it simple and fixed.
    ts_sh = ts_utc.astimezone(timezone(timedelta(hours=8)))

    title, summary = extract_title_summary(text)
    domain = extract_domain(text)

    # Embed as a single request.
    E = fetch_embeddings(base_url, api_key, model_name, [title, summary, domain])
    X_title, X_sum, X_dom = E[0], E[1], E[2]

    meta = np.asarray([ts_sh.weekday(), ts_sh.hour, len(title), len(summary)], dtype=np.float32)
    X = np.hstack([X_title, X_sum, X_dom, meta]).reshape(1, -1)

    model_path = args.model
    best_iter = None
    if not model_path and args.model_report:
        rep = json.loads(Path(args.model_report).read_text(encoding="utf-8"))
        model_path = rep.get("model")
        best_iter = (rep.get("report") or {}).get("best", {}).get("iteration")

    if not model_path:
        raise SystemExit("Provide --model or --model-report")

    booster = xgb.Booster()
    booster.load_model(model_path)

    kwargs = {}
    if best_iter is not None:
        kwargs["iteration_range"] = (0, int(best_iter) + 1)

    yhat = float(booster.predict(xgb.DMatrix(X), **kwargs)[0])

    stars = None
    score10 = None
    if args.rating_scale:
        scale = json.loads(Path(args.rating_scale).read_text(encoding="utf-8"))
        method = scale.get("method")

        if method == "decile_quantiles":
            thr = scale.get("thresholds") or []
            ys = [float(t["y_threshold"]) for t in thr]
            score10 = 1
            for t in ys:
                if yhat >= t:
                    score10 += 1
            if score10 < 1:
                score10 = 1
            if score10 > 10:
                score10 = 10

        else:
            # Default: 5-star mu/sigma thresholds (legacy)
            mu = float(scale["scale"]["mu"])
            sigma = float(scale["scale"]["sigma"])
            t1 = mu - 2 * sigma
            t2 = mu - 1 * sigma
            t4 = mu + 1 * sigma
            t5 = mu + 2 * sigma
            if yhat < t1:
                stars = 1
            elif yhat < t2:
                stars = 2
            elif yhat < t4:
                stars = 3
            elif yhat < t5:
                stars = 4
            else:
                stars = 5

    out = {
        "ts_utc": ts_utc.isoformat(),
        "ts_shanghai": ts_sh.isoformat(),
        "title": title,
        "domain": domain,
        "meta": {
            "weekday": int(meta[0]),
            "hour": int(meta[1]),
            "title_len": int(meta[2]),
            "summary_len": int(meta[3]),
        },
        "yhat_log1p_reactions": yhat,
        "pred_reactions_total": math.expm1(yhat),
        "stars": stars,
        "score10": score10,
    }

    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
