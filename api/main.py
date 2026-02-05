from __future__ import annotations

import json
import math
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import lxml.html
import numpy as np
import urllib.request
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

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

    text = "\n".join([s.strip() for s in b.itertext() if s and s.strip()])

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

    if len(lines) > 1 and len(first) <= 80:
        title = first
        summary = "\n".join(lines[1:]).strip() or title
        return title, summary

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


class _EmbedLRU:
    def __init__(self, maxsize: int = 2048):
        self.maxsize = maxsize
        self._d: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()

    def get(self, model: str, text: str) -> Optional[np.ndarray]:
        k = (model, text)
        v = self._d.get(k)
        if v is None:
            return None
        self._d.move_to_end(k)
        return v

    def set(self, model: str, text: str, emb: np.ndarray) -> None:
        k = (model, text)
        self._d[k] = emb
        self._d.move_to_end(k)
        while len(self._d) > self.maxsize:
            self._d.popitem(last=False)


class Predictor:
    def __init__(self, model_report_path: str, rating_scale_path: str):
        rep = json.loads(Path(model_report_path).read_text(encoding="utf-8"))
        self.model_path = rep.get("model")
        if not self.model_path:
            raise RuntimeError("model_report missing 'model' path")
        self.best_iter = (rep.get("report") or {}).get("best", {}).get("iteration")

        self.scale = json.loads(Path(rating_scale_path).read_text(encoding="utf-8"))
        self.scale_method = self.scale.get("method")

        self.booster = xgb.Booster()
        self.booster.load_model(self.model_path)
        self._embed_cache = _EmbedLRU(maxsize=int(os.environ.get("EMBED_CACHE_SIZE", "2048")))

    def score10(self, yhat: float) -> Optional[int]:
        if self.scale_method != "decile_quantiles":
            return None
        thr = self.scale.get("thresholds") or []
        ys = [float(t["y_threshold"]) for t in thr]
        s = 1
        for t in ys:
            if yhat >= t:
                s += 1
        return max(1, min(10, s))

    def predict(self, *, base_url: str, api_key: str, emb_model: str, text: str, ts_utc: datetime) -> dict[str, Any]:
        ts_sh = ts_utc.astimezone(timezone(timedelta(hours=8)))

        title, summary = extract_title_summary(text)
        domain = extract_domain(text)

        # Embed title/summary/domain; cache per string.
        texts = [title, summary, domain]
        embs: list[Optional[np.ndarray]] = [self._embed_cache.get(emb_model, t) for t in texts]
        missing_idx = [i for i, e in enumerate(embs) if e is None]
        if missing_idx:
            missing_texts = [texts[i] for i in missing_idx]
            E = fetch_embeddings(base_url, api_key, emb_model, missing_texts)
            for j, i in enumerate(missing_idx):
                self._embed_cache.set(emb_model, texts[i], E[j])
                embs[i] = E[j]

        X_title, X_sum, X_dom = (embs[0], embs[1], embs[2])
        assert X_title is not None and X_sum is not None and X_dom is not None

        meta = np.asarray([ts_sh.weekday(), ts_sh.hour, len(title), len(summary)], dtype=np.float32)
        X = np.hstack([X_title, X_sum, X_dom, meta]).reshape(1, -1).astype(np.float32)

        kwargs = {}
        if self.best_iter is not None:
            kwargs["iteration_range"] = (0, int(self.best_iter) + 1)
        yhat = float(self.booster.predict(xgb.DMatrix(X), **kwargs)[0])

        return {
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
            "score10": self.score10(yhat),
        }


class PredictRequest(BaseModel):
    text: Optional[str] = Field(default=None, description="Raw text: title + summary + links")
    tme_url: Optional[str] = Field(default=None, description="Public Telegram post URL: https://t.me/<channel>/<id>")
    ts: Optional[str] = Field(default=None, description="ISO8601 timestamp override")


app = FastAPI(title="interaction-index-api", version="0.1.0")

_predictor: Optional[Predictor] = None


@app.on_event("startup")
def _startup() -> None:
    global _predictor
    model_report = os.environ.get("MODEL_REPORT_PATH", "outputs/pref_xgb_fe_report.json")
    rating_scale = os.environ.get("RATING_SCALE_PATH", "outputs/pref_rating_scale_10pt.json")
    _predictor = Predictor(model_report_path=model_report, rating_scale_path=rating_scale)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True}


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    if not _predictor:
        raise HTTPException(status_code=503, detail="predictor not ready")

    if not req.text and not req.tme_url:
        raise HTTPException(status_code=400, detail="Provide 'text' or 'tme_url'")

    base_url = os.environ.get("EMBEDDINGS_BASE_URL")
    emb_model = os.environ.get("EMBEDDINGS_MODEL", "bge-m3")
    api_key = os.environ.get("EMBEDDINGS_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not base_url or not api_key:
        raise HTTPException(status_code=500, detail="Missing EMBEDDINGS_BASE_URL and/or EMBEDDINGS_API_KEY")

    ts_utc: Optional[datetime] = None
    text = ""

    if req.tme_url:
        try:
            p = fetch_telegram_post(req.tme_url)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"tme_url fetch failed: {e}")
        text = p.text
        ts_utc = p.ts_utc
    else:
        text = req.text or ""

    if req.ts:
        try:
            ts_utc = datetime.fromisoformat(req.ts)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid ts; expected ISO8601")

    if ts_utc is None:
        ts_utc = datetime.now(timezone.utc)

    out = _predictor.predict(base_url=base_url, api_key=api_key, emb_model=emb_model, text=text, ts_utc=ts_utc)
    out["input"] = {"tme_url": req.tme_url, "text": req.text}
    return out
