#!/usr/bin/env python3
"""XGBoost training with feature-engineering on Telegram export messages.

Requested features:
- weekday (0-6)
- hour (0-23)
- title_len (chars)
- summary_len (chars)
- emb_title: embedding of first non-empty line
- emb_summary: embedding of remaining lines (or full text fallback)
- emb_domain: embedding of source_domain extracted from first URL in text

Model:
- XGBoost regressor on concatenated features.

Notes:
- Uses the last 2 years relative to max timestamp found.
- Split: random shuffle with fixed seed; 0.8/0.1/0.1.
- Label: y = log1p(max(reactions_total, 0)).
- Embedding API: OpenAI-compatible /v1/embeddings.

Env (or pass flags):
- EMBEDDINGS_API_KEY (or OPENAI_API_KEY)
- EMBEDDINGS_BASE_URL
- EMBEDDINGS_MODEL
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import numpy as np
import urllib.error
import urllib.request


URL_RE = re.compile(r"https?://[^\s)\]>\"']+", re.IGNORECASE)


def _iso_parse(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = (np.linalg.norm(ra) * np.linalg.norm(rb))
    if denom == 0:
        return float("nan")
    return float((ra @ rb) / denom)


def _metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    err = (yhat - y).astype(np.float64)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    sp = _spearman(y, yhat)
    return {"mae": mae, "rmse": rmse, "spearman": sp}


@dataclass
class Row:
    title: str
    summary: str
    domain: str
    weekday: int
    hour: int
    title_len: int
    summary_len: int
    y: float


def _extract_title_summary(text: str) -> tuple[str, str]:
    lines = [ln.strip() for ln in (text or "").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return "", ""
    title = lines[0]
    summary = "\n".join(lines[1:]).strip()
    if not summary:
        summary = title
    return title, summary


def _extract_domain(text: str) -> str:
    m = URL_RE.search(text or "")
    if not m:
        return "(none)"
    try:
        netloc = urlparse(m.group(0)).netloc.lower().strip()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc or "(none)"
    except Exception:
        return "(none)"


def load_rows(path: Path, *, min_text_len: int = 10) -> list[Row]:
    # First pass: find max timestamp.
    max_ts: Optional[datetime] = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if o.get("is_service"):
                continue
            ts = _iso_parse(o.get("timestamp") or "")
            if ts is None:
                continue
            if max_ts is None or ts > max_ts:
                max_ts = ts

    if max_ts is None:
        raise SystemExit("No valid timestamps found")

    cutoff = max_ts - timedelta(days=365 * 2)

    out: list[Row] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if o.get("is_service"):
                continue
            text = (o.get("text") or "").strip()
            if len(text) < min_text_len:
                continue
            ts = _iso_parse(o.get("timestamp") or "")
            if ts is None or ts < cutoff:
                continue

            title, summary = _extract_title_summary(text)
            domain = _extract_domain(text)

            score = float(o.get("reactions_total") or 0)
            if score < 0:
                score = 0.0
            y = math.log1p(score)

            out.append(
                Row(
                    title=title,
                    summary=summary,
                    domain=domain,
                    weekday=int(ts.weekday()),
                    hour=int(ts.hour),
                    title_len=len(title),
                    summary_len=len(summary),
                    y=y,
                )
            )

    return out


def fetch_embeddings(
    *,
    base_url: str,
    api_key: str,
    model: str,
    texts: list[str],
    timeout_s: int = 60,
) -> np.ndarray:
    url = base_url.rstrip("/") + "/v1/embeddings"
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read()
    obj = json.loads(body)
    data = sorted(obj.get("data") or [], key=lambda d: d.get("index", 0))
    embs = [d["embedding"] for d in data]
    return np.asarray(embs, dtype=np.float32)


def embed_all(
    *,
    texts: list[str],
    base_url: str,
    api_key: str,
    model: str,
    batch: int,
) -> np.ndarray:
    Xs: list[np.ndarray] = []
    t0 = time.time()
    i = 0
    while i < len(texts):
        chunk = texts[i : i + batch]
        try:
            X = fetch_embeddings(base_url=base_url, api_key=api_key, model=model, texts=chunk)
            Xs.append(X)
            i += len(chunk)
        except urllib.error.HTTPError as e:
            if e.code == 413 and batch > 1:
                batch = max(1, batch // 2)
                print(f"[embed] 413; reduce batch -> {batch}", flush=True)
                continue
            raise

        if i % (batch * 50) == 0 or i == len(texts):
            dt = time.time() - t0
            rate = i / dt if dt > 0 else 0
            print(f"[embed] {i}/{len(texts)} ({rate:.1f} items/s)", flush=True)

    return np.vstack(Xs) if Xs else np.zeros((0, 0), dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base-url", default=os.environ.get("EMBEDDINGS_BASE_URL", ""))
    ap.add_argument("--emb-model", default=os.environ.get("EMBEDDINGS_MODEL", ""))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--min-text-len", type=int, default=10)
    ap.add_argument("--cache", default="outputs/pref_xgb_fe_cache_seed42.npz")
    ap.add_argument("--out-model", default="outputs/pref_xgb_fe.json")
    ap.add_argument("--n-estimators", type=int, default=4000, help="Total boosting rounds (not per-resume chunk)")
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--early-stopping-rounds", type=int, default=200)
    ap.add_argument("--min-delta", type=float, default=0.0, help="Min RMSE improvement to reset early stopping")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-dir", default="outputs/pref_xgb_fe_ckpt")
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in --ckpt-dir")
    args = ap.parse_args()

    api_key = os.environ.get("EMBEDDINGS_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing EMBEDDINGS_API_KEY (or OPENAI_API_KEY)")
    if not args.base_url:
        raise SystemExit("Missing --base-url or EMBEDDINGS_BASE_URL")
    if not args.emb_model:
        raise SystemExit("Missing --emb-model or EMBEDDINGS_MODEL")

    rows = load_rows(Path(args.data), min_text_len=args.min_text_len)
    print(json.dumps({"examples_last_2y": len(rows)}, ensure_ascii=False), flush=True)

    rng = random.Random(args.seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    rows = [rows[i] for i in idx]

    n = len(rows)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        X_title = z["X_title"]
        X_sum = z["X_sum"]
        X_dom = z["X_dom"]
        meta = z["meta"]
        y = z["y"]
        print(json.dumps({"cache": "hit", "dims": int(X_title.shape[1])}, ensure_ascii=False), flush=True)
    else:
        titles = [r.title for r in rows]
        sums = [r.summary for r in rows]
        doms = [r.domain for r in rows]

        X_title = embed_all(texts=titles, base_url=args.base_url, api_key=api_key, model=args.emb_model, batch=args.batch)
        X_sum = embed_all(texts=sums, base_url=args.base_url, api_key=api_key, model=args.emb_model, batch=args.batch)
        X_dom = embed_all(texts=doms, base_url=args.base_url, api_key=api_key, model=args.emb_model, batch=args.batch)

        meta = np.asarray(
            [[r.weekday, r.hour, r.title_len, r.summary_len] for r in rows],
            dtype=np.float32,
        )
        y = np.asarray([r.y for r in rows], dtype=np.float32)

        np.savez_compressed(cache_path, X_title=X_title, X_sum=X_sum, X_dom=X_dom, meta=meta, y=y)
        print(json.dumps({"cache": "write", "path": str(cache_path)}, ensure_ascii=False), flush=True)

    X = np.hstack([X_title, X_sum, X_dom, meta])

    Xtr, ytr = X[:n_train], y[:n_train]
    Xva, yva = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    Xte, yte = X[n_train + n_val :], y[n_train + n_val :]

    import xgboost as xgb

    # Use xgb.train to get progress logging + early stopping + checkpointing.
    dtr = xgb.DMatrix(Xtr, label=ytr)
    dva = xgb.DMatrix(Xva, label=yva)
    dte = xgb.DMatrix(Xte, label=yte)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": args.learning_rate,
        "max_depth": args.max_depth,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "alpha": 0.0,
        "min_child_weight": 1.0,
        "seed": args.seed,
        "tree_method": "hist",
    }

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"pref_xgb_fe_seed{args.seed}"

    resume_model: Optional[str] = None
    already_rounds = 0
    if args.resume:
        # xgboost saves checkpoints as: <name>_<iteration>.ubj
        cand = list(ckpt_dir.glob(f"{ckpt_name}_*.ubj"))
        if cand:
            def it_num(p: Path) -> int:
                m = re.search(r"_(\d+)\.ubj$", p.name)
                return int(m.group(1)) if m else -1
            cand.sort(key=it_num)
            resume_model = str(cand[-1])
            try:
                b0 = xgb.Booster()
                b0.load_model(resume_model)
                already_rounds = int(b0.num_boosted_rounds())
            except Exception:
                already_rounds = 0
            print(json.dumps({"resume": True, "from": resume_model, "already_rounds": already_rounds}, ensure_ascii=False), flush=True)

    remaining = max(0, int(args.n_estimators) - int(already_rounds))
    if remaining == 0 and resume_model:
        booster = xgb.Booster()
        booster.load_model(resume_model)
        fit_s = 0.0
    else:
        callbacks = [
            xgb.callback.TrainingCheckPoint(directory=str(ckpt_dir), name=ckpt_name, interval=int(args.ckpt_every)),
            xgb.callback.EarlyStopping(
                rounds=int(args.early_stopping_rounds),
                metric_name="rmse",
                data_name="val",
                maximize=False,
                save_best=True,
                min_delta=float(args.min_delta),
            ),
        ]

        t0 = time.time()
        booster = xgb.train(
            params,
            dtr,
            num_boost_round=int(remaining),
            evals=[(dva, "val")],
            verbose_eval=int(args.log_every),
            xgb_model=resume_model,
            callbacks=callbacks,
        )
        fit_s = time.time() - t0

    best_iter = getattr(booster, "best_iteration", None)
    best_score = getattr(booster, "best_score", None)

    # Predict with the best iteration range when early stopping is on.
    it_end = int(best_iter) + 1 if best_iter is not None else 0
    pred_kwargs = {}
    if it_end > 0:
        pred_kwargs["iteration_range"] = (0, it_end)

    yhat_tr = booster.predict(dtr, **pred_kwargs)
    yhat_va = booster.predict(dva, **pred_kwargs)
    yhat_te = booster.predict(dte, **pred_kwargs)

    report = {
        "seed": args.seed,
        "params": {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "early_stopping_rounds": args.early_stopping_rounds,
            "log_every": args.log_every,
        },
        "n": {"all": int(n), "train": int(len(Xtr)), "val": int(len(Xva)), "test": int(len(Xte))},
        "dims": {"emb": int(X_title.shape[1]), "total": int(X.shape[1])},
        "fit_seconds": fit_s,
        "best": {"iteration": best_iter, "val_rmse": best_score},
        "metrics": {
            "train": _metrics(ytr, yhat_tr),
            "val": _metrics(yva, yhat_va),
            "test": _metrics(yte, yhat_te),
        },
    }

    out_model = Path(args.out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    # Save model in XGBoost's native JSON format.
    model_path = out_model.with_suffix(".xgb.json")
    booster.save_model(model_path)

    out_model.write_text(
        json.dumps({"report": report, "model": str(model_path), "cache": str(cache_path)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(json.dumps({"ok": True, "model": str(model_path), "report": str(out_model)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
