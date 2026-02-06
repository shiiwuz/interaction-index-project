#!/usr/bin/env python3
"""Convert Telegram HTML export (messages*.html) into training-friendly JSONL/CSV.

Designed for the standard Telegram Desktop HTML export format.
- Extracts sender, timestamp, text (incl. captions), and basic media/file references.
- Keeps the raw HTML of text blocks out of the output; stores plain text.

Usage:
  python3 scripts/clean_telegram_html_export.py \
    --input-dir inbound/chatexport \
    --out-jsonl outputs/chat_clean.jsonl \
    --out-csv outputs/chat_clean.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Iterable, Any

# Prefer lxml for speed; fallback to BeautifulSoup if needed.
try:
    import lxml.html  # type: ignore
except Exception:  # pragma: no cover
    lxml = None  # type: ignore

from bs4 import BeautifulSoup


_TZ_RE = re.compile(r"UTC([+-])(\d{1,2})(?::(\d{2}))?\s*$")


def _parse_telegram_title_datetime(s: str) -> Optional[str]:
    """Parse Telegram export date title like: '01.08.2017 05:46:52 UTC+10:00'.

    Returns ISO 8601 string with offset, or None.
    """
    if not s:
        return None
    s = s.strip()

    m = _TZ_RE.search(s)
    tzinfo = None
    if m:
        sign = 1 if m.group(1) == "+" else -1
        hours = int(m.group(2))
        mins = int(m.group(3) or "0")
        tzinfo = timezone(sign * timedelta(hours=hours, minutes=mins))
        s_no_tz = s[: m.start()].rstrip()
    else:
        s_no_tz = s

    # Telegram uses DD.MM.YYYY HH:MM:SS
    for fmt in ("%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M"):
        try:
            dt = datetime.strptime(s_no_tz, fmt)
            if tzinfo is not None:
                dt = dt.replace(tzinfo=tzinfo)
            return dt.isoformat()
        except ValueError:
            continue
    return None


def _text_of(node) -> str:
    # lxml elements are falsy when they have no children; we only want to treat None as empty.
    if node is None:
        return ""
    # BeautifulSoup node
    if hasattr(node, "get_text"):
        txt = node.get_text("\n", strip=True)
    else:
        # lxml element
        txt = "\n".join([t.strip() for t in node.itertext() if t and t.strip()])
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt


@dataclass
class CleanMsg:
    chat_title: Optional[str]
    file: str
    message_dom_id: Optional[str]
    sender: Optional[str]
    timestamp: Optional[str]
    is_service: bool
    text: str
    reply_to: Optional[str]
    forward_from: Optional[str]
    media_type: Optional[str]
    media_path: Optional[str]
    reactions: Optional[dict[str, int]]
    reactions_total: int


def _detect_chat_title_bs4(soup: BeautifulSoup) -> Optional[str]:
    # Telegram exports usually have a header with .text bold; fallback to <title>.
    h = soup.select_one(".page_header .text")
    if h:
        t = _text_of(h)
        return t or None
    if soup.title and soup.title.string:
        return soup.title.string.strip() or None
    return None


def _detect_chat_title_lxml(doc) -> Optional[str]:
    # XPath variants for Telegram export.
    try:
        nodes = doc.xpath("//*[contains(@class,'page_header')]//*[contains(@class,'text')]")
        if nodes:
            t = _text_of(nodes[0])
            return t or None
        title = doc.xpath("string(//title)")
        title = title.strip() if isinstance(title, str) else ""
        return title or None
    except Exception:
        return None


def _guess_media_type(href: str) -> str:
    if href.startswith("video_files/"):
        return "video"
    if href.startswith("files/"):
        return "file"
    if href.startswith("photos/") or href.startswith("photo/"):
        return "photo"
    if href.startswith("stickers/"):
        return "sticker"
    if href.startswith("voice_messages/"):
        return "voice"
    return "media"


def _extract_media_bs4(msg) -> tuple[Optional[str], Optional[str]]:
    a = msg.select_one(".media_wrap a[href]")
    if a and a.get("href"):
        href = a.get("href")
        return _guess_media_type(href), href

    a2 = msg.select_one(".file_wrap a[href]")
    if a2 and a2.get("href"):
        return "file", a2.get("href")

    return None, None


def _extract_media_lxml(msg) -> tuple[Optional[str], Optional[str]]:
    try:
        nodes = msg.xpath(".//*[contains(@class,'media_wrap')]//a[@href]")
        if nodes:
            href = nodes[0].get("href")
            if href:
                return _guess_media_type(href), href
        nodes = msg.xpath(".//*[contains(@class,'file_wrap')]//a[@href]")
        if nodes:
            href = nodes[0].get("href")
            if href:
                return "file", href
    except Exception:
        pass
    return None, None


def _extract_forward_from_bs4(msg) -> Optional[str]:
    f = msg.select_one(".forwarded")
    if not f:
        return None
    t = _text_of(f)
    return t or None


def _extract_forward_from_lxml(msg) -> Optional[str]:
    try:
        nodes = msg.xpath(".//*[contains(@class,'forwarded')]")
        if nodes:
            t = _text_of(nodes[0])
            return t or None
    except Exception:
        pass
    return None


def _extract_reply_to_bs4(msg) -> Optional[str]:
    r = msg.select_one(".reply_to")
    if not r:
        return None
    t = _text_of(r)
    return t or None


def _extract_reply_to_lxml(msg) -> Optional[str]:
    try:
        nodes = msg.xpath(".//*[contains(@class,'reply_to')]")
        if nodes:
            t = _text_of(nodes[0])
            return t or None
    except Exception:
        pass
    return None


def _extract_text_bs4(msg) -> str:
    t = msg.select_one(".text")
    if t:
        return _text_of(t)

    c = msg.select_one(".media_wrap .caption")
    if c:
        return _text_of(c)

    return ""


def _extract_text_lxml(msg) -> str:
    try:
        nodes = msg.xpath(".//*[contains(@class,'text')]")
        if nodes:
            return _text_of(nodes[0])
        nodes = msg.xpath(".//*[contains(@class,'media_wrap')]//*[contains(@class,'caption')]")
        if nodes:
            return _text_of(nodes[0])
    except Exception:
        pass
    return ""


def _extract_reactions_lxml(msg) -> tuple[Optional[dict[str, int]], int]:
    """Extract Telegram reactions.

    In Telegram HTML exports, reactions can be:
    - inside the message DOM subtree
    - or as an immediate following sibling <span class="reactions"> after the message.

    Returns (reactions_dict, total_count).
    """

    def parse_reaction_container(container) -> tuple[Optional[dict[str, int]], int]:
        reaction_nodes = container.xpath(".//*[contains(@class,'reaction')]")
        if not reaction_nodes:
            return None, 0
        out: dict[str, int] = {}
        total = 0
        for r in reaction_nodes:
            e = r.xpath(".//*[contains(@class,'emoji')]")
            c = r.xpath(".//*[contains(@class,'count')]")
            emoji = _text_of(e[0]) if e else ""
            count_txt = _text_of(c[0]) if c else ""
            emoji = (emoji or "").strip()
            if not emoji:
                continue
            try:
                n = int(re.sub(r"[^0-9]", "", count_txt or "") or "0")
            except ValueError:
                n = 0
            out[emoji] = out.get(emoji, 0) + n
            total += n
        return (out or None), total

    try:
        # 1) Descendant container
        containers = msg.xpath(".//*[contains(@class,'reactions')]")
        if containers:
            d, t = parse_reaction_container(containers[0])
            if t:
                return d, t

        # 2) Following sibling container
        sib = msg.getnext()
        # Skip non-elements
        while sib is not None and not hasattr(sib, 'tag'):
            sib = sib.getnext()
        if sib is not None:
            cls = sib.get('class') or ''
            if sib.tag == 'span' and 'reactions' in cls.split():
                return parse_reaction_container(sib)

        return None, 0
    except Exception:
        return None, 0


def iter_messages_from_html(path: Path) -> Iterable[CleanMsg]:
    raw = path.read_bytes()

    # Fast path: lxml
    if "lxml" in globals() and globals().get("lxml") is not None:  # type: ignore
        doc = lxml.html.fromstring(raw)  # type: ignore
        chat_title = _detect_chat_title_lxml(doc)

        for msg in doc.xpath("//div[contains(concat(' ', normalize-space(@class), ' '), ' message ')]"):
            cls = msg.get("class") or ""
            classes = set(cls.split())
            is_service = "service" in classes

            sender = None
            if not is_service:
                nodes = msg.xpath(".//*[contains(@class,'from_name')]")
                sender = _text_of(nodes[0]) if nodes else None
                sender = sender or None

            date_nodes = msg.xpath(".//*[contains(@class,'date')]")
            title = date_nodes[0].get("title") if date_nodes else ""
            ts = _parse_telegram_title_datetime(title or "")

            text = _extract_text_lxml(msg)
            reply_to = _extract_reply_to_lxml(msg)
            forward_from = _extract_forward_from_lxml(msg)
            media_type, media_path = _extract_media_lxml(msg)
            reactions, reactions_total = _extract_reactions_lxml(msg)

            yield CleanMsg(
                chat_title=chat_title,
                file=path.name,
                message_dom_id=msg.get("id"),
                sender=sender,
                timestamp=ts,
                is_service=is_service,
                text=text,
                reply_to=reply_to,
                forward_from=forward_from,
                media_type=media_type,
                media_path=media_path,
                reactions=reactions,
                reactions_total=reactions_total,
            )

        return

    # Fallback: BeautifulSoup
    soup = BeautifulSoup(raw, "html.parser")
    chat_title = _detect_chat_title_bs4(soup)

    for msg in soup.select("div.message"):
        classes = set(msg.get("class") or [])
        is_service = "service" in classes

        sender = None
        if not is_service:
            fn = msg.select_one(".from_name")
            sender = _text_of(fn) or None

        date = msg.select_one(".date")
        ts = _parse_telegram_title_datetime(date.get("title") if date else "")

        text = _extract_text_bs4(msg)
        reply_to = _extract_reply_to_bs4(msg)
        forward_from = _extract_forward_from_bs4(msg)
        media_type, media_path = _extract_media_bs4(msg)

        yield CleanMsg(
            chat_title=chat_title,
            file=path.name,
            message_dom_id=msg.get("id"),
            sender=sender,
            timestamp=ts,
            is_service=is_service,
            text=text,
            reply_to=reply_to,
            forward_from=forward_from,
            media_type=media_type,
            media_path=media_path,
            reactions=None,
            reactions_total=0,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--min-text-len", type=int, default=0)
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    html_files = sorted(in_dir.glob("messages*.html"), key=lambda p: p.name)
    if not html_files:
        raise SystemExit(f"No messages*.html found under {in_dir}")

    out_jsonl = Path(args.out_jsonl)
    out_csv = Path(args.out_csv)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Stream outputs to avoid holding everything in memory.
    n = 0
    n_text = 0
    n_media = 0

    with out_jsonl.open("w", encoding="utf-8") as jf, out_csv.open("w", encoding="utf-8", newline="") as cf:
        w = csv.DictWriter(
            cf,
            fieldnames=[
                "chat_title",
                "file",
                "message_dom_id",
                "sender",
                "timestamp",
                "is_service",
                "text",
                "reply_to",
                "forward_from",
                "media_type",
                "media_path",
                "reactions",
                "reactions_total",
            ],
        )
        w.writeheader()

        for p in html_files:
            # Progress marker for large exports.
            print(f"[clean] parsing {p.name} ...")
            for m in iter_messages_from_html(p):
                if args.min_text_len and len(m.text) < args.min_text_len and not m.media_path:
                    continue
                d = asdict(m)
                jf.write(json.dumps(d, ensure_ascii=False) + "\n")
                w.writerow(d)

                n += 1
                if m.text:
                    n_text += 1
                if m.media_path:
                    n_media += 1

    print(
        json.dumps(
            {
                "ok": True,
                "messages": n,
                "with_text": n_text,
                "with_media": n_media,
                "out_jsonl": str(out_jsonl),
                "out_csv": str(out_csv),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
