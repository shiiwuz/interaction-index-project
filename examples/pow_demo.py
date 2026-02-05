#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request


def sha_ok(difficulty: int, data: str) -> bool:
    h = hashlib.sha256(data.encode("utf-8")).hexdigest()
    return h.startswith("0" * difficulty)


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

    ch = json.loads(urllib.request.urlopen(base + "/pow/challenge").read())
    pow_id = ch["pow_id"]
    challenge = ch["challenge"]
    diff = int(ch["difficulty"])

    start = time.time()
    nonce = 0
    while True:
        s = str(nonce)
        if sha_ok(diff, f"{pow_id}:{challenge}:{s}"):
            break
        nonce += 1

    solved_ms = int((time.time() - start) * 1000)

    req = urllib.request.Request(
        base + "/predict",
        data=json.dumps({"text": "demo title\ndemo body\nhttps://example.com"}).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-pow-id": pow_id,
            "x-pow-nonce": str(nonce),
        },
        method="POST",
    )
    out = urllib.request.urlopen(req).read().decode("utf-8", "ignore")
    print(json.dumps({"solved_ms": solved_ms, "nonce": nonce, "response": json.loads(out)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
