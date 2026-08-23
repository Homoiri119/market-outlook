"""J-Quants connectivity self-test (V2 API key).

Confirms which base URL works and what data/schema is available, so the
sector/stock backtest can be built on a verified foundation. Reads the key from
the environment (JQUANTS_API_KEY) and NEVER prints it.

Run locally:  JQUANTS_API_KEY=... python backend/scripts/jquants_check.py
Or in CI:     provided via the JQUANTS_API_KEY secret.
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

import httpx

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

RAW_KEY = os.environ.get("JQUANTS_API_KEY", "")
API_KEY = RAW_KEY.strip()

CANDIDATE_BASE_URLS = [
    "https://api.jquants-pro.com/v2",
    "https://api.jquants.com/v2",
    "https://api.jquants.com/v1",  # (would need a token, listed here only to detect 401 vs 404)
]

# Auth header styles to try (the docs disagree; we resolve it empirically).
AUTH_STYLES = [
    ("x-api-key", lambda k: {"x-api-key": k}),
    ("Authorization: Bearer", lambda k: {"Authorization": f"Bearer {k}"}),
    ("Authorization: raw", lambda k: {"Authorization": k}),
]


def _fingerprint(raw: str) -> str:
    stripped = raw.strip()
    ws = "YES (leading/trailing whitespace!)" if raw != stripped else "no"
    mask = f"{stripped[:2]}…{stripped[-2:]}" if len(stripped) >= 4 else "(too short)"
    return f"raw_len={len(raw)} stripped_len={len(stripped)} whitespace={ws} fingerprint={mask}"

SECTOR_FIELDS = [
    "Sector17Code", "Sector17CodeName", "Sector33Code", "Sector33CodeName",
    "MarketCode", "MarketCodeName", "CompanyName", "Code",
]


def _get(base: str, path: str, params: dict, headers: dict) -> tuple[int, dict | str]:
    try:
        r = httpx.get(f"{base}{path}", params=params, headers=headers, timeout=30)
    except Exception as e:  # pragma: no cover
        return -1, f"request error: {type(e).__name__}"
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, r.text[:200]


def main() -> int:
    if not API_KEY:
        print("FAIL: JQUANTS_API_KEY is not set in the environment.")
        return 1
    print("Key diagnostics:", _fingerprint(RAW_KEY), "\n")

    # Step 1: find a (base URL, auth style) combination that authenticates.
    working = None  # (base, style_name, headers_fn)
    for base in CANDIDATE_BASE_URLS:
        for style_name, hfn in AUTH_STYLES:
            status, body = _get(base, "/listed/info", {}, hfn(API_KEY))
            n = len(body.get("info", [])) if isinstance(body, dict) else 0
            msg = body.get("message", "")[:70] if isinstance(body, dict) else str(body)[:70]
            print(f"[{base}] [{style_name}] /listed/info -> HTTP {status}, rows={n}  {msg}")
            if status == 200 and n > 0 and working is None:
                working = (base, style_name, hfn)
        print()

    if not working:
        print("FAIL: No (base URL, auth style) combination authenticated.")
        print("→ If key diagnostics show whitespace or an unexpected length, re-copy the key from the")
        print("  J-Quants dashboard into the Secret (no quotes/spaces). Otherwise verify the plan/key.")
        return 1

    working_base, style_name, hfn = working
    print(f"WORKING: base={working_base}  auth={style_name}\n")

    def get(path, params):
        return _get(working_base, path, params, hfn(API_KEY))

    # Inspect the listed/info schema (which sector fields are present).
    _, info = get("/listed/info", {})
    sample = (info.get("info") or [{}])[0] if isinstance(info, dict) else {}
    print("listed/info sample fields present:")
    for f in SECTOR_FIELDS:
        print(f"  {f:18s}: {'YES' if f in sample else 'no'}  {repr(sample.get(f))[:40]}")

    # Daily quotes for one liquid name (Toyota 7203 / 72030).
    end = dt.date.today()
    start = end - dt.timedelta(days=20)
    for code in ("7203", "72030"):
        s, body = get("/prices/daily_quotes",
                      {"code": code, "from": start.isoformat(), "to": end.isoformat()})
        n = len(body.get("daily_quotes", [])) if isinstance(body, dict) else 0
        print(f"\n/prices/daily_quotes code={code} -> HTTP {s}, rows={n}")
        if n:
            print("  columns:", sorted((body["daily_quotes"][0]).keys()))
            break

    # Index endpoint (TOPIX) — may be premium.
    for path, params in [("/indices", {"code": "0000", "from": start.isoformat(), "to": end.isoformat()}),
                         ("/indices/topix", {"from": start.isoformat(), "to": end.isoformat()})]:
        s, body = get(path, params)
        keys = [k for k in (body.keys() if isinstance(body, dict) else []) if k != "pagination_key"]
        print(f"\n{path} -> HTTP {s}, keys={keys}")

    print("\nPASS: connectivity OK. Paste this output (no secrets shown) so the sector/stock backtest can be built.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
