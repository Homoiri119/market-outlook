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

API_KEY = os.environ.get("JQUANTS_API_KEY", "")

CANDIDATE_BASE_URLS = [
    "https://api.jquants-pro.com/v2",
    "https://api.jquants.com/v2",
    "https://api.jquants.com/v1",  # (would need a token, listed here only to detect 401 vs 404)
]

SECTOR_FIELDS = [
    "Sector17Code", "Sector17CodeName", "Sector33Code", "Sector33CodeName",
    "MarketCode", "MarketCodeName", "CompanyName", "Code",
]


def _get(base: str, path: str, params: dict) -> tuple[int, dict | str]:
    try:
        r = httpx.get(f"{base}{path}", params=params, headers={"x-api-key": API_KEY}, timeout=30)
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
    print(f"API key detected (length={len(API_KEY)}). Testing base URLs with x-api-key header...\n")

    working_base = None
    for base in CANDIDATE_BASE_URLS:
        status, body = _get(base, "/listed/info", {})
        n = len(body.get("info", [])) if isinstance(body, dict) else 0
        note = ""
        if isinstance(body, dict) and "message" in body:
            note = f" msg={body['message'][:80]}"
        print(f"[{base}] /listed/info -> HTTP {status}, rows={n}{note}")
        if status == 200 and n > 0 and working_base is None:
            working_base = base

    if not working_base:
        print("\nFAIL: No base URL returned listed/info with the given key.")
        print("→ Check the key value in Secrets/.env, and whether your plan is Pro (api.jquants-pro.com) or standard (api.jquants.com).")
        return 1

    print(f"\nWORKING BASE URL: {working_base}\n")

    # Inspect the listed/info schema (which sector fields are present).
    _, info = _get(working_base, "/listed/info", {})
    sample = (info.get("info") or [{}])[0]
    print("listed/info sample fields present:")
    for f in SECTOR_FIELDS:
        print(f"  {f:18s}: {'YES' if f in sample else 'no'}  {repr(sample.get(f))[:40]}")

    # Daily quotes for one liquid name (Toyota 7203 / 72030).
    end = dt.date.today()
    start = end - dt.timedelta(days=20)
    for code in ("7203", "72030"):
        s, body = _get(working_base, "/prices/daily_quotes",
                       {"code": code, "from": start.isoformat(), "to": end.isoformat()})
        n = len(body.get("daily_quotes", [])) if isinstance(body, dict) else 0
        print(f"\n/prices/daily_quotes code={code} -> HTTP {s}, rows={n}")
        if n:
            print("  columns:", sorted((body["daily_quotes"][0]).keys()))
            break

    # Index endpoint (TOPIX) — may be premium.
    for path, params in [("/indices", {"code": "0000", "from": start.isoformat(), "to": end.isoformat()}),
                         ("/indices/topix", {"from": start.isoformat(), "to": end.isoformat()})]:
        s, body = _get(working_base, path, params)
        keys = [k for k in (body.keys() if isinstance(body, dict) else []) if k != "pagination_key"]
        print(f"\n{path} -> HTTP {s}, keys={keys}")

    print("\nPASS: connectivity OK. Paste this output (no secrets shown) so the sector/stock backtest can be built.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
