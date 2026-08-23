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


MAIL = os.environ.get("JQUANTS_MAIL", "").strip()
PASSWORD = os.environ.get("JQUANTS_PASSWORD", "")

# Hosts to try for the mail/password -> refreshToken -> idToken -> Bearer flow.
# Pro V2 first — confirmed to host the working auth endpoint.
TOKEN_FLOW_BASES = [
    "https://api.jquants-pro.com/v2",   # J-Quants Pro
    "https://api.jquants.com/v1",       # standard J-Quants
]


def _fingerprint(raw: str) -> str:
    stripped = raw.strip()
    ws = "YES (leading/trailing whitespace!)" if raw != stripped else "no"
    mask = f"{stripped[:2]}…{stripped[-2:]}" if len(stripped) >= 4 else "(too short)"
    return f"raw_len={len(raw)} stripped_len={len(stripped)} whitespace={ws} fingerprint={mask}"


def _post(url: str, json_body=None, params=None) -> tuple[int, dict | str]:
    try:
        r = httpx.post(url, json=json_body, params=params, timeout=30)
    except Exception as e:  # pragma: no cover
        return -1, f"request error: {type(e).__name__}"
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, r.text[:200]


def _cred_diagnostics() -> None:
    raw_mail = os.environ.get("JQUANTS_MAIL", "")
    raw_pw = os.environ.get("JQUANTS_PASSWORD", "")
    mail_ws = "YES" if raw_mail != raw_mail.strip() else "no"
    pw_ws = "YES" if raw_pw != raw_pw.strip() else "no"
    domain = MAIL.split("@", 1)[1] if "@" in MAIL else "(no @ !)"
    first = MAIL[0] if MAIL else "?"
    print("Credential diagnostics (values hidden):")
    print(f"  MAIL: len={len(raw_mail)} whitespace={mail_ws} masked={first}***@{domain}")
    print(f"  PASSWORD: len={len(raw_pw)} whitespace={pw_ws}")
    print()


def try_token_flow(base: str):
    """mail/password -> refreshToken -> idToken -> GET /listed/info. Returns (ok, idToken)."""
    print(f"--- token flow @ {base} ---")
    s1, b1 = _post(f"{base}/token/auth_user", json_body={"mailaddress": MAIL, "password": PASSWORD})
    rt = b1.get("refreshToken") if isinstance(b1, dict) else None
    msg1 = b1.get("message", "")[:70] if isinstance(b1, dict) else str(b1)[:70]
    print(f"  auth_user      -> HTTP {s1}, refreshToken={'YES' if rt else 'no'} {msg1}")
    if not rt:
        return False, None
    s2, b2 = _post(f"{base}/token/auth_refresh", params={"refreshtoken": rt})
    idt = b2.get("idToken") if isinstance(b2, dict) else None
    msg2 = b2.get("message", "")[:70] if isinstance(b2, dict) else str(b2)[:70]
    print(f"  auth_refresh   -> HTTP {s2}, idToken={'YES' if idt else 'no'} {msg2}")
    if not idt:
        return False, None
    s3, b3 = _get(base, "/listed/info", {}, {"Authorization": f"Bearer {idt}"})
    n = len(b3.get("info", [])) if isinstance(b3, dict) else 0
    print(f"  /listed/info   -> HTTP {s3}, rows={n}")
    return (s3 == 200 and n > 0), idt

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
    # --- Preferred: mail/password token flow (the chosen method) ---
    working_base = None
    id_token = None
    if MAIL and PASSWORD:
        print("=== Mail/password token flow ===")
        _cred_diagnostics()
        for base in TOKEN_FLOW_BASES:
            ok, idt = try_token_flow(base)
            print()
            if ok:
                working_base, id_token = base, idt
                break
    else:
        print("JQUANTS_MAIL / JQUANTS_PASSWORD not set — skipping token flow.\n")

    # --- Fallback probe: API-key styles (kept for diagnostics) ---
    if not working_base and API_KEY:
        print("Key diagnostics:", _fingerprint(RAW_KEY))
        print("=== API-key probe ===")
        for base in CANDIDATE_BASE_URLS:
            for style_name, hfn in AUTH_STYLES:
                status, body = _get(base, "/listed/info", {}, hfn(API_KEY))
                n = len(body.get("info", [])) if isinstance(body, dict) else 0
                msg = body.get("message", "")[:60] if isinstance(body, dict) else str(body)[:60]
                print(f"[{base}] [{style_name}] -> HTTP {status}, rows={n}  {msg}")
                if status == 200 and n > 0 and working_base is None:
                    working_base = base
            print()

    if not working_base:
        print("FAIL: Could not authenticate with mail/password token flow nor API key.")
        print("→ Verify JQUANTS_MAIL/JQUANTS_PASSWORD are correct, and whether the account is")
        print("  standard (api.jquants.com/v1) or Pro (api.jquants-pro.com/v2).")
        return 1

    print(f"WORKING BASE URL: {working_base}\n")

    def get(path, params):
        headers = {"Authorization": f"Bearer {id_token}"} if id_token else {"x-api-key": API_KEY}
        return _get(working_base, path, params, headers)

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
