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


V2_HOSTS = ["https://api.jquants.com/v2", "https://api.jquants-pro.com/v2"]


def _summarize(body) -> str:
    if not isinstance(body, dict):
        return str(body)[:80]
    parts = []
    for k, v in body.items():
        if k == "pagination_key":
            continue
        if isinstance(v, list):
            cols = sorted(v[0].keys()) if v and isinstance(v[0], dict) else []
            parts.append(f"{k}[{len(v)}] cols={cols}")
        else:
            parts.append(f"{k}={repr(v)[:50]}")
    return " ; ".join(parts) if parts else "(empty)"


def main() -> int:
    end = dt.date.today()
    start = end - dt.timedelta(days=20)
    d1, d2 = start.isoformat(), end.isoformat()

    print("Key diagnostics:", _fingerprint(RAW_KEY), "\n")

    # Confirmed V2 paths (renamed from V1). /fins/details is the key-validity canary
    # (403 "not available on your subscription" = key OK but endpoint not in plan).
    probes = [
        ("/fins/details", {"code": "86970", "date": "20230130"}),        # key canary
        ("/equities/master", {}),                                        # listed info + sectors
        ("/equities/bars/daily", {"code": "7203", "from": d1, "to": d2}),  # daily quotes
        ("/indices/bars/daily/topix", {"from": d1, "to": d2}),           # TOPIX
        ("/indices/bars/daily", {"code": "0028", "from": d1, "to": d2}),  # sector index
        # Fundamentals — needed for PER/PBR/ROE/dividend analysis:
        ("/fins/statements", {"code": "7203"}),
        ("/fins/dividend", {"code": "7203"}),
        # External-info features (calendar/alerts + supply-demand):
        ("/fins/announcement", {}),                       # earnings announcement calendar
        ("/markets/weekly-margin-interest", {"code": "7203"}),  # 信用残
        ("/markets/short-selling", {"from": d1, "to": d2}),     # 空売り
        ("/markets/trades-spec", {"from": d1, "to": d2}),       # 投資部門別売買
    ]

    working_host = None
    for host in ["https://api.jquants.com/v2"]:
        print(f"=== {host}  (x-api-key) ===")
        for path, params in probes:
            status, body = _get(host, path, params, {"x-api-key": API_KEY})
            pstr = str(params) if params else ""
            print(f"  {path:28s} {pstr:<38} -> HTTP {status}  {_summarize(body)[:200]}")
            # Key is valid if any endpoint authenticates (200) or reports a plan limit.
            msg = body.get("message", "") if isinstance(body, dict) else ""
            if working_host is None and (status == 200 or "subscription" in msg):
                working_host = host
        print()

    # Show which sector fields appear in /equities/master.
    s, body = _get("https://api.jquants.com/v2", "/equities/master", {}, {"x-api-key": API_KEY})
    if isinstance(body, dict):
        arr = next((v for v in body.values() if isinstance(v, list)), [])
        sample = arr[0] if arr else {}
        if sample:
            print("equities/master ALL fields:", sorted(sample.keys()))
            print("sector-related:", {k: sample[k] for k in sample if "ector" in k or "arket" in k or "17" in k or "33" in k})
        print()

    # Fundamentals schema (if the plan allows it).
    s, body = _get("https://api.jquants.com/v2", "/fins/statements", {"code": "7203"}, {"x-api-key": API_KEY})
    print(f"/fins/statements -> HTTP {s}")
    if isinstance(body, dict):
        arr = next((v for v in body.values() if isinstance(v, list)), [])
        if arr:
            print("  fins/statements ALL fields:", sorted(arr[-1].keys()))
    print()

    # Also try the mail/password token flow (in case the account still supports it).
    if MAIL and PASSWORD:
        print("=== Mail/password token flow (secondary) ===")
        _cred_diagnostics()
        for base in TOKEN_FLOW_BASES:
            try_token_flow(base)
            print()

    if working_host:
        print(f"PASS: key works on {working_host} via x-api-key. "
              f"Paste this whole output so the correct V2 paths + sector fields can be wired.")
        return 0
    print("FAIL: key did not authenticate on /fins/details on any V2 host.")
    print("→ Paste this output; if even /fins/details is 401/403, the dashboard's exact sample is needed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
