"""Generate the per-stock analysis page (J-Quants V2, x-api-key).

Universe = large caps (TOPIX Core30 + Large70) + the codes in data/watchlist.json.
For each stock: fetch ~1.5y of daily bars + financial statements (best effort),
compute technical + fundamental analysis and ATR-based entry/TP/SL, then publish a
browsable static page (docs/analysis.html) + JSON.

Run from repo root:  python backend/scripts/run_stock_analysis.py
Requires JQUANTS_API_KEY in the environment.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.clients.jquants_client import jquants_client  # noqa: E402
from app.services.stock_analysis import analyze_stock  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_stock_analysis")

DOCS_DIR = REPO_ROOT / "docs"
TEMPLATE = BACKEND_DIR / "app" / "static" / "analysis.html"
WATCHLIST = BACKEND_DIR / "data" / "watchlist.json"
SCALE_CATS = ("TOPIX Core30", "TOPIX Large70")


def _load_watchlist() -> list[str]:
    try:
        data = json.loads(WATCHLIST.read_text(encoding="utf-8"))
        return [str(c) for c in data.get("codes", [])]
    except Exception:
        return []


def main() -> int:
    end = dt.date.today()
    start = end - dt.timedelta(days=1900)  # ~5y so weekly/monthly charts have enough bars

    master = jquants_client.fetch_listed_info()
    if master.empty or "Code" not in master.columns:
        raise RuntimeError("Could not fetch equities master.")

    # Universe: large caps by ScaleCat + watchlist codes.
    large = master[master["ScaleCat"].isin(SCALE_CATS)] if "ScaleCat" in master.columns else master.head(0)
    # Index master rows (as plain dicts) by both the full code and its 4-digit form,
    # so 4-digit watchlist codes match 5-digit master codes.
    info_by_code: dict[str, dict] = {}
    for _, r in master.iterrows():
        d = r.to_dict()
        c = str(d.get("Code"))
        info_by_code.setdefault(c, d)
        info_by_code.setdefault(c[:4], d)

    def canon(c: str) -> str:
        d = info_by_code.get(c) or info_by_code.get(c[:4])
        return str(d.get("Code")) if d else c

    # Canonicalize to the master code so 7203 (watchlist) and 72030 (master) don't duplicate.
    codes = list(dict.fromkeys(canon(c) for c in list(large["Code"].astype(str)) + _load_watchlist()))
    logger.info("Universe size: %d (large caps %d + watchlist)", len(codes), len(large))

    results = []
    fund_on, empty_streak = True, 0
    for i, code in enumerate(codes):
        info = info_by_code.get(code) or info_by_code.get(code[:4]) or {}
        name = str(info.get("CoName", code))
        sector = str(info.get("S17Nm", ""))
        try:
            bars = jquants_client.fetch_daily_quotes(code, start, end)
        except Exception:
            logger.exception("bars failed for %s", code)
            continue
        if bars.empty:
            continue
        statements = []
        if fund_on:
            statements = jquants_client.fetch_statements(code)
            if statements:
                empty_streak = 0
            else:
                empty_streak += 1
                if empty_streak >= 5:  # plan clearly lacks statements — stop trying
                    fund_on = False
                    logger.info("Disabling fundamentals (fins/statements not in plan)")
        res = analyze_stock(code, name, sector, bars, statements)
        if res:
            results.append(res)
        if (i + 1) % 20 == 0:
            logger.info("  analyzed %d/%d", i + 1, len(codes))

    # Sort: strongest buys first, then by score.
    order = {"BUY": 0, "WEAK_BUY": 1, "NEUTRAL": 2, "WEAK_SELL": 3, "SELL": 4}
    results.sort(key=lambda r: (order.get(r["direction"], 9), -r["score"]))

    has_fund = sum(1 for r in results if r.get("fundamentals"))
    payload = {
        "generated_at": dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M JST"),
        "count": len(results),
        "fundamentals_available": has_fund > 0,
        "stocks": results,
    }
    logger.info("Analyzed %d stocks (%d with fundamentals)", len(results), has_fund)

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "analysis.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    template = TEMPLATE.read_text(encoding="utf-8")
    html = template.replace("</head>", "<script>window.__ANALYSIS__ = " + json.dumps(payload, ensure_ascii=False) + ";</script>\n</head>", 1)
    (DOCS_DIR / "analysis.html").write_text(html, encoding="utf-8")
    (DOCS_DIR / ".nojekyll").write_text("", encoding="utf-8")
    logger.info("Wrote docs/analysis.html and docs/analysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
