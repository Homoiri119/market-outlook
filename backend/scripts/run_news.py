"""Build the watchlist news page (Google News RSS + material tags).

For each code in data/watchlist.json, fetch recent headlines, tag 好/悪材料, and
publish a static page (docs/news.html) + JSON. Company names come from the
J-Quants master (JQUANTS_API_KEY); news itself needs no key.

Run from repo root:  python backend/scripts/run_news.py
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.clients.jquants_client import jquants_client  # noqa: E402
from app.services.news import fetch_news, summarize  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_news")

DOCS_DIR = REPO_ROOT / "docs"
TEMPLATE = BACKEND_DIR / "app" / "static" / "news.html"
WATCHLIST = BACKEND_DIR / "data" / "watchlist.json"


def _watchlist() -> list[str]:
    try:
        return [str(c) for c in json.loads(WATCHLIST.read_text(encoding="utf-8")).get("codes", [])]
    except Exception:
        return []


def main() -> int:
    codes = _watchlist()
    if not codes:
        logger.warning("watchlist is empty")
    name_by_code: dict[str, str] = {}
    sector_by_code: dict[str, str] = {}
    try:
        master = jquants_client.fetch_listed_info()
        for _, r in master.iterrows():
            c = str(r.get("Code"))
            name_by_code[c] = str(r.get("CoName", c)); name_by_code[c[:4]] = str(r.get("CoName", c))
            sector_by_code[c] = str(r.get("S17Nm", "")); sector_by_code[c[:4]] = str(r.get("S17Nm", ""))
    except Exception:
        logger.info("master fetch failed; using codes as names", exc_info=True)

    stocks = []
    for code in dict.fromkeys(codes):
        name = name_by_code.get(code) or name_by_code.get(code[:4]) or code
        sector = sector_by_code.get(code) or sector_by_code.get(code[:4]) or ""
        headlines = fetch_news(name, limit=6)
        summ = summarize(headlines)
        stocks.append({"code": code, "name": name, "sector": sector, **summ, "headlines": headlines})
        logger.info("  %s %s: %d件 (好%d/悪%d)", code, name, summ["count"], summ["good"], summ["bad"])
        time.sleep(0.5)  # be gentle to the RSS endpoint

    order = {"bad": 0, "good": 1, "neutral": 2}
    stocks.sort(key=lambda s: (order.get(s["sentiment"], 3), -abs(s["score"])))
    payload = {
        "generated_at": dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M JST"),
        "count": len(stocks), "stocks": stocks,
    }

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "news.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    template = TEMPLATE.read_text(encoding="utf-8")
    html = template.replace("</head>", "<script>window.__NEWS__ = " + json.dumps(payload, ensure_ascii=False) + ";</script>\n</head>", 1)
    (DOCS_DIR / "news.html").write_text(html, encoding="utf-8")
    (DOCS_DIR / ".nojekyll").write_text("", encoding="utf-8")
    logger.info("Wrote docs/news.html and docs/news.json (%d stocks)", len(stocks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
