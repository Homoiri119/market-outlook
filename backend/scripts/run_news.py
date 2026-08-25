"""Build the theme-based news page (Google News RSS + material tags).

Reads themes from data/themes.json (name + search query), fetches recent headlines
per theme, tags 好/悪材料 with materiality weights, and publishes a static page
(docs/news.html) + JSON. No API key needed.

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

from app.services.news import fetch_news, summarize  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_news")

DOCS_DIR = REPO_ROOT / "docs"
TEMPLATE = BACKEND_DIR / "app" / "static" / "news.html"
THEMES_FILE = BACKEND_DIR / "data" / "themes.json"

DEFAULT_THEMES = [
    {"name": "半導体・AI", "query": "半導体 関連株 OR AI半導体"},
    {"name": "防衛", "query": "防衛 関連株 OR 防衛費"},
]


def _load_themes() -> list[dict]:
    try:
        data = json.loads(THEMES_FILE.read_text(encoding="utf-8"))
        themes = [t for t in data.get("themes", []) if t.get("name") and t.get("query")]
        return themes or DEFAULT_THEMES
    except Exception:
        logger.info("themes.json missing/invalid; using defaults", exc_info=True)
        return DEFAULT_THEMES


def main() -> int:
    themes = _load_themes()
    results = []
    for t in themes:
        headlines = fetch_news(t["query"], limit=8, days=7)
        summ = summarize(headlines)
        results.append({"name": t["name"], "query": t["query"], **summ, "headlines": headlines})
        logger.info("  %s: %d件 (好%d/悪%d, score%+d)", t["name"], summ["count"], summ["good"], summ["bad"], summ["score"])
        time.sleep(0.5)

    order = {"bad": 0, "good": 1, "neutral": 2}
    results.sort(key=lambda s: (order.get(s["sentiment"], 3), -abs(s["score"])))
    payload = {
        "generated_at": dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M JST"),
        "count": len(results),
        "themes_config": themes,
        "themes": results,
    }

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "news.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    template = TEMPLATE.read_text(encoding="utf-8")
    html = template.replace("</head>", "<script>window.__NEWS__ = " + json.dumps(payload, ensure_ascii=False) + ";</script>\n</head>", 1)
    (DOCS_DIR / "news.html").write_text(html, encoding="utf-8")
    (DOCS_DIR / ".nojekyll").write_text("", encoding="utf-8")
    logger.info("Wrote docs/news.html and docs/news.json (%d themes)", len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
