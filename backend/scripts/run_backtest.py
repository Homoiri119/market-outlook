"""Generate the backtest page for GitHub Pages.

Run from the repo root:  python backend/scripts/run_backtest.py
Writes docs/backtest.html (self-contained, data embedded) and docs/backtest.json.
Index-level (Nikkei 225), yfinance only, no API key.
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

from app.services.backtest import run_index_backtest  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_backtest")

DOCS_DIR = REPO_ROOT / "docs"
TEMPLATE = BACKEND_DIR / "app" / "static" / "backtest.html"

YEARS = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0


def _jst_now() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M JST")


def main() -> int:
    bt = run_index_backtest(years=YEARS)
    bt["generated_at"] = _jst_now()
    logger.info(
        "Backtest %s..%s: hit-rate %.1f%%, long/flat total %.1f%% vs buy&hold %.1f%%",
        bt["start"], bt["end"], bt["hit_rate"] * 100,
        bt["strategies"]["long_flat"]["total_return"] * 100,
        bt["strategies"]["buy_hold"]["total_return"] * 100,
    )

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "backtest.json").write_text(json.dumps(bt, ensure_ascii=False, indent=2), encoding="utf-8")

    template = TEMPLATE.read_text(encoding="utf-8")
    inject = "<script>window.__BT__ = " + json.dumps(bt, ensure_ascii=False) + ";</script>\n</head>"
    html = template.replace("</head>", inject, 1)
    (DOCS_DIR / "backtest.html").write_text(html, encoding="utf-8")
    (DOCS_DIR / ".nojekyll").write_text("", encoding="utf-8")
    logger.info("Wrote docs/backtest.html and docs/backtest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
