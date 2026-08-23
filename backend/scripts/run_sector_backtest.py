"""Generate the sector-level backtest page (J-Quants V2, x-api-key).

Run from the repo root:  python backend/scripts/run_sector_backtest.py [years]
Writes docs/sector_backtest.html + docs/sector_backtest.json.
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

from app.services.sector_backtest import run_sector_backtest  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_sector_backtest")

DOCS_DIR = REPO_ROOT / "docs"
TEMPLATE = BACKEND_DIR / "app" / "static" / "sector_backtest.html"
YEARS = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0


def main() -> int:
    bt = run_sector_backtest(years=YEARS)
    bt["generated_at"] = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M JST")
    logger.info("Sector backtest: %d sectors, universe=%d, %s..%s",
                len(bt["sectors"]), bt["universe_size"], bt["start"], bt["end"])
    for s in bt["sectors"][:5]:
        logger.info("  %s: hit=%.1f%% strat=%.1f%% (%d stocks, %d days)",
                    s["name"], s["hit_rate"] * 100, s["strategy_return"] * 100, s["n_stocks"], s["n_days"])

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "sector_backtest.json").write_text(json.dumps(bt, ensure_ascii=False, indent=2), encoding="utf-8")
    template = TEMPLATE.read_text(encoding="utf-8")
    html = template.replace("</head>", "<script>window.__SBT__ = " + json.dumps(bt, ensure_ascii=False) + ";</script>\n</head>", 1)
    (DOCS_DIR / "sector_backtest.html").write_text(html, encoding="utf-8")
    (DOCS_DIR / ".nojekyll").write_text("", encoding="utf-8")
    logger.info("Wrote docs/sector_backtest.html and docs/sector_backtest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
