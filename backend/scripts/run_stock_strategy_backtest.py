"""Generate the individual-stock strategy backtest page (J-Quants V2).

Run from repo root:  python backend/scripts/run_stock_strategy_backtest.py [years]
Writes docs/stock_strategy.html + docs/stock_strategy.json. Requires JQUANTS_API_KEY.
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

from app.services.stock_strategy_backtest import run_stock_strategy_backtest  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_stock_strategy")

DOCS_DIR = REPO_ROOT / "docs"
TEMPLATE = BACKEND_DIR / "app" / "static" / "stock_strategy.html"
YEARS = float(sys.argv[1]) if len(sys.argv) > 1 else 3.5


def main() -> int:
    bt = run_stock_strategy_backtest(years=YEARS)
    bt["generated_at"] = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M JST")
    for name, st in bt["configs"].items():
        if st.get("trades"):
            logger.info("  %-7s trades=%d win=%.1f%% expR=%.3f PF=%s totalR=%.1f",
                        name, st["trades"], st["win_rate"] * 100, st["expectancy_R"],
                        round(st["profit_factor"], 2) if st.get("profit_factor") else "—", st["total_R"])

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "stock_strategy.json").write_text(json.dumps(bt, ensure_ascii=False), encoding="utf-8")
    template = TEMPLATE.read_text(encoding="utf-8")
    html = template.replace("</head>", "<script>window.__SS__ = " + json.dumps(bt, ensure_ascii=False) + ";</script>\n</head>", 1)
    (DOCS_DIR / "stock_strategy.html").write_text(html, encoding="utf-8")
    (DOCS_DIR / ".nojekyll").write_text("", encoding="utf-8")
    logger.info("Wrote docs/stock_strategy.html and docs/stock_strategy.json (universe=%d)", bt["universe_size"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
