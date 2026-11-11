"""Survivorship-free backtest of legendary-trader style rules (J-Quants V2).

Run from repo root:  python backend/scripts/run_legend_backtest.py
Writes reports/legend_backtest_<date>.md and reports/legend_backtest_results.json.
Requires JQUANTS_API_KEY. Fetches ~3.8 years of bars-by-date (one call per trading
day), so it is a heavy, occasional batch.
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

from app.services.legend_backtest import render_markdown, run_legend_backtest  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_legend")

REPORTS_DIR = REPO_ROOT / "reports"


def main() -> int:
    res = run_legend_backtest()
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=9)))
    res["generated_at"] = now.strftime("%Y-%m-%d %H:%M JST")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "legend_backtest_results.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    md = render_markdown(res, res["generated_at"])
    out = REPORTS_DIR / f"legend_backtest_{now.strftime('%Y-%m-%d')}.md"
    out.write_text(md, encoding="utf-8")
    logger.info("Wrote %s", out)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
