"""Survivorship-free backtests of individual-stock picking (J-Quants V2), on one panel:

  1. The inverse of losing methods (app/services/inverse_backtest.py)
  2. Picks with a reason: catalysts and themes (app/services/catalyst_backtest.py)
  3. Popular entries (golden cross, MACD, volume surge) with filters (combo_backtest.py)

Run from repo root:  python backend/scripts/run_inverse_backtest.py
Writes reports/{inverse,combo,catalyst}_backtest_<date>.md and their
*_results.json. Requires JQUANTS_API_KEY. Fetches ~3.8 years of bars-by-date (one call
per trading day) plus financial disclosures by date, so it is a heavy, occasional batch.
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

from app.services import catalyst_backtest as cb  # noqa: E402
from app.services import combo_backtest as co  # noqa: E402
from app.services import inverse_backtest as ib  # noqa: E402
from app.services import legend_backtest as lb  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_inverse")

REPORTS_DIR = REPO_ROOT / "reports"


def _write(name: str, res: dict, md: str, stamp: str) -> None:
    (REPORTS_DIR / f"{name}_results.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    out = REPORTS_DIR / f"{name}_{stamp}.md"
    out.write_text(md, encoding="utf-8")
    logger.info("Wrote %s", out)
    print(md)


def main() -> int:
    panel = lb.fetch_panel()
    bench = lb.fetch_benchmark(panel)
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=9)))
    gen, stamp = now.strftime("%Y-%m-%d %H:%M JST"), now.strftime("%Y-%m-%d")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    res = ib.run_on_panel(panel, bench)
    res["generated_at"] = gen
    _write("inverse_backtest", res, ib.render_markdown(res, gen), stamp)

    res3 = co.run_on_panel(panel, bench)
    res3["generated_at"] = gen
    _write("combo_backtest", res3, co.render_markdown(res3, gen), stamp)

    dates = [d.date() for d in panel["close"].index]
    disc, fins_info = cb.fetch_disclosures(dates)
    sectors = cb.fetch_sectors([dates[0], dates[len(dates) // 2], dates[-1]])
    res2 = cb.run_on_panel(panel, bench, disc, fins_info, sectors)
    res2["generated_at"] = gen
    _write("catalyst_backtest", res2, cb.render_markdown(res2, gen), stamp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
