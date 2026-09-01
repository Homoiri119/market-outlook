"""Backfill the paper-portfolio tracker from past Discord recommendations.

Reads backend/data/recos_backfill.json — a list of the candidates that were sent
to Discord on past mornings — seeds them as paper positions, then re-prices every
position against J-Quants bars from its recommendation date, and writes
docs/portfolio.json + docs/portfolio.html so the equity curve starts from that day.

Seed file format:
  [
    {"date": "2026-08-24", "code": "8604", "name": "野村HD",
     "entry": 1642, "stop_loss": 1593, "take_profit": 1723, "trail_stop": 1561},
    ...
  ]

Idempotent: positions are keyed by code-date, so re-running does not duplicate them.
Run from repo root:  python backend/scripts/run_portfolio_backfill.py   (needs JQUANTS_API_KEY)
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

from app.services import portfolio  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("portfolio_backfill")

DOCS_DIR = REPO_ROOT / "docs"
SEED_FILE = BACKEND_DIR / "data" / "recos_backfill.json"
TEMPLATE = BACKEND_DIR / "app" / "static" / "portfolio.html"


def _jst_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9)))


def main() -> int:
    if not SEED_FILE.exists():
        logger.error("Seed file not found: %s", SEED_FILE)
        return 1
    recos = json.loads(SEED_FILE.read_text(encoding="utf-8"))
    recos.sort(key=lambda r: (r.get("date", ""), str(r.get("code"))))

    # Group by date and replay day-by-day (record that day's picks, then re-price
    # through that date). This mirrors the live morning flow exactly, so a name that
    # is still open is not re-bought, and closes happen in chronological order.
    by_date: dict[str, list[dict]] = {}
    for r in recos:
        by_date.setdefault(r["date"], []).append(r)

    pdata = portfolio.load(DOCS_DIR)
    added = 0
    for d in sorted(by_date):
        for r in by_date[d]:
            if portfolio.open_position(
                pdata, d, str(r["code"]), r.get("name", ""),
                r.get("entry"), r.get("stop_loss"), r.get("take_profit"), r.get("trail_stop"),
            ):
                added += 1
        portfolio.update_positions(pdata, today=dt.date.fromisoformat(d))
    logger.info("Seeded %d new positions (%d total)", added, len(pdata["positions"]))

    # Final re-price to bring every still-open position up to today.
    portfolio.update_positions(pdata)
    summary = portfolio.summarize(pdata)
    summary["generated_at"] = _jst_now().strftime("%Y-%m-%d %H:%M JST")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "portfolio.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    template = TEMPLATE.read_text(encoding="utf-8")
    html = template.replace("</head>", "<script>window.__PF__ = " + json.dumps(summary, ensure_ascii=False) + ";</script>\n</head>", 1)
    (DOCS_DIR / "portfolio.html").write_text(html, encoding="utf-8")
    (DOCS_DIR / ".nojekyll").write_text("", encoding="utf-8")
    logger.info("Portfolio: equity ¥%s (%d open / %d closed)",
                f"{summary['equity']:,}", summary["n_open"], summary["n_closed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
