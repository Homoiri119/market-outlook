"""Generate the IPO secondary-investment backtest page (J-Quants V2).

Run from repo root:  python backend/scripts/run_ipo_backtest.py
Writes docs/ipo.html + docs/ipo.json. Requires JQUANTS_API_KEY.

Detection scans the listed universe, so this is a heavier job — run it as an
occasional batch (monthly), not every morning.
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

from app.services.ipo_backtest import run_ipo_backtest  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_ipo")

DOCS_DIR = REPO_ROOT / "docs"
TEMPLATE = BACKEND_DIR / "app" / "static" / "ipo.html"


def main() -> int:
    bt = run_ipo_backtest()
    bt["generated_at"] = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M JST")
    for h in bt["holds"]:
        st = bt["by_hold"][str(h)]
        if st.get("n"):
            logger.info("  hold=%-2d n=%d win=%.0f%% mean=%.1f%% median=%.1f%%",
                        h, st["n"], st["win_rate"] * 100, st["mean"] * 100, st["median"] * 100)

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "ipo.json").write_text(json.dumps(bt, ensure_ascii=False), encoding="utf-8")
    template = TEMPLATE.read_text(encoding="utf-8")
    html = template.replace("</head>", "<script>window.__IPO__ = " + json.dumps(bt, ensure_ascii=False) + ";</script>\n</head>", 1)
    (DOCS_DIR / "ipo.html").write_text(html, encoding="utf-8")
    (DOCS_DIR / ".nojekyll").write_text("", encoding="utf-8")
    logger.info("Wrote docs/ipo.html and docs/ipo.json (IPOs=%d, excluded=%d)", bt["n_ipos"], bt["n_excluded"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
