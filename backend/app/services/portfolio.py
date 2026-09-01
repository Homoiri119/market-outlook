"""Paper-trading tracker for the daily Discord buy candidates.

Each morning the outlook job proposes buy candidates (entry / stop-loss / take-profit).
This module records them as simulated positions, then re-prices open positions against
subsequent J-Quants bars: a position closes at the stop-loss (損切) or take-profit (利確)
if the day's range reaches it, otherwise it is marked-to-market at the latest close.

Sizing follows the same 1%-risk rule shown on the strategy page:
  shares = floor( initial_capital * risk_pct / (entry - stop_loss) ) rounded to a 100-lot.

The result feeds docs/portfolio.json and the 実績 dashboard page. Simplified paper
tracking — no cash/margin constraint beyond a concurrent-position cap, entry assumed at
the recommended price, fees/slippage ignored. Not investment advice.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

from app.clients.jquants_client import jquants_client

logger = logging.getLogger(__name__)

INITIAL_CAPITAL = 1_000_000
RISK_PCT = 0.01
LOT = 100             # Japanese round lot


def load(docs_dir: Path) -> dict:
    f = docs_dir / "portfolio.json"
    if f.exists():
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            d.setdefault("initial_capital", INITIAL_CAPITAL)
            d.setdefault("risk_pct", RISK_PCT)
            d.setdefault("positions", [])
            return d
        except (ValueError, OSError):
            logger.warning("Could not parse portfolio.json; starting fresh")
    return {"initial_capital": INITIAL_CAPITAL, "risk_pct": RISK_PCT, "positions": []}


def _shares(capital: float, risk_pct: float, entry: float, sl: float) -> int:
    if entry is None or sl is None or entry <= sl:
        return 0
    raw = capital * risk_pct / (entry - sl)
    lots = int(raw // LOT) * LOT
    return max(LOT, lots)


def open_position(data: dict, date: str, code: str, name: str, entry: float,
                  sl: float, tp: float, trail: float | None = None) -> bool:
    """Append one paper position (equal-risk sizing). Skips duplicates (same
    code-date id) and codes already held open. Returns True if a position was added."""
    code = str(code)
    pos_id = f"{code}-{date}"
    if any(p["id"] == pos_id for p in data["positions"]):
        return False
    if any(p["code"] == code and p["status"] == "open" for p in data["positions"]):
        return False
    sh = _shares(data["initial_capital"], data["risk_pct"], entry, sl)
    if not sh or not entry or not tp:
        return False
    data["positions"].append({
        "id": pos_id, "code": code, "name": name or code, "date": date,
        "entry": entry, "stop_loss": sl, "take_profit": tp, "trail_stop": trail,
        "shares": sh, "status": "open", "exit_date": None, "exit_price": None,
        "reason": None, "last_price": entry, "pnl": 0.0, "pnl_pct": 0.0,
    })
    return True


def record_recommendations(data: dict, top: list[dict], date: str) -> None:
    """Open a paper position for EVERY candidate (not already held). Sizing is
    equal-risk (1% of the reference capital per trade), so it does not depend on
    whether the capital could actually "afford" the shares — every candidate is
    recorded, and positions are not capped."""
    for s in top:
        open_position(data, date, str(s.get("code")), s.get("name", ""),
                      s.get("current_price"), s.get("stop_loss"),
                      s.get("take_profit"), s.get("trail_stop"))


def update_positions(data: dict, today: dt.date | None = None) -> None:
    """Re-price open positions against bars since their recommendation date."""
    today = today or dt.date.today()
    for p in data["positions"]:
        if p["status"] != "open":
            continue
        try:
            start = dt.date.fromisoformat(p["date"])
        except ValueError:
            continue
        try:
            bars = jquants_client.fetch_daily_quotes(p["code"], start, today)
        except Exception:
            logger.exception("bars failed for %s", p["code"])
            continue
        if bars.empty:
            continue
        entry, sl, tp, sh = p["entry"], p["stop_loss"], p["take_profit"], p["shares"]
        exited = False
        for d, row in bars.iterrows():
            hi, lo, cl = row.get("high"), row.get("low"), row.get("close")
            dd = d.isoformat() if hasattr(d, "isoformat") else str(d)
            # Stop-loss checked first (conservative), then take-profit.
            if lo is not None and sl is not None and lo <= sl:
                p.update(status="closed", exit_date=dd, exit_price=sl, reason="損切")
                exited = True; break
            if hi is not None and tp is not None and hi >= tp:
                p.update(status="closed", exit_date=dd, exit_price=tp, reason="利確")
                exited = True; break
            if cl is not None:
                p["last_price"] = float(cl)
        px = p["exit_price"] if exited else p.get("last_price", entry)
        p["pnl"] = float((px - entry) * sh)
        p["pnl_pct"] = float(px / entry - 1) if entry else 0.0


def summarize(data: dict) -> dict:
    """Compute summary KPIs and an equity curve (¥) over time."""
    positions = data["positions"]
    init = data["initial_capital"]
    closed = [p for p in positions if p["status"] == "closed"]
    open_ps = [p for p in positions if p["status"] == "open"]
    realized = sum(p["pnl"] for p in closed)
    unrealized = sum(p["pnl"] for p in open_ps)
    wins = [p for p in closed if p["pnl"] > 0]

    # Equity curve: initial capital, stepping at each closed exit (realized), then a
    # final point today that also folds in current unrealized P/L.
    curve = []
    if positions:
        first_date = min(p["date"] for p in positions)
        curve.append({"date": first_date, "equity": round(init)})
        cum = 0.0
        for p in sorted(closed, key=lambda x: x["exit_date"] or ""):
            cum += p["pnl"]
            curve.append({"date": p["exit_date"], "equity": round(init + cum)})
        today = dt.date.today().isoformat()
        curve.append({"date": today, "equity": round(init + realized + unrealized)})

    equity_now = init + realized + unrealized
    return {
        "initial_capital": init,
        "risk_pct": data["risk_pct"],
        "equity": round(equity_now),
        "total_pnl": round(realized + unrealized),
        "total_pnl_pct": (realized + unrealized) / init if init else 0.0,
        "realized": round(realized),
        "unrealized": round(unrealized),
        "n_positions": len(positions),
        "n_open": len(open_ps),
        "n_closed": len(closed),
        "win_rate": (len(wins) / len(closed)) if closed else None,
        "curve": curve,
        "positions": sorted(positions, key=lambda p: p["date"], reverse=True),
    }
