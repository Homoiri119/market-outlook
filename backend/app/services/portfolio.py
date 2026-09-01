"""Paper-trading tracker for the daily Discord buy candidates.

Each morning the outlook job proposes buy candidates (entry / stop-loss / take-profit).
This module records them as simulated positions, then re-prices open positions against
subsequent J-Quants bars: a position closes at the stop-loss (損切) or take-profit (利確)
if the day's range reaches it, otherwise it is marked-to-market at the latest close.

Sizing: a fixed 100 shares (one Japanese round lot) per position. The reference capital
(運用資金) is auto-derived so it just exceeds the peak simultaneous cost of holding 100
shares of everything (rounded up to ¥10,000) — this keeps 投下資金 ≤ 運用資金 while never
dropping a candidate for lack of cash.

The result feeds docs/portfolio.json and the 実績 dashboard page. Entry assumed at the
recommended price; fees/slippage ignored. Not investment advice.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
from pathlib import Path

from app.clients.jquants_client import jquants_client

logger = logging.getLogger(__name__)

RISK_PCT = 0.01       # informational only (sizing is a fixed 100-share lot)
LOT = 100             # Japanese round lot; every position = 1 lot
CAPITAL_ROUND = 10_000  # 運用資金 is rounded up to this unit


def load(docs_dir: Path) -> dict:
    f = docs_dir / "portfolio.json"
    if f.exists():
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            d.setdefault("risk_pct", RISK_PCT)
            d.setdefault("positions", [])
            return d
        except (ValueError, OSError):
            logger.warning("Could not parse portfolio.json; starting fresh")
    return {"risk_pct": RISK_PCT, "positions": []}


def regime_by_date(docs_dir: Path) -> dict[str, dict]:
    """Map each date to that morning's outlook {dir, move} from docs/history.json."""
    f = docs_dir / "history.json"
    if not f.exists():
        return {}
    try:
        hist = json.loads(f.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    return {h["date"]: {"dir": h.get("direction"), "move": h.get("expected_move")}
            for h in hist if h.get("date")}


def attach_regime(positions: list[dict], docs_dir: Path) -> None:
    """Tag each position with the recommendation-day outlook (地合い)."""
    reg = regime_by_date(docs_dir)
    for p in positions:
        r = reg.get(p.get("date"))
        p["reg_dir"] = r["dir"] if r else None
        p["reg_move"] = r["move"] if r else None


def open_position(data: dict, date: str, code: str, name: str, entry: float,
                  sl: float, tp: float, trail: float | None = None) -> bool:
    """Append one paper position (fixed 100-share lot). Skips duplicates (same
    code-date id) and codes already held open. Returns True if a position was added."""
    code = str(code)
    pos_id = f"{code}-{date}"
    if any(p["id"] == pos_id for p in data["positions"]):
        return False
    if any(p["code"] == code and p["status"] == "open" for p in data["positions"]):
        return False
    if not entry or not tp or not sl or entry <= sl:
        return False
    data["positions"].append({
        "id": pos_id, "code": code, "name": name or code, "date": date,
        "entry": entry, "stop_loss": sl, "take_profit": tp, "trail_stop": trail,
        "shares": LOT, "status": "open", "exit_date": None, "exit_price": None,
        "reason": None, "last_price": entry, "pnl": 0.0, "pnl_pct": 0.0,
    })
    return True


def record_recommendations(data: dict, top: list[dict], date: str) -> None:
    """Open a 100-share paper position for EVERY candidate (not already held)."""
    for s in top:
        open_position(data, date, str(s.get("code")), s.get("name", ""),
                      s.get("current_price"), s.get("stop_loss"),
                      s.get("take_profit"), s.get("trail_stop"))


def _auto_capital(positions: list[dict]) -> int:
    """運用資金 = smallest ¥10,000 multiple that exceeds the PEAK simultaneous cost of
    holding 100 shares of every open position over the tracked period."""
    if not positions:
        return CAPITAL_ROUND
    # Concurrency changes only on open/exit dates; the peak is at some open date.
    dates = sorted({p["date"] for p in positions}
                   | {p["exit_date"] for p in positions if p.get("exit_date")})
    peak = 0.0
    for d in dates:
        cost = 0.0
        for p in positions:
            end = p.get("exit_date") or "9999-12-31"
            if p["date"] <= d <= end:  # ISO strings compare correctly
                cost += LOT * p["entry"]
        peak = max(peak, cost)
    return int(math.floor(peak / CAPITAL_ROUND) + 1) * CAPITAL_ROUND


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
        p["shares"] = LOT  # normalize (fixed 100-share lot)
        entry, sl, tp, sh = p["entry"], p["stop_loss"], p["take_profit"], LOT
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
    # Normalize every position to a fixed 100-share lot (recompute P/L accordingly).
    for p in positions:
        p["shares"] = LOT
        px = p["exit_price"] if p["status"] == "closed" else p.get("last_price", p["entry"])
        if px is None:
            px = p["entry"]
        p["pnl"] = float((px - p["entry"]) * LOT)
        p["pnl_pct"] = float(px / p["entry"] - 1) if p["entry"] else 0.0

    init = _auto_capital(positions)
    data["initial_capital"] = init
    closed = [p for p in positions if p["status"] == "closed"]
    open_ps = [p for p in positions if p["status"] == "open"]
    realized = sum(p["pnl"] for p in closed)
    unrealized = sum(p["pnl"] for p in open_ps)
    deployed = sum(LOT * p["entry"] for p in open_ps)
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
        "shares_per_trade": LOT,
        "deployed": round(deployed),
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
