"""Paper-trading tracker for the daily Discord buy candidates.

Each morning the outlook job proposes buy candidates. This module records them as
simulated positions and, on every run, re-simulates every position from scratch
against J-Quants bars (idempotent, so rule changes apply to past positions too):

  * Entry at the recommendation day's OPEN; stop/target re-anchored to it
    (stop = open - 1.5×ATR, target = open + 2.5×ATR, ATR taken from the recommendation).
  * Three exit methods in parallel:
      fixed  — stop / target, no time limit
      trail  — stop that ratchets up with the high (trail width from the recommendation), no target
      fixed5 — stop / target, and at the latest the close EXIT_MAX_HOLD bars after entry (推奨)
  * Realistic gap fills: if the open is already beyond a level, fill at the open.
  * Share splits: J-Quants back-adjusts prices; if the prior close no longer matches the
    recommended price, the bars are rescaled to the recommendation's price basis.

Sizing: a fixed 100 shares per position; the page derives other sizings. Fees and
slippage are ignored. Not investment advice.
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
SL_ATR, TP_ATR = 1.5, 2.5
EXIT_MAX_HOLD = 5     # 推奨ルール: exit at the close 5 bars after the entry day at the latest
SPLIT_TH = 0.20       # prior close vs recommended price beyond this = share split


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
    # "Still held" follows the recommended rule (fixed5), so a name can be re-bought
    # once its 5-day position has been closed.
    if any(p["code"] == code and p.get("f5_status", p["status"]) == "open" for p in data["positions"]):
        return False
    if not entry or not tp or not sl or entry <= sl:
        return False
    data["positions"].append({
        "id": pos_id, "code": code, "name": name or code, "date": date,
        "rec_entry": entry, "rec_stop_loss": sl, "rec_take_profit": tp,
        "entry": entry, "stop_loss": sl, "take_profit": tp, "trail_stop": trail,
        "filled": False, "split_adj": None,
        "shares": LOT, "status": "open", "exit_date": None, "exit_price": None,
        "reason": None, "last_price": entry, "pnl": 0.0, "pnl_pct": 0.0,
        "trail_status": "open", "trail_exit_date": None, "trail_exit_price": None,
        "trail_reason": None, "trail_cur_stop": None,
        "f5_status": "open", "f5_exit_date": None, "f5_exit_price": None, "f5_reason": None,
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


def _bars_by_code(positions: list[dict], today: dt.date) -> dict:
    """One J-Quants fetch per code, from a little before its earliest recommendation
    (the prior close is needed for the split check) up to `today`."""
    first: dict[str, dt.date] = {}
    for p in positions:
        d = dt.date.fromisoformat(p["date"])
        first[p["code"]] = min(first.get(p["code"], d), d)
    out = {}
    for code, d0 in first.items():
        try:
            out[code] = jquants_client.fetch_daily_quotes(code, d0 - dt.timedelta(days=14), today)
        except Exception:
            logger.exception("bars failed for %s", code)
    return out


def _resimulate(p: dict, bars) -> None:
    """Re-run all three exit methods for one position from its recommendation day."""
    rec_e, rec_sl = p["rec_entry"], p["rec_stop_loss"]
    atr = (rec_e - rec_sl) / SL_ATR
    trail_dist = (rec_e - p["trail_stop"]) if p.get("trail_stop") else TP_ATR * atr
    d0 = dt.date.fromisoformat(p["date"])
    prev, seg = bars[bars.index < d0], bars[bars.index >= d0]

    # Share split: the back-adjusted prior close no longer matches the recommended price.
    scale = 1.0
    if not prev.empty:
        ratio = float(prev["close"].iloc[-1]) / rec_e
        if abs(ratio - 1) > SPLIT_TH:
            scale = 1 / ratio

    # Reset to "recommended, not yet traded", then replay.
    p.update(entry=rec_e, stop_loss=rec_sl, take_profit=p["rec_take_profit"], last_price=rec_e,
             filled=False, split_adj=round(1 / scale, 4) if scale != 1.0 else None, shares=LOT,
             status="open", exit_date=None, exit_price=None, reason=None,
             trail_status="open", trail_exit_date=None, trail_exit_price=None, trail_reason=None,
             f5_status="open", f5_exit_date=None, f5_exit_price=None, f5_reason=None,
             pnl=0.0, pnl_pct=0.0)
    if seg.empty:
        return  # the recommendation day has not traded yet

    E = float(seg["open"].iloc[0]) * scale  # buy at the recommendation day's open
    sl, tp = E - SL_ATR * atr, E + TP_ATR * atr
    p.update(entry=round(E, 1), stop_loss=round(sl, 1), take_profit=round(tp, 1), filled=True)

    fixed_open = trail_open = f5_open = True
    t_stop, hh, last = sl, E, E
    for i, (d, row) in enumerate(seg.iterrows()):
        op, hi, lo, cl = (float(row[k]) * scale for k in ("open", "high", "low", "close"))
        dd = d.isoformat() if hasattr(d, "isoformat") else str(d)
        gap = i > 0  # the entry bar's open IS the entry, so no gap fill on it
        if fixed_open or f5_open:
            px = why = None
            if lo <= sl:  # stop first (conservative)
                px, why = (op if gap and op < sl else sl), "損切"
            elif hi >= tp:
                px, why = (op if gap and op > tp else tp), "利確"
            if px is not None:
                if fixed_open:
                    p.update(status="closed", exit_date=dd, exit_price=round(px, 1), reason=why)
                    fixed_open = False
                if f5_open:
                    p.update(f5_status="closed", f5_exit_date=dd, f5_exit_price=round(px, 1), f5_reason=why)
                    f5_open = False
        if f5_open and i >= EXIT_MAX_HOLD:
            p.update(f5_status="closed", f5_exit_date=dd, f5_exit_price=round(cl, 1), f5_reason="期限")
            f5_open = False
        if trail_open and lo <= t_stop:
            px = op if gap and op < t_stop else t_stop
            p.update(trail_status="closed", trail_exit_date=dd, trail_exit_price=round(px, 1),
                     trail_reason="トレール" if px > E else "損切")
            trail_open = False
        hh = max(hh, hi)
        t_stop = max(t_stop, hh - trail_dist)  # effective from the next bar (conservative)
        last = cl
        if not (fixed_open or trail_open or f5_open):
            break
    p["last_price"] = round(last, 1)
    p["trail_cur_stop"] = round(t_stop)
    px = p["exit_price"] if p["status"] == "closed" else last
    p["pnl"] = float((px - E) * LOT)
    p["pnl_pct"] = float(px / E - 1)


def update_positions(data: dict, today: dt.date | None = None) -> None:
    """Re-simulate every position from scratch (open-anchored entry, split-safe,
    fixed / trail / fixed5 exits). Positions whose bars cannot be fetched keep their
    previous state."""
    today = today or dt.date.today()
    positions = data["positions"]
    for p in positions:  # keep the original (pre-open) recommendation levels
        p.setdefault("rec_entry", p["entry"])
        p.setdefault("rec_stop_loss", p["stop_loss"])
        p.setdefault("rec_take_profit", p["take_profit"])
    bars_by_code = _bars_by_code(positions, today)
    for p in positions:
        bars = bars_by_code.get(p["code"])
        if bars is None or bars.empty:
            continue
        _resimulate(p, bars[bars.index <= today])


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
