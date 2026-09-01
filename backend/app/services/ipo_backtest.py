"""IPO secondary-investment backtest (J-Quants V2 bars).

Strategy studied: you do NOT rely on winning the IPO allocation lottery. Instead
you buy on the open market at the debut ("secondary"): enter at the first traded
price (初値 = debut-day open) and hold N trading days, then sell at the close.

IPO detection: J-Quants has no listing-date field, so we snapshot the set of codes
that already traded at the start of the lookback window (via bars-by-date) and treat
any master code missing from that set as "listed later". For each such candidate we
fetch its full bars; the first bar is the debut. Codes whose first bar is close to the
window start are treated as pre-existing (false positives) and skipped.

Kioxia (285A) and mega-cap debuts are once-in-years outliers and are excluded from the
headline numbers (listed separately). Not investment advice; fees/slippage ignored.
"""

from __future__ import annotations

import datetime as dt
import logging
import time

import numpy as np
import pandas as pd

from app.clients.jquants_client import jquants_client

logger = logging.getLogger(__name__)

LOOKBACK_DAYS = 1300          # ~3.5y; within the J-Quants plan horizon
REF_TRADING_DAYS = 12         # trading days near the window start used as the "pre-existing" snapshot
MIN_DEBUT_OFFSET_DAYS = 30    # first bar must be this far after window start to count as an IPO
HOLDS = [0, 1, 3, 5, 10, 20]  # trading days held (0 = buy 初値 / sell same-day close)
MIN_BARS_AFTER = 1            # need at least this many bars after debut
NOTIONAL = 300_000            # ¥ invested per IPO for the money-terms equity curve
EXCLUDE_CODES = {"285A"}      # キオクシア(数年に一度の外れ値)だけ除外。他の大型IPOは含める


def _debut_mktcap(bars: pd.DataFrame) -> float | None:
    if "mkt_cap" in bars.columns and not bars["mkt_cap"].isna().all():
        v = bars["mkt_cap"].dropna()
        return float(v.iloc[0]) if len(v) else None
    return None


def _pre_existing_codes(window_start: dt.date) -> set[str]:
    """Union of codes that traded on the first few trading days from window_start."""
    codes: set[str] = set()
    got = 0
    d = window_start
    guard = 0
    while got < REF_TRADING_DAYS and guard < 40:
        rows = jquants_client.fetch_quotes_by_date(d)
        if rows:
            codes |= {str(r.get("Code")) for r in rows if r.get("Code")}
            got += 1
        d += dt.timedelta(days=1)
        guard += 1
        time.sleep(0.15)
    return codes


def _trades_for_ipo(bars: pd.DataFrame) -> dict[int, float] | None:
    """Return {hold: return_pct} for one IPO, entering at the debut open (初値)."""
    if bars.empty:
        return None
    o = bars["open"].to_numpy(); c = bars["close"].to_numpy()
    entry = o[0]
    if not np.isfinite(entry) or entry <= 0:
        return None
    n = len(bars)
    out: dict[int, float] = {}
    for h in HOLDS:
        if h == 0:
            exit_px = c[0]
        else:
            idx = min(h, n - 1)          # clamp to last available bar
            exit_px = c[idx]
        if n - 1 < MIN_BARS_AFTER and h > 0:
            continue
        out[h] = float(exit_px / entry - 1)
    return out or None


def _stats(returns: list[float]) -> dict:
    if not returns:
        return {"n": 0}
    a = np.array(returns, dtype=float)
    return {
        "n": int(a.size),
        "win_rate": float((a > 0).mean()),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "best": float(a.max()),
        "worst": float(a.min()),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
    }


def run_ipo_backtest(lookback_days: int = LOOKBACK_DAYS, max_candidates: int = 600) -> dict:
    end = dt.date.today()
    window_start = end - dt.timedelta(days=lookback_days)

    master = jquants_client.fetch_listed_info()
    if master.empty or "Code" not in master.columns:
        raise RuntimeError("Could not fetch equities master.")
    name_col = "CoName" if "CoName" in master.columns else ("CompanyName" if "CompanyName" in master.columns else None)
    names = {str(r["Code"]): (str(r[name_col]) if name_col else str(r["Code"])) for _, r in master.iterrows()}
    all_codes = [str(c) for c in dict.fromkeys(master["Code"].astype(str))]

    pre_existing = _pre_existing_codes(window_start)
    logger.info("Pre-existing snapshot: %d codes", len(pre_existing))
    # Safety: if the by-date snapshot is empty, the /equities/bars/daily `date` query
    # is unavailable on this plan. Abort rather than probe every listed code (thousands).
    if len(pre_existing) < 100:
        raise RuntimeError(
            "IPO detection needs the bars-by-date snapshot, but it returned "
            f"{len(pre_existing)} codes — the endpoint may be unavailable on this plan. "
            "Provide a curated IPO list instead."
        )
    # Candidates = listed after the window start (missing from the early snapshot).
    candidates = [c for c in all_codes if c not in pre_existing][:max_candidates]
    logger.info("IPO candidates to probe: %d", len(candidates))

    ipos: list[dict] = []       # detected, kept
    excluded: list[dict] = []   # detected but excluded (Kioxia / mega-cap)
    for code in candidates:
        try:
            bars = jquants_client.fetch_daily_quotes(code, window_start, end)
        except Exception:
            logger.exception("bars failed for %s", code)
            continue
        time.sleep(0.12)
        if bars.empty:
            continue
        debut = bars.index[0]
        debut_date = debut if isinstance(debut, dt.date) else pd.to_datetime(debut).date()
        # Skip false positives (already existed near the window start).
        if (debut_date - window_start).days < MIN_DEBUT_OFFSET_DAYS:
            continue
        rets = _trades_for_ipo(bars)
        if not rets:
            continue
        mcap = _debut_mktcap(bars)
        rec = {
            "code": code,
            "name": names.get(code, code),
            "debut": debut_date.isoformat(),
            "ipo_price": float(bars["open"].iloc[0]),
            "mkt_cap": mcap,
            "returns": rets,
        }
        # Only Kioxia (a once-in-years outlier) is excluded; all other IPOs, including
        # other very large ones, are kept.
        if code in EXCLUDE_CODES:
            rec["excluded_reason"] = "キオクシア(数年に一度の外れ値)"
            excluded.append(rec)
        else:
            ipos.append(rec)

    ipos.sort(key=lambda r: r["debut"])
    excluded.sort(key=lambda r: r["debut"])

    # Per-hold stats (headline set = excluding outliers).
    by_hold = {}
    equity = {}
    for h in HOLDS:
        rets = [r["returns"][h] for r in ipos if h in r["returns"]]
        by_hold[str(h)] = _stats(rets)
        # money-terms equity curve for this hold, in debut order
        eq, cum = [], 0.0
        for r in ipos:
            if h not in r["returns"]:
                continue
            cum += NOTIONAL * r["returns"][h]
            eq.append({"date": r["debut"], "pnl": round(cum)})
        equity[str(h)] = eq

    # choose a representative headline hold: highest median return
    best_hold = max(HOLDS, key=lambda h: by_hold[str(h)].get("median", -9) if by_hold[str(h)].get("n") else -9)

    return {
        "level": "ipo",
        "start": window_start.isoformat(),
        "end": end.isoformat(),
        "n_ipos": len(ipos),
        "n_excluded": len(excluded),
        "holds": HOLDS,
        "notional": NOTIONAL,
        "best_hold": best_hold,
        "by_hold": by_hold,
        "equity": equity,
        "ipos": ipos,
        "excluded": excluded,
        "params": {"lookback_days": lookback_days, "excluded_codes": sorted(EXCLUDE_CODES),
                   "min_debut_offset_days": MIN_DEBUT_OFFSET_DAYS},
    }
