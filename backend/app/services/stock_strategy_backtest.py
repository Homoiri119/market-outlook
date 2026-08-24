"""Individual-stock trade-simulation backtest (long-only) with regime filter and
trailing stop, using J-Quants V2 bars.

Entry rule (matches the analysis tool's bullish setup): a fresh signal where
close > 25MA > 75MA and MACD histogram > 0. Trades are simulated bar-by-bar with
intrabar TP/SL checks. Three configurations are compared:

  base   : fixed SL = entry - 1.5*ATR, TP = entry + 2.5*ATR
  regime : base, but only enter when TOPIX is above its 200-day MA
  trail  : regime + a chandelier trailing stop (no fixed TP, let winners run)

Results are reported in R-multiples (return / initial risk), which are position-size
agnostic: win-rate, expectancy (avg R), profit factor, max drawdown (in R), and a
pooled equity curve (cumulative R). Indicators are causal, so there is no look-ahead.
Not investment advice; fees/slippage are not modeled.
"""

from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

from app.clients.jquants_client import jquants_client
from app.services.stock_analysis import _atr, _rsi  # reuse indicator helpers

logger = logging.getLogger(__name__)

SL_ATR = 1.5
TP_ATR = 2.5
TRAIL_ATR = 2.5
MAX_HOLD = 40
MAX_HOLD_TRAIL = 60
SCALE_CATS = ("TOPIX Core30", "TOPIX Large70")


def _signals(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    sma25 = close.rolling(25).mean()
    sma75 = close.rolling(75).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_hist = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
    out = df.copy()
    out["atr"] = _atr(df)
    out["long_sig"] = (close > sma25) & (sma25 > sma75) & (macd_hist > 0)
    return out


def simulate(df: pd.DataFrame, regime_ok=None, trailing: bool = False) -> list[dict]:
    """Return a list of closed trades for one stock."""
    s = _signals(df)
    close = s["close"].to_numpy(); high = s["high"].to_numpy(); low = s["low"].to_numpy()
    atr = s["atr"].to_numpy(); sig = s["long_sig"].to_numpy()
    dates = list(s.index)
    n = len(s)
    trades: list[dict] = []
    i = 1
    max_hold = MAX_HOLD_TRAIL if trailing else MAX_HOLD
    while i < n:
        fresh = sig[i] and not sig[i - 1]
        if fresh and not np.isnan(atr[i]) and atr[i] > 0:
            if regime_ok is not None and not regime_ok(dates[i]):
                i += 1
                continue
            entry = close[i]; a = atr[i]
            init_sl = entry - SL_ATR * a
            tp = entry + TP_ATR * a
            cur_sl = init_sl
            hh = high[i]
            exit_price = None
            j = i + 1
            while j < n:
                hh = max(hh, high[j])
                if trailing:
                    cur_sl = max(cur_sl, hh - TRAIL_ATR * a)
                if low[j] <= cur_sl:
                    exit_price = cur_sl; break
                if not trailing and high[j] >= tp:
                    exit_price = tp; break
                if j - i >= max_hold:
                    exit_price = close[j]; break
                j += 1
            if exit_price is None:
                break  # open at series end — skip
            R = (exit_price - entry) / (entry - init_sl)
            trades.append({
                "R": float(R),
                "ret": float(exit_price / entry - 1),
                "bars": int(j - i),
                "win": bool(exit_price > entry),
                "exit_date": dates[j].isoformat() if hasattr(dates[j], "isoformat") else str(dates[j]),
            })
            i = j + 1
        else:
            i += 1
    return trades


def _stats(trades: list[dict]) -> dict:
    n = len(trades)
    if n == 0:
        return {"trades": 0}
    Rs = np.array([t["R"] for t in trades])
    wins = Rs[Rs > 0]; losses = Rs[Rs <= 0]
    gross_win = float(wins.sum()); gross_loss = float(-losses.sum())
    ordered = sorted(trades, key=lambda t: t["exit_date"])
    eq = np.cumsum([t["R"] for t in ordered])
    peak = np.maximum.accumulate(eq)
    max_dd = float((eq - peak).min()) if len(eq) else 0.0
    step = max(1, len(ordered) // 200)
    curve = [{"date": ordered[k]["exit_date"], "cumR": float(eq[k])} for k in range(0, len(ordered), step)]
    if curve and curve[-1]["date"] != ordered[-1]["exit_date"]:
        curve.append({"date": ordered[-1]["exit_date"], "cumR": float(eq[-1])})
    return {
        "trades": n,
        "win_rate": float((Rs > 0).mean()),
        "expectancy_R": float(Rs.mean()),
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else None,
        "avg_win_R": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_R": float(losses.mean()) if len(losses) else 0.0,
        "total_R": float(Rs.sum()),
        "max_dd_R": max_dd,
        "avg_bars": float(np.mean([t["bars"] for t in trades])),
        "equity": curve,
    }


def run_stock_strategy_backtest(years: float = 3.5, universe_cap: int = 90) -> dict:
    end = dt.date.today()
    start = end - dt.timedelta(days=int(365 * years) + 90)

    topix = jquants_client.fetch_topix_history(start, end)
    if topix.empty:
        raise RuntimeError("Could not fetch TOPIX for the regime filter.")
    tclose = topix["close"]
    tma = tclose.rolling(200).mean()
    regime = {d.isoformat() if hasattr(d, "isoformat") else str(d): bool(tclose.loc[d] > tma.loc[d])
              for d in tclose.index if not pd.isna(tma.loc[d])}

    def regime_ok(d) -> bool:
        key = d.isoformat() if hasattr(d, "isoformat") else str(d)
        return regime.get(key, True)

    master = jquants_client.fetch_listed_info()
    if master.empty or "Code" not in master.columns:
        raise RuntimeError("Could not fetch equities master.")
    universe = master[master["ScaleCat"].isin(SCALE_CATS)] if "ScaleCat" in master.columns else master.head(universe_cap)
    codes = list(dict.fromkeys(universe["Code"].astype(str)))[:universe_cap]

    pools = {"base": [], "regime": [], "trail": []}
    n_stocks = 0
    for code in codes:
        try:
            bars = jquants_client.fetch_daily_quotes(code, start, end)
        except Exception:
            logger.exception("bars failed for %s", code)
            continue
        if bars.empty or len(bars) < 120:
            continue
        n_stocks += 1
        pools["base"].extend(simulate(bars, regime_ok=None, trailing=False))
        pools["regime"].extend(simulate(bars, regime_ok=regime_ok, trailing=False))
        pools["trail"].extend(simulate(bars, regime_ok=regime_ok, trailing=True))

    return {
        "level": "stock-strategy",
        "years": years,
        "universe_size": n_stocks,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "params": {"sl_atr": SL_ATR, "tp_atr": TP_ATR, "trail_atr": TRAIL_ATR,
                   "max_hold": MAX_HOLD, "max_hold_trail": MAX_HOLD_TRAIL},
        "configs": {k: _stats(v) for k, v in pools.items()},
    }
