"""Fetches US market / FX indicators using yfinance (no API key required)."""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# yfinance ticker symbols
TICKERS = {
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "dow": "^DJI",
    "vix": "^VIX",
    "usdjpy": "USDJPY=X",
}

# Nikkei 225 cash index (previous Tokyo close) — used as the gap reference.
NIKKEI_INDEX_TICKER = "^N225"

# CME Nikkei 225 futures. Both quote in Nikkei index points; NIY=F settles in yen,
# NKD=F in USD. They trade overnight, so at ~08:00 JST they already price in the
# US close and are the market's own estimate of where Tokyo will open.
NIKKEI_FUTURES_TICKERS = ["NIY=F", "NKD=F"]


def fetch_us_market_history(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Return a DataFrame indexed by date with columns:
    sp500_close, nasdaq_close, dow_close, vix_close, usdjpy_close,
    sp500_return, nasdaq_return, dow_return, vix_change, usdjpy_return
    """
    frames = {}
    for name, ticker in TICKERS.items():
        try:
            data = yf.Ticker(ticker).history(
                start=start.isoformat(),
                end=(end + dt.timedelta(days=1)).isoformat(),
                interval="1d",
            )
            if data.empty:
                logger.warning("No data returned for %s (%s)", name, ticker)
                continue
            close = data["Close"].copy()
            close.index = close.index.date
            frames[f"{name}_close"] = close
        except Exception:  # pragma: no cover - network dependent
            logger.exception("Failed to fetch %s (%s)", name, ticker)

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df = df.sort_index()

    for name in TICKERS:
        close_col = f"{name}_close"
        if close_col not in df.columns:
            continue
        if name == "vix":
            df["vix_change"] = df[close_col].diff()
        else:
            df[f"{name}_return"] = df[close_col].pct_change()

    df.index.name = "date"
    return df


def fetch_latest_us_market(reference_date: dt.date | None = None) -> dict | None:
    """Return the most recent row of US market data as a dict, or None if unavailable."""
    end = reference_date or dt.date.today()
    start = end - dt.timedelta(days=14)
    df = fetch_us_market_history(start, end)
    if df.empty:
        return None
    df = df.dropna(subset=["sp500_close"])
    if df.empty:
        return None
    last = df.iloc[-1]
    result = {"date": df.index[-1]}
    for col in df.columns:
        value = last[col]
        result[col] = None if pd.isna(value) else float(value)
    return result


def fetch_nikkei_prev_close(reference_date: dt.date | None = None) -> dict | None:
    """Return the most recent Nikkei 225 cash-index daily close (the previous Tokyo
    close when called pre-open) as {"date", "close"}, or None if unavailable."""
    try:
        tk = yf.Ticker(NIKKEI_INDEX_TICKER)
        data = tk.history(period="10d", interval="1d")
        close = data["Close"].dropna() if not data.empty else None
        if close is None or close.empty:
            return None
        return {"date": close.index[-1].date(), "close": float(close.iloc[-1])}
    except Exception:  # pragma: no cover - network dependent
        logger.exception("Failed to fetch Nikkei 225 index")
        return None


def fetch_nikkei_futures() -> dict | None:
    """Return the latest overnight Nikkei 225 futures price as {"ticker", "price"},
    or None if unavailable. Tries intraday, then fast_info, then the last daily bar,
    across the configured futures tickers."""
    for ticker in NIKKEI_FUTURES_TICKERS:
        try:
            tk = yf.Ticker(ticker)
            price: float | None = None

            intraday = tk.history(period="1d", interval="5m")
            if not intraday.empty:
                closes = intraday["Close"].dropna()
                if not closes.empty:
                    price = float(closes.iloc[-1])

            if price is None:
                fast_info = getattr(tk, "fast_info", None)
                last = None
                if fast_info is not None:
                    try:
                        last = fast_info.get("last_price")  # type: ignore[union-attr]
                    except Exception:
                        last = getattr(fast_info, "last_price", None)
                if last:
                    price = float(last)

            if price is None:
                daily = tk.history(period="5d", interval="1d")
                if not daily.empty:
                    closes = daily["Close"].dropna()
                    if not closes.empty:
                        price = float(closes.iloc[-1])

            if price and price > 0:
                return {"ticker": ticker, "price": price}
        except Exception:  # pragma: no cover - network dependent
            logger.exception("Failed to fetch Nikkei futures %s", ticker)
    return None


def fetch_overnight_snapshot(reference_date: dt.date | None = None) -> dict:
    """Bundle everything needed for the pre-open morning outlook:
    the latest US market row, the previous Nikkei cash close, and the current
    Nikkei futures price. Any piece may be None if its source is unavailable."""
    return {
        "us": fetch_latest_us_market(reference_date),
        "nikkei_prev": fetch_nikkei_prev_close(reference_date),
        "futures": fetch_nikkei_futures(),
    }
