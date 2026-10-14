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
# NKD=F in USD. Their close for the last US session already prices in the US close and
# is the market's own estimate of where Tokyo will open. NKD=F first: over 2021-2026
# its close tracked the Tokyo opening gap better (corr 0.84 vs 0.77).
NIKKEI_FUTURES_TICKERS = ["NKD=F", "NIY=F"]


def _bars_before(data: pd.DataFrame, reference_date: dt.date | None) -> pd.Series:
    """Daily closes strictly BEFORE reference_date (the Tokyo session being forecast),
    so a bar still trading — e.g. when the job runs after the Tokyo open — is ignored
    and the result is the same whenever the job runs."""
    close = data["Close"].dropna()
    close.index = [d.date() for d in close.index]
    if reference_date is not None:
        close = close[[d < reference_date for d in close.index]]
    return close

# Extra overnight indicators a JP investor checks pre-open. Each entry is a list of
# fallback tickers tried in order.
#  - sox: Philadelphia Semiconductor Index (drives Tokyo semis: TEL, Advantest, ...)
#  - us10y: US 10Y Treasury yield (quoted in percent by ^TNX)
#  - wti: WTI crude oil front-month future
EXTRA_TICKERS = {
    "sox": ["^SOX", "SOXX"],
    "us10y": ["^TNX"],
    "wti": ["CL=F"],
}


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
    if reference_date is not None:
        df = df[[d < reference_date for d in df.index]]
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
        data = tk.history(period="15d", interval="1d")
        if data.empty:
            return None
        # Strictly before the forecast day: after 09:00 JST today's in-progress bar
        # would otherwise be taken as the "previous close".
        close = _bars_before(data, reference_date)
        if close.empty:
            return None
        return {"date": close.index[-1], "close": float(close.iloc[-1])}
    except Exception:  # pragma: no cover - network dependent
        logger.exception("Failed to fetch Nikkei 225 index")
        return None


def fetch_nikkei_futures(reference_date: dt.date | None = None,
                         min_date: dt.date | None = None) -> dict | None:
    """Return the Nikkei 225 futures price for the gap estimate as
    {"ticker", "price", "asof", "method"}, or None.

    Preferred: the close of the last CME session completed before reference_date (the
    futures' own read of the US close), provided that session is not older than
    min_date (the previous Tokyo session). Fallback: the old intraday snapshot."""
    for ticker in NIKKEI_FUTURES_TICKERS:
        try:
            daily = yf.Ticker(ticker).history(period="10d", interval="1d")
        except Exception:  # pragma: no cover - network dependent
            logger.exception("Failed to fetch Nikkei futures %s", ticker)
            continue
        if daily.empty:
            continue
        close = _bars_before(daily, reference_date)
        if close.empty or (min_date is not None and close.index[-1] < min_date):
            continue
        price = float(close.iloc[-1])
        if price > 0:
            return {"ticker": ticker, "price": price, "asof": close.index[-1].isoformat(),
                    "method": "daily_close"}
    logger.warning("No fresh futures daily close; falling back to the intraday snapshot")
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
                return {"ticker": ticker, "price": price, "asof": None, "method": "intraday"}
        except Exception:  # pragma: no cover - network dependent
            logger.exception("Failed to fetch Nikkei futures %s", ticker)
    return None


def fetch_extra_overnight(reference_date: dt.date | None = None) -> dict:
    """Fetch extra overnight indicators (SOX semis, US 10Y yield, WTI crude) from bars
    completed before reference_date. Returns a flat dict; missing pieces are absent."""
    out: dict = {}
    for name, tickers in EXTRA_TICKERS.items():
        for ticker in tickers:
            try:
                data = yf.Ticker(ticker).history(period="10d", interval="1d")
            except Exception:  # pragma: no cover - network dependent
                logger.exception("Failed to fetch %s (%s)", name, ticker)
                continue
            if data.empty:
                continue
            close = _bars_before(data, reference_date)
            if close.empty:
                continue
            out[f"{name}_level"] = float(close.iloc[-1])
            if len(close) >= 2:
                prev = float(close.iloc[-2])
                if name == "us10y":
                    # yield change in basis points (^TNX is quoted in percent)
                    out["us10y_change_bp"] = (float(close.iloc[-1]) - prev) * 100.0
                else:
                    out[f"{name}_return"] = float(close.iloc[-1]) / prev - 1 if prev else None
            break  # first working ticker wins
    return out


def fetch_nikkei_ohlc(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Return Nikkei 225 cash-index daily OHLC as a DataFrame with columns
    date (datetime64), open, close. Empty on failure."""
    try:
        data = yf.Ticker(NIKKEI_INDEX_TICKER).history(
            start=start.isoformat(),
            end=(end + dt.timedelta(days=1)).isoformat(),
            interval="1d",
        )
    except Exception:  # pragma: no cover - network dependent
        logger.exception("Failed to fetch Nikkei 225 OHLC")
        return pd.DataFrame()
    if data.empty:
        return pd.DataFrame()
    df = pd.DataFrame(
        {
            "date": pd.to_datetime([idx.date() for idx in data.index]),
            "open": data["Open"].to_numpy(),
            "close": data["Close"].to_numpy(),
        }
    )
    return df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)


def fetch_overnight_snapshot(reference_date: dt.date | None = None) -> dict:
    """Bundle everything needed for the pre-open morning outlook:
    the latest US market row, the previous Nikkei cash close, the current Nikkei
    futures price, and extra overnight indicators. Any piece may be None/empty."""
    nikkei_prev = fetch_nikkei_prev_close(reference_date)
    return {
        "us": fetch_latest_us_market(reference_date),
        "nikkei_prev": nikkei_prev,
        "futures": fetch_nikkei_futures(reference_date, nikkei_prev["date"] if nikkei_prev else None),
        "extra": fetch_extra_overnight(reference_date),
    }
