"""Stateless morning-outlook computation for ephemeral cloud runners (GitHub Actions).

Unlike `morning_outlook.compute_morning_outlook`, this needs no database: it fetches
~1 year of US and Nikkei history from yfinance on every run, trains the regression
model in memory, and returns a plain dict. Perfect for a daily serverless cron job.
"""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression

from app.clients.us_market_client import (
    NIKKEI_INDEX_TICKER,
    fetch_overnight_snapshot,
    fetch_us_market_history,
)
from app.config import settings
from app.services.morning_outlook import (
    FEATURE_COLUMNS,
    _build_narrative,
    _classify,
    _format_us_detail,
    us_market_struct,
)

logger = logging.getLogger(__name__)


def _feature_frame(start: dt.date, end: dt.date) -> pd.DataFrame:
    df = fetch_us_market_history(start, end)
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index()  # 'date' column
    cols = ["date"] + [c for c in FEATURE_COLUMNS if c in df.columns]
    df = df[cols].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.dropna(subset=[c for c in FEATURE_COLUMNS if c in df.columns])


def _nikkei_next_return_frame(start: dt.date, end: dt.date) -> pd.DataFrame:
    try:
        data = yf.Ticker(NIKKEI_INDEX_TICKER).history(
            start=start.isoformat(),
            end=(end + dt.timedelta(days=1)).isoformat(),
            interval="1d",
        )
    except Exception:
        logger.exception("Failed to fetch Nikkei history")
        return pd.DataFrame()
    if data.empty:
        return pd.DataFrame()
    close = data["Close"].dropna()
    df = pd.DataFrame({"date": [i.date() for i in close.index], "close": close.values})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["nikkei_return"] = df["close"].pct_change()
    df["feature_date"] = df["date"].shift(1)
    out = df[["feature_date", "nikkei_return"]].dropna(subset=["feature_date"])
    return out.rename(columns={"feature_date": "date"})


def _train(start: dt.date, end: dt.date) -> tuple[float | None, float]:
    features = _feature_frame(start, end)
    nikkei = _nikkei_next_return_frame(start, end)
    if features.empty or nikkei.empty:
        return None, 0.0
    merged = pd.merge(features, nikkei, on="date", how="inner").dropna(
        subset=FEATURE_COLUMNS + ["nikkei_return"]
    )
    if len(merged) < 20:
        return None, 0.0
    X = merged[FEATURE_COLUMNS].to_numpy()
    y = merged["nikkei_return"].to_numpy()
    model = LinearRegression().fit(X, y)
    r2 = float(model.score(X, y))
    latest = features.iloc[-1]
    predicted = float(model.predict(latest[FEATURE_COLUMNS].to_numpy().reshape(1, -1))[0])
    return predicted, max(0.0, min(1.0, r2))


def compute_stateless_outlook(date: dt.date | None = None) -> dict:
    """Compute the morning outlook without any database, returning a dict with the
    same shape as the MorningOutlook API model (plus `us_market`)."""
    date = date or dt.date.today()
    end = date
    start = end - dt.timedelta(days=settings.history_days)

    snapshot = fetch_overnight_snapshot(date)
    us = snapshot.get("us")
    nikkei_prev = snapshot.get("nikkei_prev")
    futures = snapshot.get("futures")

    nikkei_prev_close = nikkei_prev.get("close") if nikkei_prev else None
    nikkei_futures = futures.get("price") if futures else None
    futures_source = futures.get("ticker") if futures else None
    implied_gap = None
    if nikkei_prev_close and nikkei_futures:
        implied_gap = (nikkei_futures - nikkei_prev_close) / nikkei_prev_close

    model_return, r2 = _train(start, end)

    if implied_gap is None and model_return is None:
        raise RuntimeError("No Nikkei futures data and not enough history to train the model.")

    w = settings.futures_gap_weight
    if implied_gap is not None and model_return is not None:
        expected_move = w * implied_gap + (1 - w) * model_return
        confidence = max(0.6, r2)
    elif implied_gap is not None:
        expected_move = implied_gap
        confidence = 0.6
    else:
        expected_move = float(model_return)
        confidence = max(0.2, r2)

    direction = _classify(expected_move)
    implied_open_level = (
        nikkei_prev_close * (1 + expected_move) if nikkei_prev_close else nikkei_futures
    )

    return {
        "date": date.isoformat(),
        "direction": direction,
        "expected_move": float(expected_move),
        "confidence": float(confidence),
        "implied_gap": implied_gap,
        "model_return": model_return,
        "nikkei_prev_close": nikkei_prev_close,
        "nikkei_futures": nikkei_futures,
        "implied_open_level": implied_open_level,
        "futures_source": futures_source,
        "us_detail": _format_us_detail(us),
        "us_market": us_market_struct(us),
        "narrative": _build_narrative(
            direction=direction,
            expected_move=expected_move,
            implied_gap=implied_gap,
            model_return=model_return,
            nikkei_prev_close=nikkei_prev_close,
            implied_open_level=implied_open_level,
            us=us,
        ),
    }
