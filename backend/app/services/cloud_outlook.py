"""Stateless morning-outlook computation for ephemeral cloud runners (GitHub Actions).

Unlike `morning_outlook.compute_morning_outlook`, this needs no database: it fetches
~1 year of US and Nikkei history from yfinance on every run, trains the regression
model in memory, and returns a plain dict. Perfect for a daily serverless cron job.
"""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
from sklearn.linear_model import LinearRegression

from app.clients.us_market_client import (
    fetch_nikkei_ohlc,
    fetch_overnight_snapshot,
    fetch_us_market_history,
)
from app.config import settings
from app.services import analytics
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


def _next_return_frame(nikkei_ohlc: pd.DataFrame) -> pd.DataFrame:
    """From Nikkei OHLC, build (feature_date -> next-session return) for training."""
    if nikkei_ohlc.empty:
        return pd.DataFrame()
    df = nikkei_ohlc.sort_values("date").copy()
    df["nikkei_return"] = df["close"].pct_change()
    df["feature_date"] = df["date"].shift(1)
    out = df[["feature_date", "nikkei_return"]].dropna(subset=["feature_date"])
    return out.rename(columns={"feature_date": "date"})


def _train(features: pd.DataFrame, nikkei_ohlc: pd.DataFrame) -> tuple[float | None, float]:
    nikkei = _next_return_frame(nikkei_ohlc)
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
    same shape as the MorningOutlook API model plus enrichments (`us_market`, expected
    range, trend context). The returned dict also carries `actual_gaps` (a date->gap
    map) so the caller can backfill prediction-accuracy history; strip it before
    persisting the outlook itself."""
    date = date or dt.date.today()
    end = date
    start = end - dt.timedelta(days=settings.history_days)

    snapshot = fetch_overnight_snapshot(date)
    us = snapshot.get("us")
    nikkei_prev = snapshot.get("nikkei_prev")
    futures = snapshot.get("futures")
    extra = snapshot.get("extra") or {}

    nikkei_ohlc = fetch_nikkei_ohlc(start, end)
    features = _feature_frame(start, end)

    nikkei_prev_close = nikkei_prev.get("close") if nikkei_prev else None
    nikkei_futures = futures.get("price") if futures else None
    futures_source = futures.get("ticker") if futures else None
    implied_gap = None
    if nikkei_prev_close and nikkei_futures:
        implied_gap = (nikkei_futures - nikkei_prev_close) / nikkei_prev_close

    model_return, r2 = _train(features, nikkei_ohlc)

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

    # Enrichments: expected range from recent gap volatility, trend & vol regime.
    actual_gaps = analytics.compute_actual_gaps(nikkei_ohlc)
    recent_gaps = [actual_gaps[d] for d in sorted(actual_gaps)]
    range_low, range_high = analytics.expected_range(expected_move, recent_gaps)
    context = analytics.nikkei_context(nikkei_ohlc)
    vix_close = us.get("vix_close") if us else None
    us_struct = us_market_struct(us, extra)

    result = {
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
        "expected_range_low": range_low,
        "expected_range_high": range_high,
        "expected_open_low": nikkei_prev_close * (1 + range_low) if nikkei_prev_close else None,
        "expected_open_high": nikkei_prev_close * (1 + range_high) if nikkei_prev_close else None,
        "nikkei_ma25": context.get("ma25"),
        "nikkei_vs_ma25": context.get("vs_ma25"),
        "vix_regime": analytics.vix_regime(vix_close),
        "us_detail": _format_us_detail(us),
        "us_market": us_struct,
        "sectors": analytics.sector_signals(us_struct),
        "narrative": _build_narrative(
            direction=direction,
            expected_move=expected_move,
            implied_gap=implied_gap,
            model_return=model_return,
            nikkei_prev_close=nikkei_prev_close,
            implied_open_level=implied_open_level,
            us=us,
        ),
        "actual_gaps": actual_gaps,
    }
    return result
