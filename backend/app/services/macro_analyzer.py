"""Builds a simple linear-regression model that predicts the next TOPIX session's
return from the previous day's US market indicators (S&P500 return, VIX change,
USD/JPY return), and classifies the prediction into BUY / SELL / NEUTRAL."""

from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sqlalchemy.orm import Session

from app.config import settings
from app.models import MacroIndicator, MacroSignal, PriceHistory
from app.services.data_collector import TOPIX_CODE

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = ["sp500_return", "vix_change", "usdjpy_return"]


class InsufficientDataError(RuntimeError):
    pass


def _load_macro_df(db: Session) -> pd.DataFrame:
    rows = db.query(MacroIndicator).order_by(MacroIndicator.date).all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        [
            {
                "date": r.date,
                "sp500_return": r.sp500_return,
                "vix_change": r.vix_change,
                "usdjpy_return": r.usdjpy_return,
            }
            for r in rows
        ]
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_topix_next_return(db: Session) -> pd.DataFrame:
    rows = (
        db.query(PriceHistory)
        .filter(PriceHistory.code == TOPIX_CODE)
        .order_by(PriceHistory.date)
        .all()
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([{"date": r.date, "topix_return": r.return_pct} for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    # feature_date = the previous TOPIX trading date; topix_return on this row is the
    # "next session return" relative to feature_date.
    df["date"] = df["date"]
    df["feature_date"] = df["date"].shift(1)
    out = df[["feature_date", "topix_return"]].dropna(subset=["feature_date"])
    out = out.rename(columns={"feature_date": "date"})
    return out


def _build_training_set(db: Session) -> pd.DataFrame:
    macro_df = _load_macro_df(db)
    topix_df = _load_topix_next_return(db)
    if macro_df.empty or topix_df.empty:
        return pd.DataFrame()
    merged = pd.merge(macro_df, topix_df, on="date", how="inner")
    merged = merged.dropna(subset=FEATURE_COLUMNS + ["topix_return"])
    return merged


def compute_macro_signal(db: Session, date: dt.date | None = None) -> MacroSignal:
    """Compute and persist today's macro signal (predicted TOPIX return for the next
    JP trading session, based on the latest available macro indicators)."""
    date = date or dt.date.today()

    training = _build_training_set(db)
    if len(training) < 20:
        raise InsufficientDataError(
            f"Not enough historical data to train the macro model (have {len(training)} rows, need >= 20)"
        )

    X = training[FEATURE_COLUMNS].to_numpy()
    y = training["topix_return"].to_numpy()

    model = LinearRegression()
    model.fit(X, y)
    r2 = float(model.score(X, y))
    confidence = max(0.0, min(1.0, r2))

    macro_df = _load_macro_df(db)
    latest = macro_df.dropna(subset=FEATURE_COLUMNS).iloc[-1]
    latest_date = latest["date"].date()
    X_latest = latest[FEATURE_COLUMNS].to_numpy().reshape(1, -1)
    predicted_return = float(model.predict(X_latest)[0])

    threshold = settings.macro_signal_threshold
    if predicted_return > threshold:
        signal = "BUY"
    elif predicted_return < -threshold:
        signal = "SELL"
    else:
        signal = "NEUTRAL"

    coef_text = ", ".join(
        f"{col}={coef:+.4f}" for col, coef in zip(FEATURE_COLUMNS, model.coef_)
    )
    detail = (
        f"based on {latest_date.isoformat()} US market data "
        f"(SP500 return={latest['sp500_return']:+.4%}, VIX change={latest['vix_change']:+.2f}, "
        f"USDJPY return={latest['usdjpy_return']:+.4%}); "
        f"model coefficients: {coef_text}; intercept={model.intercept_:+.5f}; "
        f"R2={r2:.3f}; trained on {len(training)} samples"
    )

    record = db.query(MacroSignal).filter(MacroSignal.date == date).first()
    if record is None:
        record = MacroSignal(date=date)
        db.add(record)
    record.predicted_return = predicted_return
    record.confidence = confidence
    record.signal = signal
    record.detail = detail
    db.commit()
    db.refresh(record)
    return record


def get_latest_macro_signal(db: Session) -> MacroSignal | None:
    return db.query(MacroSignal).order_by(MacroSignal.date.desc()).first()


def get_macro_signal_history(db: Session, limit: int = 30) -> list[MacroSignal]:
    return (
        db.query(MacroSignal)
        .order_by(MacroSignal.date.desc())
        .limit(limit)
        .all()
    )
