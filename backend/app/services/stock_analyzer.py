"""Combines the macro signal with each target stock's historical beta to TOPIX and
recent EDINET disclosures to produce a per-stock BUY / SELL / HOLD judgement."""

from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Disclosure, MacroSignal, PriceHistory, StockSignal, TargetStock
from app.services.data_collector import TOPIX_CODE
from app.services.target_stocks import list_target_stocks

logger = logging.getLogger(__name__)


class InsufficientDataError(RuntimeError):
    pass


def _load_returns(db: Session, code: str) -> pd.DataFrame:
    rows = (
        db.query(PriceHistory.date, PriceHistory.return_pct)
        .filter(PriceHistory.code == code)
        .order_by(PriceHistory.date)
        .all()
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", "return_pct"])
    return df.dropna(subset=["return_pct"])


def _compute_beta(stock_returns: pd.DataFrame, topix_returns: pd.DataFrame) -> tuple[float, int]:
    merged = pd.merge(stock_returns, topix_returns, on="date", suffixes=("_stock", "_topix"))
    if len(merged) < 20:
        return 1.0, len(merged)
    topix = merged["return_pct_topix"].to_numpy()
    stock = merged["return_pct_stock"].to_numpy()
    variance = float(np.var(topix))
    if variance == 0:
        return 1.0, len(merged)
    covariance = float(np.cov(stock, topix)[0, 1])
    beta = covariance / variance
    return beta, len(merged)


def _recent_disclosures(db: Session, code: str, date: dt.date, lookback_days: int = 3) -> list[Disclosure]:
    start = date - dt.timedelta(days=lookback_days)
    return (
        db.query(Disclosure)
        .filter(Disclosure.code == code, Disclosure.date >= start, Disclosure.date <= date)
        .all()
    )


def compute_stock_signals(
    db: Session, macro_signal: MacroSignal, date: dt.date | None = None
) -> list[StockSignal]:
    date = date or dt.date.today()

    topix_returns = _load_returns(db, TOPIX_CODE)
    if topix_returns.empty:
        raise InsufficientDataError("No TOPIX history available to compute stock betas")

    results = []
    for stock in list_target_stocks(db):
        stock_returns = _load_returns(db, stock.code)
        if stock_returns.empty:
            logger.warning("No price history for %s; skipping", stock.code)
            continue

        beta, sample_size = _compute_beta(stock_returns, topix_returns)
        expected_return = macro_signal.predicted_return * beta

        threshold = settings.stock_signal_threshold
        if expected_return > threshold:
            signal = "BUY"
        elif expected_return < -threshold:
            signal = "SELL"
        else:
            signal = "HOLD"

        reason = (
            f"macro signal={macro_signal.signal} (predicted TOPIX return={macro_signal.predicted_return:+.4%}, "
            f"confidence={macro_signal.confidence:.2f}); beta={beta:.2f} (n={sample_size}); "
            f"expected return={expected_return:+.4%}"
        )

        disclosures = _recent_disclosures(db, stock.code, date)
        if disclosures:
            titles = "; ".join(f"{d.doc_type}: {d.title}" for d in disclosures)
            reason += f" | 注意: 直近の開示あり ({titles})"

        record = (
            db.query(StockSignal)
            .filter(StockSignal.date == date, StockSignal.code == stock.code)
            .first()
        )
        if record is None:
            record = StockSignal(date=date, code=stock.code, name=stock.name)
            db.add(record)
        record.name = stock.name
        record.expected_return = expected_return
        record.beta = beta
        record.signal = signal
        record.reason = reason
        results.append(record)

    db.commit()
    for record in results:
        db.refresh(record)
    return results


def get_latest_stock_signals(db: Session) -> list[StockSignal]:
    latest = db.query(StockSignal.date).order_by(StockSignal.date.desc()).first()
    if latest is None:
        return []
    return (
        db.query(StockSignal)
        .filter(StockSignal.date == latest[0])
        .order_by(StockSignal.code)
        .all()
    )


def get_stock_signal_history(db: Session, code: str, limit: int = 30) -> list[StockSignal]:
    return (
        db.query(StockSignal)
        .filter(StockSignal.code == code)
        .order_by(StockSignal.date.desc())
        .limit(limit)
        .all()
    )
