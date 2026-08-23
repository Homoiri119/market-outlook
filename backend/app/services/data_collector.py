"""Fetches data from external sources and stores it in the database."""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
from sqlalchemy.orm import Session

from app.clients import edinet_client
from app.clients.jquants_client import JQuantsAuthError, jquants_client
from app.clients.us_market_client import (
    NIKKEI_INDEX_TICKER,
    fetch_us_market_history,
)
from app.config import settings
from app.models import Disclosure, MacroIndicator, PriceHistory
from app.services.target_stocks import list_target_stocks

import yfinance as yf

logger = logging.getLogger(__name__)

TOPIX_CODE = "TOPIX"
NIKKEI_CODE = "NK225"


def _earliest_missing_date(db: Session, code: str, default_start: dt.date, end: dt.date) -> dt.date:
    """Return the date to start fetching from for a given price-history code.

    If we already have recent data, only fetch a small recent window; otherwise
    backfill from `default_start`.
    """
    latest = (
        db.query(PriceHistory.date)
        .filter(PriceHistory.code == code)
        .order_by(PriceHistory.date.desc())
        .first()
    )
    if latest is None:
        return default_start
    # Refetch the last few days too, in case of late revisions.
    return min(latest[0] - dt.timedelta(days=5), end)


def _upsert_price_history(db: Session, code: str, df: pd.DataFrame) -> int:
    count = 0
    for date_value, row in df.iterrows():
        record = (
            db.query(PriceHistory)
            .filter(PriceHistory.code == code, PriceHistory.date == date_value)
            .first()
        )
        if record is None:
            record = PriceHistory(code=code, date=date_value)
            db.add(record)
        record.open = row.get("open")
        record.high = row.get("high")
        record.low = row.get("low")
        record.close = row["close"]
        record.return_pct = None if pd.isna(row.get("return_pct")) else row.get("return_pct")
        count += 1
    db.commit()
    return count


def _upsert_macro_indicator(db: Session, df: pd.DataFrame) -> int:
    count = 0
    for date_value, row in df.iterrows():
        record = db.query(MacroIndicator).filter(MacroIndicator.date == date_value).first()
        if record is None:
            record = MacroIndicator(date=date_value)
            db.add(record)

        def _val(col: str) -> float | None:
            v = row.get(col)
            return None if v is None or pd.isna(v) else float(v)

        record.sp500_return = _val("sp500_return")
        record.nasdaq_return = _val("nasdaq_return")
        record.dow_return = _val("dow_return")
        record.vix_close = _val("vix_close")
        record.vix_change = _val("vix_change")
        record.usdjpy_return = _val("usdjpy_return")
        count += 1
    db.commit()
    return count


def collect_us_market(db: Session, end: dt.date | None = None) -> int:
    end = end or dt.date.today()
    latest = db.query(MacroIndicator.date).order_by(MacroIndicator.date.desc()).first()
    start = latest[0] - dt.timedelta(days=5) if latest else end - dt.timedelta(days=settings.history_days)
    start = min(start, end)

    df = fetch_us_market_history(start, end)
    if df.empty:
        logger.warning("No US market data fetched")
        return 0
    return _upsert_macro_indicator(db, df)


def collect_nikkei(db: Session, end: dt.date | None = None) -> int:
    """Collect Nikkei 225 cash-index daily history from yfinance (no API key needed)
    and store it under the NK225 price-history code. This is the target series for
    the morning outlook's US -> next-day-Nikkei regression."""
    end = end or dt.date.today()
    start = _earliest_missing_date(db, NIKKEI_CODE, end - dt.timedelta(days=settings.history_days), end)

    try:
        data = yf.Ticker(NIKKEI_INDEX_TICKER).history(
            start=start.isoformat(),
            end=(end + dt.timedelta(days=1)).isoformat(),
            interval="1d",
        )
    except Exception:
        logger.exception("Failed to fetch Nikkei 225 index history")
        return 0
    if data.empty:
        logger.warning("No Nikkei 225 index data fetched")
        return 0

    df = pd.DataFrame(
        {
            "open": data["Open"],
            "high": data["High"],
            "low": data["Low"],
            "close": data["Close"],
        }
    )
    df.index = [idx.date() for idx in data.index]
    df = df.dropna(subset=["close"])
    df["return_pct"] = df["close"].pct_change()
    return _upsert_price_history(db, NIKKEI_CODE, df)


def collect_topix(db: Session, end: dt.date | None = None) -> int:
    end = end or dt.date.today()
    start = _earliest_missing_date(db, TOPIX_CODE, end - dt.timedelta(days=settings.history_days), end)

    try:
        df = jquants_client.fetch_topix_history(start, end)
    except JQuantsAuthError:
        logger.warning("Skipping TOPIX collection: J-Quants credentials not configured")
        return 0
    if df.empty:
        logger.warning("No TOPIX data fetched")
        return 0
    return _upsert_price_history(db, TOPIX_CODE, df)


def collect_stock_quotes(db: Session, end: dt.date | None = None) -> int:
    end = end or dt.date.today()
    total = 0
    for stock in list_target_stocks(db):
        start = _earliest_missing_date(db, stock.code, end - dt.timedelta(days=settings.history_days), end)
        try:
            df = jquants_client.fetch_daily_quotes(stock.code, start, end)
        except JQuantsAuthError:
            logger.warning("Skipping stock quote collection: J-Quants credentials not configured")
            break
        except Exception:
            logger.exception("Failed to fetch daily quotes for %s", stock.code)
            continue
        if df.empty:
            continue
        total += _upsert_price_history(db, stock.code, df)
    return total


def collect_disclosures(db: Session, date: dt.date | None = None) -> int:
    date = date or dt.date.today()
    stocks = list_target_stocks(db)
    company_names = {s.edinet_company_name for s in stocks if s.edinet_company_name}
    name_to_code = {s.edinet_company_name: s.code for s in stocks if s.edinet_company_name}

    try:
        matches = edinet_client.fetch_notable_disclosures(date, company_names)
    except Exception:
        logger.exception("Failed to fetch EDINET disclosures")
        return 0

    count = 0
    for match in matches:
        existing = db.query(Disclosure).filter(Disclosure.doc_id == match["doc_id"]).first()
        if existing:
            continue
        db.add(
            Disclosure(
                doc_id=match["doc_id"],
                date=date,
                code=name_to_code.get(match["matched_company"], ""),
                doc_type=match["doc_type"],
                title=match["title"],
                edinet_code=match.get("edinet_code"),
            )
        )
        count += 1
    db.commit()
    return count


def collect_market_only(db: Session, date: dt.date | None = None) -> dict[str, int]:
    """Lightweight, key-free collection used by the morning outlook: only the US
    market (yfinance) and the Nikkei 225 cash index (yfinance). No J-Quants / EDINET."""
    date = date or dt.date.today()
    return {
        "us_market": collect_us_market(db, date),
        "nikkei": collect_nikkei(db, date),
    }


def collect_all(db: Session, date: dt.date | None = None) -> dict[str, int]:
    date = date or dt.date.today()
    results = {}
    results["us_market"] = collect_us_market(db, date)
    results["nikkei"] = collect_nikkei(db, date)
    results["topix"] = collect_topix(db, date)
    results["stock_quotes"] = collect_stock_quotes(db, date)
    try:
        results["disclosures"] = collect_disclosures(db, date)
    except edinet_client.EdinetAuthError:
        logger.warning("Skipping EDINET disclosure collection: API key not configured")
        results["disclosures"] = 0
    return results
