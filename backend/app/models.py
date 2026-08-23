from datetime import date as date_type

from sqlalchemy import Date, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class PriceHistory(Base):
    """Daily OHLC + return for a stock code or an index code (e.g. TOPIX, NK225)."""

    __tablename__ = "price_history"
    __table_args__ = (UniqueConstraint("code", "date", name="uq_price_history_code_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(16), index=True)
    date: Mapped[date_type] = mapped_column(Date, index=True)
    open: Mapped[float | None] = mapped_column(Float, nullable=True)
    high: Mapped[float | None] = mapped_column(Float, nullable=True)
    low: Mapped[float | None] = mapped_column(Float, nullable=True)
    close: Mapped[float] = mapped_column(Float)
    return_pct: Mapped[float | None] = mapped_column(Float, nullable=True)


class MacroIndicator(Base):
    """Daily US market / FX indicators used as model features."""

    __tablename__ = "macro_indicator"
    __table_args__ = (UniqueConstraint("date", name="uq_macro_indicator_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date_type] = mapped_column(Date, index=True)
    sp500_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    nasdaq_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    dow_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    vix_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    vix_change: Mapped[float | None] = mapped_column(Float, nullable=True)
    usdjpy_return: Mapped[float | None] = mapped_column(Float, nullable=True)


class MacroSignal(Base):
    """Daily macro-based prediction for the next JP trading session."""

    __tablename__ = "macro_signal"
    __table_args__ = (UniqueConstraint("date", name="uq_macro_signal_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date_type] = mapped_column(Date, index=True)
    predicted_return: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    signal: Mapped[str] = mapped_column(String(16))  # BUY / SELL / NEUTRAL
    detail: Mapped[str | None] = mapped_column(String(1024), nullable=True)


class MorningOutlook(Base):
    """Pre-open (~08:00 JST) outlook for the Tokyo session: how the Nikkei 225 is
    expected to open, driven by the overnight US market and Nikkei futures."""

    __tablename__ = "morning_outlook"
    __table_args__ = (UniqueConstraint("date", name="uq_morning_outlook_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date_type] = mapped_column(Date, index=True)
    # STRONG_UP / UP / FLAT / DOWN / STRONG_DOWN
    direction: Mapped[str] = mapped_column(String(16))
    expected_move: Mapped[float] = mapped_column(Float)  # blended expected open move (fraction)
    confidence: Mapped[float] = mapped_column(Float)
    implied_gap: Mapped[float | None] = mapped_column(Float, nullable=True)  # futures-vs-prev-close gap
    model_return: Mapped[float | None] = mapped_column(Float, nullable=True)  # regression estimate
    nikkei_prev_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    nikkei_futures: Mapped[float | None] = mapped_column(Float, nullable=True)
    implied_open_level: Mapped[float | None] = mapped_column(Float, nullable=True)
    futures_source: Mapped[str | None] = mapped_column(String(16), nullable=True)
    us_detail: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    # JSON-serialized structured US market snapshot (returns/levels) for rich display.
    us_market_json: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    # JSON-serialized per-sector tailwind/headwind signals.
    sectors_json: Mapped[str | None] = mapped_column(String(4000), nullable=True)
    narrative: Mapped[str | None] = mapped_column(String(4000), nullable=True)
    # Expected opening range (fractions + Nikkei levels).
    expected_range_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    expected_range_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    expected_open_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    expected_open_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Trend / volatility context.
    nikkei_ma25: Mapped[float | None] = mapped_column(Float, nullable=True)
    nikkei_vs_ma25: Mapped[float | None] = mapped_column(Float, nullable=True)
    vix_regime: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # Prediction-accuracy tracking (filled in once the real Tokyo open is known).
    actual_move: Mapped[float | None] = mapped_column(Float, nullable=True)
    hit: Mapped[bool | None] = mapped_column(nullable=True)

    @property
    def us_market(self) -> dict | None:
        """Parsed structured US market snapshot (or None)."""
        import json

        if not self.us_market_json:
            return None
        try:
            return json.loads(self.us_market_json)
        except (ValueError, TypeError):
            return None

    @property
    def sectors(self) -> list | None:
        """Parsed per-sector tailwind/headwind signals (or None)."""
        import json

        if not self.sectors_json:
            return None
        try:
            return json.loads(self.sectors_json)
        except (ValueError, TypeError):
            return None


class StockSignal(Base):
    """Daily per-stock buy/sell/hold judgement."""

    __tablename__ = "stock_signal"
    __table_args__ = (UniqueConstraint("date", "code", name="uq_stock_signal_date_code"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date_type] = mapped_column(Date, index=True)
    code: Mapped[str] = mapped_column(String(16), index=True)
    name: Mapped[str] = mapped_column(String(128))
    expected_return: Mapped[float] = mapped_column(Float)
    beta: Mapped[float] = mapped_column(Float)
    signal: Mapped[str] = mapped_column(String(16))  # BUY / SELL / HOLD
    reason: Mapped[str] = mapped_column(String(1024))


class Disclosure(Base):
    """EDINET disclosure documents relevant to target stocks."""

    __tablename__ = "disclosure"
    __table_args__ = (
        UniqueConstraint("doc_id", name="uq_disclosure_doc_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(String(32), index=True)
    date: Mapped[date_type] = mapped_column(Date, index=True)
    code: Mapped[str] = mapped_column(String(16), index=True)
    doc_type: Mapped[str] = mapped_column(String(128))
    title: Mapped[str] = mapped_column(String(512))
    edinet_code: Mapped[str | None] = mapped_column(String(16), nullable=True)


class TargetStock(Base):
    """Stocks that are monitored / analyzed."""

    __tablename__ = "target_stock"
    __table_args__ = (UniqueConstraint("code", name="uq_target_stock_code"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(16), index=True)
    name: Mapped[str] = mapped_column(String(128))
    edinet_company_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
