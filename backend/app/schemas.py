from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, ConfigDict


class MacroSignalOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    date: dt.date
    predicted_return: float
    confidence: float
    signal: str
    detail: str | None = None


class StockSignalOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    date: dt.date
    code: str
    name: str
    expected_return: float
    beta: float
    signal: str
    reason: str


class TodaySignalsOut(BaseModel):
    macro_signal: MacroSignalOut | None
    stock_signals: list[StockSignalOut]


class MorningOutlookOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    date: dt.date
    direction: str
    expected_move: float
    confidence: float
    implied_gap: float | None = None
    model_return: float | None = None
    nikkei_prev_close: float | None = None
    nikkei_futures: float | None = None
    implied_open_level: float | None = None
    futures_source: str | None = None
    us_detail: str | None = None
    us_market: dict | None = None
    sectors: list | None = None
    narrative: str | None = None
    expected_range_low: float | None = None
    expected_range_high: float | None = None
    expected_open_low: float | None = None
    expected_open_high: float | None = None
    nikkei_ma25: float | None = None
    nikkei_vs_ma25: float | None = None
    vix_regime: str | None = None
    actual_move: float | None = None
    hit: bool | None = None


class OutlookRunResultOut(BaseModel):
    date: str
    collection_results: dict[str, int]
    direction: str
    expected_move: float
    confidence: float
    stock_signal_count: int
    notification_sent: bool


class TargetStockOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    code: str
    name: str
    edinet_company_name: str | None = None


class TargetStockCreate(BaseModel):
    code: str
    name: str
    edinet_company_name: str | None = None


class RunResultOut(BaseModel):
    date: str
    collection_results: dict[str, int]
    macro_signal: str
    stock_signal_count: int
    notification_sent: bool
