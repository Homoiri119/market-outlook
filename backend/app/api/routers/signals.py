from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.schemas import MacroSignalOut, StockSignalOut, TodaySignalsOut
from app.services.macro_analyzer import get_latest_macro_signal, get_macro_signal_history
from app.services.stock_analyzer import get_latest_stock_signals, get_stock_signal_history

router = APIRouter(prefix="/api/signals", tags=["signals"])


@router.get("/today", response_model=TodaySignalsOut)
def today_signals(db: Session = Depends(get_db)) -> TodaySignalsOut:
    macro_signal = get_latest_macro_signal(db)
    stock_signals = get_latest_stock_signals(db)
    return TodaySignalsOut(
        macro_signal=MacroSignalOut.model_validate(macro_signal) if macro_signal else None,
        stock_signals=[StockSignalOut.model_validate(s) for s in stock_signals],
    )


@router.get("/macro/history", response_model=list[MacroSignalOut])
def macro_history(limit: int = 30, db: Session = Depends(get_db)) -> list[MacroSignalOut]:
    return [MacroSignalOut.model_validate(s) for s in get_macro_signal_history(db, limit)]


@router.get("/stock/{code}/history", response_model=list[StockSignalOut])
def stock_history(code: str, limit: int = 30, db: Session = Depends(get_db)) -> list[StockSignalOut]:
    history = get_stock_signal_history(db, code, limit)
    if not history:
        raise HTTPException(status_code=404, detail=f"No signal history for stock code {code}")
    return [StockSignalOut.model_validate(s) for s in history]
