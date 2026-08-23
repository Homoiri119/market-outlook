from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.schemas import RunResultOut
from app.services.macro_analyzer import InsufficientDataError as MacroInsufficientDataError
from app.services.pipeline import run_daily_pipeline
from app.services.stock_analyzer import InsufficientDataError as StockInsufficientDataError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/run", tags=["run"])


@router.post("/daily", response_model=RunResultOut)
def run_daily(db: Session = Depends(get_db)) -> RunResultOut:
    try:
        result = run_daily_pipeline(db)
    except (MacroInsufficientDataError, StockInsufficientDataError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Daily pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return RunResultOut(**result)
