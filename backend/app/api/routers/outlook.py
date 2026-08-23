from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.schemas import MorningOutlookOut, OutlookRunResultOut
from app.services.morning_outlook import (
    OutlookUnavailableError,
    get_latest_morning_outlook,
    get_morning_outlook_history,
)
from app.services.pipeline import run_morning_outlook_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/outlook", tags=["outlook"])


@router.post("/run", response_model=OutlookRunResultOut)
def run_outlook(db: Session = Depends(get_db)) -> OutlookRunResultOut:
    try:
        result = run_morning_outlook_pipeline(db)
    except OutlookUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Morning outlook pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return OutlookRunResultOut(**result)


@router.get("/latest", response_model=MorningOutlookOut | None)
def latest_outlook(db: Session = Depends(get_db)) -> MorningOutlookOut | None:
    record = get_latest_morning_outlook(db)
    return MorningOutlookOut.model_validate(record) if record else None


@router.get("/history", response_model=list[MorningOutlookOut])
def outlook_history(limit: int = 30, db: Session = Depends(get_db)) -> list[MorningOutlookOut]:
    return [MorningOutlookOut.model_validate(o) for o in get_morning_outlook_history(db, limit)]
