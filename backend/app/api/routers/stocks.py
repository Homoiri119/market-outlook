from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.schemas import TargetStockCreate, TargetStockOut
from app.services.target_stocks import add_target_stock, list_target_stocks, remove_target_stock, seed_target_stocks

router = APIRouter(prefix="/api/stocks", tags=["stocks"])


@router.get("", response_model=list[TargetStockOut])
def get_stocks(db: Session = Depends(get_db)) -> list[TargetStockOut]:
    seed_target_stocks(db)
    return [TargetStockOut.model_validate(s) for s in list_target_stocks(db)]


@router.post("", response_model=TargetStockOut)
def create_stock(payload: TargetStockCreate, db: Session = Depends(get_db)) -> TargetStockOut:
    stock = add_target_stock(db, payload.code, payload.name, payload.edinet_company_name)
    return TargetStockOut.model_validate(stock)


@router.delete("/{code}")
def delete_stock(code: str, db: Session = Depends(get_db)) -> dict[str, bool]:
    removed = remove_target_stock(db, code)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Stock code {code} not found")
    return {"removed": True}
