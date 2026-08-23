"""Manages the list of target stocks (TargetStock table), seeded from data/target_stocks.json."""

from __future__ import annotations

import json

from sqlalchemy.orm import Session

from app.config import settings
from app.models import TargetStock


def load_default_target_stocks() -> list[dict]:
    with open(settings.target_stocks_file, encoding="utf-8") as f:
        return json.load(f)


def seed_target_stocks(db: Session) -> None:
    if db.query(TargetStock).count() > 0:
        return
    for item in load_default_target_stocks():
        db.add(
            TargetStock(
                code=item["code"],
                name=item["name"],
                edinet_company_name=item.get("edinet_company_name"),
            )
        )
    db.commit()


def list_target_stocks(db: Session) -> list[TargetStock]:
    return db.query(TargetStock).order_by(TargetStock.code).all()


def add_target_stock(db: Session, code: str, name: str, edinet_company_name: str | None = None) -> TargetStock:
    existing = db.query(TargetStock).filter(TargetStock.code == code).first()
    if existing:
        return existing
    stock = TargetStock(code=code, name=name, edinet_company_name=edinet_company_name)
    db.add(stock)
    db.commit()
    db.refresh(stock)
    return stock


def remove_target_stock(db: Session, code: str) -> bool:
    stock = db.query(TargetStock).filter(TargetStock.code == code).first()
    if not stock:
        return False
    db.delete(stock)
    db.commit()
    return True
