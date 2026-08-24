from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.api.routers import outlook, run, signals, stocks
from app.db import SessionLocal, init_db
from app.services.scheduler import shutdown_scheduler, start_scheduler
from app.services.target_stocks import seed_target_stocks

STATIC_DIR = Path(__file__).resolve().parent / "static"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    db = SessionLocal()
    try:
        seed_target_stocks(db)
    finally:
        db.close()
    start_scheduler()
    yield
    shutdown_scheduler()


app = FastAPI(title="日本株売買判断アプリ", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(signals.router)
app.include_router(stocks.router)
app.include_router(run.router)
app.include_router(outlook.router)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    """Serve the no-build HTML dashboard (open http://localhost:8000/)."""
    return FileResponse(STATIC_DIR / "dashboard.html")


@app.get("/backtest.html", include_in_schema=False)
def backtest_page() -> FileResponse:
    """Serve the backtest page (falls back to /api/backtest for data in live mode)."""
    return FileResponse(STATIC_DIR / "backtest.html")


@app.get("/analysis.html", include_in_schema=False)
def analysis_page() -> FileResponse:
    """Serve the per-stock analysis page."""
    return FileResponse(STATIC_DIR / "analysis.html")


@app.get("/sector_backtest.html", include_in_schema=False)
def sector_backtest_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "sector_backtest.html")


@app.get("/stock_strategy.html", include_in_schema=False)
def stock_strategy_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "stock_strategy.html")


@app.get("/news.html", include_in_schema=False)
def news_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "news.html")
