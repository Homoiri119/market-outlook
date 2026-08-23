"""APScheduler setup: runs the daily pipeline on weekday mornings (JST)."""

from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.config import settings
from app.db import SessionLocal
from app.services.pipeline import run_morning_outlook_pipeline

logger = logging.getLogger(__name__)

scheduler = BackgroundScheduler(timezone=settings.scheduler_timezone)


def _scheduled_job() -> None:
    db = SessionLocal()
    try:
        result = run_morning_outlook_pipeline(db)
        logger.info("Morning outlook pipeline finished: %s", result)
    except Exception:
        logger.exception("Morning outlook pipeline failed")
    finally:
        db.close()


def start_scheduler() -> None:
    if scheduler.running:
        return
    scheduler.add_job(
        _scheduled_job,
        trigger=CronTrigger(
            day_of_week="mon-fri",
            hour=settings.daily_run_hour,
            minute=settings.daily_run_minute,
        ),
        id="morning_outlook",
        replace_existing=True,
    )
    scheduler.start()
    logger.info(
        "Scheduler started: morning outlook runs Mon-Fri at %02d:%02d %s",
        settings.daily_run_hour,
        settings.daily_run_minute,
        settings.scheduler_timezone,
    )


def shutdown_scheduler() -> None:
    if scheduler.running:
        scheduler.shutdown(wait=False)
