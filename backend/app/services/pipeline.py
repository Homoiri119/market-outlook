"""Orchestrates the daily run: collect data, compute signals, notify Discord."""

from __future__ import annotations

import datetime as dt
import logging

from sqlalchemy.orm import Session

from app.clients.discord_notifier import send_discord_message
from app.services import data_collector, macro_analyzer, morning_outlook, stock_analyzer
from app.services.morning_outlook import DIRECTION_LABELS_JP
from app.services.target_stocks import seed_target_stocks

logger = logging.getLogger(__name__)


def _format_notification(macro_signal, stock_signals, date: dt.date) -> str:
    lines = [f"**日本株 売買判断レポート ({date.isoformat()})**", ""]
    lines.append(
        f"マクロシグナル: **{macro_signal.signal}** "
        f"(予測TOPIXリターン: {macro_signal.predicted_return:+.3%}, 信頼度: {macro_signal.confidence:.2f})"
    )
    lines.append("")
    lines.append("銘柄別判断:")
    for s in sorted(stock_signals, key=lambda x: x.expected_return, reverse=True):
        lines.append(f"- {s.code} {s.name}: **{s.signal}** (期待リターン {s.expected_return:+.3%}, β={s.beta:.2f})")
    return "\n".join(lines)


DIRECTION_EMOJI = {
    "STRONG_UP": "🚀",
    "UP": "📈",
    "FLAT": "➡️",
    "DOWN": "📉",
    "STRONG_DOWN": "⚠️",
}


def _best_effort_stock_signals(db: Session, date: dt.date) -> list:
    """Compute per-stock signals if J-Quants data allows; never fail the morning run."""
    try:
        seed_target_stocks(db)
        data_collector.collect_topix(db, date)
        data_collector.collect_stock_quotes(db, date)
        macro_signal = macro_analyzer.compute_macro_signal(db, date)
        return stock_analyzer.compute_stock_signals(db, macro_signal, date)
    except Exception:
        logger.info("Skipping per-stock signals in morning run (data unavailable)", exc_info=True)
        return []


def _format_morning_brief(outlook, stock_signals, date: dt.date) -> str:
    emoji = DIRECTION_EMOJI.get(outlook.direction, "")
    label = DIRECTION_LABELS_JP.get(outlook.direction, outlook.direction)
    lines = [
        f"{emoji} **東京市場 寄り付き前アウトルック ({date.isoformat()})**",
        "",
        f"見通し: **{label}**(予想寄り付き {outlook.expected_move:+.2%}, 信頼度 {outlook.confidence:.2f})",
    ]
    if outlook.implied_open_level and outlook.nikkei_prev_close:
        lines.append(
            f"日経225: 前日終値 {outlook.nikkei_prev_close:,.0f} → 予想寄り {outlook.implied_open_level:,.0f} 近辺"
        )
    if outlook.us_detail:
        lines.append(outlook.us_detail)
    lines.append("")
    if outlook.narrative:
        lines.append(outlook.narrative)

    if stock_signals:
        lines.append("")
        lines.append("参考: 注目銘柄(期待リターン順)")
        ranked = sorted(stock_signals, key=lambda x: x.expected_return, reverse=True)
        for s in ranked[:3] + ranked[-2:] if len(ranked) > 5 else ranked:
            lines.append(f"- {s.code} {s.name}: **{s.signal}** ({s.expected_return:+.3%})")
    return "\n".join(lines)


def run_morning_outlook_pipeline(db: Session, date: dt.date | None = None) -> dict:
    """Pre-open (~08:00 JST) run: fetch the overnight US market + Nikkei futures,
    compute the Tokyo-open outlook, and push a morning brief to Discord.

    Uses only yfinance (no API keys) for the core outlook; per-stock signals are
    best-effort and only appear when J-Quants data is available."""
    date = date or dt.date.today()

    collection_results = data_collector.collect_market_only(db, date)
    logger.info("Market data collection results: %s", collection_results)

    outlook = morning_outlook.compute_morning_outlook(db, date)
    stock_signals = _best_effort_stock_signals(db, date)

    message = _format_morning_brief(outlook, stock_signals, date)
    sent = send_discord_message(message)

    return {
        "date": date.isoformat(),
        "collection_results": collection_results,
        "direction": outlook.direction,
        "expected_move": outlook.expected_move,
        "confidence": outlook.confidence,
        "stock_signal_count": len(stock_signals),
        "notification_sent": sent,
    }


def run_daily_pipeline(db: Session, date: dt.date | None = None) -> dict:
    date = date or dt.date.today()

    seed_target_stocks(db)

    collection_results = data_collector.collect_all(db, date)
    logger.info("Data collection results: %s", collection_results)

    macro_signal = macro_analyzer.compute_macro_signal(db, date)
    stock_signals = stock_analyzer.compute_stock_signals(db, macro_signal, date)

    message = _format_notification(macro_signal, stock_signals, date)
    sent = send_discord_message(message)

    return {
        "date": date.isoformat(),
        "collection_results": collection_results,
        "macro_signal": macro_signal.signal,
        "stock_signal_count": len(stock_signals),
        "notification_sent": sent,
    }
