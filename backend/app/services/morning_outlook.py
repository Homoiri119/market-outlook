"""Pre-open (~08:00 JST) morning outlook for the Tokyo session.

The centerpiece is the overnight Nikkei 225 futures gap: at 08:00 JST the CME
Nikkei futures have already priced in the US close, so
    implied gap = (futures price now - previous Nikkei cash close) / previous close
is the market's own estimate of where Tokyo will open. We blend that forward-
looking gap with a US-market -> next-day-Nikkei linear-regression estimate (for
corroboration and to keep working when futures data is momentarily unavailable),
then classify the blended move into STRONG_UP / UP / FLAT / DOWN / STRONG_DOWN.

Unlike the older macro signal, this degrades gracefully: the futures gap alone is
enough to produce an outlook from day one, before enough history exists to train
the regression model.
"""

from __future__ import annotations

import datetime as dt
import json
import logging

import pandas as pd
from sklearn.linear_model import LinearRegression
from sqlalchemy.orm import Session

from app.clients.us_market_client import fetch_nikkei_ohlc, fetch_overnight_snapshot
from app.config import settings
from app.models import MorningOutlook, PriceHistory
from app.services import analytics
from app.services.data_collector import NIKKEI_CODE
from app.services.macro_analyzer import FEATURE_COLUMNS, _load_macro_df

logger = logging.getLogger(__name__)


class OutlookUnavailableError(RuntimeError):
    """Raised when neither the futures gap nor the regression model can be computed."""


DIRECTION_LABELS_JP = {
    "STRONG_UP": "大幅上昇",
    "UP": "上昇",
    "FLAT": "ほぼ横ばい",
    "DOWN": "下落",
    "STRONG_DOWN": "大幅下落",
}


def _load_nikkei_next_return(db: Session) -> pd.DataFrame:
    """Return rows of (date, nikkei_return) where nikkei_return is the return of the
    session *following* `date` — so it can be joined onto US features observed on
    `date`."""
    rows = (
        db.query(PriceHistory)
        .filter(PriceHistory.code == NIKKEI_CODE)
        .order_by(PriceHistory.date)
        .all()
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([{"date": r.date, "nikkei_return": r.return_pct} for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["feature_date"] = df["date"].shift(1)
    out = df[["feature_date", "nikkei_return"]].dropna(subset=["feature_date"])
    return out.rename(columns={"feature_date": "date"})


def _train_model(db: Session) -> tuple[float | None, float]:
    """Train US-features -> next-day-Nikkei-return regression on stored history.

    Returns (predicted_return_for_latest_us_row, r2). predicted_return is None when
    there is not enough history yet (< 20 merged samples)."""
    macro_df = _load_macro_df(db)
    nikkei_df = _load_nikkei_next_return(db)
    if macro_df.empty or nikkei_df.empty:
        return None, 0.0

    merged = pd.merge(macro_df, nikkei_df, on="date", how="inner")
    merged = merged.dropna(subset=FEATURE_COLUMNS + ["nikkei_return"])
    if len(merged) < 20:
        return None, 0.0

    X = merged[FEATURE_COLUMNS].to_numpy()
    y = merged["nikkei_return"].to_numpy()
    model = LinearRegression()
    model.fit(X, y)
    r2 = float(model.score(X, y))

    latest = macro_df.dropna(subset=FEATURE_COLUMNS).iloc[-1]
    X_latest = latest[FEATURE_COLUMNS].to_numpy().reshape(1, -1)
    predicted = float(model.predict(X_latest)[0])
    return predicted, max(0.0, min(1.0, r2))


def _classify(move: float) -> str:
    strong = settings.outlook_strong_threshold
    flat = settings.outlook_flat_threshold
    if move >= strong:
        return "STRONG_UP"
    if move >= flat:
        return "UP"
    if move <= -strong:
        return "STRONG_DOWN"
    if move <= -flat:
        return "DOWN"
    return "FLAT"


def us_market_struct(us: dict | None, extra: dict | None = None) -> dict | None:
    """Build a JSON-safe, display-oriented structured view of the US market snapshot,
    including a coarse risk-on / risk-off read used by the dashboards. `extra` carries
    the additional overnight indicators (SOX, US 10Y, WTI)."""
    if not us:
        return None
    extra = extra or {}
    date = us.get("date")
    sp = us.get("sp500_return")
    vix_chg = us.get("vix_change")

    # Simple risk read: equities up + VIX not rising => risk-on, and vice versa.
    sentiment = "neutral"
    if sp is not None and vix_chg is not None:
        if sp > 0 and vix_chg <= 0:
            sentiment = "risk_on"
        elif sp < 0 and vix_chg >= 0:
            sentiment = "risk_off"
        elif sp > 0:
            sentiment = "risk_on"
        elif sp < 0:
            sentiment = "risk_off"

    return {
        "date": date.isoformat() if isinstance(date, dt.date) else None,
        "sp500_return": sp,
        "nasdaq_return": us.get("nasdaq_return"),
        "dow_return": us.get("dow_return"),
        "vix_close": us.get("vix_close"),
        "vix_change": vix_chg,
        "usdjpy_return": us.get("usdjpy_return"),
        "sentiment": sentiment,
        # extra overnight indicators (may be absent)
        "sox_return": extra.get("sox_return"),
        "sox_level": extra.get("sox_level"),
        "us10y_level": extra.get("us10y_level"),
        "us10y_change_bp": extra.get("us10y_change_bp"),
        "wti_return": extra.get("wti_return"),
        "wti_level": extra.get("wti_level"),
    }


def _format_us_detail(us: dict | None) -> str:
    if not us:
        return "米国市場データ取得不可"
    def pct(key: str) -> str:
        v = us.get(key)
        return "N/A" if v is None else f"{v:+.2%}"
    parts = [
        f"S&P500 {pct('sp500_return')}",
        f"NASDAQ {pct('nasdaq_return')}",
        f"Dow {pct('dow_return')}",
    ]
    vix = us.get("vix_close")
    vix_chg = us.get("vix_change")
    if vix is not None:
        vix_txt = f"VIX {vix:.1f}"
        if vix_chg is not None:
            vix_txt += f" ({vix_chg:+.1f})"
        parts.append(vix_txt)
    usdjpy = us.get("usdjpy_return")
    if usdjpy is not None:
        parts.append(f"USD/JPY {usdjpy:+.2%}")
    date = us.get("date")
    prefix = f"{date.isoformat()} 米国終値: " if isinstance(date, dt.date) else "米国終値: "
    return prefix + " / ".join(parts)


def _build_narrative(
    *,
    direction: str,
    expected_move: float,
    implied_gap: float | None,
    model_return: float | None,
    nikkei_prev_close: float | None,
    implied_open_level: float | None,
    us: dict | None,
) -> str:
    label = DIRECTION_LABELS_JP[direction]
    lines: list[str] = []

    # Headline
    move_pct = f"{expected_move:+.2%}"
    if implied_open_level and nikkei_prev_close:
        lines.append(
            f"本日の東京市場は、日経225が前日終値({nikkei_prev_close:,.0f}円)に対し "
            f"約 {move_pct}({implied_open_level:,.0f}円 近辺)で寄り付く見通しです。方向感: 【{label}】"
        )
    else:
        lines.append(f"本日の東京市場の寄り付き見通し: 約 {move_pct} 【{label}】")

    # Drivers
    driver_bits: list[str] = []
    if us:
        sp = us.get("sp500_return")
        if sp is not None:
            driver_bits.append("米国株の上昇" if sp > 0 else "米国株の下落")
        usdjpy = us.get("usdjpy_return")
        if usdjpy is not None:
            driver_bits.append("円安" if usdjpy > 0 else "円高")
        vix_chg = us.get("vix_change")
        if vix_chg is not None and abs(vix_chg) >= 1.0:
            driver_bits.append("VIX上昇(警戒感)" if vix_chg > 0 else "VIX低下(安心感)")
    if driver_bits:
        lines.append("主な背景: " + "、".join(driver_bits) + "。")

    # Signal breakdown
    detail_bits: list[str] = []
    if implied_gap is not None:
        detail_bits.append(f"日経先物ギャップ {implied_gap:+.2%}(主指標)")
    if model_return is not None:
        detail_bits.append(f"回帰モデル予測 {model_return:+.2%}")
    if detail_bits:
        lines.append("内訳: " + " / ".join(detail_bits) + "。")
    elif implied_gap is None:
        lines.append("※日経先物が取得できず、回帰モデルのみに基づく参考値です。")

    lines.append("(寄り付き前の推定値であり、投資判断の最終責任は利用者本人にあります。)")
    return "\n".join(lines)


def compute_morning_outlook(db: Session, date: dt.date | None = None) -> MorningOutlook:
    date = date or dt.date.today()

    snapshot = fetch_overnight_snapshot(date)
    us = snapshot.get("us")
    nikkei_prev = snapshot.get("nikkei_prev")
    futures = snapshot.get("futures")
    extra = snapshot.get("extra") or {}

    # Primary signal: overnight futures gap vs. the previous Nikkei cash close.
    implied_gap: float | None = None
    nikkei_prev_close = nikkei_prev.get("close") if nikkei_prev else None
    nikkei_futures = futures.get("price") if futures else None
    futures_source = futures.get("ticker") if futures else None
    if nikkei_prev_close and nikkei_futures:
        implied_gap = (nikkei_futures - nikkei_prev_close) / nikkei_prev_close

    # Secondary signal: US -> next-day-Nikkei regression.
    model_return, r2 = _train_model(db)

    if implied_gap is None and model_return is None:
        raise OutlookUnavailableError(
            "No Nikkei futures data and not enough history to train the model."
        )

    # Blend.
    w = settings.futures_gap_weight
    if implied_gap is not None and model_return is not None:
        expected_move = w * implied_gap + (1 - w) * model_return
        confidence = max(0.6, r2)
    elif implied_gap is not None:
        expected_move = implied_gap
        confidence = 0.6  # futures gap is a strong, market-implied signal
    else:
        expected_move = float(model_return)
        confidence = max(0.2, r2)

    direction = _classify(expected_move)
    implied_open_level = (
        nikkei_prev_close * (1 + expected_move) if nikkei_prev_close else nikkei_futures
    )
    us_detail = _format_us_detail(us)
    narrative = _build_narrative(
        direction=direction,
        expected_move=expected_move,
        implied_gap=implied_gap,
        model_return=model_return,
        nikkei_prev_close=nikkei_prev_close,
        implied_open_level=implied_open_level,
        us=us,
    )

    # Enrichments: expected range, trend/vol context (from live Nikkei OHLC).
    nikkei_ohlc = fetch_nikkei_ohlc(date - dt.timedelta(days=settings.history_days), date)
    actual_gaps = analytics.compute_actual_gaps(nikkei_ohlc)
    recent_gaps = [actual_gaps[d] for d in sorted(actual_gaps)]
    range_low, range_high = analytics.expected_range(expected_move, recent_gaps)
    context = analytics.nikkei_context(nikkei_ohlc)
    vix_close = us.get("vix_close") if us else None

    record = db.query(MorningOutlook).filter(MorningOutlook.date == date).first()
    if record is None:
        record = MorningOutlook(date=date)
        db.add(record)
    record.direction = direction
    record.expected_move = float(expected_move)
    record.confidence = float(confidence)
    record.implied_gap = implied_gap
    record.model_return = model_return
    record.nikkei_prev_close = nikkei_prev_close
    record.nikkei_futures = nikkei_futures
    record.implied_open_level = implied_open_level
    record.futures_source = futures_source
    record.us_detail = us_detail
    struct = us_market_struct(us, extra)
    record.us_market_json = json.dumps(struct, ensure_ascii=False) if struct else None
    sectors = analytics.sector_signals(struct)
    record.sectors_json = json.dumps(sectors, ensure_ascii=False) if sectors else None
    record.narrative = narrative
    record.expected_range_low = range_low
    record.expected_range_high = range_high
    record.expected_open_low = nikkei_prev_close * (1 + range_low) if nikkei_prev_close else None
    record.expected_open_high = nikkei_prev_close * (1 + range_high) if nikkei_prev_close else None
    record.nikkei_ma25 = context.get("ma25")
    record.nikkei_vs_ma25 = context.get("vs_ma25")
    record.vix_regime = analytics.vix_regime(vix_close)
    db.commit()
    db.refresh(record)

    _backfill_actuals(db, nikkei_ohlc)
    db.refresh(record)
    return record


def _backfill_actuals(db: Session, nikkei_ohlc=None) -> int:
    """Fill actual_move / hit on past outlook records once the real Tokyo open is known.
    Returns the number of records updated."""
    if nikkei_ohlc is None:
        end = dt.date.today()
        nikkei_ohlc = fetch_nikkei_ohlc(end - dt.timedelta(days=settings.history_days), end)
    actual_gaps = analytics.compute_actual_gaps(nikkei_ohlc)
    if not actual_gaps:
        return 0
    updated = 0
    rows = db.query(MorningOutlook).filter(MorningOutlook.actual_move.is_(None)).all()
    for row in rows:
        actual = actual_gaps.get(row.date.isoformat())
        if actual is None:
            continue
        row.actual_move = actual
        row.hit = analytics.directional_hit(row.expected_move, actual)
        updated += 1
    if updated:
        db.commit()
    return updated


def get_latest_morning_outlook(db: Session) -> MorningOutlook | None:
    return db.query(MorningOutlook).order_by(MorningOutlook.date.desc()).first()


def get_morning_outlook_history(db: Session, limit: int = 30) -> list[MorningOutlook]:
    return (
        db.query(MorningOutlook)
        .order_by(MorningOutlook.date.desc())
        .limit(limit)
        .all()
    )
