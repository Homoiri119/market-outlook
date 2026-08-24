"""Per-stock technical (+ best-effort fundamental) analysis.

Given a stock's daily bars, computes standard technical indicators, a direction
call (買い / 中立 / 売り with strength), and ATR-based trade levels: entry,
take-profit, stop-loss, plus the resulting risk/reward. Fundamentals are attached
when the J-Quants plan exposes financial statements (fuzzy-matched, so it degrades
gracefully if fields are missing).

Not investment advice — a rules-based reference only.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ATR multiples for trade levels and the flat band for direction.
SL_ATR = 1.5
TP_ATR = 2.5

DIRECTION_JP = {
    "BUY": "買い", "WEAK_BUY": "やや買い", "NEUTRAL": "中立",
    "WEAK_SELL": "やや売り", "SELL": "売り",
}


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _round_price(x: float) -> float:
    if x is None or np.isnan(x):
        return None
    if x >= 5000:
        return round(x / 5) * 5
    if x >= 1000:
        return round(x)
    return round(x, 1)


def compute_indicators(bars: pd.DataFrame) -> dict:
    df = bars.sort_index().copy()
    close, high, low = df["close"], df["high"], df["low"]
    sma5, sma25, sma75 = _sma(close, 5), _sma(close, 25), _sma(close, 75)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    rsi = _rsi(close)
    atr = _atr(df)
    std20 = close.rolling(20).std()
    mid20 = _sma(close, 20)
    window = min(len(close), 250)
    hi_252 = high.tail(window).max()
    lo_252 = low.tail(window).min()
    vol = df["volume"] if "volume" in df.columns else None
    vol_avg = vol.tail(25).mean() if vol is not None else None
    last = close.iloc[-1]

    def val(s):
        v = s.iloc[-1] if len(s) else np.nan
        return None if pd.isna(v) else float(v)

    return {
        "price": float(last),
        "sma5": val(sma5), "sma25": val(sma25), "sma75": val(sma75),
        "dev25": (float(last) / val(sma25) - 1) if val(sma25) else None,
        "macd": val(macd), "macd_signal": val(macd_signal),
        "macd_hist": (val(macd) - val(macd_signal)) if val(macd) is not None and val(macd_signal) is not None else None,
        "rsi": val(rsi),
        "atr": val(atr),
        "atr_pct": (val(atr) / float(last)) if val(atr) else None,
        "bb_upper": (val(mid20) + 2 * val(std20)) if val(mid20) and val(std20) else None,
        "bb_lower": (val(mid20) - 2 * val(std20)) if val(mid20) and val(std20) else None,
        "high_252": float(hi_252) if not pd.isna(hi_252) else None,
        "low_252": float(lo_252) if not pd.isna(lo_252) else None,
        "pct_from_high": (float(last) / float(hi_252) - 1) if not pd.isna(hi_252) else None,
        "swing_low_10": float(low.tail(10).min()),
        "swing_high_10": float(high.tail(10).max()),
        "rel_volume": (float(vol.iloc[-1]) / vol_avg) if vol is not None and vol_avg else None,
        "mkt_cap": float(df["mkt_cap"].iloc[-1]) if "mkt_cap" in df.columns and not pd.isna(df["mkt_cap"].iloc[-1]) else None,
    }


def _direction(ind: dict) -> tuple[str, int, list[str]]:
    """Score trend + momentum into a direction, with human-readable reasons."""
    score = 0
    reasons: list[str] = []
    price, s25, s75 = ind["price"], ind.get("sma25"), ind.get("sma75")
    if s25 and s75:
        if price > s25 > s75:
            score += 2; reasons.append("株価 > 25日線 > 75日線(上昇トレンド)")
        elif price < s25 < s75:
            score -= 2; reasons.append("株価 < 25日線 < 75日線(下降トレンド)")
        elif price > s25:
            score += 1; reasons.append("株価が25日線の上")
        elif price < s25:
            score -= 1; reasons.append("株価が25日線の下")
    hist = ind.get("macd_hist")
    if hist is not None:
        if hist > 0:
            score += 1; reasons.append("MACDが上向き(シグナル上)")
        else:
            score -= 1; reasons.append("MACDが下向き(シグナル下)")
    rsi = ind.get("rsi")
    if rsi is not None:
        if rsi >= 70:
            score -= 1; reasons.append(f"RSI {rsi:.0f}(買われすぎ・過熱)")
        elif rsi <= 30:
            score += 1; reasons.append(f"RSI {rsi:.0f}(売られすぎ・反発余地)")
        elif rsi >= 55:
            score += 1; reasons.append(f"RSI {rsi:.0f}(強い)")
        elif rsi <= 45:
            score -= 1; reasons.append(f"RSI {rsi:.0f}(弱い)")
    pfh = ind.get("pct_from_high")
    if pfh is not None and pfh >= -0.03:
        score += 1; reasons.append("年初来高値圏(±3%)")

    if score >= 3:
        d = "BUY"
    elif score >= 1:
        d = "WEAK_BUY"
    elif score <= -3:
        d = "SELL"
    elif score <= -1:
        d = "WEAK_SELL"
    else:
        d = "NEUTRAL"
    return d, score, reasons


def _trade_levels(direction: str, ind: dict) -> dict:
    """Pure ATR levels for a consistent risk/reward (~TP_ATR:SL_ATR). Recent swing
    low/high are reported separately as support/resistance context."""
    price = ind["price"]
    atr = ind.get("atr") or price * 0.02
    bullish = direction in ("BUY", "WEAK_BUY")
    bearish = direction in ("SELL", "WEAK_SELL")
    entry = price

    if bearish:
        side = "ショート"
        sl = price + SL_ATR * atr
        tp = price - TP_ATR * atr
        support = ind.get("swing_low_10")       # potential cover / TP zone
        resistance = ind.get("swing_high_10")   # invalidation zone
    else:
        side = "ロング" if bullish else "中立(参考:押し目買い)"
        sl = price - SL_ATR * atr
        tp = price + TP_ATR * atr
        support = ind.get("swing_low_10")       # natural stop reference
        resistance = ind.get("swing_high_10")   # potential TP zone

    rr = abs(tp - entry) / abs(entry - sl) if entry != sl else None
    return {
        "side": side,
        "entry": _round_price(entry),
        "take_profit": _round_price(tp),
        "stop_loss": _round_price(sl),
        "risk_reward": round(rr, 2) if rr else None,
        "sl_pct": (sl / entry - 1) if entry else None,
        "tp_pct": (tp / entry - 1) if entry else None,
        "support": _round_price(support) if support else None,
        "resistance": _round_price(resistance) if resistance else None,
    }


def extract_fundamentals(statements: list[dict], price: float, mkt_cap: float | None) -> dict | None:
    """Best-effort fundamentals from J-Quants /fins/statements (fuzzy field names)."""
    if not statements:
        return None
    latest = statements[-1]

    def find(*keywords):
        for k, v in latest.items():
            kl = k.lower()
            if all(w in kl for w in keywords):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None
        return None

    eps = find("earnings", "pershare") or find("eps")
    bps = find("book", "pershare") or find("bps")
    div = find("dividend", "annual") or find("dividendpershare")
    out = {
        "eps": eps, "bps": bps,
        "per": (price / eps) if eps and eps > 0 else None,
        "pbr": (price / bps) if bps and bps > 0 else None,
        "dividend_yield": (div / price) if div and price else None,
        "period": latest.get("DisclosedDate") or latest.get("Date") or latest.get("CurrentPeriodEndDate"),
        "raw_fields": sorted(latest.keys()),
    }
    return out


def analyze_stock(code: str, name: str, sector: str, bars: pd.DataFrame,
                  statements: list[dict] | None = None) -> dict | None:
    if bars is None or bars.empty or len(bars) < 30:
        return None
    ind = compute_indicators(bars)
    direction, score, reasons = _direction(ind)
    levels = _trade_levels(direction, ind)
    fundamentals = None
    try:
        if statements:
            fundamentals = extract_fundamentals(statements, ind["price"], ind.get("mkt_cap"))
    except Exception:
        logger.exception("fundamentals extraction failed for %s", code)

    return {
        "code": code, "name": name, "sector": sector,
        "direction": direction, "direction_jp": DIRECTION_JP[direction], "score": score,
        "current_price": _round_price(ind["price"]),
        **levels,
        "indicators": ind,
        "reasons": reasons,
        "fundamentals": fundamentals,
    }
