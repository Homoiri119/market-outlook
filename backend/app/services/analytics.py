"""Pure analytics helpers shared by the DB-backed and stateless (cloud) paths.

No database access and no network here — callers pass in dataframes / lists. This
keeps prediction-accuracy tracking, expected-range, trend context and VIX-regime
logic consistent across both the local API and the GitHub Actions run.
"""

from __future__ import annotations

import datetime as dt
import statistics

import pandas as pd

# Directional flat band: |move| below this counts as "flat" when scoring hits.
HIT_FLAT_BAND = 0.0015  # 0.15%
DEFAULT_RANGE_SIGMA = 0.004  # 0.4% fallback when gap history is too short


def compute_actual_gaps(nikkei_ohlc: pd.DataFrame) -> dict[str, float]:
    """Map ISO date -> actual opening gap for the Nikkei 225 cash index.

    actual_gap(D) = (open(D) - close(prev trading day)) / close(prev trading day).
    `nikkei_ohlc` must have columns: date (datetime-like), open, close.
    """
    if nikkei_ohlc.empty or not {"date", "open", "close"}.issubset(nikkei_ohlc.columns):
        return {}
    df = nikkei_ohlc.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["prev_close"] = df["close"].shift(1)
    df["actual_gap"] = (df["open"] - df["prev_close"]) / df["prev_close"]
    out: dict[str, float] = {}
    for d, g in zip(df["date"], df["actual_gap"]):
        if pd.notna(g):
            out[d.date().isoformat()] = float(g)
    return out


def directional_hit(expected: float, actual: float, flat: float = HIT_FLAT_BAND) -> bool:
    """True if the predicted direction matches the realized direction (with a flat band)."""

    def sign(v: float) -> int:
        return 1 if v > flat else -1 if v < -flat else 0

    return sign(expected) == sign(actual)


def backfill_history_actuals(history: list[dict], actual_gaps: dict[str, float]) -> list[dict]:
    """Fill `actual_move` and `hit` on past history entries once the real open is known.
    Mutates and returns the same list. Entries whose open is not yet available are left
    as-is (actual_move stays None)."""
    for entry in history:
        if entry.get("actual_move") is not None:
            continue
        actual = actual_gaps.get(entry.get("date"))
        if actual is None:
            continue
        entry["actual_move"] = actual
        exp = entry.get("expected_move")
        entry["hit"] = directional_hit(exp, actual) if exp is not None else None
    return history


def accuracy_summary(history: list[dict], window: int = 20) -> dict:
    """Directional hit-rate and mean absolute error over the most recent scored entries."""
    scored = [h for h in history if h.get("actual_move") is not None and h.get("expected_move") is not None]
    scored = scored[-window:]
    n = len(scored)
    if n == 0:
        return {"n": 0, "hit_rate": None, "mae": None}
    hits = sum(1 for h in scored if h.get("hit"))
    mae = sum(abs(h["expected_move"] - h["actual_move"]) for h in scored) / n
    return {"n": n, "hits": hits, "hit_rate": hits / n, "mae": mae}


def expected_range(expected_move: float, recent_gaps: list[float]) -> tuple[float, float]:
    """±1σ range around the point estimate, where σ is the stdev of recent actual gaps."""
    vals = [g for g in recent_gaps if g is not None][-20:]
    if len(vals) >= 5:
        sigma = statistics.pstdev(vals)
        sigma = max(sigma, 0.001)  # floor so the range is never degenerate
    else:
        sigma = DEFAULT_RANGE_SIGMA
    return expected_move - sigma, expected_move + sigma


def nikkei_context(nikkei_ohlc: pd.DataFrame) -> dict:
    """Trend context: 25-day moving average and the latest close's deviation from it."""
    if nikkei_ohlc.empty or "close" not in nikkei_ohlc.columns:
        return {"ma25": None, "vs_ma25": None}
    close = nikkei_ohlc.sort_values("date")["close"].dropna()
    if len(close) < 25:
        return {"ma25": None, "vs_ma25": None}
    ma25 = float(close.tail(25).mean())
    last = float(close.iloc[-1])
    return {"ma25": ma25, "vs_ma25": (last - ma25) / ma25 if ma25 else None}


def vix_regime(vix_level: float | None) -> str | None:
    """Coarse volatility regime label key for the VIX level."""
    if vix_level is None:
        return None
    if vix_level < 15:
        return "calm"
    if vix_level < 20:
        return "watch"
    if vix_level < 30:
        return "elevated"
    return "fear"
