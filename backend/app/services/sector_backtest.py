"""Sector-level walk-forward backtest using J-Quants V2 data.

Sector indices are not in the subscription, so we build cap-weighted sector series
from constituent stocks (17-sector / S17 classification), then run the same honest
model as the index backtest per sector: predict the OPENING GAP from the US
overnight close, score direction (hit-rate), and measure a tradeable open->close
strategy. This reveals WHICH Tokyo sectors the US signal predicts best.

Universe is limited to large caps (TOPIX Core30 + Large70 by default) to keep the
J-Quants call count modest (~100 stock requests).
"""

from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from app.clients.jquants_client import jquants_client
from app.clients.us_market_client import fetch_us_market_history
from app.services.macro_analyzer import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

DEFAULT_SCALE_CATS = ("TOPIX Core30", "TOPIX Large70")


def _select_universe(master: pd.DataFrame, scale_cats, cap: int) -> pd.DataFrame:
    if master.empty:
        return master
    df = master.copy()
    if "ScaleCat" in df.columns and scale_cats:
        sel = df[df["ScaleCat"].isin(scale_cats)]
        if sel.empty:  # unknown ScaleCat labels -> substring fallback
            mask = df["ScaleCat"].astype(str).str.contains("Core30|Large70", regex=True, na=False)
            sel = df[mask]
        df = sel if not sel.empty else df
    # De-dupe by Code, cap the count.
    df = df.drop_duplicates(subset=["Code"]).head(cap)
    return df


def _us_feature_frame(start: dt.date, end: dt.date) -> pd.DataFrame:
    us = fetch_us_market_history(start, end)
    if us.empty:
        return pd.DataFrame()
    us = us.reset_index()
    us["date"] = pd.to_datetime(us["date"])
    cols = ["date"] + [c for c in FEATURE_COLUMNS if c in us.columns]
    return us[cols].dropna(subset=[c for c in FEATURE_COLUMNS if c in us.columns])


def _walk_forward(feature_df: pd.DataFrame, target: pd.Series, intraday: pd.Series,
                  min_train: int) -> dict | None:
    """feature_df: date + FEATURE_COLUMNS. target/intraday: indexed by target_date.
    Pairs US(date=t) with the sector's next-session gap, walks forward, and returns
    hit-rate + tradeable (open->close) long/flat return."""
    tgt = pd.DataFrame({"target_date": target.index, "gap": target.values, "intraday": intraday.values})
    tgt = tgt.sort_values("target_date")
    tgt["date"] = tgt["target_date"].shift(1)  # feature date = previous session
    tgt = tgt.dropna(subset=["date", "gap", "intraday"])
    df = pd.merge(feature_df, tgt, on="date", how="inner").dropna(subset=FEATURE_COLUMNS + ["gap"])
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) < min_train + 30:
        return None

    X = df[FEATURE_COLUMNS].to_numpy()
    y = df["gap"].to_numpy()
    preds = np.empty(len(df) - min_train)
    for j, i in enumerate(range(min_train, len(df))):
        preds[j] = LinearRegression().fit(X[:i], y[:i]).predict(X[i:i + 1])[0]

    test = df.iloc[min_train:]
    gap = test["gap"].to_numpy()
    intra = test["intraday"].to_numpy()
    hit = float(((np.sign(preds) == np.sign(gap)) & (preds != 0)).mean())
    strat = (preds > 0).astype(float) * intra
    total = float(np.cumprod(1 + strat)[-1] - 1)
    return {"n_days": int(len(test)), "hit_rate": hit,
            "mae": float(np.abs(preds - gap).mean()), "strategy_return": total}


def run_sector_backtest(years: float = 4.0, min_train: int = 100, universe_cap: int = 120) -> dict:
    end = dt.date.today()
    start = end - dt.timedelta(days=int(365 * years) + 60)

    master = jquants_client.fetch_listed_info()
    if master.empty or "S17" not in master.columns:
        raise RuntimeError("Could not fetch equities master (S17 sector) from J-Quants.")
    universe = _select_universe(master, DEFAULT_SCALE_CATS, universe_cap)
    if universe.empty:
        raise RuntimeError("Empty stock universe after filtering.")

    feature_df = _us_feature_frame(start, end)
    if feature_df.empty:
        raise RuntimeError("Could not fetch US market features.")

    # Fetch per-stock bars, accumulate gap/intraday/mktcap tagged by sector.
    frames = []
    fetched = 0
    for _, row in universe.iterrows():
        code = str(row["Code"])
        try:
            bars = jquants_client.fetch_daily_quotes(code, start, end)
        except Exception:
            logger.exception("bars fetch failed for %s", code)
            continue
        if bars.empty or "open" not in bars.columns:
            continue
        b = bars.reset_index()
        b["date"] = pd.to_datetime(b["date"])
        b = b.sort_values("date")
        b["prev_close"] = b["close"].shift(1)
        b["gap"] = (b["open"] - b["prev_close"]) / b["prev_close"]
        b["intraday"] = (b["close"] - b["open"]) / b["open"]
        b["w"] = b["mkt_cap"] if "mkt_cap" in b.columns else 1.0
        b["S17"] = row.get("S17")
        b["S17Nm"] = row.get("S17Nm")
        frames.append(b[["date", "S17", "S17Nm", "gap", "intraday", "w"]].dropna(subset=["gap", "intraday"]))
        fetched += 1

    if not frames:
        raise RuntimeError("No stock bars fetched.")
    allbars = pd.concat(frames, ignore_index=True)
    allbars["w"] = allbars["w"].fillna(0).clip(lower=0)

    # Cap-weighted sector aggregation per date.
    def _wavg(g, col):
        w = g["w"].to_numpy()
        v = g[col].to_numpy()
        return float((v * w).sum() / w.sum()) if w.sum() > 0 else float(v.mean())

    stocks_per_sector = universe.groupby("S17")["Code"].nunique().to_dict()

    results = []
    for (s17, s17nm), g in allbars.groupby(["S17", "S17Nm"]):
        gap = g.groupby("date").apply(lambda x: _wavg(x, "gap")).sort_index()
        intra = g.groupby("date").apply(lambda x: _wavg(x, "intraday")).sort_index()
        stats = _walk_forward(feature_df, gap, intra, min_train)
        if stats is None:
            continue
        stats.update({
            "code": str(s17),
            "name": str(s17nm),
            "n_stocks": int(stocks_per_sector.get(s17, 0)),
        })
        results.append(stats)

    results.sort(key=lambda r: r["hit_rate"], reverse=True)
    return {
        "level": "sector",
        "classification": "S17 (17 sectors)",
        "years": years,
        "universe_size": int(fetched),
        "start": feature_df["date"].min().date().isoformat(),
        "end": feature_df["date"].max().date().isoformat(),
        "sectors": results,
    }
