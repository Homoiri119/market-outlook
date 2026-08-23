"""Walk-forward backtest of the US-overnight -> Tokyo-open model.

Index level (Nikkei 225), yfinance only, no API key. Expanding-window walk-forward
(train on all data strictly before each test day) so there is no look-ahead bias.

What it measures, and why it is framed carefully:
  * The product predicts the OPENING GAP (how Tokyo opens vs. the previous close).
    So the headline metric is the directional hit-rate of that gap. This is what the
    tool actually claims to do.
  * The opening gap itself is NOT tradeable: by the time you can buy (at the open),
    the gap has already happened. So the strategy P&L is measured on the TRADEABLE
    window only — enter at the open, exit at the close (open->close, gap excluded).
    (An earlier version compounded close-to-close returns, which silently included
    the un-tradeable gap and produced absurdly inflated results.)

Past performance is not indicative of future results; fees/slippage are not modeled.
"""

from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from app.clients.us_market_client import fetch_nikkei_ohlc, fetch_us_market_history
from app.services.macro_analyzer import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


def _dataset(years: float) -> pd.DataFrame:
    """Rows of (date=US feature day, FEATURE_COLUMNS, target_date, gap, intraday, c2c).

    For each Nikkei session d:
      gap[d]      = (open[d] - close[d-1]) / close[d-1]   -> what the model predicts
      intraday[d] = (close[d] - open[d]) / open[d]        -> the tradeable window
      c2c[d]      = (close[d] - close[d-1]) / close[d-1]  -> buy & hold benchmark
    The gap on session d reacts to the US close of the prior evening (US date d-1),
    so it is paired with US features on the previous calendar trading date.
    """
    end = dt.date.today()
    start = end - dt.timedelta(days=int(365 * years) + 60)

    us = fetch_us_market_history(start, end)
    if us.empty:
        return pd.DataFrame()
    us = us.reset_index()
    us["date"] = pd.to_datetime(us["date"])
    feats = us[["date"] + [c for c in FEATURE_COLUMNS if c in us.columns]].dropna()

    nk = fetch_nikkei_ohlc(start, end)
    if nk.empty:
        return pd.DataFrame()
    nk = nk.sort_values("date").copy()
    nk["prev_close"] = nk["close"].shift(1)
    nk["gap"] = (nk["open"] - nk["prev_close"]) / nk["prev_close"]
    nk["intraday"] = (nk["close"] - nk["open"]) / nk["open"]
    nk["c2c"] = (nk["close"] - nk["prev_close"]) / nk["prev_close"]
    nk["feat_date"] = nk["date"].shift(1)  # gap on `date` reacts to prev day's US close
    tgt = nk[["feat_date", "date", "gap", "intraday", "c2c"]].dropna(subset=["feat_date", "gap"])
    tgt = tgt.rename(columns={"feat_date": "date", "date": "target_date"})

    df = pd.merge(feats, tgt, on="date", how="inner").dropna(
        subset=FEATURE_COLUMNS + ["gap", "intraday", "c2c"]
    )
    return df.sort_values("date").reset_index(drop=True)


def _metrics(ret: np.ndarray) -> dict:
    if len(ret) == 0:
        return {"total_return": None, "cagr": None, "sharpe": None, "mdd": None}
    eq = np.cumprod(1 + ret)
    total = float(eq[-1] - 1)
    n = len(ret)
    cagr = float(eq[-1] ** (TRADING_DAYS / n) - 1) if eq[-1] > 0 else None
    vol = float(ret.std(ddof=0))
    sharpe = float(ret.mean() / vol * np.sqrt(TRADING_DAYS)) if vol > 0 else None
    peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1).min())
    return {"total_return": total, "cagr": cagr, "sharpe": sharpe, "mdd": mdd}


def run_index_backtest(years: float = 6.0, min_train: int = 120) -> dict:
    df = _dataset(years)
    if len(df) < min_train + 40:
        raise RuntimeError(
            f"Not enough history for a backtest (have {len(df)} usable rows, need > {min_train + 40})."
        )

    X = df[FEATURE_COLUMNS].to_numpy()
    y = df["gap"].to_numpy()  # model predicts the opening gap

    preds = np.empty(len(df) - min_train)
    for j, i in enumerate(range(min_train, len(df))):
        model = LinearRegression().fit(X[:i], y[:i])
        preds[j] = model.predict(X[i : i + 1])[0]

    test = df.iloc[min_train:].copy().reset_index(drop=True)
    test["pred"] = preds
    gap = test["gap"].to_numpy()
    intraday = test["intraday"].to_numpy()  # tradeable: open -> close
    c2c = test["c2c"].to_numpy()  # buy & hold benchmark
    pred = test["pred"].to_numpy()

    # Headline accuracy: does the model call the OPENING GAP direction correctly?
    hit = (np.sign(pred) == np.sign(gap)) & (pred != 0)
    hit_rate = float(hit.mean())
    mae = float(np.abs(pred - gap).mean())

    # Tradeable strategies: enter at the open, exit at the close (gap NOT captured).
    pos_lf = (pred > 0).astype(float)  # long / flat
    pos_ls = np.sign(pred)  # long / short
    ret_lf = pos_lf * intraday
    ret_ls = pos_ls * intraday

    strategies = {
        "long_flat": _metrics(ret_lf) | {"hit_rate": hit_rate},
        "long_short": _metrics(ret_ls) | {"hit_rate": hit_rate},
        "buy_hold": _metrics(c2c),
    }

    # Per-year distribution: gap-direction hit-rate + tradeable (open->close) return.
    test["year"] = test["target_date"].dt.year
    by_year = []
    for year, g in test.groupby("year"):
        a = g["gap"].to_numpy()
        p = g["pred"].to_numpy()
        intra = g["intraday"].to_numpy()
        cc = g["c2c"].to_numpy()
        h = ((np.sign(p) == np.sign(a)) & (p != 0)).mean()
        lf = float(np.cumprod(1 + (p > 0).astype(float) * intra)[-1] - 1)
        bh = float(np.cumprod(1 + cc)[-1] - 1)
        by_year.append(
            {"year": int(year), "n": int(len(g)), "hit_rate": float(h),
             "strategy_return": lf, "buy_hold_return": bh}
        )

    # Equity curves (downsampled for embedding).
    eq_lf = np.cumprod(1 + ret_lf)
    eq_ls = np.cumprod(1 + ret_ls)
    eq_bh = np.cumprod(1 + c2c)
    step = max(1, len(test) // 300)
    equity = []
    for i in range(0, len(test), step):
        equity.append(
            {
                "date": test["target_date"].iloc[i].date().isoformat(),
                "lf": float(eq_lf[i]),
                "ls": float(eq_ls[i]),
                "bh": float(eq_bh[i]),
            }
        )
    # Always include the final point.
    if (len(test) - 1) % step != 0:
        i = len(test) - 1
        equity.append(
            {"date": test["target_date"].iloc[i].date().isoformat(),
             "lf": float(eq_lf[i]), "ls": float(eq_ls[i]), "bh": float(eq_bh[i])}
        )

    return {
        "level": "index",
        "instrument": "Nikkei 225",
        "target": "opening_gap",  # what the model predicts
        "strategy_window": "open_to_close",  # tradeable window used for P&L
        "years": years,
        "n": int(len(test)),
        "train_min": min_train,
        "start": test["target_date"].iloc[0].date().isoformat(),
        "end": test["target_date"].iloc[-1].date().isoformat(),
        "hit_rate": hit_rate,
        "mae": mae,
        "features": FEATURE_COLUMNS,
        "strategies": strategies,
        "by_year": by_year,
        "equity": equity,
    }
