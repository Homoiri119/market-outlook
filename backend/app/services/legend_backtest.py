"""Survivorship-free backtest of "legendary trader" style rules (J-Quants V2).

Why: an earlier yfinance study ran these rules on the *current* Nikkei 225 members
and found that simply equal-weighting today's members beat the index by ~3%/yr —
i.e. the universe itself was biased toward winners, inflating breakout/momentum
results. Here the universe is built from bars-by-date: each day contains only the
stocks that actually traded that day, including those later delisted.

Rules (signal on day i's close, entry at day i+1's open unless noted):
  turtle55   55-day high breakout, stop 2N (N=20d ATR), exit on close < prior 20-day low
  duke52     first close at a 52-week high, stop -10%, exit on close < 25-day MA (max 120d)
  minervini  trend template (price>150MA>200MA, 200MA rising, 50MA>150MA, price>50MA,
             >=+30% off 52w low, within 25% of 52w high, RS rank >= 70th pct) + first
             50-day closing high on volume >= 1.5x; stop -8%, exit on close < 50MA (max 60d)
  cis_vol    first 20-day closing high on volume >= 2x; exit on close < prior 10-day low (max 60d)
  bnf_dev    close <= -20% below the 25-day MA (first cross); exit on close >= 25MA or 10 days
  bnf_small  BNF's "small account" idea: turnover >= 3x its 20d average and day return
             >= +5% (a 売買代金急増 / 値上がり率 ranking proxy); stop -8%, exit on close
             < 10MA (max 10d)
  ep         Qullamaggie episodic pivot: gap >= +10% on volume >= 3x and close > open;
             buy that close, stop at that day's low, exit on close < 10MA (max 60d)

Outputs trade-level stats (vs TOPIX), first/second-half split, and 10-slot portfolio
simulations compared with the survivorship-free equal-weight baseline. Not investment
advice; costs are a flat round-trip assumption, no slippage model, no taxes.
"""

from __future__ import annotations

import datetime as dt
import logging
import time

import numpy as np
import pandas as pd

from app.clients.jquants_client import jquants_client
from app.services.ipo_backtest import _equity_codes

logger = logging.getLogger(__name__)

LOOKBACK_DAYS = 1400          # ~3.8y; J-Quants plan horizon (earlier dates simply return [])
WARMUP_BARS = 250             # 52-week indicators need this many bars before signals count
MIN_TURNOVER = 100_000_000    # ¥100M 20-day average turnover at signal time (tradable)
COST = 0.002                  # round-trip cost per trade
SLOTS = 10                    # concurrent positions in the portfolio simulation
RETRY_WAITS = (5, 15, 30, 60, 120, 180)  # seconds; J-Quants 429s lasted minutes in run #2
STRATEGIES = ["turtle55", "duke52", "minervini", "cis_vol", "bnf_dev", "bnf_small", "ep"]
LABELS = {
    "turtle55": "タートル(55日高値ブレイク)",
    "duke52": "DUKE。(52週高値ブレイク)",
    "minervini": "ミネルヴィニ(トレンドテンプレート)",
    "cis_vol": "cis(出来高を伴う20日高値)",
    "bnf_dev": "BNF(25日線乖離−20%の逆張り)",
    "bnf_small": "BNF(売買代金急増+値上がりの順張り)",
    "ep": "クラマギー(EP:+10%の窓)",
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _num(row: dict, *keys: str) -> float:
    for k in keys:
        v = row.get(k)
        if v is not None and v != "":
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return np.nan


def fetch_panel(lookback_days: int = LOOKBACK_DAYS, sleep: float = 0.15) -> dict[str, pd.DataFrame]:
    """Wide (date x code) frames of adjusted O/H/L/C, volume and turnover, built from
    bars-by-date so the universe is point-in-time. Non-stock listings are dropped using
    masters taken at several dates (so delisted stocks are still classified)."""
    end = dt.date.today()
    start = end - dt.timedelta(days=lookback_days)
    # Trading calendar from TOPIX (one call), so an empty response on a calendar day is
    # an error (e.g. HTTP 429), never a holiday. Run #2 silently lost 54 days that way.
    topix = jquants_client.fetch_topix_history(start, end)
    calendar = sorted(pd.to_datetime(topix.index).date) if not topix.empty else []
    if not calendar:
        raise RuntimeError("TOPIX calendar unavailable (plan/credentials?)")
    recs: list[tuple] = []
    missing: list[dt.date] = []
    for n_days, d in enumerate(calendar, 1):
        rows = jquants_client.fetch_quotes_by_date(d)
        # Before the first non-empty day we may simply be outside the plan's horizon:
        # retry briefly and skip. After it, back off on rate limits / transient errors.
        waits = RETRY_WAITS if recs else (2.0,)
        for wait in waits:
            if rows:
                break
            logger.info("no bars for %s; retrying in %.0fs", d, wait)
            time.sleep(wait)
            rows = jquants_client.fetch_quotes_by_date(d)
        if not rows and recs:
            missing.append(d)
        for r in rows:
            code = str(r.get("Code") or "")
            if not code:
                continue
            o = _num(r, "AdjO", "AdjustmentOpen", "O")
            h = _num(r, "AdjH", "AdjustmentHigh", "H")
            lo = _num(r, "AdjL", "AdjustmentLow", "L")
            c = _num(r, "AdjC", "AdjustmentClose", "C")
            v = _num(r, "AdjVo", "AdjustmentVolume", "Vo")
            va = _num(r, "Va", "TurnoverValue")
            if np.isnan(va) and not (np.isnan(c) or np.isnan(v)):
                va = c * v
            recs.append((d, code, o, h, lo, c, v, va))
        if n_days % 50 == 0:
            logger.info("fetched %d/%d trading days (at %s, %d rows so far)",
                        n_days, len(calendar), d, len(recs))
        time.sleep(sleep)
    if missing and missing[-1] == calendar[-1]:
        missing.pop()  # today's bars may not be published yet
    if missing:
        # A gap distorts every signal and exit around it, so fail instead of reporting.
        raise RuntimeError(f"bars-by-date missing for {len(missing)} trading days: "
                           f"{', '.join(map(str, missing[:20]))}")
    if not recs:
        raise RuntimeError("No bars-by-date returned (plan/credentials?)")

    df = pd.DataFrame(recs, columns=["date", "code", "open", "high", "low", "close", "volume", "turnover"])
    df["date"] = pd.to_datetime(df["date"])

    # Equity filter: union of masters at several dates (keeps delisted names classified).
    dates = sorted(df["date"].dt.date.unique())
    keep: set[str] = set()
    for md in {dates[0], dates[len(dates) // 2], dates[-1], None}:
        try:
            m = jquants_client.fetch_listed_info(md)
        except Exception:
            logger.info("master unavailable for %s", md, exc_info=True)
            continue
        if not m.empty and "Code" in m.columns:
            eq = _equity_codes(m)
            if eq:
                keep |= eq
    if keep:
        before = df["code"].nunique()
        df = df[df["code"].isin(keep)]
        logger.info("equity filter: %d -> %d codes", before, df["code"].nunique())

    panel = {k: df.pivot_table(index="date", columns="code", values=k, aggfunc="last").astype("float32")
             for k in ["open", "high", "low", "close", "volume", "turnover"]}
    logger.info("panel: %d days x %d codes", *panel["close"].shape)
    return panel


def fetch_benchmark(panel: dict[str, pd.DataFrame]) -> pd.Series:
    idx = panel["close"].index
    topix = jquants_client.fetch_topix_history(idx[0].date(), idx[-1].date())
    s = pd.Series(topix["close"].values, index=pd.to_datetime(topix.index))
    return s.reindex(idx).ffill()


# ---------------------------------------------------------------------------
# Signals and trades
# ---------------------------------------------------------------------------

def _trades_for_stock(code: str, o, h, l, c, v, va, dates, rs, start_i: int) -> dict[str, list]:
    out: dict[str, list] = {k: [] for k in STRATEGIES}
    n = len(c)
    s = pd.Series(c)
    ma10, ma25, ma50, ma150, ma200 = (s.rolling(k).mean().values for k in (10, 25, 50, 150, 200))
    hi20 = s.rolling(20).max().values
    hi50 = s.rolling(50).max().values
    hi250 = s.rolling(250).max().values
    lo250 = s.rolling(250).min().values
    lo10 = s.rolling(10).min().values
    hs, ls = pd.Series(h), pd.Series(l)
    hi55 = hs.rolling(55).max().shift(1).values
    lo20 = ls.rolling(20).min().values
    prev = np.roll(c, 1)
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    atr = pd.Series(tr).rolling(20).mean().values
    v20 = pd.Series(v).rolling(20).mean().shift(1).values
    v50 = pd.Series(v).rolling(50).mean().shift(1).values
    va20 = pd.Series(va).rolling(20).mean().shift(1).values
    ret1 = s.pct_change().values

    def run(i, entry, stop, ma=None, low=None, maxh=250, ma_ge=None):
        j = i + 1
        while j < n - 1 and j - (i + 1) < maxh:
            if c[j] <= stop:
                break
            if ma is not None and c[j] < ma[j]:
                break
            if low is not None and c[j] < low[j - 1]:
                break
            if ma_ge is not None and c[j] >= ma_ge[j]:
                break
            j += 1
        return j

    def rec(key, ent_i, entry, exit_i, exit_px):
        if not (entry > 0 and exit_px > 0):
            return
        out[key].append((dates[ent_i], dates[exit_i], code, float(exit_px / entry - 1),
                         int(exit_i - ent_i + 1), float(rs[ent_i - 1]) if ent_i > 0 else np.nan,
                         float(entry), float(exit_px)))

    def exit_open(j):
        # sell at the next open after the exit signal; if the data ends (delisting), last close
        return (j + 1, o[j + 1]) if j + 1 < n and o[j + 1] > 0 else (j, c[j])

    for i in range(max(start_i, WARMUP_BARS), n - 1):
        if not (va20[i] >= MIN_TURNOVER) or not (c[i] > 0):
            continue
        e = o[i + 1]
        if not (e > 0):
            continue
        # turtle55
        if h[i] > hi55[i] and h[i - 1] <= hi55[i - 1] and atr[i] > 0:
            j = run(i, e, e - 2 * atr[i], low=lo20)
            rec("turtle55", i + 1, e, *exit_open(j))
        # duke52
        if c[i] >= hi250[i] and c[i - 1] < hi250[i - 1]:
            j = run(i, e, e * 0.90, ma=ma25, maxh=120)
            rec("duke52", i + 1, e, *exit_open(j))
        # minervini
        tt = (c[i] > ma150[i] > ma200[i] and ma200[i] > ma200[i - 20] and ma50[i] > ma150[i]
              and c[i] > ma50[i] and c[i] >= 1.3 * lo250[i] and c[i] >= 0.75 * hi250[i] and rs[i] >= 0.7)
        if tt and c[i] >= hi50[i] and c[i - 1] < hi50[i - 1] and v[i] > 1.5 * v50[i]:
            j = run(i, e, e * 0.92, ma=ma50, maxh=60)
            rec("minervini", i + 1, e, *exit_open(j))
        # cis_vol
        if c[i] >= hi20[i] and c[i - 1] < hi20[i - 1] and v[i] > 2 * v20[i]:
            j = run(i, e, -np.inf, low=lo10, maxh=60)
            rec("cis_vol", i + 1, e, *exit_open(j))
        # bnf_dev
        dev, devp = c[i] / ma25[i] - 1, c[i - 1] / ma25[i - 1] - 1
        if dev <= -0.20 < devp:
            j = run(i, e, -np.inf, maxh=10, ma_ge=ma25)
            rec("bnf_dev", i + 1, e, j, c[j])
        # bnf_small
        if va[i] >= 3 * va20[i] and ret1[i] >= 0.05:
            j = run(i, e, e * 0.92, ma=ma10, maxh=10)
            rec("bnf_small", i + 1, e, *exit_open(j))
        # ep: gap on day i, buy that close
        if c[i - 1] > 0 and o[i] / c[i - 1] - 1 >= 0.10 and v[i] > 3 * v50[i] and c[i] > o[i]:
            j = run(i, c[i], l[i], ma=ma10, maxh=60)
            rec("ep", i, c[i], *exit_open(j))
    return out


def build_trades(panel: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    close = panel["close"]
    dates = close.index
    # RS rank: 12-month return percentile among names with enough history, per date
    r250 = close / close.shift(250) - 1
    rs = r250.rank(axis=1, pct=True)
    all_tr: dict[str, list] = {k: [] for k in STRATEGIES}
    for k, code in enumerate(close.columns):
        cl = close[code]
        valid = cl.notna().values
        if valid.sum() < WARMUP_BARS + 5:
            continue
        idx = np.where(valid)[0]
        sl = slice(idx[0], idx[-1] + 1)
        sub = {f: panel[f][code].iloc[sl].ffill() for f in ["open", "high", "low", "close"]}
        vol = panel["volume"][code].iloc[sl].fillna(0).values.astype("float64")
        va = panel["turnover"][code].iloc[sl].fillna(0).values.astype("float64")
        res = _trades_for_stock(
            code,
            sub["open"].values.astype("float64"), sub["high"].values.astype("float64"),
            sub["low"].values.astype("float64"), sub["close"].values.astype("float64"),
            vol, va, dates[sl], rs[code].iloc[sl].values, 0,
        )
        for key, lst in res.items():
            all_tr[key].extend(lst)
        if k % 500 == 0:
            logger.info("signals: %d / %d codes", k, close.shape[1])
    frames = {k: pd.DataFrame(v, columns=["entry", "exit", "code", "ret", "bars", "rs", "entry_px", "exit_px"])
              for k, v in all_tr.items()}
    return frames, rs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _trade_stats(t: pd.DataFrame, bench: pd.Series, split: pd.Timestamp) -> dict:
    if t.empty:
        return {"n": 0}
    t = t.copy()
    t["net"] = t["ret"] - COST
    b = bench
    # benchmark over the same window (close before entry -> close of exit day)
    bi = b.index
    pe = bi.searchsorted(t["entry"].values) - 1
    px = bi.searchsorted(t["exit"].values)
    pe = np.clip(pe, 0, len(bi) - 1)
    px = np.clip(px, 0, len(bi) - 1)
    t["mkt"] = b.values[px] / b.values[pe] - 1
    t["x"] = t["net"] - t["mkt"]
    byd = t.groupby("entry")["x"].mean()
    tstat = float(byd.mean() / byd.std() * np.sqrt(len(byd))) if len(byd) > 2 and byd.std() > 0 else None
    win = t.loc[t["net"] > 0, "net"].sum()
    loss = -t.loc[t["net"] < 0, "net"].sum()
    first, second = t[t["entry"] < split], t[t["entry"] >= split]
    return {
        "n": int(len(t)),
        "days": int(len(byd)),
        "win_rate": float((t["net"] > 0).mean()),
        "avg": float(t["net"].mean()),
        "pf": float(win / loss) if loss > 0 else None,
        "hold": float(t["bars"].mean()),
        "mkt": float(t["mkt"].mean()),
        "excess": float(t["x"].mean()),
        "t": tstat,
        "excess_first": float(first["x"].mean()) if len(first) else None,
        "excess_second": float(second["x"].mean()) if len(second) else None,
    }


def _portfolio(t: pd.DataFrame, panel: dict[str, pd.DataFrame], cal: pd.DatetimeIndex,
               priority: str = "rs", seed: int = 1) -> pd.Series:
    """Equal-slot portfolio that fills each trade at the trade's own entry/exit prices
    (so rules that enter at the close, or exit at a close, are not re-priced at the open
    — re-pricing an "enter at the close if close > open" rule at the open is look-ahead)."""
    close = panel["close"]
    if t.empty:
        return pd.Series(1.0, index=cal)
    t = t.copy()
    if priority == "random":
        t["rs"] = np.random.default_rng(seed).random(len(t))
    t = t.sort_values(["entry", "rs"], ascending=[True, False])
    groups = {d: g for d, g in t.groupby("entry")}
    cash, eq = 1.0, []
    pos: dict[str, tuple[float, pd.Timestamp, float]] = {}  # code -> (shares, exit date, exit price)
    last_px: dict[str, float] = {}
    for d in cal:
        for code in [c for c, (_, xd, _) in pos.items() if xd <= d]:
            sh, _, xp = pos.pop(code)
            cash += sh * xp * (1 - COST / 2)
        if d in groups:
            val = cash + sum(sh * last_px.get(c, 0.0) for c, (sh, _, _) in pos.items())
            for _, r in groups[d].iterrows():
                if len(pos) >= SLOTS or r.code in pos:
                    continue
                if not (r.entry_px > 0 and r.exit_px > 0):
                    continue
                amt = min(val / SLOTS, cash)
                if amt <= 0:
                    break
                pos[r.code] = (amt * (1 - COST / 2) / r.entry_px, r.exit, float(r.exit_px))
                last_px[r.code] = float(r.entry_px)
                cash -= amt
        for c in pos:
            p = close.at[d, c]
            if p > 0:
                last_px[c] = float(p)
        eq.append(cash + sum(sh * last_px.get(c, 0.0) for c, (sh, _, _) in pos.items()))
    return pd.Series(eq, index=cal)


def _curve_stats(e: pd.Series) -> dict:
    e = e[e > 0]
    yrs = max((e.index[-1] - e.index[0]).days / 365.25, 1e-9)
    r = e.pct_change().dropna()
    yearly = e.resample("YE").last()
    yearly_ret = yearly.pct_change()
    yearly_ret.iloc[0] = yearly.iloc[0] / e.iloc[0] - 1
    return {
        "cagr": float((e.iloc[-1] / e.iloc[0]) ** (1 / yrs) - 1),
        "max_dd": float((e / e.cummax() - 1).min()),
        "sharpe": float(r.mean() / r.std() * np.sqrt(245)) if r.std() > 0 else None,
        "final": float(e.iloc[-1] / e.iloc[0]),
        "yearly": {str(d.year): float(v) for d, v in yearly_ret.items()},
    }


def run_on_panel(panel: dict[str, pd.DataFrame], bench: pd.Series) -> dict:
    close = panel["close"]
    trades, rs = build_trades(panel)
    cal = close.index[WARMUP_BARS:]
    split = cal[len(cal) // 2]

    # Survivorship-free equal-weight baseline: every liquid stock that traded that day.
    va20 = panel["turnover"].rolling(20, min_periods=10).mean().shift(1)
    rets = close.pct_change(fill_method=None)
    elig = (va20 >= MIN_TURNOVER) & rets.notna()
    ew = (rets.where(elig).mean(axis=1)).reindex(cal).fillna(0)
    ew_curve = (1 + ew).cumprod()
    b = bench.reindex(cal).ffill()
    bench_curve = b / b.iloc[0]

    result = {
        "period": {"start": str(cal[0].date()), "end": str(cal[-1].date()), "split": str(split.date())},
        "universe": {"codes": int(close.shape[1]), "avg_eligible": float(elig.reindex(cal).sum(axis=1).mean())},
        "params": {"min_turnover": MIN_TURNOVER, "cost": COST, "slots": SLOTS, "warmup_bars": WARMUP_BARS},
        "baselines": {"topix": _curve_stats(bench_curve), "equal_weight": _curve_stats(ew_curve)},
        "strategies": {},
    }
    for key in STRATEGIES:
        t = trades[key]
        t = t[t["entry"] >= cal[0]]
        logger.info("%s: %d trades", key, len(t))
        st = {"label": LABELS[key], "trades": _trade_stats(t, bench, split)}
        if len(t):
            st["portfolio_rs"] = _curve_stats(_portfolio(t, panel, cal, "rs"))
            st["portfolio_random"] = _curve_stats(_portfolio(t, panel, cal, "random"))
        result["strategies"][key] = st
    return result


def run_legend_backtest(lookback_days: int = LOOKBACK_DAYS) -> dict:
    panel = fetch_panel(lookback_days)
    bench = fetch_benchmark(panel)
    return run_on_panel(panel, bench)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pct(x, d=2, sign=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"


def render_markdown(res: dict, generated_at: str) -> str:
    p, u, bl = res["period"], res["universe"], res["baselines"]
    lines = [
        f"# 伝説的トレーダーの手法:生存者バイアスなしの検証({generated_at[:10]})",
        "",
        f"生成:{generated_at} / 計算:`python backend/scripts/run_legend_backtest.py`(J-Quants、GitHub Actions「Legend Backtest」)",
        "",
        "- **銘柄**:日付指定の全銘柄日足から作った「その日に実際に上場していた銘柄」(後に上場廃止になった銘柄も含む)。ETF・REIT等は除外。",
        f"- **期間**:{p['start']}〜{p['end']}(前半/後半の境目 {p['split']})。延べ{u['codes']:,}銘柄、売買代金の条件を満たす銘柄は1日平均{u['avg_eligible']:.0f}社。",
        f"- **条件**:シグナル日までの20日平均売買代金が{res['params']['min_turnover'] / 1e8:.0f}億円以上。往復コスト{res['params']['cost'] * 100:.1f}%。超過=同期間のTOPIXとの差。",
        f"- **口座の試算**:同時{res['params']['slots']}銘柄・均等。シグナルが多い日は相対強度の高い順(RS順)と、ランダムな順の2通り。",
        "",
        "## 1. 口座全体の成績",
        "",
        "| | 年率 | 最大下落 | シャープ | 期間で |",
        "|---|---|---|---|---|",
        f"| TOPIX | {_pct(bl['topix']['cagr'], 1)} | {_pct(bl['topix']['max_dd'], 1, False)} | {bl['topix']['sharpe']:.2f} | {bl['topix']['final']:.2f}倍 |",
        f"| **売買代金1億円以上の全銘柄を均等(バイアスなしの基準)** | **{_pct(bl['equal_weight']['cagr'], 1)}** | {_pct(bl['equal_weight']['max_dd'], 1, False)} | {bl['equal_weight']['sharpe']:.2f} | {bl['equal_weight']['final']:.2f}倍 |",
    ]
    for key, st in res["strategies"].items():
        for kind, nm in (("portfolio_rs", "RS順"), ("portfolio_random", "ランダム順")):
            c = st.get(kind)
            if c:
                sh = f"{c['sharpe']:.2f}" if c["sharpe"] is not None else "—"
                lines.append(f"| {st['label']}({nm}) | {_pct(c['cagr'], 1)} | {_pct(c['max_dd'], 1, False)} | {sh} | {c['final']:.2f}倍 |")
    lines += [
        "",
        "## 2. トレード単位の成績",
        "",
        "| 手法 | 件数 | 勝率 | 平均 | PF | 平均保有 | 同期間TOPIX | **超過** | t値 | 前半/後半の超過 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, st in res["strategies"].items():
        t = st["trades"]
        if not t.get("n"):
            lines.append(f"| {st['label']} | 0 | — | — | — | — | — | — | — | — |")
            continue
        pf = f"{t['pf']:.2f}" if t["pf"] is not None else "—"
        ts = f"{t['t']:+.1f}" if t["t"] is not None else "—"
        lines.append(
            f"| {st['label']} | {t['n']:,} | {t['win_rate'] * 100:.0f}% | {_pct(t['avg'])} | {pf} | {t['hold']:.1f}日 | "
            f"{_pct(t['mkt'])} | **{_pct(t['excess'])}** | {ts} | {_pct(t['excess_first'])} / {_pct(t['excess_second'])} |"
        )
    lines += [
        "",
        "## 3. 読み方",
        "",
        "- 判断の基準は「売買代金1億円以上の全銘柄を均等に持つ」場合。これを年率で上回れなければ、銘柄選びの優位性はない。",
        "- RS順とランダム順の差は「強い銘柄から選ぶ」効果。手法そのものの効果はランダム順で見る。",
        "- 超過のt値が2未満、または前半と後半で符号が違う手法は、偶然の可能性が高い。",
        "- 期間は契約の取得範囲(約3.8年)から、52週の計算に使う最初の約1年を除いた分。1つの相場局面に近いので、結果は目安として扱う。",
        "",
        "一般的な手法の検証であり、個別の投資助言ではない。",
    ]
    return "\n".join(lines) + "\n"
