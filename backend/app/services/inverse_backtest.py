"""Survivorship-free test of the *inverse* of losing trading methods (J-Quants V2).

reports/losing_rules_2026-11-12.md lists methods that lose money (chasing attention and
lottery-like stocks, picking the strongest names, frequent technical signals). This
module tests the opposite of each on the same point-in-time universe as
legend_backtest (every stock that traded that day, delisted names included):

  A. Monthly quintile portfolios (rebalanced at each month-end close, held one month):
     low vs high 21-day MAX return (lottery), 20/120-day volume ratio (attention),
     1-month return (short-term reversal vs chasing), 12-1 month momentum (hold
     winners / cut losers = the inverse of the disposition effect), 60-day volatility,
     distance from the 52-week high. Plus "avoid attention" (below-median MAX and
     volume ratio) against "lottery & attention" (top quintile of both).
  B. Fading surges: short the trades of the losing chase rules (bnf_small, ep) at the
     same prices, with a borrow fee.
  C. Inverse technical signals: hold through MACD / 5-25 MA dead crosses instead of
     golden crosses.

Not investment advice; flat cost assumptions, no slippage model, no taxes.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from app.services import legend_backtest as lb

logger = logging.getLogger(__name__)

COST = lb.COST                 # round trip
BORROW = 0.015                 # yearly borrow fee for shorts (信用売りの貸株料の目安)
QUINTILE = 0.2

FEATURES = {
    "max21": "過去21日の最大上昇率(宝くじ度)",
    "abvol": "出来高の急増(20日平均÷120日平均)",
    "r21": "直近1か月のリターン",
    "mom": "12-1か月のリターン(モメンタム)",
    "vol60": "60日の値動きの大きさ(ボラティリティ)",
    "dist52": "52週高値からの距離",
}
# Which side is the "inverse of a losing method", for the report.
INVERSE_SIDE = {"max21": "low", "abvol": "low", "r21": "low", "mom": "high",
                "vol60": "low", "dist52": "low"}
BAD_SIDE_NOTE = {
    "max21": "急騰した株(宝くじ株)を買う",
    "abvol": "出来高が急増した注目株を買う",
    "r21": "直近で一番上がった株を買う(飛び乗り)",
    "mom": "値上がり株を売り、値下がり株を持ち続ける(処分効果)",
    "vol60": "値動きの大きい株を買う",
    "dist52": "高値を更新している一番強い株を買う",
}


# ---------------------------------------------------------------------------
# A. Monthly cross-sectional portfolios
# ---------------------------------------------------------------------------
def _features(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    close, vol = panel["close"].astype("float64"), panel["volume"].astype("float64")
    rets = close.pct_change(fill_method=None)
    return {
        "max21": rets.rolling(21, min_periods=15).max(),
        "abvol": vol.rolling(20, min_periods=15).mean() / vol.rolling(120, min_periods=80).mean(),
        "r21": close / close.shift(21) - 1,
        "mom": close.shift(21) / close.shift(252) - 1,
        "vol60": rets.rolling(60, min_periods=40).std(),
        "dist52": close / close.rolling(250, min_periods=200).max() - 1,
    }


def _month_ends(idx: pd.DatetimeIndex, start: pd.Timestamp) -> list[pd.Timestamp]:
    s = pd.Series(idx, index=idx)
    me = s.groupby(idx.to_period("M")).last().tolist()
    return [d for d in me if d >= start]


def _hold(picks_by_month: list[tuple[pd.Timestamp, pd.Timestamp, list[str]]],
          rets: pd.DataFrame) -> tuple[pd.Series, float]:
    """Equal weight within each month; turnover cost charged on the first day."""
    out, prev, turns = [], set(), []
    for a, b, picks in picks_by_month:
        pk = set(picks)
        if not pk:
            continue
        t = len(pk.symmetric_difference(prev)) / (2 * len(pk)) if prev else 1.0
        prev = pk
        r = rets.loc[(rets.index > a) & (rets.index <= b), list(pk)].mean(axis=1).fillna(0)
        if len(r):
            r.iloc[0] -= t * COST
        out.append(r)
        turns.append(t)
    return (pd.concat(out) if out else pd.Series(dtype=float)), float(np.mean(turns) if turns else 0)


def _excess_stats(r: pd.Series, base: pd.Series, split: pd.Timestamp) -> dict:
    x = (r - base.reindex(r.index).fillna(0)).dropna()
    return {
        "excess_yr": float(x.mean() * 245),
        "t": float(x.mean() / x.std() * np.sqrt(len(x))) if x.std() > 0 else None,
        "excess_first": float(x[x.index < split].mean() * 245),
        "excess_second": float(x[x.index >= split].mean() * 245),
    }


def cross_section(panel: dict[str, pd.DataFrame], cal: pd.DatetimeIndex,
                  split: pd.Timestamp) -> dict:
    close = panel["close"]
    rets = close.astype("float64").pct_change(fill_method=None)
    va20 = panel["turnover"].rolling(20, min_periods=10).mean()
    feats = _features(panel)
    me = _month_ends(close.index, cal[0])
    months = list(zip(me[:-1], me[1:]))

    def eligible(a):
        ok = (va20.loc[a] >= lb.MIN_TURNOVER) & close.loc[a].notna()
        return ok[ok].index

    base_picks = [(a, b, list(eligible(a))) for a, b in months]
    base_raw = pd.concat([rets.loc[(rets.index > a) & (rets.index <= b), p].mean(axis=1).fillna(0)
                          for a, b, p in base_picks])  # no turnover cost for the baseline
    out = {"baseline": {"label": "売買代金1億円以上の全銘柄を均等(月末に入れ替え)",
                        **lb._curve_stats((1 + base_raw).cumprod())}, "quintiles": {}, "combos": {}}

    for key, f in feats.items():
        res = {}
        for side in ("low", "high"):
            picks = []
            for a, b in months:
                s = f.loc[a].reindex(eligible(a)).dropna()
                if len(s) < 50:
                    picks.append((a, b, []))
                    continue
                k = max(int(len(s) * QUINTILE), 1)
                picks.append((a, b, list((s.nsmallest(k) if side == "low" else s.nlargest(k)).index)))
            r, turn = _hold(picks, rets)
            if r.empty:
                continue
            res[side] = {**lb._curve_stats((1 + r).cumprod()), **_excess_stats(r, base_raw, split),
                         "turnover": turn, "n_avg": float(np.mean([len(p) for _, _, p in picks]))}
        out["quintiles"][key] = {"label": FEATURES[key], "inverse_side": INVERSE_SIDE[key],
                                 "bad_note": BAD_SIDE_NOTE[key], **res}
        logger.info("quintile %s done", key)

    for name, label, rule in (
        ("avoid", "注目を避ける(宝くじ度と出来高急増が、どちらも中央値以下)",
         lambda m, v: (m <= m.median()) & (v <= v.median())),
        ("lottery", "宝くじ株かつ注目株(どちらも上位20%)",
         lambda m, v: (m >= m.quantile(0.8)) & (v >= v.quantile(0.8))),
    ):
        picks = []
        for a, b in months:
            el = eligible(a)
            m, v = feats["max21"].loc[a].reindex(el), feats["abvol"].loc[a].reindex(el)
            ok = m.notna() & v.notna()
            m, v = m[ok], v[ok]
            picks.append((a, b, list(m[rule(m, v)].index) if len(m) >= 50 else []))
        r, turn = _hold(picks, rets)
        if not r.empty:
            out["combos"][name] = {"label": label, **lb._curve_stats((1 + r).cumprod()),
                                   **_excess_stats(r, base_raw, split), "turnover": turn,
                                   "n_avg": float(np.mean([len(p) for _, _, p in picks]))}
    return out


# ---------------------------------------------------------------------------
# B. Fading the losing chase rules (short at the same prices)
# ---------------------------------------------------------------------------
def fade(trades: dict[str, pd.DataFrame], bench: pd.Series, cal: pd.DatetimeIndex,
         split: pd.Timestamp) -> dict:
    out = {}
    bi = bench.index
    for key in ("bnf_small", "ep"):
        t = trades[key]
        t = t[(t["entry"] >= cal[0]) & (t["entry_px"] > 0) & (t["exit_px"] > 0)].copy()
        if t.empty:
            continue
        t["short"] = 1 - t["exit_px"] / t["entry_px"] - COST - BORROW * t["bars"] / 245
        pe = np.clip(bi.searchsorted(t["entry"].values) - 1, 0, len(bi) - 1)
        px = np.clip(bi.searchsorted(t["exit"].values), 0, len(bi) - 1)
        t["mkt"] = bench.values[px] / bench.values[pe] - 1
        t["hedged"] = t["short"] + t["mkt"]          # short the stock, hold TOPIX
        byd = t.groupby("entry")["short"].mean()
        first, second = t[t["entry"] < split], t[t["entry"] >= split]
        out[key] = {
            "label": "逆張りで空売り:" + lb.LABELS[key],
            "n": int(len(t)), "win_rate": float((t["short"] > 0).mean()),
            "avg_short": float(t["short"].mean()), "avg_hedged": float(t["hedged"].mean()),
            "hold": float(t["bars"].mean()),
            "worst": float(t["short"].min()),
            "t": float(byd.mean() / byd.std() * np.sqrt(len(byd))) if len(byd) > 2 and byd.std() > 0 else None,
            "short_first": float(first["short"].mean()) if len(first) else None,
            "short_second": float(second["short"].mean()) if len(second) else None,
        }
    return out


# ---------------------------------------------------------------------------
# C. Inverse technical signals (long only)
# ---------------------------------------------------------------------------
TECH = {
    "macd_gc": "MACDのゴールデンクロスで買い→デッドクロスで売り(悪い手法)",
    "macd_dc": "MACDのデッドクロスで買い→ゴールデンクロスで売り(逆)",
    "ma_gc": "5日線と25日線のゴールデンクロスで買い→デッドクロスで売り(悪い手法)",
    "ma_dc": "5日線と25日線のデッドクロスで買い→ゴールデンクロスで売り(逆)",
}


def _tech_trades(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    close, opn, to = panel["close"], panel["open"], panel["turnover"]
    dates = close.index
    rows: dict[str, list] = {k: [] for k in TECH}
    for code in close.columns:
        cl = close[code]
        valid = cl.notna().values
        if valid.sum() < lb.WARMUP_BARS + 5:
            continue
        idx = np.where(valid)[0]
        sl = slice(idx[0], idx[-1] + 1)
        c = cl.iloc[sl].ffill().values.astype("float64")
        o = opn[code].iloc[sl].ffill().values.astype("float64")
        va20 = to[code].iloc[sl].fillna(0).rolling(20).mean().values
        d = dates[sl]
        s = pd.Series(c)
        macd = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
        above = {"macd": (macd > macd.ewm(span=9, adjust=False).mean()).values,
                 "ma": (s.rolling(5).mean() > s.rolling(25).mean()).values}
        n = len(c)
        for fam, ab in above.items():
            for kind, want in (("gc", True), ("dc", False)):
                key = f"{fam}_{kind}"
                i = max(lb.WARMUP_BARS, 1)
                while i < n - 2:
                    if ab[i] == want and ab[i - 1] != want and va20[i] >= lb.MIN_TURNOVER:
                        e = i + 1
                        j = e
                        while j < n - 1 and ab[j] == want:
                            j += 1
                        x = min(j + 1, n - 1)  # cross back seen at close j -> sell next open
                        if o[e] > 0 and o[x] > 0:
                            rows[key].append((d[e], d[x], code, o[x] / o[e] - 1, x - e, 0.5,
                                              o[e], o[x]))
                        i = x
                        continue
                    i += 1
    return {k: pd.DataFrame(v, columns=["entry", "exit", "code", "ret", "bars", "rs", "entry_px", "exit_px"])
            for k, v in rows.items()}


def technical(panel, bench, cal, split) -> dict:
    out = {}
    for key, t in _tech_trades(panel).items():
        t = t[t["entry"] >= cal[0]]
        st = {"label": TECH[key], "trades": lb._trade_stats(t, bench, split)}
        if len(t):
            st["portfolio_random"] = lb._curve_stats(lb._portfolio(t, panel, cal, "random"))
        out[key] = st
        logger.info("technical %s: %d trades", key, len(t))
    return out


# ---------------------------------------------------------------------------
def run_on_panel(panel: dict[str, pd.DataFrame], bench: pd.Series) -> dict:
    close = panel["close"]
    cal = close.index[lb.WARMUP_BARS:]
    split = cal[len(cal) // 2]
    trades, _ = lb.build_trades(panel)
    b = bench.reindex(cal).ffill()
    return {
        "period": {"start": str(cal[0].date()), "end": str(cal[-1].date()), "split": str(split.date())},
        "universe": {"codes": int(close.shape[1])},
        "params": {"min_turnover": lb.MIN_TURNOVER, "cost": COST, "borrow": BORROW, "quintile": QUINTILE,
                   "slots": lb.SLOTS},
        "topix": lb._curve_stats(b / b.iloc[0]),
        "cross_section": cross_section(panel, cal, split),
        "fade": fade(trades, bench, cal, split),
        "technical": technical(panel, bench, cal, split),
    }


def run_inverse_backtest() -> dict:
    panel = lb.fetch_panel()
    bench = lb.fetch_benchmark(panel)
    return run_on_panel(panel, bench)


# ---------------------------------------------------------------------------
def _p(x, d=1, sign=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x * 100:{'+' if sign else ''}.{d}f}%"


def _t(x):
    return "—" if x is None else f"{x:+.1f}"


def render_markdown(res: dict, generated_at: str) -> str:
    p, xs = res["period"], res["cross_section"]
    bl = xs["baseline"]
    L = [
        f"# 悪いトレード手法の「逆」:生存者バイアスなしの検証({generated_at[:10]})",
        "",
        f"生成:{generated_at} / 計算:`python backend/scripts/run_inverse_backtest.py`(J-Quants、GitHub Actions「Inverse Backtest」)",
        "",
        "- **銘柄**:日付指定の全銘柄日足から作った、その日に実際に上場していた銘柄(上場廃止を含む)。ETF・REIT等は除外。"
        f"月末時点の20日平均売買代金が{res['params']['min_turnover'] / 1e8:.0f}億円以上。",
        f"- **期間**:{p['start']}〜{p['end']}(前半/後半の境目 {p['split']})。",
        f"- **コスト**:往復{res['params']['cost'] * 100:.1f}%。空売りは貸株料 年{res['params']['borrow'] * 100:.1f}%も差し引く。",
        "",
        "## A. 銘柄の選び方:月末に上位・下位20%を選んで1か月持つ",
        "",
        "| | 年率 | 最大下落 | シャープ |",
        "|---|---|---|---|",
        f"| TOPIX | {_p(res['topix']['cagr'])} | {_p(res['topix']['max_dd'], 1, False)} | {res['topix']['sharpe']:.2f} |",
        f"| **{bl['label']}(基準)** | **{_p(bl['cagr'])}** | {_p(bl['max_dd'], 1, False)} | {bl['sharpe']:.2f} |",
        "",
        "| 指標 | 悪い手法 | 選んだ側 | 年率 | 最大下落 | 基準との差(年) | t値 | 前半/後半 | 月の入れ替え |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key, q in xs["quintiles"].items():
        for side in ("low", "high"):
            r = q.get(side)
            if not r:
                continue
            tag = "**逆**" if side == q["inverse_side"] else "悪い側"
            nm = ("低い20%" if side == "low" else "高い20%") + f"({tag})"
            L.append(f"| {q['label']} | {q['bad_note']} | {nm} | {_p(r['cagr'])} | {_p(r['max_dd'], 1, False)} | "
                     f"**{_p(r['excess_yr'])}** | {_t(r['t'])} | {_p(r['excess_first'])} / {_p(r['excess_second'])} | "
                     f"{r['turnover'] * 100:.0f}% |")
    for key, r in xs["combos"].items():
        L.append(f"| {r['label']} | — | 平均{r['n_avg']:.0f}社 | {_p(r['cagr'])} | {_p(r['max_dd'], 1, False)} | "
                 f"**{_p(r['excess_yr'])}** | {_t(r['t'])} | {_p(r['excess_first'])} / {_p(r['excess_second'])} | "
                 f"{r['turnover'] * 100:.0f}% |")
    L += ["", "## B. 急騰への飛び乗りの逆:同じ場面で空売りする", "",
          "| 手法 | 件数 | 勝率 | 1回の平均(空売り) | 空売り+TOPIX買い | 平均保有 | 最悪の1回 | t値 | 前半/後半 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for key, r in res["fade"].items():
        L.append(f"| {r['label']} | {r['n']:,} | {r['win_rate'] * 100:.0f}% | **{_p(r['avg_short'], 2)}** | "
                 f"{_p(r['avg_hedged'], 2)} | {r['hold']:.1f}日 | {_p(r['worst'], 0)} | {_t(r['t'])} | "
                 f"{_p(r['short_first'], 2)} / {_p(r['short_second'], 2)} |")
    L += ["", "## C. テクニカルの逆:デッドクロスで買う", "",
          "| 手法 | 件数 | 勝率 | 1回の平均 | TOPIXとの超過 | t値 | 前半/後半の超過 | 口座の年率(同時10銘柄・ランダム順) |",
          "|---|---|---|---|---|---|---|---|"]
    for key, r in res["technical"].items():
        t = r["trades"]
        if not t.get("n"):
            continue
        pr = r.get("portfolio_random", {})
        L.append(f"| {r['label']} | {t['n']:,} | {t['win_rate'] * 100:.0f}% | {_p(t['avg'], 2)} | "
                 f"**{_p(t['excess'], 2)}** | {_t(t['t'])} | {_p(t['excess_first'], 2)} / {_p(t['excess_second'], 2)} | "
                 f"{_p(pr.get('cagr'))} |")
    L += ["", "## 読み方", "",
          "- 基準は「売買代金1億円以上の全銘柄を均等に持つ」場合。逆の手法がこれを上回り、t値が2以上で、前半と後半の符号がそろっていれば、優位性がある可能性がある。",
          "- 期間は約2.8年で、1つの相場局面に近い。結果は目安として扱う。",
          "- 上場廃止になった月は、最後に取引された日までのリターンで計算している(整理売買の最後の下落は含まれない場合がある)。",
          "- 空売りは、銘柄によっては借りられない(制度信用の対象外・在庫切れ)。逆日歩も含めていない。",
          "", "一般的な手法の検証であり、個別の投資助言ではない。"]
    return "\n".join(L) + "\n"
