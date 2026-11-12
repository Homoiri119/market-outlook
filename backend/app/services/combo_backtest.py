"""Combinations of popular-but-losing entries with the filters that the inverse study
suggests (J-Quants V2, survivorship-free universe from legend_backtest).

Golden crosses, MACD crosses and volume surges are widely taught as profitable, but on
their own they lost to TOPIX. The idea tested here: keep the popular trigger, and only
take it in the context where it should work.

Event rules (signal at close i, buy at open i+1):
  gc / macd           the plain trigger (5/25-day MA golden cross; MACD 12-26-9 cross),
                      sold at the next dead cross
  *_trend             + stock above its 200-day MA and TOPIX above its 200-day MA
  *_quiet             + not a lottery stock (largest daily gain over the past 21 days <= 6%)
  *_strong            + 12-1 month momentum in the top 30% and stock above its 200-day MA
                      (a pullback in a leader that turns up again)
  *_best              trend + quiet + strong
  vol                 volume >= 3x the 20-day average on an up day (>= +2%), held 20 days
  vol_quiet_trend     + modest move (+2% to +6%), first surge (no prior 21-day gain > 6%),
                      stock and TOPIX above their 200-day MAs
  vol_lottery         + big move (>= +8%): the attention/lottery case
  vol_flush_trend     volume >= 3x on a down day (<= -2%) in a stock above its 200-day MA
Monthly portfolios (month-end, held one month):
  mom_pullback        momentum top 30% and 1-month return in the bottom half
  mom_quiet           momentum top 30% and 21-day MAX in the bottom half
  mom_quiet_filter    mom_quiet, but in cash while TOPIX is below its 200-day MA
Not investment advice; flat costs, no slippage, no taxes.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from app.services import inverse_backtest as ib
from app.services import legend_backtest as lb

logger = logging.getLogger(__name__)

QUIET_MAX = 0.06
MOM_TOP = 0.70
VOL_MULT = 3.0
VOL_HOLD = 20
MAX_HOLD = 120

EVENTS = {
    "gc": "ゴールデンクロス(5日・25日線)で買い→デッドクロスで売り",
    "gc_trend": "ゴールデンクロス+上昇トレンド(株価もTOPIXも200日線の上)",
    "gc_quiet": "ゴールデンクロス+宝くじ株でない(過去21日の最大上昇6%以下)",
    "gc_strong": "ゴールデンクロス+強い株の押し目明け(12か月モメンタム上位30%・200日線の上)",
    "gc_best": "ゴールデンクロス+上の3条件すべて",
    "macd": "MACDのゴールデンクロスで買い→デッドクロスで売り",
    "macd_trend": "MACD+上昇トレンド(株価もTOPIXも200日線の上)",
    "macd_quiet": "MACD+宝くじ株でない",
    "macd_strong": "MACD+強い株の押し目明け",
    "macd_best": "MACD+上の3条件すべて",
    "vol": "出来高急増(3倍)+上昇(+2%以上)→20日持つ",
    "vol_quiet_trend": "出来高急増+小幅上昇(+2〜6%)+初めての急増+上昇トレンド",
    "vol_lottery": "出来高急増+大幅上昇(+8%以上)=宝くじ化した注目株",
    "vol_flush_trend": "出来高急増+下落(−2%以下)+200日線の上=上昇トレンド中の投げ売りを拾う",
}
PORTFOLIOS = {
    "mom_pullback": "強い株の押し目(12-1か月モメンタム上位30%で、直近1か月は下位半分)",
    "mom_quiet": "静かに強い株(モメンタム上位30%で、宝くじ度は下位半分)",
    "mom_quiet_filter": "静かに強い株+相場フィルター(TOPIXが200日線の下なら現金)",
}


def _trades(panel: dict[str, pd.DataFrame], bench: pd.Series) -> dict[str, pd.DataFrame]:
    close, opn, vol, to = panel["close"], panel["open"], panel["volume"], panel["turnover"]
    dates = close.index
    c_all = close.astype("float64")
    mom_rank = (c_all.shift(21) / c_all.shift(252) - 1).rank(axis=1, pct=True)
    b = bench.reindex(dates).ffill()
    mkt_up = (b > b.rolling(200, min_periods=150).mean()).to_numpy()
    rows: dict[str, list] = {k: [] for k in EVENTS}
    for code in close.columns:
        cl = close[code]
        valid = cl.notna().values
        if valid.sum() < lb.WARMUP_BARS + 5:
            continue
        idx = np.where(valid)[0]
        sl = slice(idx[0], idx[-1] + 1)
        c = cl.iloc[sl].ffill().to_numpy("float64")
        o = opn[code].iloc[sl].ffill().to_numpy("float64")
        v = vol[code].iloc[sl].fillna(0).to_numpy("float64")
        va20 = to[code].iloc[sl].fillna(0).rolling(20).mean().shift(1).to_numpy()
        d = dates[sl]
        mk = mkt_up[sl]
        mr = mom_rank[code].iloc[sl].to_numpy()
        s = pd.Series(c)
        r1 = s.pct_change().to_numpy()
        max21_prev = s.pct_change().rolling(21).max().shift(1).to_numpy()
        ma200 = s.rolling(200).mean().to_numpy()
        v20 = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
        macd = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
        above = {"macd": (macd > macd.ewm(span=9, adjust=False).mean()).to_numpy(),
                 "gc": (s.rolling(5).mean() > s.rolling(25).mean()).to_numpy()}
        n = len(c)
        with np.errstate(invalid="ignore"):
            trend = (c > ma200) & mk
            quiet = max21_prev <= QUIET_MAX
            strong = (mr >= MOM_TOP) & (c > ma200)
            liquid = va20 >= lb.MIN_TURNOVER
            surge = v >= VOL_MULT * v20
        filters = {"": None, "_trend": trend, "_quiet": quiet, "_strong": strong,
                   "_best": trend & quiet & strong}

        def add(key, e, x):
            if e < n and x < n and x > e and o[e] > 0 and o[x] > 0:
                rows[key].append((d[e], d[x], code, o[x] / o[e] - 1, x - e, 0.5, o[e], o[x]))

        for fam, ab in above.items():
            cross = np.zeros(n, dtype=bool)
            cross[1:] = ab[1:] & ~ab[:-1]
            for suf, flt in filters.items():
                key = fam + suf
                i = lb.WARMUP_BARS
                while i < n - 2:
                    if cross[i] and liquid[i] and (flt is None or flt[i]):
                        e, j = i + 1, i + 1
                        while j < n - 1 and ab[j] and j - e < MAX_HOLD:
                            j += 1
                        x = min(j + 1, n - 1)
                        add(key, e, x)
                        i = x
                        continue
                    i += 1

        conds = {
            "vol": surge & (r1 >= 0.02),
            "vol_quiet_trend": surge & (r1 >= 0.02) & (r1 <= 0.06) & quiet & trend,
            "vol_lottery": surge & (r1 >= 0.08),
            "vol_flush_trend": surge & (r1 <= -0.02) & (c > ma200),
        }
        for key, cond in conds.items():
            i = lb.WARMUP_BARS
            while i < n - 2:
                if cond[i] and liquid[i]:
                    e = i + 1
                    x = min(e + VOL_HOLD, n - 1)
                    add(key, e, x)
                    i = x
                    continue
                i += 1
    return {k: pd.DataFrame(v, columns=["entry", "exit", "code", "ret", "bars", "rs", "entry_px", "exit_px"])
            for k, v in rows.items()}


def _portfolios(panel, bench, cal, split) -> dict:
    close = panel["close"].astype("float64")
    rets = close.pct_change(fill_method=None)
    va20 = panel["turnover"].rolling(20, min_periods=10).mean()
    feats = ib._features(panel)
    b = bench.reindex(close.index).ffill()
    mkt_up = b > b.rolling(200, min_periods=150).mean()
    me = ib._month_ends(close.index, cal[0])
    months = list(zip(me[:-1], me[1:]))
    picks = {k: [] for k in PORTFOLIOS}
    base = []
    for a, bb in months:
        el = close.columns[((va20.loc[a] >= lb.MIN_TURNOVER) & close.loc[a].notna()).values]
        base.append((a, bb, list(el)))
        f = pd.DataFrame({k: feats[k].loc[a].reindex(el) for k in ("mom", "r21", "max21")}).dropna()
        if len(f) < 100:
            for k in picks:
                picks[k].append((a, bb, []))
            continue
        strong = f["mom"] >= f["mom"].quantile(MOM_TOP)
        pull = strong & (f["r21"] <= f["r21"].median())
        quiet = strong & (f["max21"] <= f["max21"].median())
        picks["mom_pullback"].append((a, bb, list(f.index[pull])))
        picks["mom_quiet"].append((a, bb, list(f.index[quiet])))
        picks["mom_quiet_filter"].append((a, bb, list(f.index[quiet]) if bool(mkt_up.loc[a]) else []))
    base_raw = pd.concat([rets.loc[(rets.index > a) & (rets.index <= bb), p].mean(axis=1).fillna(0)
                          for a, bb, p in base])
    out = {}
    for k, pk in picks.items():
        # Months in cash contribute zero return (not dropped) so the curve is comparable.
        parts, prev, turns = [], set(), []
        for a, bb, p in pk:
            days = rets.index[(rets.index > a) & (rets.index <= bb)]
            if not p:
                cash = pd.Series(0.0, index=days)
                if prev and len(cash):
                    cash.iloc[0] -= lb.COST / 2  # sell everything into cash
                parts.append(cash)
                prev = set()
                continue
            pkset = set(p)
            t = len(pkset.symmetric_difference(prev)) / (2 * len(pkset)) if prev else 1.0
            r = rets.loc[days, p].mean(axis=1).fillna(0)
            if len(r):
                r.iloc[0] -= t * lb.COST
            parts.append(r)
            turns.append(t)
            prev = pkset
        r = pd.concat(parts)
        out[k] = {"label": PORTFOLIOS[k], **lb._curve_stats((1 + r).cumprod()),
                  **ib._excess_stats(r, base_raw, split),
                  "invested": float(np.mean([bool(p) for _, _, p in pk])),
                  "n_avg": float(np.mean([len(p) for _, _, p in pk if p] or [0]))}
    return {"baseline": lb._curve_stats((1 + base_raw).cumprod()), "portfolios": out}



def run_on_panel(panel: dict[str, pd.DataFrame], bench: pd.Series) -> dict:
    close = panel["close"]
    cal = close.index[lb.WARMUP_BARS:]
    split = cal[len(cal) // 2]
    ev = {}
    for key, t in _trades(panel, bench).items():
        t = t[t["entry"] >= cal[0]]
        st = {"label": EVENTS[key], "trades": lb._trade_stats(t, bench, split)}
        if len(t):
            st["portfolio_random"] = lb._curve_stats(lb._portfolio(t, panel, cal, "random"))
        ev[key] = st
        logger.info("combo %s: %d trades", key, len(t))
    b = bench.reindex(cal).ffill()
    return {
        "period": {"start": str(cal[0].date()), "end": str(cal[-1].date()), "split": str(split.date())},
        "params": {"min_turnover": lb.MIN_TURNOVER, "cost": lb.COST, "slots": lb.SLOTS,
                   "quiet_max": QUIET_MAX, "mom_top": MOM_TOP, "vol_mult": VOL_MULT, "vol_hold": VOL_HOLD},
        "topix": lb._curve_stats(b / b.iloc[0]),
        "events": ev,
        "monthly": _portfolios(panel, bench, cal, split),
    }


def render_markdown(res: dict, generated_at: str) -> str:
    p, P = res["period"], ib._p
    mo = res["monthly"]
    L = [
        f"# 人気の手法(ゴールデンクロス・MACD・出来高急増)に条件を足す:生存者バイアスなしの検証({generated_at[:10]})",
        "",
        f"生成:{generated_at} / 計算:`python backend/scripts/run_inverse_backtest.py`(J-Quants、GitHub Actions「Inverse Backtest」)",
        "",
        f"- **銘柄**:その日に実際に上場していた全銘柄(上場廃止を含む)。20日平均売買代金{res['params']['min_turnover'] / 1e8:.0f}億円以上。",
        f"- **期間**:{p['start']}〜{p['end']}(前半/後半の境目 {p['split']})。往復コスト{res['params']['cost'] * 100:.1f}%。シグナルの翌日の寄り付きで買う。",
        f"- **基準**:TOPIX 年率{P(res['topix']['cagr'])}、全銘柄の均等保有 年率{P(mo['baseline']['cagr'])}。",
        "",
        "## 1. シグナルで買う手法",
        "",
        "| 手法 | 件数 | 勝率 | 1回の平均 | 保有 | TOPIXとの超過 | t値 | 前半/後半の超過 | 口座の年率(同時10銘柄・ランダム順) | 最大下落 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, st in res["events"].items():
        t = st["trades"]
        if not t.get("n"):
            L.append(f"| {st['label']} | 0 | — | — | — | — | — | — | — | — |")
            continue
        pr = st.get("portfolio_random", {})
        L.append(f"| {st['label']} | {t['n']:,} | {t['win_rate'] * 100:.0f}% | {P(t['avg'], 2)} | {t['hold']:.0f}日 | "
                 f"**{P(t['excess'], 2)}** | {ib._t(t['t'])} | {P(t['excess_first'], 2)} / {P(t['excess_second'], 2)} | "
                 f"{P(pr.get('cagr'))} | {P(pr.get('max_dd'), 1, False)} |")
    L += ["", "## 2. 月末に選んで1か月持つ手法", "",
          "| 手法 | 平均銘柄数 | 投資していた月 | 年率 | 最大下落 | 基準との差(年) | t値 | 前半/後半 |",
          "|---|---|---|---|---|---|---|---|"]
    for k, r in mo["portfolios"].items():
        L.append(f"| {r['label']} | {r['n_avg']:.0f} | {r['invested'] * 100:.0f}% | {P(r['cagr'])} | "
                 f"{P(r['max_dd'], 1, False)} | **{P(r['excess_yr'])}** | {ib._t(r['t'])} | "
                 f"{P(r['excess_first'])} / {P(r['excess_second'])} |")
    L += ["", "## 読み方", "",
          "- 条件を足すほど件数が減り、偶然の影響が大きくなる。t値2以上と、前半・後半の符号がそろうことを確認する。",
          "- 条件はいくつも試しているので、一番良いものだけを選ぶと過大評価になる(データの使い回し)。",
          "- 期間は約2.8年で、1つの相場局面に近い。",
          "", "一般的な手法の検証であり、個別の投資助言ではない。"]
    return "\n".join(L) + "\n"
