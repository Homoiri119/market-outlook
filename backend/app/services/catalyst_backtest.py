"""Survivorship-free test of stock picks that have a *reason*: news/catalysts and themes.

A stock is usually bought for a reason, so the "inverse" study (inverse_backtest.py)
is extended with conditions a discretionary trader would actually use:

  Catalysts (J-Quants financial disclosures, if the plan provides them):
    rev_up / rev_down     company forecast of operating profit revised by >= +10% / <= -10%
    div_up / div_down     forecast annual dividend raised / cut
    earn_pos / earn_neg   results day reaction >= +5% / <= -5% (bought the day after)
  Price-only "news-like" moves (always available):
    gap_up / gap_down     open gaps >= +5% / <= -5% on volume >= 3x the 20-day average,
                          on a day with no financial disclosure (if disclosures are known)
  Themes (33-sector as a theme proxy; true themes such as AI or defence are not tagged):
    hot sector leaders / laggards, hot sector, cold sector, 12-1 month sector momentum.

Entries use the first open after the information is public (no look-ahead). Trade-level
excess is measured against TOPIX over the same window; monthly portfolios against the
equal-weight universe. Not investment advice; flat costs, no slippage, no taxes.
"""

from __future__ import annotations

import datetime as dt
import logging
import time

import numpy as np
import pandas as pd

from app.clients.jquants_client import jquants_client
from app.services import inverse_backtest as ib
from app.services import legend_backtest as lb
from app.services.ipo_backtest import _EMPTY_SECTOR

logger = logging.getLogger(__name__)

COST = lb.COST
HOLDS = (5, 20, 60)
PORTFOLIO_HOLD = 20
REV_TH = 0.10
GAP_TH = 0.05
REACT_TH = 0.05
VOL_MULT = 3.0
N_SECTORS = 3
FIN_PATHS = ("/fins/summary", "/fins/statements")

EVENTS = {
    "rev_up": "上方修正(会社予想の営業利益+10%以上)→翌朝買う",
    "rev_down": "下方修正(−10%以下)→翌朝買う(逆張り)",
    "div_up": "増配(予想配当の引き上げ)→翌朝買う",
    "div_down": "減配(予想配当の引き下げ)→翌朝買う(逆張り)",
    "earn_pos": "決算で+5%以上上昇→その翌朝買う(決算後のトレンドに乗る)",
    "earn_neg": "決算で−5%以下に下落→その翌朝買う(逆張り)",
    "gap_up": "決算以外で+5%以上の窓上げ+出来高3倍(好材料らしき動き)→翌朝買う",
    "gap_down": "決算以外で−5%以下の窓下げ+出来高3倍(悪材料らしき動き)→翌朝買う(逆張り)",
}


# ---------------------------------------------------------------------------
# Data: financial disclosures and sectors
# ---------------------------------------------------------------------------
def _g(r: dict, *keys):
    for k in keys:
        v = r.get(k)
        if v not in (None, "", "-"):
            return v
    return None


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


FIELDS = {
    "date": ("DisclosedDate", "DiscDate", "Date"),
    "time": ("DisclosedTime", "DiscTime", "Time"),
    "code": ("LocalCode", "Code"),
    "doc": ("TypeOfDocument", "DocType"),
    "fy_end": ("CurrentFiscalYearEndDate", "CurFYEn", "CurFYEnd"),
    "op": ("OperatingProfit", "OP"),
    "fop": ("ForecastOperatingProfit", "FOP"),
    "fdiv": ("ForecastDividendPerShareAnnual", "FDivAnn", "FDivPerShareAnn"),
}


def fetch_disclosures(calendar: list[dt.date], sleep: float = 0.15) -> tuple[pd.DataFrame, dict]:
    """All financial disclosures by date. Returns (rows, info). Empty frame if the
    plan does not include financial statements (the price-only tests still run)."""
    info: dict = {"path": None, "raw_keys": [], "rows": 0}
    path = None
    probe_dates = calendar[len(calendar) // 2: len(calendar) // 2 + 15]
    for p in FIN_PATHS:
        for d in probe_dates:
            try:
                rows = jquants_client._get_paginated(p, {"date": d.isoformat()}, "data")
            except Exception as e:  # 403 plan / 404 path
                logger.info("fins probe %s %s failed: %s", p, d, e)
                break
            if rows:
                path = p
                info["raw_keys"] = sorted(rows[0].keys())
                break
        if path:
            break
    info["path"] = path
    if not path:
        logger.warning("financial disclosures unavailable; catalyst tests use prices only")
        return pd.DataFrame(), info
    logger.info("fins path %s; fields: %s", path, info["raw_keys"])
    recs = []
    for n, d in enumerate(calendar, 1):
        rows = None
        for wait in (0,) + lb.RETRY_WAITS:
            if wait:
                time.sleep(wait)
            try:
                rows = jquants_client._get_paginated(path, {"date": d.isoformat()}, "data")
                break
            except Exception as e:
                logger.info("fins %s retry after error: %s", d, e)
        for r in rows or []:
            recs.append({k: _g(r, *keys) for k, keys in FIELDS.items()})
        if n % 100 == 0:
            logger.info("fins: %d/%d days, %d rows", n, len(calendar), len(recs))
        time.sleep(sleep)
    df = pd.DataFrame(recs)
    info["rows"] = int(len(df))
    info["non_null"] = {k: int(df[k].notna().sum()) for k in FIELDS} if len(df) else {}
    return df, info


def fetch_sectors(dates: list[dt.date]) -> dict[str, str]:
    """code -> 33-sector name, from masters at several dates (latest wins)."""
    out: dict[str, str] = {}
    for d in dates:
        try:
            m = jquants_client.fetch_listed_info(d)
        except Exception:
            logger.info("master unavailable for %s", d, exc_info=True)
            continue
        if m.empty or "S33Nm" not in m.columns:
            continue
        for code, sec in zip(m["Code"].astype(str), m["S33Nm"].astype(str)):
            if sec.strip() not in _EMPTY_SECTOR:
                out[code] = sec
    return out


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------
def _norm_code(c) -> str:
    s = str(c or "").strip()
    return s + "0" if len(s) == 4 else s


def disclosure_events(disc: pd.DataFrame) -> pd.DataFrame:
    """One row per (code, date, kind) for revisions, dividend changes and results."""
    if disc.empty:
        return pd.DataFrame(columns=["code", "date", "time", "kind"])
    d = disc.copy()
    d["code"] = d["code"].map(_norm_code)
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"]).sort_values(["code", "date"])
    d["fop"] = d["fop"].map(_f)
    d["fdiv"] = d["fdiv"].map(_f)
    d["op"] = d["op"].map(_f)
    ev = []
    for code, g in d.groupby("code", sort=False):
        last_fop: dict = {}
        last_div: dict = {}
        for r in g.itertuples(index=False):
            fy = r.fy_end
            doc = str(r.doc or "")
            if (doc and "FinancialStatements" in doc) or (not doc and not np.isnan(r.op)):
                ev.append((code, r.date, r.time, "results"))
            if fy is not None and not np.isnan(r.fop):
                prev = last_fop.get(fy)
                if prev is not None and prev > 0:
                    chg = r.fop / prev - 1
                    if chg >= REV_TH:
                        ev.append((code, r.date, r.time, "rev_up"))
                    elif chg <= -REV_TH:
                        ev.append((code, r.date, r.time, "rev_down"))
                last_fop[fy] = r.fop
            if fy is not None and not np.isnan(r.fdiv):
                prev = last_div.get(fy)
                if prev is not None:
                    if r.fdiv > prev + 1e-9:
                        ev.append((code, r.date, r.time, "div_up"))
                    elif r.fdiv < prev - 1e-9:
                        ev.append((code, r.date, r.time, "div_down"))
                last_div[fy] = r.fdiv
    out = pd.DataFrame(ev, columns=["code", "date", "time", "kind"]).drop_duplicates(["code", "date", "kind"])
    return out


def build_event_trades(panel: dict[str, pd.DataFrame], events: pd.DataFrame,
                       has_fins: bool) -> dict[tuple[str, int], pd.DataFrame]:
    close, opn, vol, to = panel["close"], panel["open"], panel["volume"], panel["turnover"]
    dates = close.index
    va20 = to.rolling(20, min_periods=10).mean().shift(1)
    v20 = vol.rolling(20, min_periods=10).mean().shift(1)
    rows: dict[tuple[str, int], list] = {(k, h): [] for k in EVENTS for h in HOLDS}
    code_pos = {c: i for i, c in enumerate(close.columns)}
    C, O = close.to_numpy("float64"), opn.to_numpy("float64")
    VA = va20.to_numpy("float64")
    n = len(dates)

    def rec(kind, j, e):
        """Signal known by close/disclosure before day e; buy open e, sell open e+h."""
        if e >= n - 1 or not (O[e, j] > 0) or not (VA[e - 1, j] >= lb.MIN_TURNOVER):
            return
        for h in HOLDS:
            x = min(e + h, n - 1)
            if x <= e or not (O[x, j] > 0):
                continue
            rows[(kind, h)].append((dates[e], dates[x], close.columns[j], O[x, j] / O[e, j] - 1,
                                    x - e, 0.5, O[e, j], O[x, j]))

    disc_days: dict[str, set] = {}
    if not events.empty:
        for r in events.itertuples(index=False):
            j = code_pos.get(r.code)
            if j is None:
                continue
            disc_days.setdefault(r.code, set())
            # First open strictly after the disclosure date is the first realistic fill.
            e = int(dates.searchsorted(pd.Timestamp(r.date), side="right"))
            if e >= n - 1 or e < lb.WARMUP_BARS:
                continue
            disc_days[r.code].update({e - 1, e})  # reaction can land on either day
            if r.kind == "results":
                pre = e - 1
                intraday = bool(r.time) and str(r.time)[:5] < "15:00"
                if intraday and dates[pre] == pd.Timestamp(r.date):
                    pre -= 1  # disclosed during the session: reaction starts that day
                if pre < 0 or not (C[pre, j] > 0 and C[e, j] > 0):
                    continue
                react = C[e, j] / C[pre, j] - 1
                if react >= REACT_TH:
                    rec("earn_pos", j, e + 1)
                elif react <= -REACT_TH:
                    rec("earn_neg", j, e + 1)
            else:
                rec(r.kind, j, e)

    # Price-only news-like gaps on volume, excluding disclosure days when known.
    V, V20 = vol.to_numpy("float64"), v20.to_numpy("float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = O[1:] / C[:-1] - 1
    for j, code in enumerate(close.columns):
        skip = disc_days.get(code, set())
        for i in np.where((gap[:, j] >= GAP_TH) | (gap[:, j] <= -GAP_TH))[0] + 1:
            if i < lb.WARMUP_BARS or not (V[i, j] >= VOL_MULT * V20[i, j]) or i in skip:
                continue
            rec("gap_up" if gap[i - 1, j] > 0 else "gap_down", j, i + 1)
    frames = {k: pd.DataFrame(v, columns=["entry", "exit", "code", "ret", "bars", "rs", "entry_px", "exit_px"])
              for k, v in rows.items()}
    if not has_fins:
        for k in ("rev_up", "rev_down", "div_up", "div_down", "earn_pos", "earn_neg"):
            for h in HOLDS:
                frames[(k, h)] = frames[(k, h)].iloc[0:0]
    return frames


# ---------------------------------------------------------------------------
# Themes (sector proxy), monthly portfolios
# ---------------------------------------------------------------------------
THEMES = {
    "hot_leaders": "旬のテーマ株に飛び乗る(1か月で一番上がった3業種の中で、上昇率上位20%)",
    "hot_laggards": "旬のテーマの出遅れ株(同じ3業種の中で、上昇率下位20%)",
    "hot_all": "旬のテーマ(1か月で一番上がった3業種)の全銘柄",
    "cold_all": "不人気のテーマ(1か月で一番下がった3業種)の全銘柄",
    "mom_top": "テーマの持続(12-1か月で一番強い3業種)の全銘柄",
    "mom_bottom": "12-1か月で一番弱い3業種の全銘柄",
}


def themes(panel: dict[str, pd.DataFrame], sectors: dict[str, str], cal: pd.DatetimeIndex,
           split: pd.Timestamp) -> dict:
    close = panel["close"].astype("float64")
    rets = close.pct_change(fill_method=None)
    va20 = panel["turnover"].rolling(20, min_periods=10).mean()
    r21 = close / close.shift(21) - 1
    mom = close.shift(21) / close.shift(252) - 1
    sec = pd.Series({c: sectors.get(c) for c in close.columns}).dropna()
    me = ib._month_ends(close.index, cal[0])
    months = list(zip(me[:-1], me[1:]))
    picks: dict[str, list] = {k: [] for k in THEMES}
    base = []
    for a, b in months:
        el = close.columns[((va20.loc[a] >= lb.MIN_TURNOVER) & close.loc[a].notna()).values]
        el = [c for c in el if c in sec.index]
        base.append((a, b, el))
        if len(el) < 100:
            for k in THEMES:
                picks[k].append((a, b, []))
            continue
        s = sec[el]
        f1 = r21.loc[a, el]
        fm = mom.loc[a, el]
        sec1 = f1.groupby(s).mean()[s.value_counts() >= 5].sort_values()
        secm = fm.groupby(s).mean()[s.value_counts() >= 5].dropna().sort_values()
        hot, cold = list(sec1.index[-N_SECTORS:]), list(sec1.index[:N_SECTORS])
        in_hot = [c for c in el if s[c] in hot]
        fh = f1[in_hot].dropna()
        k = max(int(len(fh) * ib.QUINTILE), 1)
        picks["hot_leaders"].append((a, b, list(fh.nlargest(k).index)))
        picks["hot_laggards"].append((a, b, list(fh.nsmallest(k).index)))
        picks["hot_all"].append((a, b, in_hot))
        picks["cold_all"].append((a, b, [c for c in el if s[c] in cold]))
        picks["mom_top"].append((a, b, [c for c in el if s[c] in list(secm.index[-N_SECTORS:])]))
        picks["mom_bottom"].append((a, b, [c for c in el if s[c] in list(secm.index[:N_SECTORS])]))
    base_raw = pd.concat([rets.loc[(rets.index > a) & (rets.index <= b), p].mean(axis=1).fillna(0)
                          for a, b, p in base])
    out = {"baseline": lb._curve_stats((1 + base_raw).cumprod()), "n_sectors": int(sec.nunique()),
           "portfolios": {}}
    for k, pk in picks.items():
        r, turn = ib._hold(pk, rets)
        if r.empty:
            continue
        out["portfolios"][k] = {"label": THEMES[k], **lb._curve_stats((1 + r).cumprod()),
                                **ib._excess_stats(r, base_raw, split), "turnover": turn,
                                "n_avg": float(np.mean([len(p) for _, _, p in pk]))}
    return out


# ---------------------------------------------------------------------------
def run_on_panel(panel: dict[str, pd.DataFrame], bench: pd.Series, disc: pd.DataFrame,
                 fins_info: dict, sectors: dict[str, str]) -> dict:
    close = panel["close"]
    cal = close.index[lb.WARMUP_BARS:]
    split = cal[len(cal) // 2]
    events = disclosure_events(disc)
    has_fins = not events.empty
    trades = build_event_trades(panel, events, has_fins)
    ev_out = {}
    for kind, label in EVENTS.items():
        st = {"label": label, "holds": {}}
        for h in HOLDS:
            t = trades[(kind, h)]
            t = t[t["entry"] >= cal[0]]
            st["holds"][str(h)] = lb._trade_stats(t, bench, split)
            if h == PORTFOLIO_HOLD and len(t):
                st["portfolio_random"] = lb._curve_stats(lb._portfolio(t, panel, cal, "random"))
        ev_out[kind] = st
        logger.info("event %s: %s", kind, {h: st["holds"][str(h)].get("n", 0) for h in HOLDS})
    b = bench.reindex(cal).ffill()
    return {
        "period": {"start": str(cal[0].date()), "end": str(cal[-1].date()), "split": str(split.date())},
        "params": {"min_turnover": lb.MIN_TURNOVER, "cost": COST, "rev_th": REV_TH, "gap_th": GAP_TH,
                   "react_th": REACT_TH, "vol_mult": VOL_MULT, "holds": list(HOLDS),
                   "n_sectors": N_SECTORS, "slots": lb.SLOTS},
        "fins": {**fins_info, "events": events["kind"].value_counts().to_dict() if has_fins else {}},
        "topix": lb._curve_stats(b / b.iloc[0]),
        "events": ev_out,
        "themes": themes(panel, sectors, cal, split) if sectors else None,
    }


# ---------------------------------------------------------------------------
def render_markdown(res: dict, generated_at: str) -> str:
    p, P = res["period"], ib._p
    L = [
        f"# 材料・テーマで個別株を選ぶ:生存者バイアスなしの検証({generated_at[:10]})",
        "",
        f"生成:{generated_at} / 計算:`python backend/scripts/run_inverse_backtest.py`(J-Quants、GitHub Actions「Inverse Backtest」)",
        "",
        "- **銘柄**:日付指定の全銘柄日足から作った、その日に実際に上場していた銘柄(上場廃止を含む)。ETF・REIT等は除外。"
        f"20日平均売買代金が{res['params']['min_turnover'] / 1e8:.0f}億円以上。",
        f"- **期間**:{p['start']}〜{p['end']}(前半/後半の境目 {p['split']})。往復コスト{res['params']['cost'] * 100:.1f}%。",
        "- **売買**:材料が公表された後の最初の寄り付きで買い、決めた日数後の寄り付きで売る(先読みなし)。超過=同じ期間のTOPIXとの差。",
    ]
    fi = res["fins"]
    if fi.get("path"):
        L.append(f"- **財務データ**:J-Quants `{fi['path']}` から{fi.get('rows', 0):,}件の開示。"
                 f"検出した材料:{', '.join(f'{k} {v:,}件' for k, v in fi.get('events', {}).items()) or 'なし'}。")
        if not fi.get("events"):
            L.append(f"  - 材料を検出できなかった(フィールド名が想定と違う可能性)。取得したフィールド:{', '.join(fi.get('raw_keys', []))}")
    else:
        L.append("- **財務データ**:契約に財務情報が含まれず取得できなかったため、決算・修正・配当の検証は空欄。値動きだけの検証(窓・テーマ)は実施。")
    L += ["", f"TOPIX:年率{P(res['topix']['cagr'])}・最大下落{P(res['topix']['max_dd'], 1, False)}", "",
          "## 1. 材料で買う", "",
          "| 材料 | 保有 | 件数 | 勝率 | 1回の平均 | TOPIXとの超過 | t値 | 前半/後半の超過 |",
          "|---|---|---|---|---|---|---|---|"]
    for kind, st in res["events"].items():
        for h, t in st["holds"].items():
            if not t.get("n"):
                L.append(f"| {st['label']} | {h}日 | 0 | — | — | — | — | — |")
                continue
            L.append(f"| {st['label']} | {h}日 | {t['n']:,} | {t['win_rate'] * 100:.0f}% | {P(t['avg'], 2)} | "
                     f"**{P(t['excess'], 2)}** | {ib._t(t['t'])} | {P(t['excess_first'], 2)} / {P(t['excess_second'], 2)} |")
    L += ["", f"### 口座の試算({PORTFOLIO_HOLD}日保有・同時{res['params']['slots']}銘柄・ランダム順)", "",
          "| 材料 | 年率 | 最大下落 | シャープ |", "|---|---|---|---|"]
    for kind, st in res["events"].items():
        c = st.get("portfolio_random")
        if c:
            sh = f"{c['sharpe']:.2f}" if c["sharpe"] is not None else "—"
            L.append(f"| {st['label']} | {P(c['cagr'])} | {P(c['max_dd'], 1, False)} | {sh} |")
    th = res.get("themes")
    if th:
        L += ["", f"## 2. テーマで買う(東証33業種をテーマの代わりに使う。月末に選んで1か月持つ)", "",
              f"基準(全銘柄を均等):年率{P(th['baseline']['cagr'])}・最大下落{P(th['baseline']['max_dd'], 1, False)}", "",
              "| 選び方 | 平均銘柄数 | 年率 | 最大下落 | 基準との差(年) | t値 | 前半/後半 | 月の入れ替え |",
              "|---|---|---|---|---|---|---|---|"]
        for k, r in th["portfolios"].items():
            L.append(f"| {r['label']} | {r['n_avg']:.0f} | {P(r['cagr'])} | {P(r['max_dd'], 1, False)} | "
                     f"**{P(r['excess_yr'])}** | {ib._t(r['t'])} | {P(r['excess_first'])} / {P(r['excess_second'])} | "
                     f"{r['turnover'] * 100:.0f}% |")
    L += ["", "## 読み方", "",
          "- 超過がプラスで、t値が2以上、前半と後半の符号がそろっていれば、優位性がある可能性がある。",
          "- 「テーマ」は東証33業種で代用している。AI・半導体・防衛のような本来のテーマは業種をまたぐので、この検証では捉えきれない。",
          "- 材料の内容(ニュースの文面)は使っていない。決算・業績予想の修正・配当の変更と、値動き(窓+出来高)で材料を判定している。",
          "- 期間は約2.8年で、1つの相場局面に近い。",
          "", "一般的な手法の検証であり、個別の投資助言ではない。"]
    return "\n".join(L) + "\n"
