"""Executes reports/strategy_review_prompt.md (Q1-Q7) and writes the numbers to
reports/strategy_review_results.json, printing readable tables.

Run from repo root:  py reports/strategy_review.py
Fills: the live tracker assumes idealized fills (exit exactly at SL/TP). The
counterfactual grid here uses realistic gap fills (if the open is already beyond
the level, fill at the open) and a conservative trailing update (today's low is
checked against the stop set from PRIOR highs).
"""

from __future__ import annotations

import datetime as dt
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = ROOT / "reports" / "strategy_review_results.json"

pf = json.loads((DOCS / "portfolio.json").read_text(encoding="utf-8"))
hist = json.loads((DOCS / "history.json").read_text(encoding="utf-8"))
ss = json.loads((DOCS / "stock_strategy.json").read_text(encoding="utf-8"))
analysis = json.loads((DOCS / "analysis.json").read_text(encoding="utf-8"))
P = pf["positions"]
HIST = {h["date"]: h for h in hist}
SECTOR = {str(s["code"]): s.get("sector") or "—" for s in analysis.get("stocks", [])}
R: dict = {}


def yt(code: str) -> str:
    code = str(code)
    return (code[:4] if len(code) == 5 and code.endswith("0") else code) + ".T"


def stats(rs: list[float]) -> dict:
    if not rs:
        return {"n": 0}
    a = np.array(rs, dtype=float)
    win, loss = a[a > 0], a[a <= 0]
    return {"n": int(a.size), "win": float((a > 0).mean()), "avg": float(a.mean()),
            "med": float(np.median(a)), "sum": float(a.sum()),
            "pf": float(win.sum() / -loss.sum()) if loss.sum() < 0 else None}


def fmt(s: dict, unit: str = "R") -> str:
    if not s.get("n"):
        return "n=0"
    pf_ = f"{s['pf']:.2f}" if s.get("pf") is not None else "—"
    return (f"n={s['n']:>3} 勝率{s['win']*100:5.1f}% 平均{s['avg']:+.3f}{unit} "
            f"中央{s['med']:+.3f}{unit} 合計{s['sum']:+.2f}{unit} PF {pf_}")


# ---------------------------------------------------------------- Q1 outlook
done = [h for h in hist if h.get("hit") is not None and h.get("actual_move") is not None]
exp = np.array([h["expected_move"] for h in done]); act = np.array([h["actual_move"] for h in done])
by_dir = defaultdict(list)
for h in done:
    by_dir[h["direction"]].append(1.0 if h["hit"] else 0.0)
R["q1"] = {
    "n": len(done), "hit_rate": float(np.mean([h["hit"] for h in done])),
    "mae": float(np.mean(np.abs(exp - act))), "corr": float(np.corrcoef(exp, act)[0, 1]),
    "by_dir": {k: {"n": len(v), "hit": float(np.mean(v))} for k, v in by_dir.items()},
    "avg_actual_gap": float(act.mean()),
}
print("\n=== Q1 寄り付き予想 ===")
print(f"n={R['q1']['n']} 的中{R['q1']['hit_rate']*100:.1f}% MAE {R['q1']['mae']*100:.2f}% 相関 {R['q1']['corr']:.2f}")
for k, v in R["q1"]["by_dir"].items():
    print(f"  {k:12s} n={v['n']:2d} 的中{v['hit']*100:5.1f}%")


# ---------------------------------------------------------------- Q2 live
def live_R(p: dict, method: str) -> tuple[float, float, bool]:
    e, sl = p["entry"], p["stop_loss"]
    if method == "trail":
        closed = p.get("trail_status") == "closed"
        px = p["trail_exit_price"] if closed else p["last_price"]
    else:
        closed = p["status"] == "closed"
        px = p["exit_price"] if closed else p["last_price"]
    return (px - e) / (e - sl), px / e - 1, closed


R["q2"] = {}
print("\n=== Q2 実績(トラッカーの理想約定) ===")
for m in ("fixed", "trail"):
    rs = [live_R(p, m)[0] for p in P]
    rs_closed = [live_R(p, m)[0] for p in P if live_R(p, m)[2]]
    yen100 = sum((live_R(p, m)[1]) * p["entry"] * 100 for p in P)
    R["q2"][m] = {"all": stats(rs), "closed": stats(rs_closed), "yen_100sh": yen100}
    print(f"{m:6s} 全建玉 {fmt(stats(rs))}")
    print(f"{'':6s} 決済分 {fmt(stats(rs_closed))}  100株損益 {yen100:+,.0f}円")
bt = {k: {"win": v["win_rate"], "expR": v["expectancy_R"], "pf": v["profit_factor"]}
      for k, v in ss["configs"].items()}
R["q2"]["backtest"] = bt
print("バックテスト基準:", {k: f"勝率{v['win']*100:.0f}% 期待値{v['expR']:+.3f}R PF{v['pf']:.2f}" for k, v in bt.items()})

# ---------------------------------------------------------------- bars
start = (min(dt.date.fromisoformat(p["date"]) for p in P) - dt.timedelta(days=7)).isoformat()
end = (dt.date.today() + dt.timedelta(days=1)).isoformat()
tickers = sorted({yt(p["code"]) for p in P})
raw = yf.download(tickers, start=start, end=end, auto_adjust=False, group_by="ticker",
                  progress=False, threads=True)
BARS: dict[str, pd.DataFrame] = {}
for t in tickers:
    try:
        df = raw[t][["Open", "High", "Low", "Close"]].dropna()
    except Exception:
        continue
    if df.empty:
        continue
    df.index = [d.date() for d in pd.to_datetime(df.index)]
    BARS[t] = df
print(f"\n日足取得: {len(BARS)}/{len(tickers)} 銘柄")


SPLIT_TH = 0.20   # |prev close / entry - 1| beyond this = share split (price basis changed)
SPLIT_IDS: set[str] = set()
STALE_IDS: set[str] = set()


def seg(p: dict) -> pd.DataFrame | None:
    """Bars from the recommendation date, on the SAME price basis as the entry.
    yfinance back-adjusts splits; if the prior close is far from the entry, the
    series is rescaled by entry/prior-close (≈ split ratio)."""
    b = BARS.get(yt(p["code"]))
    if b is None:
        return None
    d0 = dt.date.fromisoformat(p["date"])
    prev = b[b.index < d0]
    s = b[b.index >= d0]
    if s.empty:
        return None
    if not prev.empty:
        ratio = float(prev["Close"].iloc[-1]) / p["entry"]
        if abs(ratio - 1) > SPLIT_TH:
            s = s * (1 / ratio)
    return s


# sanity: does the recommended entry match the prior close in yfinance?
mism = []
for p in P:
    b = BARS.get(yt(p["code"]))
    if b is None:
        continue
    prev = b[b.index < dt.date.fromisoformat(p["date"])]
    if not prev.empty:
        dev = prev["Close"].iloc[-1] / p["entry"] - 1
        mism.append(abs(dev))
        if abs(dev) > SPLIT_TH:
            SPLIT_IDS.add(p["id"])
        elif abs(dev) > 0.01:
            STALE_IDS.add(p["id"])
R["sanity_entry_vs_prevclose_median_abs"] = float(np.median(mism)) if mism else None
R["split_affected"] = sorted(SPLIT_IDS)
R["stale_entry"] = sorted(STALE_IDS)
print(f"建値と前日終値(yfinance)の乖離 中央値 {R['sanity_entry_vs_prevclose_median_abs']*100:.2f}%")
print(f"分割で価格基準が変わった建玉: {sorted(SPLIT_IDS)}")
print(f"建値が前日終値と1%超ずれ(古い推奨価格): {len(STALE_IDS)}件 {sorted(STALE_IDS)}")

# Q2b: the live tracker's numbers with split-distorted positions removed.
R["q2"]["fixed_ex_split"] = stats([live_R(p, "fixed")[0] for p in P if p["id"] not in SPLIT_IDS])
R["q2"]["trail_ex_split"] = stats([live_R(p, "trail")[0] for p in P if p["id"] not in SPLIT_IDS])
print("実績(分割の誤判定を除外) fixed", fmt(R["q2"]["fixed_ex_split"]))
print("実績(分割の誤判定を除外) trail", fmt(R["q2"]["trail_ex_split"]))

# Q2c: why the 100-share view can show a yen profit while R is negative.
contrib = sorted(((live_R(p, "fixed")[1] * p["entry"] * 100, p) for p in P if p["id"] not in SPLIT_IDS),
                 key=lambda t: t[0], reverse=True)
yen_total = sum(c for c, _ in contrib)
top5 = sum(c for c, _ in contrib[:5])
R["q2"]["yen100_ex_split"] = yen_total
R["q2"]["yen100_top5"] = top5
R["q2"]["yen100_top5_names"] = [f"{p['name']}({p['code']}) {p['date']} {c:+,.0f}" for c, p in contrib[:5]]
R["q2"]["ret_avg_equal_weight"] = float(np.mean([live_R(p, "fixed")[1] for p in P if p["id"] not in SPLIT_IDS]))
print(f"100株・円換算(分割除外) {yen_total:+,.0f}円 / うち上位5件 {top5:+,.0f}円 / 均等加重の平均騰落 {R['q2']['ret_avg_equal_weight']*100:+.2f}%")
for s_ in R["q2"]["yen100_top5_names"]:
    print("   ", s_)


# ---------------------------------------------------------------- simulator
def simulate(p: dict, sl_mult: float, kind: str, param: float, entry_mode: str = "prev",
             realistic: bool = True, max_hold: int | None = None,
             be_at: float | None = None) -> dict | None:
    """kind: fixed (param=TP×ATR) | trail (param=trail×ATR) | time (param=hold days).
    max_hold: also exit at the close after N bars. be_at: once the high reaches
    entry + be_at×ATR, raise the stop to the entry (breakeven) from the next bar."""
    s = seg(p)
    if s is None:
        return None
    atr = (p["entry"] - p["stop_loss"]) / 1.5
    E = float(s["Open"].iloc[0]) if entry_mode == "open" else p["entry"]
    sl0 = E - sl_mult * atr
    stop = sl0
    tp = E + param * atr if kind == "fixed" else None
    hh = E
    for i, (d, row) in enumerate(s.iterrows()):
        op, hi, lo, cl = map(float, (row.Open, row.High, row.Low, row.Close))
        # On the entry day with an open entry, the stop can't fire on the pre-entry gap.
        gap_ok = not (entry_mode == "open" and i == 0)
        if lo <= stop:
            px = op if (realistic and gap_ok and op < stop) else stop
            return {"R": (px - E) / (E - sl0), "ret": px / E - 1, "days": i, "why": "stop" if px <= E else "trail", "open": False}
        if kind == "fixed" and hi >= tp:
            px = op if (realistic and gap_ok and op > tp) else tp
            return {"R": (px - E) / (E - sl0), "ret": px / E - 1, "days": i, "why": "tp", "open": False}
        if (kind == "time" and i >= param) or (max_hold is not None and i >= max_hold):
            return {"R": (cl - E) / (E - sl0), "ret": cl / E - 1, "days": i, "why": "time", "open": False}
        hh = max(hh, hi)
        if kind == "trail":
            stop = max(stop, hh - param * atr)  # applies from the NEXT bar (conservative)
        if be_at is not None and hh >= E + be_at * atr:
            stop = max(stop, E)
    cl = float(s["Close"].iloc[-1])
    return {"R": (cl - E) / (E - sl0), "ret": cl / E - 1, "days": len(s) - 1, "why": "open", "open": True}


# ---------------------------------------------------------------- Q3 gap / entry
gaps, same_day_sl, rows = [], 0, 0
for p in P:
    s = seg(p)
    if s is None:
        continue
    rows += 1
    gaps.append(float(s["Open"].iloc[0]) / p["entry"] - 1)
    if float(s["Low"].iloc[0]) <= p["stop_loss"]:
        same_day_sl += 1
live_like = [simulate(p, 1.5, "fixed", 2.5, "prev", realistic=False) for p in P]
real_prev = [simulate(p, 1.5, "fixed", 2.5, "prev", realistic=True) for p in P]
real_open = [simulate(p, 1.5, "fixed", 2.5, "open", realistic=True) for p in P]
R["q3"] = {
    "n": rows, "gap_mean": float(np.mean(gaps)), "gap_median": float(np.median(gaps)),
    "gap_down_share": float(np.mean([g < 0 for g in gaps])),
    "same_day_sl_share": same_day_sl / rows if rows else None,
    "idealized_prev": stats([x["R"] for x in live_like if x]),
    "realistic_prev": stats([x["R"] for x in real_prev if x]),
    "realistic_open": stats([x["R"] for x in real_open if x]),
}
print("\n=== Q3 エントリー前提 ===")
print(f"推奨日の寄りギャップ 平均{R['q3']['gap_mean']*100:+.2f}% 中央{R['q3']['gap_median']*100:+.2f}% 下窓率{R['q3']['gap_down_share']*100:.0f}%")
print(f"推奨当日に損切ライン到達: {R['q3']['same_day_sl_share']*100:.0f}%")
print("理想約定(前日終値で買えた)   ", fmt(R["q3"]["idealized_prev"]))
print("現実約定(前日終値・窓は寄りで約定)", fmt(R["q3"]["realistic_prev"]))
print("当日寄りで買って同じATR幅      ", fmt(R["q3"]["realistic_open"]))

# ---------------------------------------------------------------- Q4 regime
R["q4"] = {"by_pred": {}, "by_actual": {}}
grp_pred, grp_act = defaultdict(list), defaultdict(list)
for p, x in zip(P, real_prev):
    if not x:
        continue
    grp_pred[p.get("reg_dir") or "?"].append(x["R"])
    h = HIST.get(p["date"])
    if h and h.get("actual_move") is not None:
        grp_act["実際は上寄り" if h["actual_move"] > 0.002 else ("実際は下寄り" if h["actual_move"] < -0.002 else "実際は横ばい")].append(x["R"])
print("\n=== Q4 地合い別(現実約定・固定利確) ===")
for k, v in sorted(grp_pred.items()):
    R["q4"]["by_pred"][k] = stats(v); print(f"予想 {k:12s}", fmt(stats(v)))
for k, v in sorted(grp_act.items()):
    R["q4"]["by_actual"][k] = stats(v); print(f"{k:12s}     ", fmt(stats(v)))

# ---------------------------------------------------------------- Q5 grid
dates = sorted(p["date"] for p in P)
mid = dates[len(dates) // 2]
DOWN_DAYS = {"DOWN", "STRONG_DOWN"}
configs = []  # (sl, kind, param, opts, label)
for slm in (1.0, 1.5, 2.0, 2.5):
    for prm in (1.0, 1.5, 2.5, 3.5):
        configs.append((slm, "fixed", prm, {}, f"固定利確{prm}ATR"))
    for prm in (1.5, 2.5):
        configs.append((slm, "trail", prm, {}, f"トレール{prm}ATR"))
    for prm in (3, 5, 10):
        configs.append((slm, "time", prm, {}, f"{prm}日で決済"))
    configs.append((slm, "fixed", 2.5, {"max_hold": 5}, "利確2.5+最長5日"))
    configs.append((slm, "fixed", 2.5, {"be_at": 1.0}, "利確2.5+建値ストップ"))
    configs.append((slm, "fixed", 1.5, {"be_at": 1.0}, "利確1.5+建値ストップ"))
    configs.append((slm, "trail", 1.5, {"max_hold": 5}, "トレール1.5+最長5日"))
grid = []
for slm, kind, prm, opts, lab in configs:
    res = [(p, simulate(p, slm, kind, prm, **opts)) for p in P]
    res = [(p, x) for p, x in res if x]
    a = stats([x["R"] for _, x in res])
    h1 = stats([x["R"] for p, x in res if p["date"] < mid])
    h2 = stats([x["R"] for p, x in res if p["date"] >= mid])
    sk = stats([x["R"] for p, x in res if p.get("reg_dir") not in DOWN_DAYS])
    grid.append({"sl": slm, "kind": kind, "param": prm, "opts": opts, "label": lab,
                 "all": a, "h1": h1, "h2": h2, "skip_down": sk,
                 "ret_avg": float(np.mean([x["ret"] for _, x in res])),
                 "days_avg": float(np.mean([x["days"] for _, x in res])),
                 "open_share": float(np.mean([x["open"] for _, x in res]))})
for key in ("h1", "h2"):  # rank in each half; robust = good in BOTH halves
    order = sorted(grid, key=lambda g: g[key]["avg"], reverse=True)
    for rank, g in enumerate(order, 1):
        g[f"rank_{key}"] = rank
for g in grid:
    g["rank_worst_half"] = max(g["rank_h1"], g["rank_h2"])
grid.sort(key=lambda g: g["all"]["avg"], reverse=True)
R["q5"] = {"mid_date": mid, "n_configs": len(grid), "grid": grid}


def pfs(s):
    return "—" if s.get("pf") is None else f"{s['pf']:.2f}"


print(f"\n=== Q5 決済ルールのグリッド(現実約定・1%リスク換算R、{len(grid)}通り、前半<{mid}≦後半) ===")
print("順位 設定                          平均R   勝率  PF  | 前半(順位) 後半(順位) | 下落予想日を見送り | 平均保有日")
for i, g in enumerate(grid, 1):
    cur = " ←現行" if (g["sl"] == 1.5 and g["kind"] == "fixed" and g["param"] == 2.5 and not g["opts"]) else ""
    print(f"{i:>3} SL{g['sl']}ATR {g['label']:20s} {g['all']['avg']:+.3f} {g['all']['win']*100:4.0f}% {pfs(g['all']):>4} | "
          f"{g['h1']['avg']:+.3f}({g['rank_h1']:>2}) {g['h2']['avg']:+.3f}({g['rank_h2']:>2}) | "
          f"{g['skip_down']['avg']:+.3f} PF{pfs(g['skip_down'])} n={g['skip_down']['n']} | {g['days_avg']:.1f}日{cur}")
robust = sorted(grid, key=lambda g: (g["rank_worst_half"], -g["all"]["avg"]))[:5]
R["q5"]["robust_top5"] = [{"sl": g["sl"], "label": g["label"], "all": g["all"]["avg"],
                           "h1": g["h1"]["avg"], "h2": g["h2"]["avg"], "worst_rank": g["rank_worst_half"],
                           "skip_down": g["skip_down"]["avg"]} for g in robust]
print("\n前半・後半の両方で上位(悪い方の順位が小さい順)トップ5:")
for g in robust:
    print(f"  SL{g['sl']}ATR {g['label']:20s} 全体{g['all']['avg']:+.3f}R 前半{g['h1']['avg']:+.3f} 後半{g['h2']['avg']:+.3f} "
          f"(悪い方の順位 {g['rank_worst_half']}/{len(grid)}) 下落日見送り{g['skip_down']['avg']:+.3f}R")

# ---------------------------------------------------------------- Q6 MFE/MAE, after-exit
mfe, mae, tp_after, sl_recover, sl_n, tp_n = [], [], [], 0, 0, 0
for p in P:
    s = seg(p)
    if s is None or p["id"] in SPLIT_IDS:  # tracker exit is a split artifact
        continue
    atr = (p["entry"] - p["stop_loss"]) / 1.5
    e = p["entry"]
    if p["status"] == "closed" and p.get("exit_date"):
        ed = dt.date.fromisoformat(p["exit_date"])
        held = s[s.index <= ed]
        after = s[s.index > ed].head(10)
    else:
        held, after = s, s.iloc[0:0]
    mfe.append((float(held["High"].max()) - e) / atr)
    mae.append((e - float(held["Low"].min())) / atr)
    if p.get("reason") == "利確" and not after.empty:
        tp_n += 1; tp_after.append(float(after["High"].max()) / p["exit_price"] - 1)
    if p.get("reason") == "損切" and not after.empty:
        sl_n += 1
        if float(after["High"].max()) >= e:
            sl_recover += 1
R["q6"] = {"mfe_atr_median": float(np.median(mfe)), "mae_atr_median": float(np.median(mae)),
           "mfe_ge_1atr_share": float(np.mean([m >= 1.0 for m in mfe])),
           "mfe_ge_2_5atr_share": float(np.mean([m >= 2.5 for m in mfe])),
           "tp_n": tp_n, "tp_further_up_median": float(np.median(tp_after)) if tp_after else None,
           "sl_n": sl_n, "sl_recover_share": sl_recover / sl_n if sl_n else None}
print("\n=== Q6 MFE/MAE と決済後の動き(固定利確の実績) ===")
print(f"保有中の最大含み益 中央 {R['q6']['mfe_atr_median']:.2f}ATR / 最大含み損 中央 {R['q6']['mae_atr_median']:.2f}ATR")
print(f"一度でも +1ATR 到達 {R['q6']['mfe_ge_1atr_share']*100:.0f}% / +2.5ATR(利確水準) 到達 {R['q6']['mfe_ge_2_5atr_share']*100:.0f}%")
if tp_after:
    print(f"利確後10日の追加上昇 中央 {R['q6']['tp_further_up_median']*100:+.1f}% (n={tp_n})")
if sl_n:
    print(f"損切後10日以内に建値回復 {R['q6']['sl_recover_share']*100:.0f}% (n={sl_n})")

# ---------------------------------------------------------------- Q7 selection
cnt = Counter(p["code"] for p in P)
rep = [x["R"] for p, x in zip(P, real_prev) if x and cnt[p["code"]] > 1]
one = [x["R"] for p, x in zip(P, real_prev) if x and cnt[p["code"]] == 1]
sec = defaultdict(list)
for p, x in zip(P, real_prev):
    if x:
        sec[SECTOR.get(str(p["code"]), "—")].append(x["R"])
R["q7"] = {"repeat": stats(rep), "single": stats(one),
           "sector": {k: stats(v) for k, v in sec.items() if len(v) >= 3}}
print("\n=== Q7 銘柄選定 ===")
print("繰り返し推奨", fmt(stats(rep))); print("1回のみ    ", fmt(stats(one)))
for k, v in sorted(R["q7"]["sector"].items(), key=lambda kv: kv[1]["avg"]):
    print(f"  {k:16s}", fmt(v))

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(f"\n→ {OUT.relative_to(ROOT)}")
