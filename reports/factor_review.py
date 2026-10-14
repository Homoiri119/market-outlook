"""What drives the Nikkei 225 opening gap? Factor review for the morning outlook.

Run from repo root:  py reports/factor_review.py
Writes reports/factor_review_results.json and prints tables.

Alignment: for Tokyo session D with previous Tokyo session P, each overnight factor is
the move of the US-side series between its last close before P and its last close
before D (i.e. the US sessions dated in [P, D)). The target is the opening gap
open(D) / close(P) - 1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "factor_review_results.json"
START, END = "2021-01-01", "2026-10-15"
FLAT = 0.002  # same flat band as the dashboard's hit judgement

TICK = {"n225": "^N225", "fut": "NKD=F", "sp500": "^GSPC", "nasdaq": "^IXIC", "dow": "^DJI",
        "sox": "^SOX", "vix": "^VIX", "usdjpy": "JPY=X", "us10y": "^TNX", "wti": "CL=F"}
LABEL = {"fut": "日経先物ギャップ(CME)", "sp500": "S&P500", "nasdaq": "NASDAQ", "dow": "NYダウ",
         "sox": "SOX(半導体)", "vix": "VIX変化(pt)", "usdjpy": "ドル円(+=円安)",
         "us10y": "米10年金利変化(bp)", "wti": "WTI原油"}

raw = yf.download(list(TICK.values()), start=START, end=END, auto_adjust=False,
                  group_by="ticker", progress=False, threads=True)


def series(t: str, col: str = "Close") -> pd.Series:
    s = raw[t][col].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s


n_open, n_close = series(TICK["n225"], "Open"), series(TICK["n225"])
days = n_close.index
rows = []
for i in range(1, len(days)):
    P, D = days[i - 1], days[i]
    if D not in n_open.index:
        continue
    r = {"date": D.date().isoformat(), "gap": n_open[D] / n_close[P] - 1,
         "intraday": n_close[D] / n_open[D] - 1}
    for k in ("fut", "sp500", "nasdaq", "dow", "sox", "vix", "usdjpy", "us10y", "wti"):
        s = series(TICK[k])
        a, b = s[s.index < P], s[s.index < D]
        if a.empty or b.empty:
            r[k] = np.nan
            continue
        if k == "fut":
            r[k] = b.iloc[-1] / n_close[P] - 1          # futures vs. the previous Tokyo close
        elif k == "vix":
            r[k] = b.iloc[-1] - a.iloc[-1]               # points
        elif k == "us10y":
            r[k] = (b.iloc[-1] - a.iloc[-1]) * 100       # bp (^TNX is in %)
        else:
            r[k] = b.iloc[-1] / a.iloc[-1] - 1
    rows.append(r)
df = pd.DataFrame(rows).dropna()
# Guard against bad futures prints (basis/roll glitches).
df = df[df["fut"].abs() < 0.08]
F = ["fut", "sp500", "nasdaq", "dow", "sox", "vix", "usdjpy", "us10y", "wti"]
R: dict = {"n_days": len(df), "start": df["date"].iloc[0], "end": df["date"].iloc[-1]}


def sgn(v, band=FLAT):
    return np.where(v > band, 1, np.where(v < -band, -1, 0))


# ---- 1) univariate: correlation, R², sign agreement on days that actually moved
print(f"\n=== 1) 各要因と寄り付きギャップの関係({R['start']}〜{R['end']}, {R['n_days']}日) ===")
moved = df[df["gap"].abs() > FLAT]
uni = {}
for k in F:
    c = float(np.corrcoef(df[k], df["gap"])[0, 1])
    agree = float((np.sign(moved[k]) == np.sign(moved["gap"])).mean())
    beta = float(np.polyfit(df[k], df["gap"], 1)[0])
    uni[k] = {"corr": c, "r2": c * c, "sign_agree": agree, "beta": beta}
for k, v in sorted(uni.items(), key=lambda kv: -kv[1]["r2"]):
    print(f"{LABEL[k]:18s} 相関{v['corr']:+.2f}  説明力R²{v['r2']*100:5.1f}%  方向一致{v['sign_agree']*100:5.1f}%")
R["univariate"] = uni

# ---- 2) multivariate: standardized coefficients and what is left after the futures gap
X = df[F].to_numpy(); y = df["gap"].to_numpy()
Xs = (X - X.mean(0)) / X.std(0)
coef_std, *_ = np.linalg.lstsq(np.c_[np.ones(len(Xs)), Xs], (y - y.mean()) / y.std(), rcond=None)
pred_all = np.c_[np.ones(len(X)), X] @ np.linalg.lstsq(np.c_[np.ones(len(X)), X], y, rcond=None)[0]
r2_all = 1 - ((y - pred_all) ** 2).sum() / ((y - y.mean()) ** 2).sum()
Fus = [k for k in F if k != "fut"]
Xu = df[Fus].to_numpy()
pred_us = np.c_[np.ones(len(Xu)), Xu] @ np.linalg.lstsq(np.c_[np.ones(len(Xu)), Xu], y, rcond=None)[0]
r2_us = 1 - ((y - pred_us) ** 2).sum() / ((y - y.mean()) ** 2).sum()
R["multi"] = {"r2_all": float(r2_all), "r2_us_only": float(r2_us), "r2_fut_only": uni["fut"]["r2"],
              "std_coef": {k: float(c) for k, c in zip(F, coef_std[1:])}}
print("\n=== 2) 同時に入れた場合(標準化係数=1σ動いた時の影響の大きさ) ===")
for k, c in sorted(R["multi"]["std_coef"].items(), key=lambda kv: -abs(kv[1])):
    print(f"{LABEL[k]:18s} {c:+.3f}")
print(f"説明力: 先物だけ {uni['fut']['r2']*100:.1f}% / 米国指標だけ(先物なし) {r2_us*100:.1f}% / 全部 {r2_all*100:.1f}%")

# ---- 3) profile of big up / big down opens
print("\n=== 3) 大きく上げた/下げた寄り付きの日、前夜に何が起きていたか(平均) ===")
prof = {}
for name, mask in (("大幅高(+0.8%超)", df["gap"] > 0.008), ("小幅高", (df["gap"] > FLAT) & (df["gap"] <= 0.008)),
                   ("横ばい", df["gap"].abs() <= FLAT), ("小幅安", (df["gap"] < -FLAT) & (df["gap"] >= -0.008)),
                   ("大幅安(-0.8%超)", df["gap"] < -0.008)):
    sub = df[mask]
    prof[name] = {"n": int(len(sub)), "gap": float(sub["gap"].mean()), **{k: float(sub[k].mean()) for k in F}}
    print(f"{name:12s} n={len(sub):4d} 寄り{sub['gap'].mean()*100:+.2f}% | 先物{sub['fut'].mean()*100:+.2f}% "
          f"S&P{sub['sp500'].mean()*100:+.2f}% NAS{sub['nasdaq'].mean()*100:+.2f}% SOX{sub['sox'].mean()*100:+.2f}% "
          f"ドル円{sub['usdjpy'].mean()*100:+.2f}% VIX{sub['vix'].mean():+.2f} 金利{sub['us10y'].mean():+.1f}bp")
R["profiles"] = prof

# ---- 4) does the blend help? futures-only vs the dashboard's 0.7/0.3 blend shape
print("\n=== 4) 予想の作り方による方向的中率(過去全期間・同じ判定方法) ===")
hits = {}
cands = {"先物ギャップだけ": df["fut"], "米国指標の回帰だけ": pd.Series(pred_us, index=df.index),
         "現行: 先物0.7+回帰0.3": 0.7 * df["fut"] + 0.3 * pd.Series(pred_us, index=df.index),
         "先物0.9+回帰0.1": 0.9 * df["fut"] + 0.1 * pd.Series(pred_us, index=df.index)}
for name, p in cands.items():
    h = float((sgn(p.to_numpy()) == sgn(df["gap"].to_numpy())).mean())
    mae = float(np.mean(np.abs(p.to_numpy() - df["gap"].to_numpy())))
    hits[name] = {"hit": h, "mae": mae}
    print(f"{name:20s} 的中{h*100:5.1f}%  平均誤差{mae*100:.2f}%")
R["blend"] = hits

# ---- 5) gap vs the rest of the day
c_int = float(np.corrcoef(df["fut"], df["intraday"])[0, 1])
c_gap_int = float(np.corrcoef(df["gap"], df["intraday"])[0, 1])
R["intraday"] = {"corr_fut_intraday": c_int, "corr_gap_intraday": c_gap_int}
print(f"\n=== 5) 寄り付き後(寄り→引け)との関係 ===\n先物ギャップ×寄り後の値動き 相関{c_int:+.2f} / 寄りギャップ×寄り後 相関{c_gap_int:+.2f}")

# ---- 6) live record: why did the dashboard's calls miss?
hist = json.loads((ROOT / "docs" / "history.json").read_text(encoding="utf-8"))
live = []
byd = df.set_index("date")
for h in hist:
    if h["date"] not in byd.index or h.get("expected_move") is None:
        continue
    f = byd.loc[h["date"]]
    # Use the FINAL daily-bar gap: some recorded opens were taken from an in-progress
    # bar after 09:00 and are wrong.
    exp, act = h["expected_move"], float(f["gap"])
    hit = sgn(np.array([exp]))[0] == sgn(np.array([act]))[0]
    if hit:
        why = "的中"
    elif abs(act) <= FLAT:
        why = "実際は横ばい(小動き)"
    elif abs(exp) <= FLAT and np.sign(f["fut"]) == np.sign(act) and abs(f["fut"]) > FLAT:
        why = "先物は当たっていたが、回帰と混ぜて横ばい判定に"
    elif abs(exp) <= FLAT:
        why = "横ばい予想→実際は動いた"
    elif np.sign(f["fut"]) != np.sign(act):
        why = "米国引け後〜寄りまでに流れが変わった(先物も外れ)"
    else:
        why = "方向は合うが幅の判定がずれた"
    live.append({"date": h["date"], "pred": exp, "actual": act, "fut": float(f["fut"]),
                 "sp500": float(f["sp500"]), "sox": float(f["sox"]), "usdjpy": float(f["usdjpy"]), "why": why})
R["live"] = live
print(f"\n=== 6) ダッシュボードの実績{len(live)}日:外れた理由 ===")
from collections import Counter
for k, v in Counter(x["why"] for x in live).most_common():
    print(f"  {k}: {v}日")
fut_hit = np.mean([sgn(np.array([x['fut']]))[0] == sgn(np.array([x['actual']]))[0] for x in live])
R["live_fut_only_hit"] = float(fut_hit)
print(f"同じ{len(live)}日で『先物ギャップだけ』なら的中 {fut_hit*100:.1f}%")
for x in live:
    if x["why"] != "的中":
        print(f"  {x['date']} 予想{x['pred']*100:+.2f}% 実際{x['actual']*100:+.2f}% 先物{x['fut']*100:+.2f}% "
              f"S&P{x['sp500']*100:+.2f}% SOX{x['sox']*100:+.2f}% ドル円{x['usdjpy']*100:+.2f}% → {x['why']}")

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(f"\n→ {OUT.relative_to(ROOT)}")
