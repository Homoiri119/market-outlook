"""Index crash-buy rulebook v1 (reports/index_crash_rulebook.md).

Three rules on the Nikkei 225 close; capital is split into 3 units (max 3 = 100%):
  T  close > 200-day MA                         -> 3 units while on
  B  close/MA25 - 1 <= -10% (once per episode)  -> 1 unit; exit the day after the close
                                                  is back >= MA25, or after 20 sessions
  C  close/250-day high - 1 <= -15% (once)      -> 1 unit; exit after 120 sessions
Target units = max(3 if T on else 0, open B/C positions), capped at 3.

Decisions use close d; orders execute at close d+1 (引成). The state is path dependent
(re-arming, holding clocks), so it is rebuilt each morning by replaying the rules over
the full history: no saved state can drift from the rulebook.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

UNITS = 3
MA_TREND = 200
B_DEV = -0.10
B_MAX_HOLD = 20
C_DD = -0.15
C_REARM = -0.05
C_HOLD = 120
WARMUP = 260
HISTORY_START = dt.date(1995, 1, 1)
LABELS = {"B": "B 乖離−10%", "C": "C 高値−15%"}


@dataclass
class Position:
    kind: str
    entry_i: int


@dataclass
class Plan:
    """What to do at the next close, given data through close `asof_i`."""
    asof_i: int
    trend_on: bool
    held: list[Position]                     # crash positions held after `asof_i`'s close
    exits: list[Position] = field(default_factory=list)
    entries: list[str] = field(default_factory=list)
    target: int = 0
    current: int = 0


def _indicators(close: pd.Series) -> dict[str, np.ndarray]:
    c = close.to_numpy(dtype=float)
    s = pd.Series(c)
    return {
        "c": c,
        "ma25": s.rolling(25).mean().to_numpy(),
        "ma200": s.rolling(MA_TREND).mean().to_numpy(),
        "hi250": s.rolling(250).max().to_numpy(),
    }


def _plan_next(x: dict, i: int, held: list[Position], arm: dict[str, bool],
               current: int) -> tuple[Plan, dict[str, bool]]:
    """Decide at close i what happens at close i+1. Returns the plan and next arm flags."""
    c, ma25, ma200, hi250 = x["c"], x["ma25"], x["ma200"], x["hi250"]
    nxt = i + 1
    exits = []
    for p in held:
        age = nxt - p.entry_i
        if p.kind == "B" and ((i > p.entry_i and c[i] >= ma25[i]) or age >= B_MAX_HOLD):
            exits.append(p)
        elif p.kind == "C" and age >= C_HOLD:
            exits.append(p)
    arm = dict(arm)
    entries = []
    if i >= WARMUP:
        dev = c[i] / ma25[i] - 1
        dd = c[i] / hi250[i] - 1
        if arm["B"] and dev <= B_DEV:
            entries.append("B"); arm["B"] = False
        elif not arm["B"] and dev >= 0:
            arm["B"] = True
        if arm["C"] and dd <= C_DD:
            entries.append("C"); arm["C"] = False
        elif not arm["C"] and dd >= C_REARM:
            arm["C"] = True
    remaining = len(held) - len(exits)
    entries = entries[: max(UNITS - remaining, 0)]  # no room -> signal is skipped
    trend_on = bool(i >= MA_TREND - 1 and c[i] > ma200[i])
    crash = remaining + len(entries)
    target = min(UNITS, max(UNITS if trend_on else 0, crash))
    return Plan(i, trend_on, held, exits, entries, target, current), arm


def replay(close: pd.Series) -> tuple[pd.Series, Plan, list[dict]]:
    """Replay the rules. Returns (units held after each close, plan for the next close,
    trade log)."""
    x = _indicators(close)
    n = len(close)
    held: list[Position] = []
    arm = {"B": True, "C": True}
    units = np.zeros(n, dtype=int)
    log: list[dict] = []
    plan = None
    for i in range(n):
        if plan is not None:  # execute yesterday's plan at today's close
            for p in plan.exits:
                held.remove(p)
                log.append({"kind": p.kind, "entry": close.index[p.entry_i].date().isoformat(),
                            "exit": close.index[i].date().isoformat(),
                            "ret": x["c"][i] / x["c"][p.entry_i] - 1})
            for k in plan.entries:
                held.append(Position(k, i))
            units[i] = plan.target
        plan, arm = _plan_next(x, i, list(held), arm, int(units[i]))
    return pd.Series(units, index=close.index), plan, log


def fetch_close(today: dt.date) -> pd.Series:
    """Nikkei 225 closes up to the last completed session (drops today's in-progress bar)."""
    import yfinance as yf  # local import: heavy, network

    data = yf.Ticker("^N225").history(start=HISTORY_START.isoformat(),
                                      end=(today + dt.timedelta(days=1)).isoformat(),
                                      interval="1d", auto_adjust=False)
    if data.empty:
        return pd.Series(dtype=float)
    s = pd.Series(data["Close"].to_numpy(dtype=float),
                  index=pd.to_datetime([d.date() for d in data.index])).dropna()
    s = s[s > 0]
    return s[s.index.date < today]


def evaluate(close: pd.Series) -> dict:
    """Current rule state for the morning brief and docs/crash_rules.json."""
    units, plan, log = replay(close)
    x = _indicators(close)
    i = plan.asof_i
    c = x["c"]

    def pos_info(p: Position) -> dict:
        age = i - p.entry_i  # sessions held so far (entry day = 0)
        d = {"kind": p.kind, "entry_date": close.index[p.entry_i].date().isoformat(),
             "entry_close": round(float(c[p.entry_i]), 2), "sessions": int(age),
             "ret": float(c[i] / c[p.entry_i] - 1)}
        hold = B_MAX_HOLD if p.kind == "B" else C_HOLD
        d["sessions_left"] = int(hold - age)
        return d

    return {
        "asof": close.index[i].date().isoformat(),
        "close": round(float(c[i]), 2),
        "ma25_dev": float(c[i] / x["ma25"][i] - 1),
        "ma200_dev": float(c[i] / x["ma200"][i] - 1),
        "dd250": float(c[i] / x["hi250"][i] - 1),
        "day_ret": float(c[i] / c[i - 1] - 1),
        "trend_on": plan.trend_on,
        "trend_was_on": bool(c[i - 1] > x["ma200"][i - 1]),
        "positions": [pos_info(p) for p in plan.held],
        "exits": [{"kind": p.kind, "entry_date": close.index[p.entry_i].date().isoformat()}
                  for p in plan.exits],
        "entries": plan.entries,
        "current_units": int(units.iloc[i]),
        "target_units": int(plan.target),
        "recent_trades": log[-10:],
    }


def compute_state(today: dt.date | None = None) -> dict | None:
    today = today or dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).date()
    close = fetch_close(today)
    if len(close) < WARMUP + 1:
        logger.warning("Not enough Nikkei history for crash rules (%d bars)", len(close))
        return None
    return evaluate(close)


def format_section(st: dict | None) -> str:
    lines = ["", "━━━━━━━━━━━", "🛡 **指数ルール(暴落買い v1)**",
             "└ 日経225ETF・資金3ユニット・本日の引成で調整", "━━━━━━━━━━━"]
    if not st:
        lines.append("⚠️ 日経平均のデータが取れず、判定できませんでした。")
        return "\n".join(lines)
    diff = st["target_units"] - st["current_units"]
    if diff > 0:
        lines.append(f"🔔 **本日の引けで {diff}ユニット買い**({st['current_units']}→{st['target_units']})")
    elif diff < 0:
        lines.append(f"🔔 **本日の引けで {-diff}ユニット売り**({st['current_units']}→{st['target_units']})")
    else:
        lines.append(f"✅ 売買なし(保有 {st['current_units']}/{UNITS}ユニット)")
    reasons = []
    if st["trend_on"] != st.get("trend_was_on", st["trend_on"]):
        reasons.append("T 200日線を" + ("上抜け(オン)" if st["trend_on"] else "割り込み(オフ)"))
    reasons += [f"{LABELS[k]} 点灯" for k in st["entries"]]
    reasons += [f"{LABELS[e['kind']]}({e['entry_date']}買い)手仕舞い" for e in st["exits"]]
    if reasons:
        lines.append("　" + " / ".join(reasons))
    lines.append(f"　日経 {st['close']:,.0f}({st['asof']}・前日比 {st['day_ret'] * 100:+.1f}%)")
    lines.append(f"　T 200日線 {st['ma200_dev'] * 100:+.1f}% → {'オン' if st['trend_on'] else 'オフ'}")
    lines.append(f"　B 25日線乖離 {st['ma25_dev'] * 100:+.1f}%(基準 −10%)")
    lines.append(f"　C 250日高値から {st['dd250'] * 100:+.1f}%(基準 −15%)")
    exiting = {(e["kind"], e["entry_date"]) for e in st["exits"]}
    for p in st["positions"]:
        if (p["kind"], p["entry_date"]) in exiting:
            continue
        lines.append(f"　保有 {LABELS[p['kind']]}:{p['entry_date']}買い {p['ret'] * 100:+.1f}%"
                     f"・手仕舞いまで最長{p['sessions_left']}営業日(本日含む)")
    lines.append("→ ルールの状態の通知です(投資助言ではありません)。")
    return "\n".join(lines)
