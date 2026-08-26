"""Daily buy-candidate selection: combine the technical signal, sector tailwind,
market regime (and, in the caller, per-stock news) into one ranked score.

Pure functions only (no network / no DB). The caller supplies the analysis stocks,
the US-market struct (for sector tailwind), and a regime bump.
"""

from __future__ import annotations


def _blend(*vals) -> float:
    xs = [v for v in vals if v is not None]
    return sum(xs) / len(xs) if xs else 0.0


def s17_tailwind(m: dict | None) -> dict[str, float]:
    """Overnight tailwind score per S17 sector name, from the US-market struct."""
    if not m:
        return {}
    sp = m.get("sp500_return"); ndx = m.get("nasdaq_return"); fx = m.get("usdjpy_return")
    sox = m.get("sox_return"); wti = m.get("wti_return"); bp = m.get("us10y_change_bp")
    rates = (bp / 2000.0) if bp is not None else None
    inv_rates = (-rates) if rates is not None else None
    return {
        "電機・精密": _blend(sox, ndx),
        "情報通信・サービスその他": _blend(ndx, inv_rates),
        "自動車・輸送機": _blend(fx, sp),
        "機械": _blend(sp, fx),
        "銀行": rates if rates is not None else 0.0,
        "金融（除く銀行）": _blend(rates, sp),
        "不動産": inv_rates if inv_rates is not None else 0.0,
        "エネルギー資源": wti if wti is not None else 0.0,
        "商社・卸売": _blend(wti, sp),
        "鉄鋼・非鉄": _blend(sp, wti),
        "素材・化学": sp if sp is not None else 0.0,
        "医薬品": (-0.3 * sp) if sp is not None else 0.0,
        "食品": (-0.3 * sp) if sp is not None else 0.0,
        "小売": (0.5 * sp) if sp is not None else 0.0,
        "電力・ガス": inv_rates if inv_rates is not None else 0.0,
        "運輸・物流": _blend(sp, (-wti) if wti is not None else None),
        "建設・資材": sp if sp is not None else 0.0,
    }


def _norm(s: str) -> str:
    return (s or "").replace("（", "(").replace("）", ")").replace(" ", "")


def lookup_tw(tailwind: dict[str, float], sector: str | None) -> float:
    if not sector:
        return 0.0
    if sector in tailwind:
        return tailwind[sector]
    n = _norm(sector)
    for k, v in tailwind.items():
        if _norm(k) == n:
            return v
    return 0.0


def base_score(stock: dict, tailwind: dict[str, float], regime_pts: float) -> tuple[float, dict]:
    """Score = technical score + sector tailwind + regime bump (news added later)."""
    tech = float(stock.get("score") or 0)
    tw = lookup_tw(tailwind, stock.get("sector"))
    sector_pts = tw * 250.0  # +0.4% tailwind ≈ +1.0 pt
    total = tech + sector_pts + regime_pts
    return total, {"tech": tech, "sector_tw": tw, "sector_pts": sector_pts, "regime": regime_pts}


def rank_base(stocks: list[dict], us_market: dict | None, regime_pts: float,
              directions=("BUY", "WEAK_BUY")) -> list[dict]:
    """Return candidates (direction in `directions`) with a base score, sorted desc."""
    tw = s17_tailwind(us_market)
    out = []
    for s in stocks:
        if s.get("direction") not in directions:
            continue
        total, comp = base_score(s, tw, regime_pts)
        out.append({**s, "sel_base": total, "sel_comp": comp})
    out.sort(key=lambda x: x["sel_base"], reverse=True)
    return out
