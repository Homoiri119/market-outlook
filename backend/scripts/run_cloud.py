"""Cloud entrypoint (GitHub Actions): compute the morning outlook with no database,
send the Discord brief, and generate the static GitHub Pages dashboard under docs/.

Run from the repo root:  python backend/scripts/run_cloud.py
Env:  DISCORD_WEBHOOK_URL (optional; logs the brief if unset)
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sys
import time
from pathlib import Path

# Make `import app...` work when run as a script or module.
BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.clients.discord_notifier import send_discord_message  # noqa: E402
from app.services import analytics, portfolio, selection  # noqa: E402
from app.services.cloud_outlook import compute_stateless_outlook  # noqa: E402
from app.services.morning_outlook import DIRECTION_LABELS_JP  # noqa: E402
from app.services.news import fetch_news, summarize  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_cloud")

DOCS_DIR = REPO_ROOT / "docs"
HISTORY_FILE = DOCS_DIR / "history.json"
TEMPLATE = BACKEND_DIR / "app" / "static" / "dashboard.html"
PORTFOLIO_TEMPLATE = BACKEND_DIR / "app" / "static" / "portfolio.html"

DIRECTION_EMOJI = {
    "STRONG_UP": "🚀",
    "UP": "📈",
    "FLAT": "➡️",
    "DOWN": "📉",
    "STRONG_DOWN": "⚠️",
}


def _jst_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9)))


def _pct(v, d=2):
    return "N/A" if v is None else f"{v * 100:.{d}f}%"


def format_brief(o: dict, accuracy: dict | None = None) -> str:
    # スマホ(Discord)で読みやすいよう、1行を短く・縦積みにする。
    emoji = DIRECTION_EMOJI.get(o["direction"], "")
    label = DIRECTION_LABELS_JP.get(o["direction"], o["direction"])
    lines = [
        f"{emoji} **東京市場 寄り前アウトルック**",
        f"📅 {o['date']}",
        "",
        f"📊 見通し: **{label}**",
        f"　予想寄り {_pct(o['expected_move'])} ・ 信頼度 {o['confidence']:.2f}",
    ]
    if o.get("implied_open_level") and o.get("nikkei_prev_close"):
        lines.append(
            f"　日経 {o['nikkei_prev_close']:,.0f} → 約 {o['implied_open_level']:,.0f}"
        )
    if o.get("expected_open_low") and o.get("expected_open_high"):
        lines.append(
            f"　レンジ {o['expected_open_low']:,.0f}〜{o['expected_open_high']:,.0f}"
        )
    if accuracy and accuracy.get("hit_rate") is not None:
        lines.append("")
        lines.append(
            f"🎯 直近{accuracy['n']}日 的中率 {accuracy['hit_rate'] * 100:.0f}% "
            f"({accuracy['hits']}/{accuracy['n']})"
        )
        lines.append(f"　平均誤差 ±{accuracy['mae'] * 100:.2f}%")
    if o.get("us_detail"):
        lines.append("")
        lines.append(o["us_detail"])
    if o.get("narrative"):
        lines.append("")
        lines.append(o["narrative"])
    return "\n".join(lines)


def _yen(v) -> str:
    return "—" if v is None else f"{v:,}"


DOWN_DAYS = ("DOWN", "STRONG_DOWN")


def _fresh_stocks(payload: dict, outlook: dict) -> tuple[list[dict], str | None]:
    """Drop stocks whose analysis is stale (so we never recommend at an old price).
    The analysis must be from today, and each stock's latest bar must be the last
    Tokyo session (outlook['nikkei_prev_date'])."""
    stocks = payload.get("stocks", [])
    gen = (payload.get("generated_at") or "")[:10]
    if gen and gen != _jst_now().date().isoformat():
        return [], f"⚠️ 個別分析が本日更新されていません(最終 {gen})。古い価格での推奨を避けるため候補なし。"
    expected = outlook.get("nikkei_prev_date")
    if not expected:
        return stocks, None
    fresh = [s for s in stocks if not s.get("data_date") or s["data_date"] >= expected]
    if not fresh:
        latest = max((s.get("data_date") or "") for s in stocks) if stocks else "—"
        return [], f"⚠️ 株価データが前営業日({expected})に未更新(最終 {latest})のため候補なし。"
    stale = len(stocks) - len(fresh)
    return fresh, (f"(株価が未更新の{stale}銘柄は除外)" if stale else None)


def rank_candidates(outlook: dict, limit: int = 5, pool: int = 10) -> tuple[list[dict], str | None]:
    """Rank buy candidates by signal × sector-tailwind × regime × per-stock news,
    excluding stale data and serious negative headlines. Returns (top list, note);
    the list is reused for both the Discord message and the paper-portfolio tracker."""
    f = DOCS_DIR / "analysis.json"
    if not f.exists():
        return [], None
    try:
        payload = json.loads(f.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return [], None
    stocks, note = _fresh_stocks(payload, outlook)
    if not stocks:
        return [], note

    us = outlook.get("us_market")
    d = outlook.get("direction", "")
    regime = 1.0 if d in ("UP", "STRONG_UP") else -1.0 if d in ("DOWN", "STRONG_DOWN") else 0.0
    if outlook.get("vix_regime") == "fear":
        regime -= 1.0
    elif outlook.get("vix_regime") == "elevated":
        regime -= 0.5

    candidates = selection.rank_base(stocks, us, regime)[:pool]

    scored = []
    for s in candidates:
        headlines = fetch_news(s.get("name", ""), limit=4, days=5)
        # Exclude candidates with a serious negative headline (下方修正・不祥事など).
        if any((h.get("polarity", 0) < 0 and h.get("weight", 0) >= 3) for h in headlines):
            continue
        summ = summarize(headlines)
        s["_final"] = s["sel_base"] + max(-3, min(3, summ.get("score", 0)))
        s["_news"] = summ
        scored.append(s)
        time.sleep(0.3)
    scored.sort(key=lambda x: x["_final"], reverse=True)
    return scored[:limit], note


def selection_section(outlook: dict, top: list[dict] | None = None, note: str | None = None,
                      limit: int = 5, pool: int = 10) -> str:
    """Format the ranked buy candidates for Discord under the review's rule:
    buy at the open, stop/target as offsets from the open, exit within 5 sessions."""
    if top is None:
        top, note = rank_candidates(outlook, limit, pool)

    lines = ["", "━━━━━━━━━━━",
             "🎯 **今日の買い候補**",
             "└ シグナル×追い風×地合い×ニュース",
             "└ ルール: 寄りで買い/損切・利確は寄り値から/最長5営業日",
             "━━━━━━━━━━━"]
    if outlook.get("direction") in DOWN_DAYS:
        lines.append("⚠️ **本日は下落予想 → 新規は見送り推奨**(以下は参考)")
    if note:
        lines.append(note)
    if not top:
        lines.append("該当なし(様子見)。")
        return "\n".join(lines)
    circled = "①②③④⑤⑥⑦⑧⑨⑩"
    for i, s in enumerate(top, 1):
        num = circled[i - 1] if i <= len(circled) else f"{i}."
        sent = s["_news"].get("sentiment")
        newstag = "  🟢好材料" if sent == "good" else ("  🔴悪材料" if sent == "bad" else "")
        rr = f" ・ R:R 1:{s['risk_reward']}" if s.get("risk_reward") else ""
        tw = s["sel_comp"]["sector_tw"] * 100
        cp, sl, tp = s.get("current_price"), s.get("stop_loss"), s.get("take_profit")
        lines.append("")
        lines.append(f"{num} **{s['name']}**({s['code']}){newstag}")
        lines.append(f"　🟢寄りで買い(前日終値 {_yen(cp)})")
        if cp and sl and tp:  # offsets from the open (= 1.5×ATR / 2.5×ATR)
            lines.append(f"　✂️損切 寄り値−{round(cp - sl):,} ・ 🎯利確 寄り値+{round(tp - cp):,}")
        lines.append(f"　⏱5営業日後の引けで手仕舞い{rr}")
        lines.append(f"　📊 シグナル{s['sel_comp']['tech']:.0f} ・ 追い風{tw:+.1f}%")
    lines.append("")
    lines.append("→ 寄りで約定したら、その値から損切(逆指値)・利確(指値)を入れる。参考情報です。")
    return "\n".join(lines)


def playbook_section(limit: int = 5) -> str:
    """Read the pre-generated per-stock analysis (docs/analysis.json) and list the
    top BUY candidates with entry / stop-loss / take-profit / trailing levels."""
    f = DOCS_DIR / "analysis.json"
    if not f.exists():
        return ""
    try:
        stocks = json.loads(f.read_text(encoding="utf-8")).get("stocks", [])
    except (ValueError, OSError):
        return ""
    buys = [s for s in stocks if s.get("direction") == "BUY"]
    # Prefer "成行可"(timing ok, not overbought), then strongest score.
    buys.sort(key=lambda s: (0 if s.get("timing") == "ok" else 1, -s.get("score", 0)))
    buys = buys[:limit]

    lines = ["", "📋 **今日のプレイブック(買い候補)**", "※リスクは資金の1%に固定・トレーリングで利を伸ばす"]
    if not buys:
        lines.append("本日の強い買い候補なし(様子見)。")
        return "\n".join(lines)
    for s in buys:
        rr = f"  R:R 1:{s['risk_reward']}" if s.get("risk_reward") else ""
        line = (f"・{s['name']}({s['code']}) 買い {_yen(s.get('current_price'))} / "
                f"損切 {_yen(s.get('stop_loss'))} / 利確 {_yen(s.get('take_profit'))}")
        if s.get("trail_stop") is not None:
            line += f" / トレール {_yen(s['trail_stop'])}"
        line += rr
        lines.append(line)
    lines.append("→ 上のアウトルックの地合い・セクター追い風を確認し、悪材料の無い銘柄から。参考情報です。")
    return "\n".join(lines)


def load_history() -> list[dict]:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            logger.warning("Could not parse existing history.json; starting fresh")
    return []


def update_history(o: dict, actual_gaps: dict[str, float]) -> list[dict]:
    history = load_history()
    # Realized opens for past sessions, always from the final daily bars (this also
    # corrects opens that were once recorded from an in-progress bar after 09:00).
    analytics.backfill_history_actuals(history, actual_gaps, before=o["date"])
    # Upsert today's entry. Today's open is NOT recorded here even if the job runs
    # after 09:00: the in-progress bar is unreliable. It is filled on the next run.
    existing = next((h for h in history if h.get("date") == o["date"]), None)
    entry = existing or {"date": o["date"]}
    entry["direction"] = o["direction"]
    entry["expected_move"] = o["expected_move"]
    # When/how the forecast was made (to audit delays and the futures source later).
    entry["computed_at"] = o.get("computed_at")
    entry["late"] = o.get("late")
    entry["futures_method"] = o.get("futures_method")
    entry["actual_move"] = None
    entry["hit"] = None
    if existing is None:
        history.append(entry)
    history.sort(key=lambda h: h["date"])
    history = history[-120:]  # cap growth
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=0), encoding="utf-8")
    return history


def render_site(o: dict, history: list[dict]) -> None:
    template = TEMPLATE.read_text(encoding="utf-8")
    payload = {
        "outlook": o,
        # dashboard chart expects newest-first (it reverses internally)
        "history": sorted(history, key=lambda h: h["date"], reverse=True),
        "generated_at": _jst_now().strftime("%Y-%m-%d %H:%M JST"),
    }
    inject = "<script>window.__DATA__ = " + json.dumps(payload, ensure_ascii=False) + ";</script>\n</head>"
    html = template.replace("</head>", inject, 1)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "index.html").write_text(html, encoding="utf-8")
    (DOCS_DIR / "outlook.json").write_text(json.dumps(o, ensure_ascii=False, indent=2), encoding="utf-8")
    # Prevent GitHub Pages/Jekyll from ignoring files and mangling the page.
    (DOCS_DIR / ".nojekyll").write_text("", encoding="utf-8")


def track_portfolio(top: list[dict], date: str) -> None:
    """Record today's buy candidates as paper positions, re-price open ones against
    the latest bars, and write docs/portfolio.json + docs/portfolio.html."""
    pdata = portfolio.load(DOCS_DIR)
    portfolio.record_recommendations(pdata, top, date)
    portfolio.update_positions(pdata)
    summary = portfolio.summarize(pdata)
    portfolio.attach_regime(summary["positions"], DOCS_DIR)  # tag recommendation-day 地合い
    summary["generated_at"] = _jst_now().strftime("%Y-%m-%d %H:%M JST")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    # portfolio.json doubles as the page's data source AND the next run's saved state
    # (it carries initial_capital / risk_pct / positions that load() reads back).
    (DOCS_DIR / "portfolio.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    template = PORTFOLIO_TEMPLATE.read_text(encoding="utf-8")
    html = template.replace("</head>", "<script>window.__PF__ = " + json.dumps(summary, ensure_ascii=False) + ";</script>\n</head>", 1)
    (DOCS_DIR / "portfolio.html").write_text(html, encoding="utf-8")
    logger.info("Portfolio: equity ¥%s (%d open / %d closed)",
                f"{summary['equity']:,}", summary["n_open"], summary["n_closed"])


def main() -> int:
    outlook = compute_stateless_outlook()
    actual_gaps = outlook.pop("actual_gaps", {})  # internal; not part of the stored outlook
    logger.info("Outlook: %s %s (conf %.2f)", outlook["direction"], _pct(outlook["expected_move"]), outlook["confidence"])

    history = update_history(outlook, actual_gaps)
    accuracy = analytics.accuracy_summary(history)
    if accuracy.get("hit_rate") is not None:
        logger.info("Accuracy (last %d): hit-rate %.0f%%, MAE %.2f%%",
                    accuracy["n"], accuracy["hit_rate"] * 100, accuracy["mae"] * 100)

    # `or` (not get's default): an unset GitHub secret is passed as an EMPTY string,
    # which would otherwise blank the URL.
    dashboard_url = os.environ.get("DASHBOARD_URL") or "https://homoiri119.github.io/market-outlook/"
    footer = (f"\n\n━━━━━━━━━━━\n🔗 **ダッシュボード**: {dashboard_url}\n"
              "└ アウトルック / 個別銘柄 / ニュース / 💼実績\n└ 戦略検証 / バックテスト / セクター / 🆕IPO")
    top, note = rank_candidates(outlook)
    late = ("⚠️ **配信遅延:寄り付き後の通知です**(GitHubの起動遅れ。予想値は前日までの確定データで計算済み)\n"
            if outlook.get("late") else "")
    # URL at the top too, so it is always in the first message even if Discord splits it.
    message = (f"🔗 {dashboard_url}\n" + late
               + format_brief(outlook, accuracy) + selection_section(outlook, top=top, note=note) + footer)
    sent = send_discord_message(message)
    logger.info("Discord notification sent: %s", sent)

    # Track the recommended candidates as a paper portfolio (best-effort; needs J-Quants).
    try:
        track_portfolio(top, outlook["date"])
    except Exception:
        logger.exception("Portfolio tracking failed (skipped)")

    render_site(outlook, history)
    logger.info("Wrote docs/index.html and docs/history.json (%d points)", len(history))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
