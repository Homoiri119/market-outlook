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
from app.services import analytics, selection  # noqa: E402
from app.services.cloud_outlook import compute_stateless_outlook  # noqa: E402
from app.services.morning_outlook import DIRECTION_LABELS_JP  # noqa: E402
from app.services.news import fetch_news, summarize  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_cloud")

DOCS_DIR = REPO_ROOT / "docs"
HISTORY_FILE = DOCS_DIR / "history.json"
TEMPLATE = BACKEND_DIR / "app" / "static" / "dashboard.html"

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


def selection_section(outlook: dict, limit: int = 5, pool: int = 10) -> str:
    """Rank buy candidates by signal × sector-tailwind × regime × per-stock news,
    and format the top ones (with entry/SL/TP/trail) for Discord."""
    f = DOCS_DIR / "analysis.json"
    if not f.exists():
        return ""
    try:
        stocks = json.loads(f.read_text(encoding="utf-8")).get("stocks", [])
    except (ValueError, OSError):
        return ""
    if not stocks:
        return ""

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
    top = scored[:limit]

    lines = ["", "━━━━━━━━━━━",
             "🎯 **今日の買い候補**",
             "└ シグナル×追い風×地合い×ニュース",
             "└ 損切=撤退 / 利確=目標 / ﾄﾚｰﾙ=上昇に追随する動く損切",
             "━━━━━━━━━━━"]
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
        lines.append("")
        lines.append(f"{num} **{s['name']}**({s['code']}){newstag}")
        lines.append(f"　🟢買 {_yen(s.get('current_price'))}　🎯利確 {_yen(s.get('take_profit'))}")
        lines.append(f"　✂️損切 {_yen(s.get('stop_loss'))}　🔻ﾄﾚｰﾙ {_yen(s.get('trail_stop'))}")
        lines.append(f"　📊 シグナル{s['sel_comp']['tech']:.0f} ・ 追い風{tw:+.1f}%{rr}")
    lines.append("")
    lines.append("→ 地合いが弱い日はサイズ縮小か見送り。参考情報です。")
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
    # Backfill realized opens for past predictions now that they are known.
    analytics.backfill_history_actuals(history, actual_gaps)
    # Upsert today's entry, preserving any already-known actual.
    existing = next((h for h in history if h.get("date") == o["date"]), None)
    entry = existing or {"date": o["date"]}
    entry["direction"] = o["direction"]
    entry["expected_move"] = o["expected_move"]
    entry.setdefault("actual_move", actual_gaps.get(o["date"]))
    entry.setdefault("hit", None)
    if entry.get("actual_move") is not None and entry.get("hit") is None:
        entry["hit"] = analytics.directional_hit(o["expected_move"], entry["actual_move"])
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


def main() -> int:
    outlook = compute_stateless_outlook()
    actual_gaps = outlook.pop("actual_gaps", {})  # internal; not part of the stored outlook
    logger.info("Outlook: %s %s (conf %.2f)", outlook["direction"], _pct(outlook["expected_move"]), outlook["confidence"])

    history = update_history(outlook, actual_gaps)
    accuracy = analytics.accuracy_summary(history)
    if accuracy.get("hit_rate") is not None:
        logger.info("Accuracy (last %d): hit-rate %.0f%%, MAE %.2f%%",
                    accuracy["n"], accuracy["hit_rate"] * 100, accuracy["mae"] * 100)

    dashboard_url = os.environ.get("DASHBOARD_URL", "https://homoiri119.github.io/market-outlook/")
    footer = (f"\n\n━━━━━━━━━━━\n🔗 **ダッシュボード**\n{dashboard_url}\n"
              "└ アウトルック / 個別銘柄 / ニュース\n└ 戦略検証 / バックテスト / セクター")
    message = format_brief(outlook, accuracy) + selection_section(outlook) + footer
    sent = send_discord_message(message)
    logger.info("Discord notification sent: %s", sent)

    render_site(outlook, history)
    logger.info("Wrote docs/index.html and docs/history.json (%d points)", len(history))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
