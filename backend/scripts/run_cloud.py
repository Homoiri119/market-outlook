"""Cloud entrypoint (GitHub Actions): compute the morning outlook with no database,
send the Discord brief, and generate the static GitHub Pages dashboard under docs/.

Run from the repo root:  python backend/scripts/run_cloud.py
Env:  DISCORD_WEBHOOK_URL (optional; logs the brief if unset)
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from pathlib import Path

# Make `import app...` work when run as a script or module.
BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.clients.discord_notifier import send_discord_message  # noqa: E402
from app.services.cloud_outlook import compute_stateless_outlook  # noqa: E402
from app.services.morning_outlook import DIRECTION_LABELS_JP  # noqa: E402

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


def format_brief(o: dict) -> str:
    emoji = DIRECTION_EMOJI.get(o["direction"], "")
    label = DIRECTION_LABELS_JP.get(o["direction"], o["direction"])
    lines = [
        f"{emoji} **東京市場 寄り付き前アウトルック ({o['date']})**",
        "",
        f"見通し: **{label}**(予想寄り付き {_pct(o['expected_move'])}, 信頼度 {o['confidence']:.2f})",
    ]
    if o.get("implied_open_level") and o.get("nikkei_prev_close"):
        lines.append(
            f"日経225: 前日終値 {o['nikkei_prev_close']:,.0f} → 予想寄り {o['implied_open_level']:,.0f} 近辺"
        )
    if o.get("us_detail"):
        lines.append(o["us_detail"])
    if o.get("narrative"):
        lines.append("")
        lines.append(o["narrative"])
    return "\n".join(lines)


def load_history() -> list[dict]:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            logger.warning("Could not parse existing history.json; starting fresh")
    return []


def update_history(o: dict) -> list[dict]:
    history = [h for h in load_history() if h.get("date") != o["date"]]
    history.append({"date": o["date"], "direction": o["direction"], "expected_move": o["expected_move"]})
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
    logger.info("Outlook: %s %s (conf %.2f)", outlook["direction"], _pct(outlook["expected_move"]), outlook["confidence"])

    sent = send_discord_message(format_brief(outlook))
    logger.info("Discord notification sent: %s", sent)

    history = update_history(outlook)
    render_site(outlook, history)
    logger.info("Wrote docs/index.html and docs/history.json (%d points)", len(history))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
