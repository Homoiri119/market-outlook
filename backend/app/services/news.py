"""Per-company news via Google News RSS + a Japanese good/bad material lexicon.

Uses only the public Google News RSS endpoint (no API key). Headlines are tagged
好材料 / 悪材料 by keyword match and aggregated into a simple sentiment score.
Fetched text is treated strictly as data (never as instructions).
"""

from __future__ import annotations

import datetime as dt
import logging
import xml.etree.ElementTree as ET
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

# Event taxonomy: (category, polarity, materiality weight, keywords). Higher weight =
# more market-moving. Order matters (first match wins), so put high-impact first.
EVENTS = [
    ("業績上方修正", +1, 3, ["上方修正", "上振れ", "業績予想の修正（増"]),
    ("業績下方修正", -1, 3, ["下方修正", "下振れ", "業績予想の修正（減"]),
    ("不祥事・法務", -1, 3, ["不正", "不祥事", "リコール", "訴訟", "提訴", "課徴金", "行政処分",
                        "業務停止", "粉飾", "改ざん", "偽装", "インサイダー", "捜索"]),
    ("最高益・増益", +1, 2, ["最高益", "過去最高", "増益", "増収", "黒字転換", "黒字化", "最終黒字"]),
    ("減益・赤字", -1, 2, ["減益", "減収", "赤字", "最終赤字", "赤字転落", "業績悪化", "特別損失", "減損"]),
    ("株主還元", +1, 2, ["増配", "自社株買い", "復配", "記念配当"]),
    ("減配・無配", -1, 2, ["減配", "無配"]),
    ("受注・提携・M&A", +1, 2, ["大型受注", "受注", "提携", "業務提携", "資本提携", "買収", "出資", "TOB"]),
    ("増資・希薄化", -1, 2, ["公募増資", "希薄化", "新株発行", "第三者割当", "転換社債"]),
    ("格上げ・目標↑", +1, 1, ["格上げ", "目標株価引き上げ", "レーティング引き上げ", "投資判断引き上げ"]),
    ("格下げ・目標↓", -1, 1, ["格下げ", "目標株価引き下げ", "レーティング引き下げ", "投資判断引き下げ"]),
    ("新製品・好材料", +1, 1, ["新製品", "新工場", "開発成功", "承認取得", "特需", "上場来高値"]),
    ("下落・安値", -1, 1, ["急落", "安値", "ストップ安", "急落"]),
]

RSS_URL = "https://news.google.com/rss/search?q={q}&hl=ja&gl=JP&ceid=JP:ja"
_UA = "Mozilla/5.0 (compatible; market-outlook/1.0)"


def classify(title: str) -> dict:
    """Classify a headline into an event category with polarity & materiality weight."""
    for cat, pol, w, kws in EVENTS:
        hit = [k for k in kws if k in title]
        if hit:
            return {"category": cat, "polarity": pol, "weight": w, "keywords": hit,
                    "tag": "good" if pol > 0 else "bad"}
    return {"category": "その他", "polarity": 0, "weight": 0, "keywords": [], "tag": "neutral"}


def fetch_news(query: str, limit: int = 6, days: int = 10) -> list[dict]:
    url = RSS_URL.format(q=quote(query))
    try:
        r = httpx.get(url, timeout=20, headers={"User-Agent": _UA}, follow_redirects=True)
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except Exception:
        logger.info("news fetch failed for %s", query, exc_info=True)
        return []

    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
    items = []
    for it in root.findall(".//item"):
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        pub = (it.findtext("pubDate") or "").strip()
        src_el = it.find("{*}source")
        source = (src_el.text if src_el is not None else "") or ""
        # Google titles are usually "Headline - Source"; strip the trailing source.
        if source and title.endswith(" - " + source):
            title = title[: -(len(source) + 3)]
        when = None
        for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z"):
            try:
                when = dt.datetime.strptime(pub, fmt)
                if when.tzinfo is None:
                    when = when.replace(tzinfo=dt.timezone.utc)
                break
            except ValueError:
                continue
        if when and when < cutoff:
            continue
        if not title or not link.startswith("http"):
            continue
        ev = classify(title)
        items.append({
            "title": title, "link": link, "source": source,
            "published": when.astimezone(dt.timezone(dt.timedelta(hours=9))).strftime("%m/%d %H:%M") if when else "",
            "tag": ev["tag"], "category": ev["category"], "weight": ev["weight"],
            "polarity": ev["polarity"], "keywords": ev["keywords"],
        })
        if len(items) >= limit:
            break
    return items


def summarize(headlines: list[dict]) -> dict:
    good = sum(1 for h in headlines if h["tag"] == "good")
    bad = sum(1 for h in headlines if h["tag"] == "bad")
    # Materiality-weighted score: sum of polarity * weight (bigger events count more).
    weighted = sum(h.get("polarity", 0) * h.get("weight", 0) for h in headlines)
    cats = []
    for h in headlines:
        if h.get("category") and h["category"] != "その他" and h["category"] not in cats:
            cats.append(h["category"])
    if weighted > 0:
        sentiment = "good"
    elif weighted < 0:
        sentiment = "bad"
    else:
        sentiment = "neutral"
    return {"good": good, "bad": bad, "count": len(headlines), "score": weighted,
            "sentiment": sentiment, "categories": cats}
