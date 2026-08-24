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

GOOD = [
    "最高益", "過去最高", "増益", "増収", "上方修正", "上振れ", "増配", "復配", "自社株買い",
    "受注", "提携", "業務提携", "資本提携", "買収", "黒字化", "黒字転換", "上場来高値", "最高値",
    "格上げ", "上げ", "好調", "回復", "新製品", "大型受注", "採用", "特需",
]
BAD = [
    "減益", "減収", "下方修正", "下振れ", "減配", "無配", "赤字", "赤字転落", "最終赤字",
    "不正", "不祥事", "リコール", "延期", "訴訟", "提訴", "課徴金", "行政処分", "業務停止",
    "格下げ", "急落", "安値", "下落", "特別損失", "減損", "希薄化", "公募増資", "内部告発",
    "業績悪化", "リストラ", "人員削減",
]

RSS_URL = "https://news.google.com/rss/search?q={q}&hl=ja&gl=JP&ceid=JP:ja"
_UA = "Mozilla/5.0 (compatible; market-outlook/1.0)"


def tag_headline(title: str) -> tuple[str, list[str]]:
    g = [k for k in GOOD if k in title]
    b = [k for k in BAD if k in title]
    if g and not b:
        return "good", g
    if b and not g:
        return "bad", b
    if g and b:
        return "mixed", g + b
    return "neutral", []


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
        tag, kws = tag_headline(title)
        items.append({
            "title": title, "link": link, "source": source,
            "published": when.astimezone(dt.timezone(dt.timedelta(hours=9))).strftime("%m/%d %H:%M") if when else "",
            "tag": tag, "keywords": kws,
        })
        if len(items) >= limit:
            break
    return items


def summarize(headlines: list[dict]) -> dict:
    good = sum(1 for h in headlines if h["tag"] == "good")
    bad = sum(1 for h in headlines if h["tag"] == "bad")
    score = good - bad
    if score > 0:
        sentiment = "good"
    elif score < 0:
        sentiment = "bad"
    else:
        sentiment = "neutral"
    return {"good": good, "bad": bad, "count": len(headlines), "score": score, "sentiment": sentiment}
