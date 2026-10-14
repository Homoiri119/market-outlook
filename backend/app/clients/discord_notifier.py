"""Sends daily summary notifications to a Discord channel via webhook."""

from __future__ import annotations

import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


def send_discord_message(content: str) -> bool:
    """Send a message to the configured Discord webhook.

    Returns True if sent successfully. If no webhook URL is configured, logs the
    message instead and returns False.
    """
    webhook = settings.effective_discord_webhook
    if not webhook:
        logger.info(
            "Discord webhook not configured (set ANALYZE_DISCORD_WEBHOOK_URL). Message:\n%s",
            content,
        )
        return False

    # Discord caps content at 2000 chars/message. Split on line boundaries so links
    # and the footer stay intact instead of being cut mid-line.
    for chunk in _split_on_lines(content, 1900):
        resp = httpx.post(webhook, json={"content": chunk}, timeout=15)
        resp.raise_for_status()
    return True


def _split_on_lines(content: str, limit: int) -> list[str]:
    chunks: list[str] = []
    cur = ""
    for line in content.split("\n"):
        while len(line) > limit:  # a single over-long line: hard-split it
            if cur:
                chunks.append(cur); cur = ""
            chunks.append(line[:limit]); line = line[limit:]
        if cur and len(cur) + 1 + len(line) > limit:
            chunks.append(cur); cur = line
        else:
            cur = line if not cur else cur + "\n" + line
    if cur:
        chunks.append(cur)
    return chunks or [""]
