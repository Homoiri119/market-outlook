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

    # Discord message content is limited to 2000 characters per message.
    chunks = [content[i : i + 1900] for i in range(0, len(content), 1900)] or [""]
    for chunk in chunks:
        resp = httpx.post(webhook, json={"content": chunk}, timeout=15)
        resp.raise_for_status()
    return True
