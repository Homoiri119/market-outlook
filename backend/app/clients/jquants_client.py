"""Client for the J-Quants API (https://jpx-jquants.com/).

Handles authentication (refresh token -> id token) and provides helpers for
fetching daily stock quotes and TOPIX index values.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import httpx
import pandas as pd

from app.config import settings

logger = logging.getLogger(__name__)

BASE_URL = "https://api.jquants.com/v1"

# J-Quants index code for TOPIX
TOPIX_CODE = "0000"


class JQuantsAuthError(RuntimeError):
    pass


class JQuantsClient:
    def __init__(self) -> None:
        self._id_token: str | None = None
        self._id_token_expiry: dt.datetime | None = None
        self._refresh_token: str | None = settings.jquants_refresh_token or None

    def _fetch_refresh_token(self) -> str:
        if not settings.jquants_mail or not settings.jquants_password:
            raise JQuantsAuthError(
                "JQUANTS_MAIL / JQUANTS_PASSWORD (or JQUANTS_REFRESH_TOKEN) is not configured"
            )
        resp = httpx.post(
            f"{BASE_URL}/token/auth_user",
            json={"mailaddress": settings.jquants_mail, "password": settings.jquants_password},
            timeout=30,
        )
        resp.raise_for_status()
        token = resp.json().get("refreshToken")
        if not token:
            raise JQuantsAuthError("Failed to obtain J-Quants refresh token")
        return token

    def _refresh_id_token(self) -> str:
        if self._refresh_token is None:
            self._refresh_token = self._fetch_refresh_token()

        resp = httpx.post(
            f"{BASE_URL}/token/auth_refresh",
            params={"refreshtoken": self._refresh_token},
            timeout=30,
        )
        if resp.status_code == 400:
            # refresh token expired - fetch a new one and retry once
            self._refresh_token = self._fetch_refresh_token()
            resp = httpx.post(
                f"{BASE_URL}/token/auth_refresh",
                params={"refreshtoken": self._refresh_token},
                timeout=30,
            )
        resp.raise_for_status()
        id_token = resp.json().get("idToken")
        if not id_token:
            raise JQuantsAuthError("Failed to obtain J-Quants id token")

        self._id_token = id_token
        self._id_token_expiry = dt.datetime.now() + dt.timedelta(hours=23)
        return id_token

    def _get_id_token(self) -> str:
        if self._id_token and self._id_token_expiry and dt.datetime.now() < self._id_token_expiry:
            return self._id_token
        return self._refresh_id_token()

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        id_token = self._get_id_token()
        resp = httpx.get(
            f"{BASE_URL}{path}",
            params=params,
            headers={"Authorization": f"Bearer {id_token}"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def fetch_daily_quotes(self, code: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Fetch daily OHLC quotes for a stock code. Returns a DataFrame indexed by date."""
        data = self._get(
            "/prices/daily_quotes",
            params={"code": code, "from": start.isoformat(), "to": end.isoformat()},
        )
        quotes = data.get("daily_quotes", [])
        if not quotes:
            return pd.DataFrame()

        df = pd.DataFrame(quotes)
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        df = df.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
            }
        )
        df = df[["date", "open", "high", "low", "close"]].dropna(subset=["close"])
        df = df.set_index("date").sort_index()
        df["return_pct"] = df["close"].pct_change()
        return df

    def fetch_topix_history(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Fetch TOPIX index history. Returns a DataFrame indexed by date with close/return_pct."""
        data = self._get(
            "/indices",
            params={"code": TOPIX_CODE, "from": start.isoformat(), "to": end.isoformat()},
        )
        rows = data.get("indices", [])
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        df = df.rename(columns={"Date": "date", "Close": "close"})
        df = df[["date", "close"]].dropna(subset=["close"])
        df = df.set_index("date").sort_index()
        df["return_pct"] = df["close"].pct_change()
        return df


jquants_client = JQuantsClient()
