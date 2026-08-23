"""Client for the J-Quants API (https://jpx-jquants.com/).

Supports two auth modes:
  * V2 "API Key Method" (preferred): send the key in the `x-api-key` header. No
    token exchange, no expiry. Enabled when JQUANTS_API_KEY is set.
  * V1 token flow (fallback): mailaddress/password -> refreshToken -> idToken
    (Bearer). Used when no API key is configured.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import httpx
import pandas as pd

from app.config import settings

logger = logging.getLogger(__name__)

V1_BASE_URL = "https://api.jquants.com/v1"

# J-Quants index code for TOPIX
TOPIX_CODE = "0000"


class JQuantsAuthError(RuntimeError):
    pass


class JQuantsClient:
    def __init__(self) -> None:
        self._id_token: str | None = None
        self._id_token_expiry: dt.datetime | None = None
        self._refresh_token: str | None = settings.jquants_refresh_token or None

    @property
    def _use_v2(self) -> bool:
        return bool(settings.jquants_api_key)

    @property
    def base_url(self) -> str:
        return settings.jquants_base_url if self._use_v2 else V1_BASE_URL

    def _fetch_refresh_token(self) -> str:
        if not settings.jquants_mail or not settings.jquants_password:
            raise JQuantsAuthError(
                "JQUANTS_MAIL / JQUANTS_PASSWORD (or JQUANTS_REFRESH_TOKEN) is not configured"
            )
        resp = httpx.post(
            f"{V1_BASE_URL}/token/auth_user",
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
            f"{V1_BASE_URL}/token/auth_refresh",
            params={"refreshtoken": self._refresh_token},
            timeout=30,
        )
        if resp.status_code == 400:
            # refresh token expired - fetch a new one and retry once
            self._refresh_token = self._fetch_refresh_token()
            resp = httpx.post(
                f"{V1_BASE_URL}/token/auth_refresh",
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

    def _headers(self) -> dict[str, str]:
        if self._use_v2:
            return {"x-api-key": settings.jquants_api_key}
        return {"Authorization": f"Bearer {self._get_id_token()}"}

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        resp = httpx.get(
            f"{self.base_url}{path}",
            params=params,
            headers=self._headers(),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def _get_paginated(self, path: str, params: dict[str, Any], key: str) -> list[dict]:
        """GET that follows J-Quants `pagination_key` and concatenates `key` arrays."""
        rows: list[dict] = []
        p = dict(params)
        for _ in range(200):  # safety bound
            data = self._get(path, params=p)
            rows.extend(data.get(key, []))
            token = data.get("pagination_key")
            if not token:
                break
            p["pagination_key"] = token
        return rows

    def fetch_listed_info(self, date: dt.date | None = None) -> pd.DataFrame:
        """Fetch the listed-company master (code, name, market and sector codes)."""
        params = {"date": date.isoformat()} if date else {}
        rows = self._get_paginated("/listed/info", params, "info")
        return pd.DataFrame(rows)

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
