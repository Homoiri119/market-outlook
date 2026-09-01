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
    def base_url(self) -> str:
        # Confirmed: J-Quants Pro V2 (api.jquants-pro.com/v2) with the mail/password
        # token flow. Configurable via JQUANTS_BASE_URL.
        return settings.jquants_base_url or V1_BASE_URL

    @property
    def _has_token_creds(self) -> bool:
        return bool(settings.jquants_mail and settings.jquants_password) or bool(settings.jquants_refresh_token)

    def _fetch_refresh_token(self) -> str:
        if not settings.jquants_mail or not settings.jquants_password:
            raise JQuantsAuthError(
                "JQUANTS_MAIL / JQUANTS_PASSWORD (or JQUANTS_REFRESH_TOKEN) is not configured"
            )
        resp = httpx.post(
            f"{self.base_url}/token/auth_user",
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
            f"{self.base_url}/token/auth_refresh",
            params={"refreshtoken": self._refresh_token},
            timeout=30,
        )
        if resp.status_code == 400:
            # refresh token expired - fetch a new one and retry once
            self._refresh_token = self._fetch_refresh_token()
            resp = httpx.post(
                f"{self.base_url}/token/auth_refresh",
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
        # Confirmed working: V2 x-api-key. Fall back to the V1 token flow if only
        # mail/password are configured.
        if settings.jquants_api_key:
            return {"x-api-key": settings.jquants_api_key}
        if self._has_token_creds:
            return {"Authorization": f"Bearer {self._get_id_token()}"}
        raise JQuantsAuthError("No J-Quants credentials configured (set JQUANTS_API_KEY)")

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
        """Fetch the listed-company master via V2 /equities/master.
        Columns include: Code, CoName, CoNameEn, Mkt/MktNm (market), S17/S17Nm and
        S33/S33Nm (17- and 33-sector code/name), ScaleCat (TOPIX size category)."""
        params = {"date": date.isoformat()} if date else {}
        rows = self._get_paginated("/equities/master", params, "data")
        return pd.DataFrame(rows)

    def fetch_statements(self, code: str) -> list[dict]:
        """Fetch financial statements for a stock (V2 /fins/statements). Returns the
        raw rows (newest last), or [] if the plan does not include this endpoint."""
        try:
            return self._get_paginated("/fins/statements", {"code": code}, "data")
        except Exception:
            logger.info("fins/statements unavailable for %s (plan?)", code, exc_info=True)
            return []

    def fetch_daily_quotes(self, code: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Fetch split-adjusted daily OHLC for a stock via V2 /equities/bars/daily.
        Uses adjusted prices (AdjO/AdjH/AdjL/AdjC). Returns a DataFrame indexed by date."""
        rows = self._get_paginated(
            "/equities/bars/daily",
            {"code": code, "from": start.isoformat(), "to": end.isoformat()},
            "data",
        )
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["Date"]).dt.date
        df = df.rename(columns={"AdjO": "open", "AdjH": "high", "AdjL": "low", "AdjC": "close",
                                "AdjVo": "volume", "MktCap": "mkt_cap"})
        keep = [c for c in ["date", "open", "high", "low", "close", "volume", "mkt_cap"] if c in df.columns]
        df = df[keep].dropna(subset=["close"]).set_index("date").sort_index()
        df["return_pct"] = df["close"].pct_change()
        return df

    def fetch_quotes_by_date(self, date: dt.date) -> list[dict]:
        """Fetch every stock's daily bar for a single date via V2 /equities/bars/daily
        (`date` param). Returns raw rows (each has a `Code`); [] if unavailable.
        Used to snapshot the set of listed codes on a day (for IPO detection)."""
        try:
            return self._get_paginated(
                "/equities/bars/daily", {"date": date.isoformat()}, "data"
            )
        except Exception:
            logger.info("bars/daily by-date unavailable for %s", date, exc_info=True)
            return []

    def fetch_topix_history(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Fetch TOPIX index history via V2 /indices/bars/daily/topix (cols O/H/L/C)."""
        rows = self._get_paginated(
            "/indices/bars/daily/topix",
            {"from": start.isoformat(), "to": end.isoformat()},
            "data",
        )
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["Date"]).dt.date
        df = df.rename(columns={"O": "open", "H": "high", "L": "low", "C": "close"})
        df = df[["date", "close"]].dropna(subset=["close"]).set_index("date").sort_index()
        df["return_pct"] = df["close"].pct_change()
        return df


jquants_client = JQuantsClient()
