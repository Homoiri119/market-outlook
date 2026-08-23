"""Client for the EDINET disclosure document API (v2).

Docs: https://disclosure.edinet-fsa.go.jp/
Requires an EDINET API subscription key (free registration).
"""

from __future__ import annotations

import datetime as dt
import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

BASE_URL = "https://disclosure.edinet-fsa.go.jp/api/v2"

# Document types that tend to move stock prices and are worth flagging.
NOTABLE_DOC_TYPE_CODES = {
    "140": "四半期報告書",
    "120": "有価証券報告書",
    "180": "臨時報告書",
    "350": "大量保有報告書",
    "360": "大量保有報告書(変更報告書)",
}


class EdinetAuthError(RuntimeError):
    pass


def fetch_documents(date: dt.date) -> list[dict]:
    """Return the list of documents submitted to EDINET on the given date."""
    if not settings.edinet_api_key:
        raise EdinetAuthError("EDINET_API_KEY is not configured")

    resp = httpx.get(
        f"{BASE_URL}/documents.json",
        params={"date": date.isoformat(), "type": 2, "Subscription-Key": settings.edinet_api_key},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("results", []) or []


def fetch_notable_disclosures(date: dt.date, company_names: set[str]) -> list[dict]:
    """Return notable disclosure documents for the given date whose filer name matches
    one of the supplied company names (substring match, case-insensitive)."""
    documents = fetch_documents(date)
    matches = []
    lowered_names = {name.lower(): name for name in company_names}

    for doc in documents:
        doc_type_code = doc.get("docTypeCode")
        if doc_type_code not in NOTABLE_DOC_TYPE_CODES:
            continue
        filer_name = (doc.get("filerName") or "").strip()
        if not filer_name:
            continue
        for lowered, original in lowered_names.items():
            if lowered and (lowered in filer_name.lower() or filer_name.lower() in lowered):
                matches.append(
                    {
                        "doc_id": doc.get("docID"),
                        "edinet_code": doc.get("edinetCode"),
                        "filer_name": filer_name,
                        "matched_company": original,
                        "doc_type": NOTABLE_DOC_TYPE_CODES[doc_type_code],
                        "title": doc.get("docDescription") or NOTABLE_DOC_TYPE_CODES[doc_type_code],
                        "submitted_at": doc.get("submitDateTime"),
                    }
                )
                break

    return matches
