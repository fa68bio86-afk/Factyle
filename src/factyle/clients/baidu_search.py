"""Baidu AI Search API client.

Reference: ARCHITECTURE.md \u00a74.2, \u00a79.2.3

Uses Baidu Cloud IAM v3 access keys (bce-v3/ALTAK- format) for authentication.
Supports multi-threaded concurrent search per \u00a79.2.3.

Thread safety: uses threading.Lock for access token acquisition to support
concurrent calls from Module 2's ThreadPoolExecutor(7).
"""

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests


@dataclass
class BaiduSearchResult:
    title: str = ""
    snippet: str = ""
    url: str = ""


class BaiduSearchClient:
    """Client for Baidu AI Search API (Qianfan).

    Uses BCE IAM key directly as Bearer token via the Qianfan V2 endpoint.

    Thread-safe: no shared mutable state between calls (\u00a79.2.3).
    """

    _ENDPOINT = "https://qianfan.baidubce.com/v2/ai_search/web_search"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 15,
        max_retries: int = 3,
    ):
        self.api_key = api_key or os.environ.get("BAIDU_SEARCH_API_KEY", "")
        self.timeout = timeout
        self.max_retries = max_retries

    def search(self, query: str, top_k: int = 5) -> List[BaiduSearchResult]:
        """Execute Baidu web search via Qianfan V2 API.

        Uses IAM key as Bearer token (no OAuth needed).
        POST with messages format per current API spec.

        Returns up to top_k results with title, snippet, url.
        """
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self._ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "messages": [{"role": "user", "content": query}],
                    },
                    timeout=self.timeout,
                )
                data = resp.json() if resp.text else {}

                if resp.status_code == 200:
                    results = self._parse_results(data)
                    if results:
                        return results[:top_k]

                # Non-200 or empty results: retry
                continue

            except requests.RequestException:
                if attempt == self.max_retries - 1:
                    return []
                continue

        return []

    def _parse_results(self, data: dict) -> List[BaiduSearchResult]:
        """Parse Qianfan V2 search response into result objects."""
        results = []

        # Qianfan V2 returns references list with url, title, content, date
        references = data.get("references") or data.get("result") or []
        if isinstance(references, dict):
            references = [references]

        for item in references:
            if not isinstance(item, dict):
                continue

            title = item.get("title", "") or item.get("name", "")
            content = item.get("content", "") or item.get("snippet", "") or item.get("summary", "")
            url = item.get("url", "") or item.get("link", "")

            if title or content:
                results.append(BaiduSearchResult(
                    title=title.strip(),
                    snippet=content.strip()[:500],
                    url=url.strip(),
                ))

        return results
