"""Small JSON/text HTTP helper over httpx (the client openai and tavily-python already use).

Turns 429 into RateLimitedError (carrying Retry-After) and other non-2xx into HTTPStatusError, so the shared retry
policy in retry.py treats every API the same way. Keys are passed per request and never logged.
"""

from __future__ import annotations

from typing import Any

import httpx

from ..retry import HTTPStatusError, RateLimitedError, parse_retry_after

USER_AGENT = "oncology-update-briefs/0.2 (research agent)"


def _check(resp: httpx.Response, label: str) -> None:
    if resp.status_code == 429:
        raise RateLimitedError(f"{label}: 429 Too Many Requests", parse_retry_after(resp.headers.get("retry-after")))
    if resp.status_code >= 400:
        body = (resp.text or "")[:200].replace("\n", " ")
        raise HTTPStatusError(f"{label}: HTTP {resp.status_code} {body}", resp.status_code)


class HTTP:
    def __init__(self, timeout_s: float = 30.0, client: httpx.AsyncClient | None = None):
        self._client = client
        self._loop: object | None = None
        self.timeout_s = timeout_s

    def client(self) -> httpx.AsyncClient:
        # Created lazily, once per event loop: the CLI runs prepare and run in separate asyncio.run() loops, and an
        # httpx client must not outlive the loop it was created in.
        import asyncio

        loop = asyncio.get_running_loop()
        if self._client is None or self._client.is_closed or (self._loop is not None and self._loop is not loop):
            self._client = httpx.AsyncClient(timeout=self.timeout_s, headers={"User-Agent": USER_AGENT},
                                             follow_redirects=True)
            self._loop = loop
        return self._client

    async def get_json(self, url: str, params: dict | None = None, label: str = "GET") -> Any:
        resp = await self.client().get(url, params=params)
        if resp.status_code == 404:
            return None
        _check(resp, label)
        return resp.json()

    async def get_text(self, url: str, params: dict | None = None, label: str = "GET") -> str | None:
        resp = await self.client().get(url, params=params)
        if resp.status_code == 404:
            return None
        _check(resp, label)
        return resp.text

    async def post_json(self, url: str, payload: dict, headers: dict | None = None, label: str = "POST") -> Any:
        resp = await self.client().post(url, json=payload, headers=headers)
        _check(resp, label)
        return resp.json()

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
