"""Retry with exponential backoff + jitter for transient API errors, with a per-attempt timeout.

A 429 that carries Retry-After (RateLimitedError.retry_after, or an httpx/openai response header) waits at least that
long, capped at retry_after_max."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")

_TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class RateLimitedError(Exception):
    """HTTP 429 from an API. retry_after is the server's Retry-After in seconds, if it sent one."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class HTTPStatusError(Exception):
    """Non-2xx response from one of our own HTTP clients (Tavily, PubMed, ClinicalTrials.gov, Crossref, openFDA)."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


def parse_retry_after(value: str | None) -> float | None:
    """Retry-After is seconds or an HTTP date; we use the seconds form and ignore dates (backoff still applies)."""
    try:
        return max(0.0, float(str(value).strip())) if value not in (None, "") else None
    except ValueError:
        return None


def retry_after_of(exc: BaseException) -> float | None:
    ra = getattr(exc, "retry_after", None)
    if ra is not None:
        return float(ra)
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if headers is not None and getattr(resp, "status_code", None) == 429:
        try:
            return parse_retry_after(headers.get("retry-after"))
        except Exception:  # noqa: BLE001 - header access must never break retry handling
            return None
    return None


def is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError, RateLimitedError)):
        return True
    if isinstance(exc, HTTPStatusError):
        return exc.status_code in _TRANSIENT_STATUS
    try:
        import openai

        if isinstance(
            exc,
            (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError, openai.InternalServerError),
        ):
            return True
        if isinstance(exc, openai.APIStatusError):
            return exc.status_code in _TRANSIENT_STATUS
    except ImportError:  # pragma: no cover
        pass
    try:
        import httpx

        if isinstance(exc, httpx.TransportError):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in _TRANSIENT_STATUS
    except ImportError:  # pragma: no cover
        pass
    name = type(exc).__name__
    return "Timeout" in name or "Connection" in name


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    timeout: float | None = None,
    transient: Callable[[BaseException], bool] = is_transient,
    on_retry: Callable[[BaseException, int, float], None] | None = None,
    retry_after_max: float = 60.0,
) -> T:
    """Call fn() up to `attempts` times. Non-transient errors are raised immediately."""
    attempt = 0
    while True:
        attempt += 1
        try:
            coro = fn()
            return await (asyncio.wait_for(coro, timeout) if timeout else coro)
        except Exception as exc:
            if attempt >= attempts or not transient(exc):
                raise
            delay = min(max_delay, base_delay * 2 ** (attempt - 1))
            delay = delay * (0.5 + random.random() / 2) if delay > 0 else 0
            ra = retry_after_of(exc)
            if ra is not None:
                delay = max(delay, min(ra, retry_after_max))
            if on_retry:
                on_retry(exc, attempt, delay)
            await asyncio.sleep(delay)
