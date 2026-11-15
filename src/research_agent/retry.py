"""Retry with exponential backoff + jitter for transient API errors, with a per-attempt timeout."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")

_TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True
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
            if on_retry:
                on_retry(exc, attempt, delay)
            await asyncio.sleep(delay)
