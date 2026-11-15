"""Per-run context shared by the pipeline steps: DB handle, clients, settings, event emission."""

from __future__ import annotations

import inspect
import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import storage
from .config import Settings
from .llm import LLMClient
from .search.base import SearchBackend

log = logging.getLogger("research_agent")

EventCallback = Callable[[dict[str, Any]], Any]


def describe_error(exc: BaseException, limit: int = 300) -> str:
    msg = str(exc).strip().replace("\n", " ")
    return f"{type(exc).__name__}: {msg[:limit]}" if msg else type(exc).__name__


@dataclass
class RunContext:
    conn: sqlite3.Connection
    task_id: int
    settings: Settings
    llm: LLMClient
    search: SearchBackend
    on_event: EventCallback | None = None

    def note(self, step: str, message: str, **detail: Any) -> None:
        """Synchronous trace-only record (used from retry callbacks)."""
        storage.add_trace(self.conn, self.task_id, step, {"message": message, "level": "warning", **detail})

    async def emit(
        self,
        step: str,
        message: str,
        *,
        level: str = "info",
        detail: dict[str, Any] | None = None,
        latency_ms: int | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        payload = {"message": message, "level": level, **(detail or {})}
        storage.add_trace(
            self.conn, self.task_id, step, payload,
            latency_ms=latency_ms, input_tokens=input_tokens, output_tokens=output_tokens,
        )
        if self.on_event is None:
            return
        event = {
            "task_id": self.task_id, "step": step, "level": level, "message": message,
            "detail": detail or {}, "latency_ms": latency_ms,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
        }
        try:
            result = self.on_event(event)
            if inspect.isawaitable(result):
                await result
        except Exception:  # a broken UI callback must never break the run
            log.exception("on_event callback failed")
