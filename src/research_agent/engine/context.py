"""Per-run context: DB handle, clients, settings, budget, and trace/event emission.

Every trace event records the prompt version and (for LLM steps) the model ID used.
"""

from __future__ import annotations

import inspect
import logging
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .. import storage
from ..backends.llm import LLMClient, LLMResult
from ..backends.search_base import SearchBackend
from ..config import Settings
from ..prompts import PROMPT_VERSION

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
    prompt_version: str = PROMPT_VERSION
    started: float = field(default_factory=time.monotonic)
    credits_used: float = 0.0

    # --- budget ----------------------------------------------------------------
    def budget_exhausted(self) -> str | None:
        b = self.settings.budget
        if time.monotonic() - self.started >= b.time_s:
            return "time_budget"
        if self.credits_used >= b.credits:
            return "credit_budget"
        return None

    # --- tracing ---------------------------------------------------------------
    def note(self, step: str, message: str, **detail: Any) -> None:
        """Synchronous trace-only record (used from retry callbacks)."""
        storage.add_trace(self.conn, self.task_id, step, {"message": message, "level": "warning", **detail},
                          prompt_version=self.prompt_version)

    async def emit(self, step: str, message: str, *, level: str = "info", detail: dict[str, Any] | None = None,
                   latency_ms: int | None = None, input_tokens: int | None = None, output_tokens: int | None = None,
                   credits: float | None = None, model_id: str | None = None) -> None:
        if credits:
            self.credits_used += credits
        storage.add_trace(self.conn, self.task_id, step, {"message": message, "level": level, **(detail or {})},
                          latency_ms=latency_ms, input_tokens=input_tokens, output_tokens=output_tokens,
                          tavily_credits=credits, model_id=model_id, prompt_version=self.prompt_version)
        if self.on_event is None:
            return
        event = {"task_id": self.task_id, "step": step, "level": level, "message": message, "detail": detail or {},
                 "latency_ms": latency_ms, "input_tokens": input_tokens, "output_tokens": output_tokens,
                 "credits": credits, "model_id": model_id, "prompt_version": self.prompt_version}
        try:
            result = self.on_event(event)
            if inspect.isawaitable(result):
                await result
        except Exception:  # a broken UI callback must never break the run
            log.exception("on_event callback failed")

    async def emit_llm(self, step: str, message: str, res: LLMResult, *, level: str = "info",
                       detail: dict[str, Any] | None = None) -> None:
        await self.emit(step, message, level=level, detail=detail, latency_ms=res.latency_ms,
                        input_tokens=res.input_tokens, output_tokens=res.output_tokens, model_id=res.model or None)
