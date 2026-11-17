"""Token Factory LLM wrapper: JSON completions validated with Pydantic, retried on transient errors."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from ..config import Network, Settings
from ..retry import with_retry

T = TypeVar("T", bound=BaseModel)


class LLMOutputError(Exception):
    """The model never produced JSON matching the required schema."""


@dataclass
class LLMResult(Generic[T]):
    data: T
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    attempts: int = 1
    model: str = ""
    cached_tokens: int | None = None  # cached input tokens, when the API reports them


class LLMClient(Protocol):
    async def complete_json(
        self, *, model: str, system: str, user: str, schema: type[T], context: dict | None = None
    ) -> LLMResult[T]: ...


def extract_json_text(text: str) -> str:
    """Pull a JSON object out of a reply that may contain <think> blocks or ``` fences."""
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, flags=re.S | re.I)
    if fence:
        t = fence.group(1).strip()
    if not t.startswith("{"):
        start, end = t.find("{"), t.rfind("}")
        if start != -1 and end > start:
            t = t[start : end + 1]
    return t


def _short_errors(exc: ValidationError) -> str:
    parts = []
    for e in exc.errors()[:5]:
        loc = ".".join(str(x) for x in e.get("loc", ()))
        parts.append(f"{loc or '(root)'}: {e.get('msg')}")
    return "; ".join(parts)


class TokenFactoryLLM:
    """Implements LLMClient over an AsyncOpenAI client pointed at Nebius Token Factory."""

    def __init__(self, client: Any, network: Network, temperature: float = 0.1, max_tokens: int = 4096):
        self.client = client
        self.network = network
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._json_mode: dict[str, bool] = {}

    @classmethod
    def from_settings(cls, settings: Settings) -> "TokenFactoryLLM":
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            base_url=settings.nebius_base_url,
            api_key=settings.nebius_api_key,
            timeout=settings.network.llm_timeout_s,
            max_retries=0,  # retries are handled by with_retry so there is one policy
        )
        return cls(client, settings.network, max_tokens=settings.network.llm_max_tokens)

    async def _create(self, model: str, messages: list[dict]) -> Any:
        import openai

        net = self.network
        kwargs: dict[str, Any] = dict(
            model=model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens
        )
        if self._json_mode.get(model, True):
            kwargs["response_format"] = {"type": "json_object"}

        def call():
            return self.client.chat.completions.create(**kwargs)

        retry = dict(
            attempts=net.max_retries, base_delay=net.backoff_base_s, max_delay=net.backoff_max_s,
            timeout=net.llm_timeout_s,
        )
        try:
            return await with_retry(call, **retry)
        except openai.BadRequestError:
            if "response_format" not in kwargs:
                raise
            # Some models reject JSON mode; fall back to prompt-only JSON and remember that.
            self._json_mode[model] = False
            kwargs.pop("response_format")
            return await with_retry(call, **retry)

    async def complete_json(
        self, *, model: str, system: str, user: str, schema: type[T], context: dict | None = None
    ) -> LLMResult[T]:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        in_tok = out_tok = 0
        cached: int | None = None
        start = time.perf_counter()
        last: ValidationError | None = None
        truncated = False
        for attempt in (1, 2):
            resp = await self._create(model, messages)
            usage = getattr(resp, "usage", None)
            in_tok += int(getattr(usage, "prompt_tokens", 0) or 0)
            out_tok += int(getattr(usage, "completion_tokens", 0) or 0)
            details = getattr(usage, "prompt_tokens_details", None)
            ct = getattr(details, "cached_tokens", None) if details is not None else None
            if ct is not None:
                cached = (cached or 0) + int(ct)
            choice = resp.choices[0] if getattr(resp, "choices", None) else None
            text = (choice.message.content or "") if choice else ""
            truncated = getattr(choice, "finish_reason", None) == "length"
            try:
                data = schema.model_validate_json(extract_json_text(text), context=context)
                return LLMResult(
                    data=data, input_tokens=in_tok, output_tokens=out_tok,
                    latency_ms=int((time.perf_counter() - start) * 1000), attempts=attempt, model=model,
                    cached_tokens=cached,
                )
            except ValidationError as exc:
                last = exc
                messages += [
                    {"role": "assistant", "content": text[:4000]},
                    {
                        "role": "user",
                        "content": f"That reply was not valid for the required JSON format: {_short_errors(exc)}. "
                        "Reply with only the corrected JSON object.",
                    },
                ]
        hint = f"; output hit the {self.max_tokens}-token limit (raise llm_max_tokens)" if truncated else ""
        raise LLMOutputError(f"{schema.__name__}: invalid output after 2 attempts ({_short_errors(last)}){hint}")
