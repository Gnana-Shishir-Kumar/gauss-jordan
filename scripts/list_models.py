"""List models available on Nebius Token Factory and check the configured IDs exist.

Usage: uv run python scripts/list_models.py
"""

from __future__ import annotations

import asyncio
import sys

from openai import AsyncOpenAI, OpenAIError

from research_agent.config import get_settings


async def main() -> int:
    settings = get_settings()
    if not settings.nebius_api_key:
        print("NEBIUS_API_KEY is not set (put it in .env).", file=sys.stderr)
        return 2
    client = AsyncOpenAI(
        base_url=settings.nebius_base_url,
        api_key=settings.nebius_api_key,
        timeout=settings.network.llm_timeout_s,
    )
    try:
        page = await client.models.list()
    except OpenAIError as exc:
        print(f"Failed to list models: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    ids = sorted(m.id for m in page.data)
    print(f"{len(ids)} models available:\n")
    for mid in ids:
        print(f"  {mid}")

    print("\nConfigured models:")
    ok = True
    for role in ("planner", "writer", "verifier"):
        mid = getattr(settings.models, role)
        found = mid in ids
        ok &= found
        print(f"  {role:9} {mid}  {'OK' if found else 'NOT FOUND'}")
        if not found:
            close = [i for i in ids if mid.split("/")[-1].split("-")[0].lower() in i.lower()]
            if close:
                print(f"            similar: {', '.join(close[:5])}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
