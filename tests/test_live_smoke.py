"""Optional live test against real Token Factory + Tavily. Skipped when keys are missing.

Run: uv run pytest -m live -s
"""

import asyncio

import pytest

from research_agent import storage
from research_agent.config import get_settings
from research_agent.pipeline import default_clients, run_research

settings = get_settings()
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(bool(settings.missing_keys()), reason="NEBIUS_API_KEY / TAVILY_API_KEY not set"),
]


def test_live_end_to_end(tmp_path):
    conn = storage.connect(str(tmp_path / "live.db"))
    llm, search = default_clients(settings)
    report = asyncio.run(run_research(
        "What is the boiling point of water at sea level in Celsius and Fahrenheit?",
        llm=llm, search=search, conn=conn, settings=settings,
    ))
    task = report["task"]
    assert task["status"] in ("done", "insufficient_evidence"), task["error"]
    assert report["sub_queries"] and report["evidence"]
    if task["status"] == "done":
        assert all(x["verdict"] for c in report["claims"] for x in c["citations"])
