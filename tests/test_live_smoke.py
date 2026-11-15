"""Optional live test against real Token Factory + Tavily. Skipped when keys are missing.

Run: uv run pytest -m live -s   (uses a few Tavily credits)
"""

import asyncio

import pytest

from research_agent import storage
from research_agent.config import get_settings
from research_agent.engine import run_all
from research_agent.engine.orchestrator import default_clients

settings = get_settings()
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(bool(settings.missing_keys()), reason="NEBIUS_API_KEY / TAVILY_API_KEY not set"),
]


def test_live_end_to_end(tmp_path):
    conn = storage.connect(str(tmp_path / "live.db"))
    llm, search = default_clients(settings)
    report = asyncio.run(run_all(
        "What has changed for first-line treatment of metastatic EGFR-mutant non-small cell lung cancer "
        "since 2024-06-01?", llm=llm, search=search, conn=conn, settings=settings))
    task = report["task"]
    assert task["status"] in ("done", "insufficient_evidence"), task["error"]
    assert task["stop_reason"] and report["sub_queries"] and report["evidence"]
    assert all(e["trust_tier"] in (1, 2, 3, None) for e in report["evidence"])
    assert all(x["verdict"] for c in report["claims"] for x in c["citations"])
    assert report["totals"]["credits"] > 0
