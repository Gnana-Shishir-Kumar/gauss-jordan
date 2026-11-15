"""Credit comparison against traditional search strategies."""

from conftest import FakeLLM, FakeSearch

from research_agent.credits import compare, extract_cost, to_markdown


def test_extract_cost_rounds_up_per_call():
    assert extract_cost(0, 20) == (0, 0)
    assert extract_cost(6, 20) == (1, 2)
    assert extract_cost(67, 20) == (4, 14)  # 20, 20, 20, 7 -> 4 + 4 + 4 + 2
    assert extract_cost(3, 1) == (3, 3)     # one call per page: 1 credit each


def test_compare_on_a_run(agent, conn):
    r = agent.full(FakeLLM(conn), FakeSearch(domains=("fda.gov", "nejm.org", "nature.com", "ascopubs.org")))
    res = compare(r)
    s = res["stats"]
    assert s["searches"] == 3 and s["raw_results"] == 12 and s["unique_results"] == 12
    assert s["extracted_urls"] == 6  # top 2 per query
    project, batched, per_page, advanced = res["rows"]
    assert project["total"] == r["totals"]["credits"] == 3 + 2
    assert batched["total"] == 3 + 3 and per_page["total"] == 3 + 12 and advanced["total"] == 6 + 12
    assert per_page["vs_project"] == 3.0
    assert res["repeat_within_cache"]["project"] == 0
    assert "| This project (billed)" in to_markdown(res)


def test_cached_rerun_still_priced_for_traditional(agent, conn):
    search = FakeSearch()
    agent.full(FakeLLM(conn), search)
    second = agent.full(FakeLLM(conn), search)  # served from cache
    res = compare(second)
    assert res["rows"][0]["total"] == 0
    assert res["stats"]["raw_results"] == 6 and res["rows"][2]["total"] == 3 + 6
