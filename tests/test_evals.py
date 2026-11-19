"""Drift gate (evals/compare.py) and eval metrics."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import FakeLLM, FakeSearch, good_item, num_notes

from research_agent.metrics import aggregate, task_counts

COMPARE = Path(__file__).parents[1] / "evals" / "compare.py"


def _summary(support, numeric):
    return {"run_at": "x", "model_ids": {}, "prompt_version": "v",
            "summary": {"citation_support_rate": support, "numeric_check_pass_rate": numeric,
                        "tier12_support_share": 90.0, "avg_credits": 10}}


def _gate(tmp_path, baseline, latest, *extra):
    b, l = tmp_path / "baseline.json", tmp_path / "latest.json"
    if baseline is not None:
        b.write_text(json.dumps(baseline))
    l.write_text(json.dumps(latest))
    return subprocess.run([sys.executable, str(COMPARE), "--baseline", str(b), "--latest", str(l), *extra],
                          capture_output=True, text=True)


@pytest.mark.parametrize("latest,code", [
    (_summary(80.0, 90.0), 0),   # unchanged
    (_summary(77.5, 88.0), 0),   # drops of 2.5 and 2.0 points: within threshold
    (_summary(76.9, 90.0), 1),   # citation support down 3.1 points
    (_summary(80.0, 86.5), 1),   # numeric pass rate down 3.5 points
    (_summary(85.0, 95.0), 0),   # improvement
    (_summary(80.0, None), 1),   # metric missing from latest run
])
def test_drift_gate(tmp_path, latest, code):
    r = _gate(tmp_path, _summary(80.0, 90.0), latest)
    assert r.returncode == code, r.stdout + r.stderr
    if code == 1:
        assert "DRIFT GATE FAILED" in r.stdout


def test_drift_gate_set_baseline_and_missing_baseline(tmp_path):
    r = _gate(tmp_path, None, _summary(80.0, 90.0))
    assert r.returncode == 2 and "--set-baseline" in r.stderr
    r = _gate(tmp_path, None, _summary(80.0, 90.0), "--set-baseline")
    assert r.returncode == 0
    assert json.loads((tmp_path / "baseline.json").read_text())["summary"]["citation_support_rate"] == 80.0


def test_metrics_from_runs(agent, conn):
    good = agent.full(FakeLLM(conn), FakeSearch())

    def wrong_number(call):
        n = num_notes(call.llm.conn)[0]
        item = good_item(call.llm.conn, n) | {"text": "HR was 0.68."}
        return {"sufficient": True, "bottom_line": {"text": "b", "confidence": "low", "reason": "r"}, "items": [item]}

    numeric_fail = agent.full(FakeLLM(conn, Draft=wrong_number, RepairAction=lambda c: {"action": "delete", "reason": "x"}), FakeSearch())
    rows = [{"counts": task_counts(good, False), "latency_s": 1.0},
            {"counts": task_counts(numeric_fail, True), "latency_s": 3.0}]
    s = aggregate(rows)
    assert s["numeric_check_pass_rate"] == pytest.approx(100 * 2 / 3, abs=0.1)  # 2 of 3 initial checks passed
    assert s["insufficient_evidence_accuracy"] == 100.0  # second run ended insufficient, as expected
    assert s["tier12_support_share"] == 100.0
    assert s["citation_support_rate"] == 100.0  # the unsourceable claim was deleted, not left flagged
    assert s["avg_latency_s"] == 2.0 and s["questions"] == 2


def test_eval_questions_file():
    rows = [json.loads(x) for x in (COMPARE.parent / "questions.jsonl").read_text(encoding="utf-8").splitlines() if x]
    assert len(rows) == 15 and sum(r["expect_insufficient"] for r in rows) == 3
    assert len({r["cancer_type"] for r in rows}) >= 8
    assert all("clinician_rating" in r and "clinician_notes" in r for r in rows)



# --- seeded-error eval and baseline comparison (mocked) --------------------------------

def _load(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, COMPARE.parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # pydantic resolves the script's postponed annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


def test_seeded_eval_catches_injected_errors(tmp_path, settings):
    import asyncio

    from conftest import Agent, FakeIds, FakeLLM, FakeSearch
    from research_agent import storage
    seeded = _load("seeded_eval")
    src_path = tmp_path / "briefs.db"
    src = storage.connect(str(src_path))
    Agent(src, settings).full(FakeLLM(src), FakeSearch())
    out = tmp_path / "seeded-test.json"

    def judge(call):  # the comparator-arm swap puts 16.6 where 23.7 was: an entailment-only error
        return {"verdict": "unsupported" if "was 16.6 months versus 16.6" in call.user else "supported", "note": "n"}

    llm = FakeLLM(src, Entailment=judge, RepairAction=lambda c: {"action": "delete", "reason": "x"})
    res = asyncio.run(seeded.run([str(src_path)], max_briefs=3, repair=True, seed=1, llm=llm, settings=settings,
                                 out=out, ids=FakeIds()))
    s = res["summary"]
    assert s["briefs"] == 1 and s["controls"] == 2 and s["false_positive_rate"] == 0.0
    t = s["per_error_type"]
    for k in ("wrong_number", "fabricated_id", "stale_date", "tier4_only_efficacy"):
        assert t[k]["cases"] >= 1 and t[k]["recall"] == 100.0, (k, t[k])
    assert t["wrong_number"]["caught_by"] == {"numbers": t["wrong_number"]["cases"]}
    assert t["fabricated_id"]["caught_by"] == {"identifiers": t["fabricated_id"]["cases"]}
    assert t["tier4_only_efficacy"]["caught_by"] == {"tier_rule": t["tier4_only_efficacy"]["cases"]}
    assert s["per_gate"]["numbers"]["precision"] == 100.0
    assert s["errors_leaked_after_repair"] == sum(v["leaked_after_repair"] for v in t.values())
    assert s["controls_supported_after_repair"] == 100.0
    assert out.is_file() and json.loads(out.read_text())["summary"]["recall"] == s["recall"]
    assert "| wrong_number |" in seeded.to_markdown(s)


def test_baseline_claims_go_through_our_verifier(conn, settings):
    import asyncio

    from conftest import FakeLLM, FakeSearch
    base_mod = _load("baseline_compare")
    url = "https://www.onclive.com/drugx-news"
    page = ("Median progression-free survival was 23.7 months versus 16.6 months (HR 0.70; p<0.001). "
            "Other text on the page.")

    def answer(call):
        return {"answer": "a", "claims": [
            {"text": "Drugx PFS was 23.7 months.", "url": url},
            {"text": "Drugx PFS was 31.2 months.", "url": url},
            {"text": "Drugx was approved in Japan.", "url": "https://made-up.example/x"}]}

    from research_agent.backends.search_base import SearchHit
    search = FakeSearch(results={"q?": [SearchHit(url=url, title="news", content="drugx", score=0.9)]},
                        pages={url: page})
    llm = FakeLLM(conn, BaselineAnswer=answer)
    from conftest import FakeIds
    import research_agent.backends.identifiers as idm
    orig = idm.IdentifierResolver.from_settings
    idm.IdentifierResolver.from_settings = classmethod(lambda cls, c, s: FakeIds())
    try:
        res = asyncio.run(base_mod.baseline("q?", "2025-01-01", llm=llm, search=search, settings=settings, conn=conn))
    finally:
        idm.IdentifierResolver.from_settings = orig
    outcomes = [r["outcome"] for r in res["claims"]]
    assert outcomes[0].startswith("shown as uncertain") and "company-reported" in outcomes[0]  # news-only efficacy
    assert outcomes[1] == "flagged: unsupported" and "numbers fail" in res["claims"][1]["checks"]
    assert outcomes[2] == "flagged: rejected" and "URL not in the search results" in res["claims"][2]["checks"]
    md = base_mod.to_markdown("q?", res, None)
    assert "1 fail a verifier gate" not in md and "2 fail a verifier gate" in md
