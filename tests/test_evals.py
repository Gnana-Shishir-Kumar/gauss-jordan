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
