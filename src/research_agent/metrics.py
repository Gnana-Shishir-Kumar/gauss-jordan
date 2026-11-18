"""Eval metrics computed from a stored report. Pooled across questions as counts, reported as percentages."""

from __future__ import annotations

from .render import SUPPORTING, claim_verdict, evidence_label
from .sources import is_preliminary

RATE_KEYS = {
    "citation_support_rate": ("supported_citations", "checked_citations"),
    "numeric_check_pass_rate": ("numeric_passed", "numeric_checked"),
    "tier12_support_share": ("tier12_supported_claims", "supported_claims"),
    "preliminary_labeled_share": ("preliminary_labeled", "preliminary_claims"),
    "insufficient_evidence_accuracy": ("insufficient_correct", "questions"),
}


def task_counts(report: dict, expect_insufficient: bool) -> dict:
    claims = [c for c in report["claims"] if c["citations"] and c.get("status") != "deleted"]
    cits = [x for c in claims for x in c["citations"] if x.get("verdict")]
    supported_claims = [c for c in claims if claim_verdict(c) in SUPPORTING]
    numeric = [t["detail"] for t in report["trace"]
               if t["step"] == "check" and t["detail"].get("gate") == "numbers"
               and t["detail"].get("phase") == "initial" and t["detail"].get("status") != "n/a"]
    prelim = [c for c in claims if any(is_preliminary(x) for x in c["citations"])]
    status = report["task"]["status"]
    return {
        "supported_citations": sum(x["verdict"] == "supported" for x in cits),
        "checked_citations": len(cits),
        "numeric_passed": sum(d.get("status") == "pass" for d in numeric),
        "numeric_checked": len(numeric),
        "tier12_supported_claims": sum(any(x.get("verdict") in SUPPORTING and x.get("trust_tier") in (1, 2)
                                           for x in c["citations"]) for c in supported_claims),
        "supported_claims": len(supported_claims),
        "preliminary_labeled": sum("PRELIMINARY" in evidence_label(c) for c in prelim),
        "preliminary_claims": len(prelim),
        # A crash or clarification is never "correct": expected-sufficient questions must end done.
        "insufficient_correct": int(status == ("insufficient_evidence" if expect_insufficient else "done")),
        "questions": 1,
        "credits": report["totals"]["credits"],
        "input_tokens": report["totals"]["input_tokens"],
        "output_tokens": report["totals"]["output_tokens"],
    }


def pct(n: float, d: float) -> float | None:
    return round(100.0 * n / d, 1) if d else None


def aggregate(rows: list[dict]) -> dict:
    total = {k: sum(r["counts"][k] for r in rows) for k in rows[0]["counts"]} if rows else {}
    out: dict = {name: pct(total.get(n, 0), total.get(d, 0)) for name, (n, d) in RATE_KEYS.items()}
    n = len(rows) or 1
    out.update(questions=len(rows),
               avg_credits=round(sum(r["counts"]["credits"] for r in rows) / n, 2),
               avg_latency_s=round(sum(r["latency_s"] for r in rows) / n, 1),
               avg_input_tokens=round(sum(r["counts"]["input_tokens"] for r in rows) / n),
               avg_output_tokens=round(sum(r["counts"]["output_tokens"] for r in rows) / n),
               counts=total)
    return out
