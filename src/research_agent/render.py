"""Plain-text rendering shared by the pipeline (final_answer) and the CLI."""

from __future__ import annotations

_RANK = ["supported", "partial", "unsupported", "source_unavailable"]


def claim_verdict(claim: dict) -> str:
    """Overall verdict for a claim: the best verdict among its citations."""
    verdicts = {c.get("verdict") for c in claim.get("citations", []) if c.get("verdict")}
    for v in _RANK:
        if v in verdicts:
            return v
    return "uncited" if not claim.get("citations") else "unverified"


def format_answer(claims: list[dict], note: str | None = None) -> str:
    lines: list[str] = []
    if note:
        lines += [f"Note: {note}", ""]
    if not claims:
        lines.append("No claims could be made from the available evidence.")
    for i, c in enumerate(claims, 1):
        cites = "; ".join(f"E{x['evidence_id']}: {x.get('verdict') or 'unchecked'}" for x in c.get("citations", []))
        suffix = f"  ({cites})" if cites else "  (no citation)"
        lines.append(f"{i}. [{c['claim_type']}] {c['text']}{suffix}")
    return "\n".join(lines).strip()


def format_report(report: dict, show_trace: bool = True) -> str:
    task = report["task"]
    out = [f"Question: {task['question']}", f"Status:   {task['status']}"]
    if task.get("error"):
        out.append(f"Error:    {task['error']}")
    out += ["", "ANSWER", "------", task.get("final_answer") or "(none)", ""]

    cited = {x["evidence_id"] for c in report["claims"] for x in c["citations"]}
    ev = {e["id"]: e for e in report["evidence"]}
    if cited:
        out += ["CITATIONS", "---------"]
        for c in report["claims"]:
            out.append(f"Claim {c['position'] + 1} [{c['claim_type']}] -> {claim_verdict(c)}")
            for x in c["citations"]:
                e = ev.get(x["evidence_id"], {})
                out.append(f"  E{x['evidence_id']} {x.get('verdict') or 'unchecked':<18} {e.get('url', '?')}")
                if x.get("verifier_note"):
                    out.append(f"      {x['verifier_note']}")
        out.append("")

    if show_trace:
        out += ["TRACE", "-----"]
        for t in report["trace"]:
            meta = []
            if t.get("latency_ms") is not None:
                meta.append(f"{t['latency_ms']}ms")
            if t.get("input_tokens"):
                meta.append(f"{t['input_tokens']}->{t['output_tokens']} tok")
            msg = t.get("message") or t.get("detail") or ""
            out.append(f"  [{t['step']:<9}] {msg}" + (f"  ({', '.join(meta)})" if meta else ""))
        tin = sum(t.get("input_tokens") or 0 for t in report["trace"])
        tout = sum(t.get("output_tokens") or 0 for t in report["trace"])
        kept = sum(1 for e in report["evidence"] if e["kept"])
        out.append(f"  totals: {len(report['sub_queries'])} sub-queries, {kept}/{len(report['evidence'])} "
                   f"evidence kept, {tin} input / {tout} output tokens")
    return "\n".join(out)
