"""Text rendering of the update brief (stored as final_answer, printed by the CLI)."""

from __future__ import annotations

from .sources import is_preliminary, tier_label, type_label

DISCLAIMER = (
    "Clinician-support tool. This brief summarizes published and public information to support a clinician's "
    "professional judgment and independent review. It is not for patients and does not give patient-specific "
    "recommendations."
)
USE_LIMITS = (
    "Use limits: clinician-facing only; not for patients; not for urgent or time-critical decisions; does not use "
    "patient data; summarizes published and public information for independent review."
)
NO_IDENTIFIERS = "Do not enter patient identifiers (names, record numbers, dates of birth). Ask about the clinical topic."

_RANK = ["supported", "partial", "unsupported", "source_unavailable"]
SUPPORTING = ("supported", "partial")


def claim_verdict(claim: dict) -> str:
    """Overall verdict for a claim: the best verdict among its citations."""
    verdicts = {c.get("verdict") for c in claim.get("citations", []) if c.get("verdict")}
    for v in _RANK:
        if v in verdicts:
            return v
    return "uncited" if not claim.get("citations") else "unverified"


def is_flagged(claim: dict) -> bool:
    return claim_verdict(claim) not in SUPPORTING


def evidence_label(claim: dict) -> str:
    """Peer-reviewed vs preliminary label, shown inside the item itself."""
    cits = [c for c in claim.get("citations", []) if c.get("verdict") in SUPPORTING] or claim.get("citations", [])
    types = [c.get("source_type") for c in cits]
    prelim = sorted({type_label(t).lower() for t in types if is_preliminary(t)})
    if types and all(is_preliminary(t) for t in types):
        return f"PRELIMINARY ({', '.join(prelim)}), not peer reviewed"
    if claim.get("peer_reviewed"):
        return "Peer-reviewed" + (f"; also cites preliminary evidence ({', '.join(prelim)})" if prelim else "")
    if prelim:
        return f"Includes PRELIMINARY evidence ({', '.join(prelim)}); not peer reviewed"
    return "Not peer reviewed (" + ", ".join(sorted({type_label(t).lower() for t in types})) + ")" if types else \
        "No verified source"


def key_numbers(claim: dict) -> list[tuple[str, int]]:
    return [(s, c["evidence_id"]) for c in claim.get("citations", [])
            for s in (c.get("source_sentence") or "").split("\n") if s.strip()]


def sort_items(claims: list[dict]) -> list[dict]:
    return sorted(claims, key=lambda c: (c.get("item_date") or "", -c["position"]), reverse=True)


def _item_lines(n: int, c: dict) -> list[str]:
    cits = c.get("citations", [])
    types = sorted({type_label(x.get("source_type")) for x in cits}) or ["no source"]
    tiers = sorted({x.get("trust_tier") for x in cits if x.get("trust_tier")})
    verdict = claim_verdict(c)
    flag = "  [FLAGGED: not verified]" if is_flagged(c) else ""
    lines = [f"{n}. {c.get('item_date') or 'date not stated'} · {', '.join(types)} · "
             f"{', '.join(tier_label(t).split(' · ')[0] for t in tiers) or 'no tier'}{flag}",
             f"   What changed: {c['text']}  [{c['claim_type']}; {verdict.replace('_', ' ')}]",
             f"   Evidence level: {c.get('evidence_level') or 'not stated'} · {evidence_label(c)}",
             f"   Population: {c.get('population') or 'population not stated'}"]
    nums = key_numbers(c)
    if nums:
        lines.append("   Key numbers (source sentences):")
        lines += [f'     E{eid}: "{s}"' for s, eid in nums]
    for x in cits:
        lines.append(f"   Source E{x['evidence_id']}: {x.get('url') or '?'} ({(x.get('verdict') or 'unchecked').replace('_', ' ')})")
    return lines


def format_answer(task: dict, claims: list[dict], answer: dict) -> str:
    anchor = task.get("anchor_date") or "the anchor date"
    out: list[str] = []
    brief = task.get("brief") or {}
    if brief.get("anchor_defaulted"):
        out.append(f"Note: no date was given, so this covers the 12 months since {anchor}.")
    if answer.get("insufficiency_note"):
        out.append(f"Insufficient evidence: {answer['insufficiency_note']}")
    bl = answer.get("bottom_line")
    out += ["", "1. BOTTOM LINE"]
    if bl:
        out.append(f"{bl['text']} (confidence: {bl['confidence']}; {bl.get('reason', '')})".rstrip())
    else:
        out.append("No bottom line: the evidence found was insufficient.")
    new = sort_items([c for c in claims if c.get("section") == "new"])
    out += ["", f"2. WHAT'S NEW SINCE {anchor} (newest first)"]
    if not new:
        out.append("No verified developments found.")
    for i, c in enumerate(new, 1):
        out += _item_lines(i, c)
    conflicts = [c for c in claims if c.get("section") == "conflict"]
    out += ["", "3. SOURCES DISAGREE"]
    out += [line for i, c in enumerate(conflicts, 1) for line in _item_lines(i, c)] or ["No conflicts found."]
    out += ["", "4. NOT YET ESTABLISHED"] + [f"- {x}" for x in answer.get("not_established") or ["(none listed)"]]
    out += ["", "5. CHECK BEFORE ACTING"] + [f"- {x}" for x in answer.get("check_before_acting") or ["(none listed)"]]
    out += ["", "6. " + DISCLAIMER, USE_LIMITS]
    return "\n".join(out).strip()


def format_report(report: dict, show_trace: bool = True) -> str:
    task = report["task"]
    brief = task.get("brief") or {}
    out = [f"Question: {task['question']}", f"Status:   {task['status']}"]
    if task.get("error"):
        out.append(f"Error:    {task['error']}")
    if task["status"] == "needs_clarification":
        out.append(f"Clarifying question: {brief.get('clarifying_question')}")
    elif brief:
        out += ["", "BRIEF", "-----", f"Cancer type: {brief.get('cancer_type')}", f"Setting: {brief.get('setting')}",
                f"Question: {brief.get('question')}",
                f"Since: {task.get('anchor_date')}" + (" (defaulted to 12 months ago)" if brief.get("anchor_defaulted") else ""),
                "Aspects: " + "; ".join(brief.get("aspects", []))]
    if task.get("final_answer"):
        out += ["", "UPDATE BRIEF", "------------", task["final_answer"]]
    totals = report.get("totals", {})
    out += ["", f"Stop reason: {task.get('stop_reason') or '-'}",
            f"Tavily credits: {totals.get('credits', 0):g} · tokens: {totals.get('input_tokens', 0)} in / "
            f"{totals.get('output_tokens', 0)} out",
            f"Models: {task.get('model_ids')} · prompt version: {task.get('prompt_version')}"]
    if show_trace:
        out += ["", "TRACE", "-----"]
        for t in report["trace"]:
            meta = []
            if t.get("latency_ms") is not None:
                meta.append(f"{t['latency_ms']}ms")
            if t.get("input_tokens"):
                meta.append(f"{t['input_tokens']}->{t['output_tokens']} tok")
            if t.get("tavily_credits") is not None:
                meta.append(f"{t['tavily_credits']:g} cr")
            if t.get("model_id"):
                meta.append(t["model_id"])
            out.append(f"  [{t['step']:<8}] {t['message']}" + (f"  ({', '.join(meta)})" if meta else ""))
    return "\n".join(out)
