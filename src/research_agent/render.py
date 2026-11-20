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
    "patient data; reads page text only (figures, image tables and supplements are not read); summarizes published "
    "and public information for independent review."
)
NO_IDENTIFIERS = "Do not enter patient identifiers (names, record numbers, dates of birth). Ask about the clinical topic."

# Only "supported" counts as verified. "partial" means a different endpoint, arm, number or population: flagged.
SUPPORTING = ("supported",)
GATE_LABELS = {"number_gate": "numbers", "id_gate": "identifiers", "entail_gate": "entailment"}


def claim_verdict(claim: dict) -> str:
    """Overall verdict: rejected (citation check), unsupported (number or identifier gate, or entailment), partial,
    supported. Set once per claim by the verifier over all cited sentences together."""
    if claim.get("verdict"):
        return claim["verdict"]
    return "uncited" if not claim.get("citations") else "unchecked"


def is_flagged(claim: dict) -> bool:
    return claim_verdict(claim) not in SUPPORTING or claim.get("status") == "unverified"


def gate_summary(claim: dict) -> str:
    """'numbers pass · identifiers n/a · entailment supported' (the failing gate is the first non-pass one)."""
    if claim.get("rejected"):
        return "citation check failed (" + "; ".join(f"{r['id'] or '-'}: {r['reason']}" for r in claim["rejected"]) + ")"
    cits = claim.get("citations") or []
    parts = []
    for g, label in GATE_LABELS.items():
        statuses = [(x.get(g) or {}).get("status") for x in cits if x.get(g)]
        if not statuses:
            continue
        st = statuses[0] if g != "id_gate" else (next((x for x in statuses if x == "fail"), statuses[0]))
        parts.append(f"{label} {st}")
    return " · ".join(parts) or "not checked"


def failed_gate(claim: dict) -> str | None:
    if claim.get("rejected"):
        return "citation"
    for g, label in GATE_LABELS.items():
        for x in claim.get("citations") or []:
            st = (x.get(g) or {}).get("status")
            if st in ("fail", "partial", "unsupported"):
                return label
    return None


COMPANY_REPORTED = "company-reported, not peer-reviewed"
PRELIMINARY = "preliminary"


def evidence_label(claim: dict) -> str:
    """Peer-reviewed / preliminary / company-reported label, shown inside the item itself. Labels set in code at
    finalize (claims.labels) take precedence: tier 3 support is always PRELIMINARY, and an efficacy number resting only
    on tier 4 sources is COMPANY-REPORTED."""
    labels = claim.get("labels") or []
    cits = [c for c in claim.get("citations", []) if c.get("verdict") in SUPPORTING] or claim.get("citations", [])
    types = sorted({type_label(c.get("source_type")).lower() for c in cits})
    if COMPANY_REPORTED in labels:
        return "COMPANY-REPORTED, not peer-reviewed (" + ", ".join(types) + ")"
    prelim = sorted({type_label(c.get("source_type")).lower() for c in cits if is_preliminary(c)})
    if cits and all(is_preliminary(c) for c in cits):
        return f"PRELIMINARY ({', '.join(prelim)}), not peer reviewed"
    if claim.get("peer_reviewed"):
        return "Peer-reviewed" + (f"; also cites PRELIMINARY evidence ({', '.join(prelim)})" if prelim else "")
    if prelim or PRELIMINARY in labels:
        return f"Includes PRELIMINARY evidence ({', '.join(prelim)}); not peer reviewed"
    return "Not peer reviewed (" + ", ".join(types) + ")" if types else "No verified source"


def cited_sentences(claim: dict) -> list[tuple[str, str]]:
    """(sentence ID, verbatim sentence) for every cited sentence, from the evidence store."""
    return [(s["id"], s["text"]) for c in claim.get("citations", []) for s in c.get("sentences", [])]


def sort_items(claims: list[dict]) -> list[dict]:
    return sorted(claims, key=lambda c: (c.get("item_date") or "", -c["position"]), reverse=True)


def review_status(claim: dict) -> str:
    if COMPANY_REPORTED in (claim.get("labels") or []):
        return "company-reported"
    cits = [c for c in claim.get("citations", []) if c.get("verdict") in SUPPORTING] or claim.get("citations", [])
    if cits and any(is_preliminary(c) for c in cits):
        return "preliminary"
    return "peer-reviewed" if claim.get("peer_reviewed") else "not peer-reviewed"


def change_rows(claims: list[dict]) -> list[dict]:
    """The "What changed" table: one row per kept 'new' item, newest first."""
    rows = []
    for c in sort_items([c for c in claims if c.get("section") == "new" and c.get("status") != "deleted"]):
        tiers = sorted({x.get("trust_tier") for x in c.get("citations", []) if x.get("trust_tier")})
        rows.append({"change": c.get("change_type") or "other", "date": c.get("item_date") or "not stated",
                     "evidence_level": c.get("evidence_level") or "not stated",
                     "tier": "/".join(str(t) for t in tiers) or "-", "status": review_status(c),
                     "verified": "yes" if not is_flagged(c) else "no",
                     "impact": f"Interpretation: {c['impact']}" if c.get("impact") else "-", "what": c["text"]})
    return rows


def change_table(claims: list[dict]) -> list[str]:
    rows = change_rows(claims)
    if not rows:
        return ["No changes found."]
    out = ["| Change | Date | Evidence level | Tier | Review status | Verified | Practice impact (interpretation) |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        cells = [r["change"], r["date"], r["evidence_level"], r["tier"], r["status"], r["verified"],
                 r["impact"].removeprefix("Interpretation: ")]
        out.append("| " + " | ".join(str(x).replace("|", "/") for x in cells) + " |")
    return out


def _item_lines(n: int, c: dict, anchor: str | None = None) -> list[str]:
    cits = c.get("citations", [])
    types = sorted({type_label(x.get("source_type")) for x in cits}) or ["no source"]
    tiers = sorted({x.get("trust_tier") for x in cits if x.get("trust_tier")})
    verdict = claim_verdict(c)
    flag = "  [UNVERIFIED]" if c.get("status") == "unverified" else "  [FLAGGED: not verified]" if is_flagged(c) else ""
    from .engine.checks import before_anchor  # local import: engine imports this module
    if before_anchor(c.get("item_date"), anchor):
        flag += f"  [DATED BEFORE {anchor}: not new]"
    if c.get("conflict_with"):
        flag += "  [SOURCES DISAGREE: see section 3]"
    lines = [f"{n}. {c.get('item_date') or 'date not stated'} · {', '.join(types)} · "
             f"{', '.join(tier_label(t).split(' · ')[0] for t in tiers) or 'no tier'}{flag}",
             f"   What changed: {c['text']}  [{c['claim_type']}; {verdict.replace('_', ' ')}]",
             f"   Evidence level: {c.get('evidence_level') or 'not stated'} · {evidence_label(c)}",
             f"   Population: {c.get('population') or 'population not stated'}",
             f"   Checks: {gate_summary(c)}"]
    if c.get("change_type"):
        lines.insert(2, f"   Change type: {c['change_type']}")
    if c.get("impact"):
        lines.append(f"   Practice impact (interpretation, not a finding): {c['impact']}")
    sents = cited_sentences(c)
    if sents:
        lines.append("   Cited sentences (verbatim):")
        lines += [f'     {sid}: "{t}"' for sid, t in sents]
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
    removed = [c for c in claims if c.get("status") == "deleted"]
    claims = [c for c in claims if c.get("status") != "deleted"]
    new = sort_items([c for c in claims if c.get("section") == "new"])
    gl = brief.get("guideline_name") or brief.get("guideline_version")
    out += ["", f"WHAT CHANGED since {anchor}" + (f" (baseline: {_guideline(brief)}, as stated by the user)" if gl else "")]
    out += change_table(claims)
    out += ["", f"2. WHAT'S NEW SINCE {anchor} (newest first)"]
    if not new:
        out.append("No verified developments found.")
    for i, c in enumerate(new, 1):
        out += _item_lines(i, c, task.get("anchor_date"))
    conflicts = [c for c in claims if c.get("section") == "conflict"]
    detected = answer.get("conflicts_detected") or []
    out += ["", "3. SOURCES DISAGREE"]
    if detected:
        from .engine.conflicts import conflict_lines  # local import: engine imports this module
        out.append("Found in code (tier 1-2 sources stating different values):")
        out += conflict_lines(detected)
    if conflicts:
        if detected:
            out.append("Reported by the writer:")
        out += [line for i, c in enumerate(conflicts, 1) for line in _item_lines(i, c, task.get("anchor_date"))]
    if not detected and not conflicts:
        out.append("No conflicts found.")
    out += ["", "4. NOT YET ESTABLISHED"] + [f"- {x}" for x in answer.get("not_established") or ["(none listed)"]]
    out += ["", "5. CHECK BEFORE ACTING"] + [f"- {x}" for x in answer.get("check_before_acting") or ["(none listed)"]]
    if removed:
        out += ["", f"Removed during verification: {len(removed)} claim(s) whose numbers or citations could not be "
                "sourced from stored evidence (see the trace)."]
    out += ["", "6. " + DISCLAIMER, USE_LIMITS]
    return "\n".join(out).strip()


def _guideline(brief: dict) -> str:
    parts = [brief.get("guideline_name"), brief.get("guideline_version"),
             brief.get("guideline_date") and f"dated {brief['guideline_date']}"]
    return " ".join(p for p in parts if p)


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
                "Guideline baseline: " + (_guideline(brief) or "not stated")
                + (" (user-supplied; guideline sites such as NCCN are never fetched)" if _guideline(brief) else ""),
                "Aspects: " + "; ".join(brief.get("aspects", []))]
    if task.get("final_answer"):
        out += ["", "UPDATE BRIEF", "------------", task["final_answer"]]
    totals = report.get("totals", {})
    usage = task.get("usage") or {}
    used = (usage.get("used") or {}) if isinstance(usage, dict) else {}
    out += ["", f"Stop reason: {task.get('stop_reason') or '-'}",
            f"Tavily credits: {totals.get('credits', 0):g} · tokens: {totals.get('input_tokens', 0)} in / "
            f"{totals.get('output_tokens', 0)} out" + (f" ({totals['cached_tokens']} cached)"
                                                       if totals.get("cached_tokens") else ""),
            "Calls: " + (", ".join(f"{k} {v}" for k, v in used.items() if v) or "-")
            + (f" (Tavily repair reserve used: {usage.get('tavily_repair_used', 0)}/"
               f"{usage.get('tavily_repair_reserve', 0)})" if used else ""),
            f"Models: {task.get('model_ids')} · prompt version: {task.get('prompt_version')}"]
    steps = report.get("steps") or []
    if steps:
        out += ["", "COST BY STEP", "------------",
                f"  {'step':<9} {'calls':>5} {'in tok':>8} {'out tok':>8} {'cached':>7} {'credits':>7} {'sec':>7}"]
        for st in steps:
            cached = f"{st['cached_share']:.0%}" if st.get("cached_share") is not None else "n/r"
            out.append(f"  {st['step']:<9} {st['llm_calls']:>5} {st['input_tokens']:>8} {st['output_tokens']:>8} "
                       f"{cached:>7} {st['credits']:>7g} {st['seconds']:>7}")
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
