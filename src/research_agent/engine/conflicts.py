"""Conflict detection between tier 1-2 sources, in code.

When two tier 1 or 2 sources state different values for the same endpoint of the same drug or trial, both are shown
side by side with their dates and sources; neither is silently picked. Two cases are checked:
  - within one claim: the claim cites several tier 1-2 sources and their sentences give different values for an
    endpoint the claim states (the number gate alone would pass if the claim's number is in either one);
  - across claims: two verified claims about the same drug or trial state different values for the same endpoint
    and rest on different tier 1-2 sources.
Recommendation disagreements (no numbers) are left to the writer's "conflicts" items, which must cite both sides.
"""

from __future__ import annotations

import re
from itertools import combinations

from ..backends.structured import drug_names
from ..render import SUPPORTING
from .numbers import norm_value

_EP = (r"median progression[- ]free survival|progression[- ]free survival|median overall survival|overall survival|"
       r"objective response rate|overall response rate|response rate|disease[- ]free survival|"
       r"event[- ]free survival|pathologic(?:al)? complete response|hazard ratio|PFS|OS|ORR|DFS|EFS|pCR|HR")
_EP_RE = re.compile(rf"\b(?P<ep>{_EP})\b(?P<gap>[^.;\d]{{0,40}}?)(?P<val>\d+(?:\.\d+)?|\.\d+)\s*(?P<unit>%|months?)?",
                    re.I)
_EP_NORM = {"progression free survival": "PFS", "median progression free survival": "PFS", "pfs": "PFS",
            "overall survival": "OS", "median overall survival": "OS", "os": "OS",
            "objective response rate": "ORR", "overall response rate": "ORR", "response rate": "ORR", "orr": "ORR",
            "disease free survival": "DFS", "dfs": "DFS", "event free survival": "EFS", "efs": "EFS",
            "pathologic complete response": "pCR", "pathological complete response": "pCR", "pcr": "pCR",
            "hazard ratio": "HR", "hr": "HR"}
_TRIAL_RE = re.compile(r"\b([A-Z][A-Z0-9]{2,}(?:-[A-Z0-9]+)*)\b")
_NOT_TRIALS = {"EGFR", "NSCLC", "SCLC", "FDA", "EMA", "HER2", "ALK", "ROS1", "KRAS", "BRAF", "MET", "RET", "NTRK",
               "PD", "PDL1", "CNS", "TKI", "TKIS", "ASCO", "ESMO", "NCCN", "AACR", "WCLC", "HR", "OS", "PFS", "ORR",
               "DFS", "EFS", "CI", "NCT", "PMID", "DOI", "USA", "US", "EU", "UK", "IV", "III", "II", "AND", "THE"}


def endpoint_values(text: str) -> dict[str, set[str]]:
    """{endpoint: {first value stated after each mention}}. "PFS was 23.7 months" -> {"PFS": {"23.7"}}."""
    out: dict[str, set[str]] = {}
    for m in _EP_RE.finditer(text or ""):
        if re.search(r"\d", m.group("gap")) or re.search(r"\b(?:phase|stage|exon|grade)\s*$", m.group("gap"), re.I):
            continue
        ep = _EP_NORM.get(m.group("ep").lower().replace("-", " "), m.group("ep").upper())
        out.setdefault(ep, set()).add(norm_value(m.group("val")))
    return out


def subjects(text: str) -> set[str]:
    trials = {t for t in _TRIAL_RE.findall(text or "") if t not in _NOT_TRIALS and not t.startswith("NCT")
              and not t.isdigit() and len(t) >= 4}
    return set(drug_names(text)) | trials


def _side(x: dict, sentence: dict, value: str) -> dict:
    return {"evidence_id": x["evidence_id"], "url": x.get("url"), "title": x.get("title"), "tier": x.get("trust_tier"),
            "date": x.get("source_date") or x.get("evidence_date"), "sentence_id": sentence["id"],
            "sentence": sentence["text"], "value": value}


def _tier12(claim: dict) -> list[dict]:
    return [x for x in claim["citations"] if x.get("verdict") in SUPPORTING and x.get("trust_tier") in (1, 2)]


def detect(claims: list[dict]) -> list[dict]:
    """Conflicts among active, verified claims. Each: {endpoint, subject, claims, sides: [side, side, ...]}."""
    active = [c for c in claims if c.get("status", "active") != "deleted" and c.get("verdict") in SUPPORTING]
    found: list[dict] = []
    seen: set[tuple] = set()

    # Within one claim: its tier 1-2 sources disagree on an endpoint it states.
    for c in active:
        wanted = endpoint_values(c["text"])
        # (endpoint, evidence id) -> [(citation, sentence, value)] from that source's cited sentences
        stated: dict[tuple[str, int], list[tuple[dict, dict, str]]] = {}
        for x in _tier12(c):
            for s in x.get("sentences", []):
                for ep, vals in endpoint_values(s["text"]).items():
                    if ep in wanted:
                        stated.setdefault((ep, x["evidence_id"]), []).extend((x, s, v) for v in sorted(vals))
        for ep in wanted:
            srcs = sorted(eid for e, eid in stated if e == ep)
            for a, b in combinations(srcs, 2):
                va = {v for _, _, v in stated[(ep, a)]}
                vb = {v for _, _, v in stated[(ep, b)]}
                if va & vb or ("within", c["id"], ep, a, b) in seen:
                    continue
                seen.add(("within", c["id"], ep, a, b))
                sides = [_side(*stated[(ep, a)][0]), _side(*stated[(ep, b)][0])]
                found.append({"endpoint": ep, "subject": ", ".join(sorted(subjects(c["text"]))) or "same claim",
                              "claims": [c["id"]], "kind": "within_claim", "sides": sides})

    # Across claims: same drug/trial, same endpoint, different values, different tier 1-2 sources.
    for c1, c2 in combinations(active, 2):
        common = subjects(c1["text"]) & subjects(c2["text"])
        if not common:
            continue
        e1, e2 = endpoint_values(c1["text"]), endpoint_values(c2["text"])
        s1, s2 = _tier12(c1), _tier12(c2)
        if not s1 or not s2 or {x["evidence_id"] for x in s1} & {x["evidence_id"] for x in s2}:
            continue
        for ep in set(e1) & set(e2):
            if e1[ep] & e2[ep]:
                continue
            key = ("across", c1["id"], c2["id"], ep)
            if key in seen:
                continue
            seen.add(key)
            sides = []
            for c, vals, srcs in ((c1, e1[ep], s1), (c2, e2[ep], s2)):
                pick = next(((x, s) for x in srcs for s in x.get("sentences", [])
                             if vals & endpoint_values(s["text"]).get(ep, set())),
                            (srcs[0], (srcs[0].get("sentences") or [{"id": "", "text": ""}])[0]))
                sides.append(_side(pick[0], pick[1], sorted(vals)[0]))
            found.append({"endpoint": ep, "subject": ", ".join(sorted(common)), "claims": [c1["id"], c2["id"]],
                          "kind": "across_claims", "sides": sides})
    return found


def conflict_lines(conflicts: list[dict]) -> list[str]:
    out = []
    for i, k in enumerate(conflicts, 1):
        out.append(f"{i}. {k['endpoint']} for {k['subject']}: sources disagree (both shown; neither is picked)")
        for s in k["sides"]:
            out.append(f"   - {s['value']} per E{s['evidence_id']} (tier {s['tier']}, {s['date'] or 'date not stated'}): "
                       f"{s['sentence_id']} \"{s['sentence']}\" {s['url'] or ''}".rstrip())
    return out
