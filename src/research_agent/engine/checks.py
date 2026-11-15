"""Deterministic checks that run before (and independently of) the LLM verifier.

- Numbers: every number in a claim must appear in its stored verbatim source sentence, and that sentence must
  appear in the cited excerpt.
- Design: a phase/design word in an evidence level must appear in a cited excerpt, else "not stated".
- Population: must come from a note whose population sentence was verified in the excerpt, else
  "population not stated".
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from ..evidence import terms

POPULATION_NOT_STATED = "population not stated"
LEVEL_NOT_STATED = "not stated"

# Numbers not glued to letters (HER2, G12C, CDK4/6, PD-1 are names, not numbers).
_NUM_RE = re.compile(r"(?<![A-Za-z0-9/\-.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?![A-Za-z0-9/])")
# Numbers that are part of a name rather than a result.
_NAME_NUM_RE = re.compile(r"\b(?:phase|exon|codon|stage|grade|line)\s+(?:\d+[a-z]?(?:/\d+)?|[ivx]+[ab]?)\b", re.I)
_ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "ib": "1b", "ia": "1a", "iia": "2a", "iib": "2b"}
_DESIGN_WORDS = (
    "randomized", "randomised", "single-arm", "single arm", "open-label", "double-blind", "placebo-controlled",
    "retrospective", "prospective", "cohort", "meta-analysis", "observational", "real-world", "case series",
    "non-randomized", "nonrandomized",
)


def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"[‐-―−]", "-", s)
    s = s.replace(" ", " ").replace(" ", " ")
    return re.sub(r"\s+", " ", s).strip()


def contains(haystack: str, needle: str) -> bool:
    n = normalize_text(needle).strip(" .")
    return bool(n) and n in normalize_text(haystack)


def _norm_number(whole: str, frac: str | None) -> str:
    raw = whole.replace(",", "") + (frac or "")
    try:
        d = Decimal(raw).normalize()
    except InvalidOperation:
        return raw
    s = format(d, "f")
    return s.rstrip("0").rstrip(".") if "." in s else s


def number_tokens(text: str) -> list[tuple[str, str]]:
    """(as written, normalized) for each result-like number in text."""
    cleaned = _NAME_NUM_RE.sub(lambda m: " " * len(m.group(0)), text or "")
    return [(m.group(0), _norm_number(m.group(1), m.group(2))) for m in _NUM_RE.finditer(cleaned)]


def numbers_in(text: str) -> set[str]:
    return {norm for _, norm in number_tokens(text)}


def check_numbers(claim_text: str, citations: list[dict]) -> tuple[bool, list[str]]:
    """citations: dicts with evidence_id, excerpt, source_sentence (newline-separated sentences).

    Returns (passed, problems). Claims without numbers pass trivially."""
    wanted = numbers_in(claim_text)
    if not wanted:
        return True, []
    problems: list[str] = []
    covered: set[str] = set()
    for c in citations:
        for sentence in filter(None, (c.get("source_sentence") or "").split("\n")):
            if not contains(c.get("excerpt") or "", sentence):
                problems.append(f"source sentence for E{c['evidence_id']} is not in the excerpt")
                continue
            covered |= numbers_in(sentence)
    missing = sorted(wanted - covered)
    if missing:
        problems.append("number(s) " + ", ".join(missing) + " not found in a verified source sentence")
    return not missing, problems


def _phases(text: str) -> set[str]:
    out = set()
    for m in re.finditer(r"\bphase\s*(\d+[ab]?(?:/\d+[ab]?)?|[ivx]+[ab]?(?:/[ivx]+[ab]?)?)\b", text, re.I):
        parts = [p.lower() for p in m.group(1).split("/")]
        out.add("/".join(_ROMAN.get(p, p) for p in parts))
    return out


def check_design(evidence_level: str | None, excerpts: list[str]) -> str:
    """Keep the level only if every design element it names is stated in a cited excerpt."""
    level = (evidence_level or "").strip()
    if not level or normalize_text(level) in ("not stated", "unknown", "n/a"):
        return LEVEL_NOT_STATED
    joined = normalize_text(" ".join(excerpts))
    for ph in _phases(level):
        if ph not in _phases(joined):
            return LEVEL_NOT_STATED
    for w in _DESIGN_WORDS:
        if w in normalize_text(level) and w not in joined:
            if not (w.startswith("randomi") and ("randomized" in joined or "randomised" in joined)):
                return LEVEL_NOT_STATED
    return level


def population_supported(population: str, sentence: str) -> bool:
    """A stated population must come from its sentence: substring, or >=80% of its terms present."""
    if not population or not sentence:
        return False
    if contains(sentence, population):
        return True
    pt = terms(population)
    return bool(pt) and len(pt & terms(sentence)) / len(pt) >= 0.8


def check_population(population: str | None, notes: list[dict], cited_ids: set[int]) -> str:
    """Accept the writer's population only if a verified note for a cited source states it."""
    if not population or normalize_text(population) in ("not stated", POPULATION_NOT_STATED, "null", "none"):
        return POPULATION_NOT_STATED
    for n in notes:
        d = n.get("details") or {}
        if not d.get("population") or not cited_ids & set(n["evidence_ids"]):
            continue
        if contains(d["population"], population) or contains(d.get("population_sentence") or "", population) \
                or population_supported(population, d.get("population_sentence") or ""):
            return population
    return POPULATION_NOT_STATED
