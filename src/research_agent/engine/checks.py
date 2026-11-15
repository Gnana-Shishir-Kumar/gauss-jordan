"""Deterministic checks that run before (and independently of) the LLM verifier.

- Numbers: every number in a claim must appear in its stored verbatim source sentence, and that sentence must
  appear in the cited excerpt (as whole words, not inside a longer number or word).
- Design: a phase/design word in an evidence level must be stated (not negated) in a cited excerpt, else "not stated".
- Population: must come from a note whose population sentence was verified in the excerpt, else
  "population not stated". A negated term ("without brain metastases", "non-small cell") never matches.
- Dates: an item's date must match a cited source's date; items dated before the anchor are not "new".
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from ..evidence import terms

POPULATION_NOT_STATED = "population not stated"
LEVEL_NOT_STATED = "not stated"
MIN_SENTENCE_WORDS = 4  # a "source sentence" shorter than this is a fragment, not a sentence

# Numbers not glued to letters (HER2, G12C, CDK4/6, PD-1 are names, not numbers). A hyphen after a letter is part
# of a name (PD-1, COVID-19); a hyphen after a digit is a range (30-40%, 0.58-0.85), so both ends are numbers.
_NUM_RE = re.compile(r"(?<![A-Za-z0-9/.])(?<![A-Za-z]-)(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?![A-Za-z0-9/])")
# ISO dates (2025-06-01, 2025-06): only the year is checked as a number.
_ISO_DATE_RE = re.compile(r"\b(\d{4})(-\d{1,2}(?:-\d{1,2})?)\b")
# Numbers that are part of a name rather than a result.
_NAME_NUM_RE = re.compile(r"\b(?:phase|exon|codon|stage|grade|line)\s+(?:\d+[a-z]?(?:/\d+)?|[ivx]+[ab]?)\b", re.I)
_ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "ib": "1b", "ia": "1a", "iia": "2a", "iib": "2b"}
_DESIGN_WORDS = (
    "randomized", "randomised", "single-arm", "single arm", "open-label", "double-blind", "placebo-controlled",
    "retrospective", "prospective", "cohort", "meta-analysis", "observational", "real-world", "case series",
    "non-randomized", "nonrandomized",
)
_NEGATION_PREFIXES = ("non-", "non ", "not ", "no ", "without ")
_NEGATION_CUES = {"without", "no", "not", "non", "excluding", "excluded", "except", "never"}


def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"[‐-―−]", "-", s)
    s = s.replace(" ", " ").replace(" ", " ")
    return re.sub(r"\s+", " ", s).strip()


def _word_char(h: str, i: int, step: int) -> bool:
    """True if h[i] continues a word: alphanumeric, or a hyphen joined to an alphanumeric (HER2-low, non-small)."""
    if not 0 <= i < len(h):
        return False
    if h[i].isalnum():
        return True
    j = i + step
    return h[i] == "-" and 0 <= j < len(h) and h[j].isalnum()


def contains(haystack: str, needle: str) -> bool:
    """Whole-word substring match after normalization: '5%' is not in '45%', 'small cell' is not in 'non-small cell'."""
    n = normalize_text(needle).strip(" .")
    if not n:
        return False
    h = normalize_text(haystack)
    start = h.find(n)
    while start != -1:
        end = start + len(n)
        left_ok = not n[0].isalnum() or not _word_char(h, start - 1, -1)
        right_ok = not n[-1].isalnum() or not _word_char(h, end, 1)
        if left_ok and right_ok:
            return True
        start = h.find(n, start + 1)
    return False


def is_sentence(s: str) -> bool:
    return len((s or "").split()) >= MIN_SENTENCE_WORDS


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
    blank = lambda m: " " * len(m.group(0))  # noqa: E731 - keep offsets so "as written" stays accurate
    cleaned = _NAME_NUM_RE.sub(blank, text or "")
    cleaned = _ISO_DATE_RE.sub(lambda m: m.group(1) + " " * len(m.group(2)), cleaned)
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
            if not is_sentence(sentence):
                problems.append(f"source sentence for E{c['evidence_id']} is a fragment, not a sentence")
                continue
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


def affirmed(word: str, text: str) -> bool:
    """True if `word` appears as a whole word in normalized `text` at least once without a negation in front
    ("non-randomized", "not randomized", "no placebo-controlled")."""
    for m in re.finditer(re.escape(word), text):
        i, j = m.start(), m.end()
        if (i > 0 and text[i - 1].isalnum()) or (j < len(text) and text[j].isalnum()):
            continue
        if not text[:i].endswith(_NEGATION_PREFIXES):
            return True
    return False


def check_design(evidence_level: str | None, excerpts: list[str]) -> str:
    """Keep the level only if every design element it names is stated (not negated) in a cited excerpt."""
    level = (evidence_level or "").strip()
    if not level or normalize_text(level) in ("not stated", "unknown", "n/a"):
        return LEVEL_NOT_STATED
    joined = normalize_text(" ".join(excerpts))
    for ph in _phases(level):
        if ph not in _phases(joined):
            return LEVEL_NOT_STATED
    norm_level = normalize_text(level)
    for w in _DESIGN_WORDS:
        if not affirmed(w, norm_level) or affirmed(w, joined):
            continue
        if w.startswith("randomi") and (affirmed("randomized", joined) or affirmed("randomised", joined)):
            continue
        return LEVEL_NOT_STATED
    return level


def _negated_terms(text: str) -> set[str]:
    """Terms directly after a negation cue: 'without brain metastases' -> {'brain'}, 'non-small cell' -> {'small'}."""
    toks = re.findall(r"[a-z0-9]+", normalize_text(text))
    return {toks[k + 1] for k, t in enumerate(toks[:-1]) if t in _NEGATION_CUES}


def population_supported(population: str, sentence: str) -> bool:
    """A stated population must come from its sentence: whole-word substring, or >=80% of its terms present with no
    term negated in the sentence that is not negated the same way in the population."""
    if not population or not sentence:
        return False
    if contains(sentence, population):
        return True
    pt = terms(population)
    if not pt or (_negated_terms(sentence) - _negated_terms(population)) & pt:
        return False
    return len(pt & terms(sentence)) / len(pt) >= 0.8


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


# --- dates -----------------------------------------------------------------------

def dates_match(a: str | None, b: str | None) -> bool:
    """Partial ISO dates match when one is a prefix of the other ('2025-06' ~ '2025-06-01')."""
    return bool(a and b) and (a.startswith(b) or b.startswith(a))


def check_item_date(item_date: str | None, source_dates: list[str]) -> str | None:
    """Keep the writer's date only if a cited source carries a matching date; otherwise use the latest source date."""
    dates = sorted(d for d in source_dates if d)
    if item_date and any(dates_match(item_date, d) for d in dates):
        return item_date
    return dates[-1] if dates else None


def before_anchor(item_date: str | None, anchor: str | None) -> bool:
    """True if the whole period of item_date ends before the anchor date ('2024' is before '2025-01-01')."""
    if not item_date or not anchor:
        return False
    end = item_date + {4: "-12-31", 7: "-31"}.get(len(item_date), "")
    return end < anchor
