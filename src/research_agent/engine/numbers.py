"""Gate 1 of the verifier: deterministic number check. No LLM call.

Every number, percentage, duration, hazard ratio, confidence interval, p-value, date and sample size in a claim must
appear in the claim's cited sentences, verbatim or after normalization:
  "0.70" = ".70" = ".7"; "1,024" = "1024"; "12 months" = "12-month"; "61%" = "61 percent"; "p<.001" = "P < 0.001";
  "June 2025" matches "June 1, 2025" (a less precise date matches a more precise one, never the reverse).
Confidence intervals in the claim must be ordered: lower < point estimate < upper.
Numbers that are names (phase 3, stage IV, exon 19, HER2, CDK4/6, PD-1, NCT IDs, PMIDs, DOIs) are not checked here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_MONTH_RE = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
             r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?")
_NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+"

# Identifiers and names whose digits are not results.
_ID_RE = re.compile(r"\bPMID:?\s*\d+|\bPMCID:?\s*PMC\d+|\bPMC\d+|\bNCT\d{8}\b|\b10\.\d{4,9}/[^\s\"<>)\]]+", re.I)
_NAME_NUM_RE = re.compile(
    r"\b(?:phase|exon|codon|stage|grade|line|cycle|cohort|part|arm|version|v)\s*"
    r"(?:\d+[a-z]?(?:\s*/\s*\d+[a-z]?)?(?:\.\d{4})?|[ivx]+[abc]?(?:/[ivx]+[abc]?)?)\b", re.I)
_DATE_RES = [
    # June 1, 2025 / Jun. 1 2025
    (re.compile(rf"\b({_MONTH_RE})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+((?:19|20)\d{{2}})\b", re.I), "mdy"),
    # 1 June 2025
    (re.compile(rf"\b(\d{{1,2}})\s+({_MONTH_RE})\s+((?:19|20)\d{{2}})\b", re.I), "dmy"),
    # June 2025
    (re.compile(rf"\b({_MONTH_RE})\s+((?:19|20)\d{{2}})\b", re.I), "my"),
    # 2025-06-01 / 2025-06
    (re.compile(r"\b((?:19|20)\d{2})-(\d{1,2})(?:-(\d{1,2}))?\b"), "iso"),
]
_P_RE = re.compile(rf"\bp\s*(<=|>=|≤|≥|<|>|=)\s*({_NUM})", re.I)
_RATIO = r"(?:hr|or|rr|hazard ratio|odds ratio|risk ratio|relative risk)"
# HR 0.70 (95% CI, 0.58-0.85) / HR, 0.70; 95% CI 0.58 to 0.85 / HR 0.70 (0.58-0.85)
_CI_RE = re.compile(
    rf"(?:\b{_RATIO}\b[\s,:=]*(?:of|was|were|is)?\s*(?P<pt>{_NUM})[\s,;(]*)?"
    rf"(?:(?P<lvl>\d{{2}}(?:\.\d+)?)\s*%\s*(?:ci|confidence interval)\b[\s,:]*|(?<=\()|(?<=\[))"
    rf"(?P<lo>-?(?:{_NUM}))\s*(?:-|–|to|,)\s*(?P<hi>-?(?:{_NUM}))", re.I)
_TOKEN_RE = re.compile(rf"(?<![\w/.])(?<![A-Za-z]-)(?P<n>{_NUM})(?P<glued>[A-Za-z/%]*)")
_UNIT_AFTER = re.compile(
    r"\s*(?:-\s*)?(%|percent\b|per cent\b|months?\b|mos?\b|years?\b|yrs?\b|weeks?\b|wks?\b|days?\b|"
    r"mg/m2\b|mg/m²|mg/kg\b|mg\b|patients\b|participants\b|subjects\b|people\b|women\b|men\b)", re.I)
_UNIT_NORM = {"%": "%", "percent": "%", "per cent": "%", "month": "month", "months": "month", "mo": "month",
              "mos": "month", "year": "year", "years": "year", "yr": "year", "yrs": "year", "week": "week",
              "weeks": "week", "wk": "week", "wks": "week", "day": "day", "days": "day", "mg/m2": "mg/m2",
              "mg/m²": "mg/m2", "mg/kg": "mg/kg", "mg": "mg", "patients": "n", "participants": "n", "subjects": "n",
              "people": "n", "women": "n", "men": "n"}
_GLUED_UNITS = {"%": "%", "mg": "mg", "mo": "month", "mos": "month", "mg/kg": "mg/kg", "mg/m2": "mg/m2"}
_N_BEFORE = re.compile(r"\bn\s*=\s*$", re.I)


def norm_value(raw: str) -> str:
    s = raw.replace(",", "")
    if s.startswith("."):
        s = "0" + s
    try:
        d = Decimal(s).normalize()
    except InvalidOperation:
        return s
    out = format(d, "f")
    return out.rstrip("0").rstrip(".") if "." in out else out


def _norm_text(s: str) -> str:
    s = (s or "").replace("−", "-").replace("‐", "-").replace("‑", "-").replace("–", "-")
    return s.replace("—", "-").replace(" ", " ").replace(" ", " ").replace(" ", " ")


@dataclass(frozen=True)
class Quantity:
    raw: str
    kind: str  # number | date | pvalue
    value: str  # normalized value; dates as YYYY, YYYY-MM or YYYY-MM-DD
    unit: str | None = None  # %, month, year, week, day, mg, mg/kg, mg/m2, n; comparator for p-values

    def label(self) -> str:
        return self.raw.strip()


@dataclass(frozen=True)
class Interval:
    raw: str
    lo: Decimal
    hi: Decimal
    point: Decimal | None = None

    def ordered(self) -> bool:
        if self.point is None:
            return self.lo < self.hi
        return self.lo < self.point < self.hi


def _dec(raw: str | None) -> Decimal | None:
    if raw is None:
        return None
    try:
        return Decimal(norm_value(raw))
    except InvalidOperation:
        return None


def _blank(s: str, start: int, end: int) -> str:
    return s[:start] + " " * (end - start) + s[end:]


def _month(tok: str) -> int:
    return _MONTHS[tok.lower().rstrip(".")[:3]]


def _iso(y: str, m: int | None = None, d: int | None = None) -> str | None:
    if m is not None and not 1 <= m <= 12:
        return None
    if d is not None and not 1 <= d <= 31:
        return None
    return y if m is None else f"{y}-{m:02d}" if d is None else f"{y}-{m:02d}-{d:02d}"


def extract(text: str) -> tuple[list[Quantity], list[Interval]]:
    """All checkable quantities in text, plus confidence intervals (for the ordering check)."""
    s = _norm_text(text)
    intervals = []
    for m in _CI_RE.finditer(s):
        lo, hi, pt = _dec(m.group("lo")), _dec(m.group("hi")), _dec(m.group("pt"))
        if lo is None or hi is None:
            continue
        if m.group("lvl") is None and pt is None:
            continue  # a bare "(3-5)" is a range, not a confidence interval
        intervals.append(Interval(m.group(0), lo, hi, pt))
    for rx in (_ID_RE, _NAME_NUM_RE):
        for m in rx.finditer(s):
            s = _blank(s, m.start(), m.end())
    out: list[Quantity] = []
    for rx, kind in _DATE_RES:
        for m in rx.finditer(s):
            g = m.groups()
            if kind == "mdy":
                iso = _iso(g[2], _month(g[0]), int(g[1]))
            elif kind == "dmy":
                iso = _iso(g[2], _month(g[1]), int(g[0]))
            elif kind == "my":
                iso = _iso(g[1], _month(g[0]))
            else:
                iso = _iso(g[0], int(g[1]), int(g[2]) if g[2] else None)
            if iso:
                out.append(Quantity(m.group(0), "date", iso))
                s = _blank(s, m.start(), m.end())
    for m in _P_RE.finditer(s):
        comp = {"≤": "<=", "≥": ">="}.get(m.group(1), m.group(1))
        out.append(Quantity(m.group(0), "pvalue", norm_value(m.group(2)), comp))
        s = _blank(s, m.start(), m.end())
    for m in _TOKEN_RE.finditer(s):
        raw, glued = m.group("n"), m.group("glued")
        unit = None
        if glued:
            unit = _GLUED_UNITS.get(glued.lower())
            if unit is None:
                continue  # glued to letters: a name (G12C-like) or an identifier, not a result
        else:
            u = _UNIT_AFTER.match(s, m.end())
            if u:
                unit = _UNIT_NORM.get(u.group(1).lower())
            elif _N_BEFORE.search(s[: m.start()]):
                unit = "n"
        value = norm_value(raw)
        if unit is None and re.fullmatch(r"(?:19|20)\d{2}", raw):
            out.append(Quantity(raw, "date", raw))  # a bare year is a date
        else:
            out.append(Quantity(raw + (glued or ""), "number", value, unit))
    return out, intervals


def quantities(text: str) -> list[Quantity]:
    return extract(text)[0]


def numbers_in(text: str) -> set[str]:
    """Normalized values of every checkable quantity (for quick subset checks, e.g. the bottom line)."""
    return {q.value for q in quantities(text)}


def _matches(c: Quantity, s: Quantity) -> bool:
    if c.kind == "date":
        if s.kind == "date":
            return s.value.startswith(c.value)  # "2025-06" in the claim matches "2025-06-01" in a sentence
        return False
    if c.kind != s.kind or c.value != s.value:
        return False
    if c.kind == "pvalue":
        return c.unit == s.unit
    return c.unit is None or s.unit is None or c.unit == s.unit


@dataclass
class NumberCheck:
    status: str  # pass | fail | n/a (no numbers in the claim)
    problems: list[str] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status != "fail"

    def as_dict(self) -> dict:
        return {"status": self.status, "problems": self.problems, "checked": self.checked}


def check(claim_text: str, sentences: list[str]) -> NumberCheck:
    """Gate 1. sentences: the texts of the claim's cited sentences (support may combine several)."""
    wanted, intervals = extract(claim_text)
    if not wanted and not intervals:
        return NumberCheck("n/a")
    have: list[Quantity] = [q for s in sentences for q in quantities(s)]
    problems = []
    for iv in intervals:
        if not iv.ordered():
            what = f"lower {iv.lo} < point {iv.point} < upper {iv.hi}" if iv.point is not None else \
                f"lower {iv.lo} < upper {iv.hi}"
            problems.append(f"confidence interval out of order in '{iv.raw.strip()}' (needs {what})")
    for q in wanted:
        if not any(_matches(q, h) for h in have):
            unit = f" {q.unit}" if q.unit and q.kind == "number" else ""
            problems.append(f"'{q.label()}' ({q.kind}{unit}) not found in the cited sentences")
    return NumberCheck("fail" if problems else "pass", problems, [q.label() for q in wanted])
