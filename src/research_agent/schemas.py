"""Pydantic models for every structured LLM output."""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

ClaimType = Literal["fact", "inference", "uncertain"]
Confidence = Literal["high", "moderate", "low"]
LLMVerdict = Literal["supported", "partial", "unsupported"]
SourceType = Literal[
    "regulatory_action", "guideline", "registry_entry", "peer_reviewed_article",
    "conference_abstract", "preprint", "press_release", "news",
]
SOURCE_TYPES: tuple[str, ...] = SourceType.__args__  # type: ignore[attr-defined]
VERDICTS = ("supported", "partial", "unsupported", "source_unavailable")


def coerce_evidence_id(v: Any) -> int:
    """Accept 12, "12", "E12", "[E12]"."""
    if isinstance(v, bool):
        raise ValueError("evidence id must be an integer")
    if isinstance(v, int):
        return v
    m = re.fullmatch(r"\[?\s*E?(\d+)\s*\]?", str(v).strip(), flags=re.I)
    if not m:
        raise ValueError(f"invalid evidence id {v!r}")
    return int(m.group(1))


def _ids(v: Any) -> list[int]:
    if v is None:
        return []
    if not isinstance(v, list):
        v = [v]
    return list(dict.fromkeys(coerce_evidence_id(x) for x in v))


def parse_partial_date(v: Any) -> str | None:
    """Normalize YYYY, YYYY-MM or YYYY-MM-DD; return None for anything else."""
    if v in (None, "", "null", "not stated"):
        return None
    s = str(v).strip()
    m = re.fullmatch(r"(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?", s)
    if not m:
        return None
    y, mo, d = m.group(1), m.group(2), m.group(3)
    try:
        if d:
            return date(int(y), int(mo), int(d)).isoformat()
        if mo:
            date(int(y), int(mo), 1)
            return f"{y}-{int(mo):02d}"
        return y
    except ValueError:
        return None


def _clean_strings(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values or []:
        s = " ".join(str(v).split())
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out


# --- brief / plan --------------------------------------------------------------

class Brief(BaseModel):
    needs_clarification: bool = False
    clarifying_question: str | None = None
    cancer_type: str | None = None
    setting: str | None = None
    question: str = ""
    anchor_date: str | None = None
    aspects: list[str] = Field(default_factory=list)

    @field_validator("anchor_date", mode="before")
    @classmethod
    def _anchor(cls, v: Any) -> str | None:
        d = parse_partial_date(v)
        if d is None:
            return None
        if len(d) == 4:
            return f"{d}-01-01"
        if len(d) == 7:
            return f"{d}-01"
        return d

    @field_validator("aspects", mode="before")
    @classmethod
    def _aspects(cls, v: Any) -> list[str]:
        return _clean_strings(v if isinstance(v, list) else [v] if v else [])[:6]

    @model_validator(mode="after")
    def _complete(self) -> "Brief":
        if self.needs_clarification:
            if not (self.clarifying_question or "").strip():
                raise ValueError("needs_clarification requires clarifying_question")
            return self
        missing = [f for f in ("cancer_type", "setting", "question") if not (getattr(self, f) or "").strip()]
        if missing:
            raise ValueError(f"missing fields: {', '.join(missing)}")
        if not self.aspects:
            raise ValueError("aspects must list at least one thing a complete answer covers")
        return self


class Plan(BaseModel):
    queries: list[str] = Field(min_length=1)

    @field_validator("queries")
    @classmethod
    def _check(cls, v: list[str], info: ValidationInfo) -> list[str]:
        ctx = info.context or {}
        out = _clean_strings(v)
        if len(out) < ctx.get("min", 1):
            raise ValueError(f"need at least {ctx.get('min', 1)} distinct queries, got {len(out)}")
        return out[: ctx.get("max", 5)]


# --- worker notes --------------------------------------------------------------

class NoteNumber(BaseModel):
    value: str
    sentence: str


class Note(BaseModel):
    text: str = Field(min_length=1)
    evidence_ids: list[int] = Field(min_length=1)
    numbers: list[NoteNumber] = Field(default_factory=list)
    design: str | None = None
    population: str | None = None
    population_sentence: str | None = None
    date: str | None = None

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def check_ids(cls, v: Any) -> list[int]:
        return _ids(v)

    @field_validator("date", mode="before")
    @classmethod
    def _date(cls, v: Any) -> str | None:
        return parse_partial_date(v)


class EvidenceMeta(BaseModel):
    evidence_id: int
    source_type: SourceType | None = None
    evidence_date: str | None = None

    @field_validator("evidence_id", mode="before")
    @classmethod
    def _id(cls, v: Any) -> int:
        return coerce_evidence_id(v)

    @field_validator("source_type", mode="before")
    @classmethod
    def _type(cls, v: Any) -> Any:
        return v if v in SOURCE_TYPES else None

    @field_validator("evidence_date", mode="before")
    @classmethod
    def _date(cls, v: Any) -> str | None:
        return parse_partial_date(v)


class WorkerNotes(BaseModel):
    notes: list[Note] = Field(default_factory=list)
    evidence: list[EvidenceMeta] = Field(default_factory=list)


# --- coverage ------------------------------------------------------------------

class AspectStatus(BaseModel):
    aspect: str
    status: Literal["covered", "thin", "conflict"]
    evidence_ids: list[int] = Field(default_factory=list)
    note: str = ""

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def check_ids(cls, v: Any) -> list[int]:
        return _ids(v)


class FollowUp(BaseModel):
    aspect: str = ""  # which thin/conflicting aspect this query is for (used for the per-aspect cap)
    query: str = Field(min_length=1)


class Coverage(BaseModel):
    aspects: list[AspectStatus] = Field(min_length=1)
    follow_up_queries: list[FollowUp] = Field(default_factory=list)

    @field_validator("follow_up_queries", mode="before")
    @classmethod
    def _q(cls, v: Any) -> list[dict]:
        out, seen = [], set()
        for item in v or []:
            d = {"query": item} if isinstance(item, str) else dict(item or {})
            q = " ".join(str(d.get("query") or "").split())
            if q and q.lower() not in seen:
                seen.add(q.lower())
                out.append({"aspect": " ".join(str(d.get("aspect") or "").split()), "query": q})
        return out


# --- writer --------------------------------------------------------------------

class ItemNumber(BaseModel):
    value: str
    sentence: str
    evidence_id: int

    @field_validator("evidence_id", mode="before")
    @classmethod
    def _id(cls, v: Any) -> int:
        return coerce_evidence_id(v)


class Item(BaseModel):
    text: str = Field(min_length=1)
    claim_type: ClaimType = "fact"
    evidence_ids: list[int] = Field(default_factory=list)
    date: str | None = None
    evidence_level: str = "not stated"
    population: str | None = None
    numbers: list[ItemNumber] = Field(default_factory=list)

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def check_ids(cls, v: Any) -> list[int]:
        return _ids(v)

    @field_validator("date", mode="before")
    @classmethod
    def _date(cls, v: Any) -> str | None:
        return parse_partial_date(v)

    @field_validator("evidence_level", mode="before")
    @classmethod
    def _level(cls, v: Any) -> str:
        s = " ".join(str(v or "").split())
        return s or "not stated"

    @model_validator(mode="after")
    def _needs_citation(self) -> "Item":
        if self.claim_type in ("fact", "inference") and not self.evidence_ids:
            raise ValueError(f"{self.claim_type} item must cite at least one evidence id")
        return self


class BottomLine(BaseModel):
    text: str
    confidence: Confidence
    reason: str = ""


class Draft(BaseModel):
    sufficient: bool
    insufficiency_note: str | None = None
    bottom_line: BottomLine | None = None
    items: list[Item] = Field(default_factory=list)
    conflicts: list[Item] = Field(default_factory=list)
    not_established: list[str] = Field(default_factory=list)
    check_before_acting: list[str] = Field(default_factory=list)

    @field_validator("not_established", "check_before_acting", mode="before")
    @classmethod
    def _strings(cls, v: Any) -> list[str]:
        return _clean_strings(v or [])


class RepairedItem(Item):
    claim_id: int


class Repair(BaseModel):
    claims: list[RepairedItem] = Field(default_factory=list)


# --- verifier ------------------------------------------------------------------

class CitationVerdict(BaseModel):
    evidence_id: int
    verdict: LLMVerdict
    note: str = ""

    @field_validator("evidence_id", mode="before")
    @classmethod
    def _id(cls, v: Any) -> int:
        return coerce_evidence_id(v)


class ClaimVerification(BaseModel):
    verdicts: list[CitationVerdict] = Field(default_factory=list)
