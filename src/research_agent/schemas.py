"""Pydantic models for every structured LLM output, plus shared literal types."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

ClaimType = Literal["fact", "inference", "uncertain"]
LLMVerdict = Literal["supported", "partial", "unsupported"]
VERDICTS = ("supported", "partial", "unsupported", "source_unavailable")


def _clean_queries(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        q = " ".join(str(v).split())
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out


def _coerce_evidence_id(v: Any) -> int:
    """Accept 12, "12", "E12", "[E12]"."""
    if isinstance(v, bool):
        raise ValueError("evidence id must be an integer")
    if isinstance(v, int):
        return v
    m = re.fullmatch(r"\[?\s*E?(\d+)\s*\]?", str(v).strip(), flags=re.I)
    if not m:
        raise ValueError(f"invalid evidence id {v!r}")
    return int(m.group(1))


class Plan(BaseModel):
    sub_queries: list[str] = Field(min_length=1)

    @field_validator("sub_queries")
    @classmethod
    def _check(cls, v: list[str], info: ValidationInfo) -> list[str]:
        ctx = info.context or {}
        lo, hi = ctx.get("min", 1), ctx.get("max", 5)
        out = _clean_queries(v)
        if len(out) < lo:
            raise ValueError(f"need at least {lo} distinct sub-queries, got {len(out)}")
        return out[:hi]


class FollowUpPlan(BaseModel):
    queries: list[str] = Field(min_length=1)

    @field_validator("queries")
    @classmethod
    def _check(cls, v: list[str], info: ValidationInfo) -> list[str]:
        out = _clean_queries(v)
        if not out:
            raise ValueError("need at least one query")
        return out[: (info.context or {}).get("max", len(out))]


class DraftClaim(BaseModel):
    text: str = Field(min_length=1)
    claim_type: ClaimType = "fact"
    evidence_ids: list[int] = Field(default_factory=list)

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def _ids(cls, v: Any) -> list[int]:
        if v is None:
            return []
        if not isinstance(v, list):
            v = [v]
        return list(dict.fromkeys(_coerce_evidence_id(x) for x in v))

    @model_validator(mode="after")
    def _needs_citation(self) -> "DraftClaim":
        if self.claim_type in ("fact", "inference") and not self.evidence_ids:
            raise ValueError(f"{self.claim_type} claim must cite at least one evidence id")
        return self


class Draft(BaseModel):
    sufficient: bool
    insufficiency_note: str | None = None
    claims: list[DraftClaim] = Field(default_factory=list)


class CitationVerdict(BaseModel):
    evidence_id: int
    verdict: LLMVerdict
    note: str = ""

    @field_validator("evidence_id", mode="before")
    @classmethod
    def _id(cls, v: Any) -> int:
        return _coerce_evidence_id(v)


class ClaimVerification(BaseModel):
    verdicts: list[CitationVerdict] = Field(default_factory=list)
