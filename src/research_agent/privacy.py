"""Input check: block questions that look like they contain patient identifiers.

Questions should be about clinical topics, not individual patients. This is a simple pattern check, not a
de-identification tool; it errs toward blocking obvious identifiers.
"""

from __future__ import annotations

import re

_DATE = (r"(?:\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}"
         r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}"
         r"|\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{4})")

PATTERNS: list[tuple[str, re.Pattern]] = [
    ("medical record number",
     re.compile(r"\b(?:MRN|medical\s+record(?:\s+(?:number|no\.?|#))?)\s*[:#]?\s*[A-Z0-9][A-Z0-9-]{3,}", re.I)),
    ("MRN-like number", re.compile(r"(?<![A-Za-z0-9.,/\-])\d{7,10}(?![A-Za-z0-9.,/\-])")),
    ("date of birth", re.compile(r"\b(?:DOB|D\.O\.B\.?|date\s+of\s+birth|born(?:\s+on)?)\s*[:\-]?\s*" + _DATE, re.I)),
    ("social security number", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("patient name",
     re.compile(r"\b(?:patient|pt\.?)\s+(?:(?:named|called)\s+)?[A-Z][a-z'\-]+\s+[A-Z][a-z'\-]+"
                r"|\b(?:patient|pt\.?)\s+(?:named|called)\s+[A-Z][a-z'\-]+"
                r"|\b(?:Mr|Mrs|Ms|Miss)\.?\s+[A-Z][a-z'\-]+")),
]


class IdentifierError(ValueError):
    def __init__(self, findings: list[str]):
        self.findings = findings
        super().__init__(
            "The question appears to contain patient identifiers (" + ", ".join(findings) + "). "
            "Ask about the clinical topic only; do not enter names, record numbers or dates of birth."
        )


def find_identifiers(text: str) -> list[str]:
    return [label for label, pat in PATTERNS if pat.search(text or "")]


def check_question(text: str) -> None:
    findings = find_identifiers(text)
    if findings:
        raise IdentifierError(findings)
