"""Sentence splitting and sentence IDs.

A fetched page's raw text is stored once per source and split deterministically into numbered sentences. Within a
brief, a source is evidence row E12 and its sentences are E12-1, E12-2, ... Claims and notes cite these IDs; the
verifier reads the cited sentences (plus one neighbour on each side for context) from the store, never from a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .evidence import terms

MAX_SENTENCES = 2000  # per source; very long pages are cut here
MAX_CHARS = 700  # longer "sentences" (tables, run-on lists) are split again at semicolons, then hard-cut

_ABBREV = {
    "al", "e.g", "i.e", "vs", "fig", "figs", "no", "nos", "dr", "approx", "ca", "inc", "ltd", "co", "corp", "st",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "u.s", "u.k", "e.u", "ref",
    "refs", "vol", "pp", "suppl", "mo", "mos", "yr", "yrs", "wk", "wks", "min", "max", "resp", "incl", "cf", "etc",
}
_SPLIT_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")
_ID_RE = re.compile(r"^E(\d+)-(\d+)$")
_ANY_ID_RE = re.compile(r"\bE(\d+)-(\d+)\b")


@dataclass(frozen=True)
class SentenceRef:
    evidence_id: int
    idx: int

    def __str__(self) -> str:
        return f"E{self.evidence_id}-{self.idx}"


def parse_ref(value: object) -> SentenceRef | None:
    """'E12-4' (also '[E12-4]', 'e12-4') -> SentenceRef(12, 4). Anything else -> None."""
    s = str(value or "").strip().strip("[]").strip()
    m = _ID_RE.match(s.upper())
    return SentenceRef(int(m.group(1)), int(m.group(2))) if m else None


def refs_in(text: str) -> list[SentenceRef]:
    return [SentenceRef(int(a), int(b)) for a, b in _ANY_ID_RE.findall(text or "")]


def fmt_refs(refs) -> str:
    return ",".join(str(r) for r in refs)


def _ends_with_abbrev(chunk: str) -> bool:
    m = re.search(r"(\S+)$", chunk)
    if not m:
        return False
    tok = m.group(1).lower().rstrip(".!?\"')]").lstrip("(\"'[")
    if tok in _ABBREV:
        return True
    return bool(re.fullmatch(r"[a-z]", tok))  # initials: "J. Smith"


def _split_block(block: str) -> list[str]:
    parts: list[str] = []
    start = 0
    for m in _SPLIT_RE.finditer(block):
        # m.start() is right after the punctuation; keep closing quotes/brackets with the sentence
        end = m.start()
        while end < len(block) and block[end] in "\"')]":
            end += 1
        chunk = block[start:end].strip()
        if _ends_with_abbrev(chunk):
            continue
        if chunk:
            parts.append(chunk)
        start = m.end()
    tail = block[start:].strip()
    if tail:
        parts.append(tail)
    out: list[str] = []
    for p in parts:
        if len(p) <= MAX_CHARS:
            out.append(p)
            continue
        buf = ""
        for piece in re.split(r"(?<=;)\s+", p):
            if buf and len(buf) + len(piece) + 1 > MAX_CHARS:
                out.append(buf)
                buf = piece
            else:
                buf = f"{buf} {piece}".strip()
        while len(buf) > MAX_CHARS:
            out.append(buf[:MAX_CHARS])
            buf = buf[MAX_CHARS:]
        if buf:
            out.append(buf)
    return out


def _useful(s: str) -> bool:
    words = s.split()
    return len(words) >= 3 or (len(words) >= 1 and any(ch.isdigit() for ch in s) and len(s) >= 4)


def split_sentences(text: str) -> list[str]:
    """Deterministic: the same text always gives the same numbered list. Lines are blocks (tables, headings)."""
    out: list[str] = []
    for line in re.split(r"\n+", (text or "").replace("\r", "\n")):
        block = " ".join(line.split())
        if not block:
            continue
        for s in _split_block(block):
            if _useful(s):
                out.append(s)
                if len(out) >= MAX_SENTENCES:
                    return out
    return out


def select(sentences: list[str], query: str, k: int = 14, max_chars: int = 2600) -> list[int]:
    """Indices (1-based, in page order) of the sentences most relevant to the query, for the worker's view.
    Sentences with numbers get a small bonus because results live there."""
    q = terms(query)
    scored = []
    for i, s in enumerate(sentences, 1):
        t = terms(s)
        score = len(q & t) + (0.5 if re.search(r"\d", s) else 0.0)
        if score > 0:
            scored.append((score, -i, i))
    chosen: list[int] = []
    total = 0
    for _, _, i in sorted(scored, reverse=True):
        if len(chosen) >= k:
            break
        n = len(sentences[i - 1])
        if chosen and total + n > max_chars:
            continue
        chosen.append(i)
        total += n
    if not chosen and sentences:
        chosen = list(range(1, min(len(sentences), 4) + 1))
    return sorted(chosen)
