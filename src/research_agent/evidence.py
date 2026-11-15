"""Pure evidence functions: URL normalization, hashing, excerpt selection, scoring, dedupe."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_TRACKING_PREFIXES = ("utm_",)
_TRACKING_KEYS = {"fbclid", "gclid", "mc_cid", "mc_eid", "ref", "ref_src", "igshid", "_ga", "spm"}
_STOPWORDS = set(
    "the and for are but not you all any can had her was one our out has his how its may new now "
    "old see two who did get let say she too use what when where which why with from that this "
    "have will your they them than then into over also more most some such only been were about "
    "after before does done just like many much other their there these those would could should "
    "between during while each very".split()
)


def normalize_url(url: str) -> str:
    """Canonical form for dedupe: lowercase host, no www, no fragment, no tracking params,
    sorted query, no trailing slash (except root)."""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url.strip()
    scheme = (parts.scheme or "https").lower()
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    netloc = host
    if parts.port and not ((scheme == "http" and parts.port == 80) or (scheme == "https" and parts.port == 443)):
        netloc = f"{host}:{parts.port}"
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith(_TRACKING_PREFIXES) and k.lower() not in _TRACKING_KEYS
    ]
    return urlunsplit((scheme, netloc, path, urlencode(sorted(query)), ""))


def domain_of(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def content_hash(text: str) -> str:
    norm = re.sub(r"\s+", " ", (text or "").lower()).strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) > 2 and w not in _STOPWORDS}


def overlap(query: str, text: str) -> float:
    q = terms(query)
    if not q:
        return 0.0
    return len(q & terms(text)) / len(q)


def best_excerpt(text: str, query: str, max_chars: int = 1500) -> str:
    """Pick the contiguous window of paragraphs (<= max_chars) with the most query-term overlap."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    # Split very long paragraphs into sentence-ish pieces so windows can stay under the limit.
    pieces: list[str] = []
    for p in paras:
        if len(p) <= max_chars:
            pieces.append(p)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", p)
            buf = ""
            for s in sentences:
                if buf and len(buf) + len(s) + 1 > max_chars:
                    pieces.append(buf)
                    buf = s
                else:
                    buf = f"{buf} {s}".strip()
            if buf:
                pieces.append(buf[:max_chars])
    q = terms(query)
    best, best_score = pieces[0][:max_chars] if pieces else text[:max_chars], -1.0
    for i in range(len(pieces)):
        window, j = pieces[i], i + 1
        while j < len(pieces) and len(window) + len(pieces[j]) + 1 <= max_chars:
            window = f"{window}\n{pieces[j]}"
            j += 1
        score = len(q & terms(window)) if q else 0
        if score > best_score:
            best, best_score = window, score
    return best


def relevance(backend_score: float, query: str, excerpt: str) -> float:
    """Blend backend score with lexical overlap; both in [0, 1]."""
    return round(0.5 * max(0.0, min(1.0, backend_score)) + 0.5 * overlap(query, excerpt), 4)


@dataclass
class Candidate:
    sub_query_id: int
    url: str  # normalized
    title: str
    published_date: str | None
    content: str  # extracted text, or the search snippet if extraction failed
    backend_score: float
    extracted: bool


@dataclass
class Decision:
    candidate: Candidate
    excerpt: str
    score: float
    content_hash: str
    kept: bool
    drop_reason: str | None


def assess(
    candidates: list[Candidate],
    query: str,
    seen_urls: set[str],
    seen_hashes: set[str],
    *,
    min_relevance: float,
    excerpt_chars: int,
) -> list[Decision]:
    """Score and dedupe candidates. Mutates seen_* with kept items. Nothing is discarded silently:
    every candidate gets a Decision with kept=False and a reason if it is dropped."""
    out: list[Decision] = []
    for c in candidates:
        excerpt = best_excerpt(c.content, query, excerpt_chars)
        h = content_hash(c.content)
        score = relevance(c.backend_score, query, excerpt)
        reason = None
        if not excerpt.strip():
            reason = "empty_content"
        elif c.url in seen_urls:
            reason = "duplicate_url"
        elif h in seen_hashes:
            reason = "duplicate_content"
        elif score < min_relevance:
            reason = "low_relevance"
        if reason is None:
            seen_urls.add(c.url)
            seen_hashes.add(h)
        out.append(Decision(c, excerpt, score, h, reason is None, reason))
    return out
