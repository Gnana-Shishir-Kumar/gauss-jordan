"""Single source of configuration: models, source tiers, limits, budgets, timeouts.

Keys are read only from the environment (optionally populated from a local .env).
Keys are never logged or included in repr().
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1"

# Search allowlist, grouped by the tier a domain's pages USUALLY get. These groups only decide which domains a search
# round may hit (Tavily include_domains, strict). A source's actual trust tier is assigned in code from its type and
# its identifier check (sources.assign_tier): a journal page is tier 2 only if its PMID/DOI resolves to the same title.
# Subdomains match the longest listed domain (meetings.asco.org -> group 3, asco.org -> group 1).
# nccn.org is deliberately absent: NCCN's terms prohibit automated access. The user supplies the guideline version.
TIER_DOMAINS: dict[int, tuple[str, ...]] = {
    1: (  # regulators, registries, guideline publishers
        "fda.gov", "ema.europa.eu", "clinicaltrials.gov", "cancer.gov", "esmo.org", "asco.org",
    ),
    2: (  # journals and PubMed / PubMed Central
        "nejm.org", "thelancet.com", "jamanetwork.com", "ascopubs.org", "annalsofoncology.org",
        "aacrjournals.org", "nature.com", "ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov",
    ),
    3: (  # conference abstract hosts
        "meetings.asco.org", "oncologypro.esmo.org", "aacr.org",
    ),
    4: (  # oncology news and company press releases (leads only)
        "ascopost.com", "onclive.com", "targetedonc.com", "cancernetwork.com",
        "prnewswire.com", "businesswire.com", "globenewswire.com",
    ),
}
BLOCKED_DOMAINS = ("nccn.org",)  # never searched or fetched
TIER_LABELS = {
    1: "Tier 1 · regulatory / registry / guideline",
    2: "Tier 2 · peer-reviewed journal (identifier confirmed)",
    3: "Tier 3 · conference abstract (preliminary)",
    4: "Tier 4 · press release / news (lower trust)",
}


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader: KEY=VALUE lines, # comments. Never overrides existing env vars."""
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class Models:
    strong: str = "deepseek-ai/DeepSeek-V4-Pro-0813"  # brief, planner, coverage, writer, repair
    fast: str = "deepseek-ai/DeepSeek-V4.1-Flash-1017"  # worker notes, citation verification


@dataclass(frozen=True)
class Limits:
    min_queries: int = 3
    max_queries_per_round: int = 5
    max_follow_ups_per_aspect: int = 1  # per follow-up round, only for thin/conflicting aspects
    max_repair_rounds: int = 2  # per failed claim, one claim at a time
    search_results: int = 10
    extract_per_query: int = 2  # top URLs per worker sent to extraction
    extract_per_round: int = 10  # cap on URLs extracted per round
    extract_batch_size: int = 20  # Tavily extract accepts up to 20 URLs per call
    min_independent_sources: int = 2  # distinct domains per aspect
    min_relevance: float = 0.15
    sentences_per_source: int = 14  # sentences each worker sees per source (by relevance, with their IDs)
    sentence_chars_per_source: int = 2600
    near_duplicate_threshold: float = 0.7
    max_notes_for_writer: int = 40
    max_concurrency: int = 5
    round1_tiers: tuple[int, ...] = (1, 2)
    follow_up_tiers: tuple[int, ...] = (1, 2, 3, 4)
    round1_depth: str = "advanced"  # evidence searches; planning probes would use basic
    follow_up_depth: str = "advanced"
    chunks_per_source: int = 3  # advanced search only
    max_research_rounds: int = 4  # safety cap; reaching it is reported as quota_reached
    repair_search_results: int = 5
    repair_extract_per_search: int = 2
    default_anchor_days: int = 365


@dataclass(frozen=True)
class Budget:
    time_s: float = 420.0  # wall clock for the research phase (after approval)
    tavily_calls: int = 40  # one pooled budget of Tavily calls (search + extract) per brief
    tavily_repair_reserve: int = 8  # of which this many are reserved for targeted repair searches
    pubmed_calls: int = 60
    ctgov_calls: int = 40
    openfda_calls: int = 20
    crossref_calls: int = 40


@dataclass(frozen=True)
class Network:
    llm_timeout_s: float = 480.0  # the writer took 269 s in a live run (2026-11-17); a ceiling, not a delay
    llm_max_tokens: int = 16384  # these models spend many output tokens on reasoning before the JSON
    search_timeout_s: float = 30.0
    extract_timeout_s: float = 60.0
    id_timeout_s: float = 20.0  # PubMed / ClinicalTrials.gov / Crossref lookups
    max_retries: int = 3
    backoff_base_s: float = 1.0
    backoff_max_s: float = 10.0
    retry_after_max_s: float = 60.0  # longest Retry-After we honor on a 429
    # Extra waits before re-trying the writer after its normal retries hit a transient error (connection drop,
    # timeout, 5xx). A writer failure would otherwise throw away the whole research phase.
    patient_waits_s: tuple[float, ...] = (15.0, 45.0)


@dataclass(frozen=True)
class Settings:
    models: Models = field(default_factory=Models)
    limits: Limits = field(default_factory=Limits)
    budget: Budget = field(default_factory=Budget)
    network: Network = field(default_factory=Network)
    db_path: str = "research.db"
    cache_ttl_days: int = 7
    nebius_base_url: str = NEBIUS_BASE_URL
    nebius_api_key: str = field(default="", repr=False)
    tavily_api_key: str = field(default="", repr=False)
    ncbi_api_key: str = field(default="", repr=False)  # optional: raises the PubMed rate limit from 3/s to 10/s
    contact_email: str = ""  # optional: sent to NCBI/Crossref as their polite-pool contact, only if set

    def missing_keys(self) -> list[str]:
        missing = []
        if not self.nebius_api_key:
            missing.append("NEBIUS_API_KEY")
        if not self.tavily_api_key:
            missing.append("TAVILY_API_KEY")
        return missing

    def model_ids(self) -> dict[str, str]:
        return {"strong": self.models.strong, "fast": self.models.fast}


def get_settings(env_file: str | Path = ".env") -> Settings:
    load_dotenv(env_file)
    defaults = Models()
    return Settings(
        models=Models(
            strong=os.environ.get("RESEARCH_STRONG_MODEL", defaults.strong),
            fast=os.environ.get("RESEARCH_FAST_MODEL", defaults.fast),
        ),
        db_path=os.environ.get("RESEARCH_DB_PATH", "research.db"),
        nebius_api_key=os.environ.get("NEBIUS_API_KEY", ""),
        tavily_api_key=os.environ.get("TAVILY_API_KEY", ""),
        ncbi_api_key=os.environ.get("NCBI_API_KEY", ""),
        contact_email=os.environ.get("RESEARCH_CONTACT_EMAIL", ""),
    )
