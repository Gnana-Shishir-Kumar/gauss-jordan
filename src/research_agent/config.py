"""Single source of configuration: models, source tiers, limits, budgets, timeouts.

Keys are read only from the environment (optionally populated from a local .env).
Keys are never logged or included in repr().
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1"

# Source allowlist by trust tier. Subdomains match (meetings.asco.org -> asco.org).
TIER_DOMAINS: dict[int, tuple[str, ...]] = {
    1: (  # regulatory, registries, guidelines
        "fda.gov", "ema.europa.eu", "clinicaltrials.gov", "cancer.gov", "nccn.org",
        "esmo.org", "asco.org", "ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov",
    ),
    2: (  # journals
        "nejm.org", "thelancet.com", "jamanetwork.com", "ascopubs.org",
        "annalsofoncology.org", "aacrjournals.org", "nature.com",
    ),
    3: (  # oncology news and company press releases (lower trust, leads only)
        "ascopost.com", "onclive.com", "targetedonc.com", "cancernetwork.com",
        "prnewswire.com", "businesswire.com", "globenewswire.com",
    ),
}
TIER_LABELS = {
    1: "Tier 1 · regulatory / registry / guideline",
    2: "Tier 2 · journal",
    3: "Tier 3 · news / press release (lower trust)",
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
    strong: str = "deepseek-ai/DeepSeek-V4-Pro"  # brief, planner, coverage, writer, repair
    fast: str = "deepseek-ai/DeepSeek-V4-Flash"  # worker notes, citation verification


@dataclass(frozen=True)
class Limits:
    min_queries: int = 3
    max_queries_per_round: int = 5
    max_rounds: int = 3
    max_repair_passes: int = 1
    search_results: int = 10
    extract_per_query: int = 2  # top URLs per worker sent to extraction
    extract_per_round: int = 10  # cap on URLs extracted per round
    extract_batch_size: int = 20  # Tavily extract accepts up to 20 URLs per call
    min_new_kept_per_round: int = 2
    min_independent_sources: int = 2  # distinct domains per aspect
    min_relevance: float = 0.15
    excerpt_chars: int = 1500
    near_duplicate_threshold: float = 0.7
    max_notes_for_writer: int = 40
    max_concurrency: int = 5
    round1_tiers: tuple[int, ...] = (1, 2)
    follow_up_tiers: tuple[int, ...] = (1, 2, 3)
    round1_depth: str = "basic"
    follow_up_depth: str = "advanced"
    default_anchor_days: int = 365


@dataclass(frozen=True)
class Budget:
    time_s: float = 420.0  # wall clock for the research phase (after approval)
    credits: float = 60.0  # Tavily credits per task


@dataclass(frozen=True)
class Network:
    llm_timeout_s: float = 120.0
    search_timeout_s: float = 30.0
    extract_timeout_s: float = 60.0
    max_retries: int = 3
    backoff_base_s: float = 1.0
    backoff_max_s: float = 10.0


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
    )
