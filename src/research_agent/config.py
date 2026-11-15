"""Single source of configuration: models, hard limits, timeouts, retry policy.

Keys are read only from the environment (optionally populated from a local .env).
Keys are never logged or included in repr().
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1"


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
    planner: str = "deepseek-ai/DeepSeek-V4-Flash"
    writer: str = "NousResearch/Hermes-4-405B"
    verifier: str = "deepseek-ai/DeepSeek-V4-Flash"


@dataclass(frozen=True)
class Limits:
    min_sub_queries: int = 3
    max_sub_queries: int = 5
    max_follow_up_rounds: int = 1
    max_results_per_search: int = 5
    min_kept_evidence_per_sub_query: int = 2  # below this, a sub-query is "weak" -> follow-up
    min_relevance: float = 0.15  # evidence scoring below this is dropped (kept=0)
    excerpt_chars: int = 1500
    max_evidence_for_writer: int = 20
    max_concurrency: int = 5


@dataclass(frozen=True)
class Network:
    llm_timeout_s: float = 90.0
    search_timeout_s: float = 30.0
    extract_timeout_s: float = 45.0
    max_retries: int = 3
    backoff_base_s: float = 1.0
    backoff_max_s: float = 10.0


@dataclass(frozen=True)
class Settings:
    models: Models = field(default_factory=Models)
    limits: Limits = field(default_factory=Limits)
    network: Network = field(default_factory=Network)
    db_path: str = "research.db"
    tavily_search_depth: str = "advanced"
    tavily_chunks_per_source: int = 3
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


def get_settings(env_file: str | Path = ".env") -> Settings:
    load_dotenv(env_file)
    defaults = Models()
    return Settings(
        models=Models(
            planner=os.environ.get("RESEARCH_PLANNER_MODEL", defaults.planner),
            writer=os.environ.get("RESEARCH_WRITER_MODEL", defaults.writer),
            verifier=os.environ.get("RESEARCH_VERIFIER_MODEL", defaults.verifier),
        ),
        db_path=os.environ.get("RESEARCH_DB_PATH", "research.db"),
        nebius_api_key=os.environ.get("NEBIUS_API_KEY", ""),
        tavily_api_key=os.environ.get("TAVILY_API_KEY", ""),
    )
