from research_agent.config import Settings, get_settings, load_dotenv


def test_dotenv_does_not_override_and_parses_quotes(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# c\nFOO_A="quoted"\nexport FOO_B=plain\nFOO_C=fromfile\n', encoding="utf-8")
    monkeypatch.delenv("FOO_A", raising=False)
    monkeypatch.delenv("FOO_B", raising=False)
    monkeypatch.setenv("FOO_C", "fromenv")
    load_dotenv(env)
    import os

    assert os.environ["FOO_A"] == "quoted"
    assert os.environ["FOO_B"] == "plain"
    assert os.environ["FOO_C"] == "fromenv"


def test_keys_not_in_repr(monkeypatch, tmp_path):
    monkeypatch.setenv("NEBIUS_API_KEY", "secret-nebius-123")
    monkeypatch.setenv("TAVILY_API_KEY", "secret-tavily-456")
    s = get_settings(tmp_path / "none.env")
    assert "secret" not in repr(s)
    assert s.missing_keys() == []


def test_missing_keys():
    assert Settings().missing_keys() == ["NEBIUS_API_KEY", "TAVILY_API_KEY"]


def test_limits_defaults():
    s = Settings()
    assert s.limits.max_queries_per_round == 5
    assert s.limits.max_rounds == 3 and s.limits.max_repair_passes == 1
    assert s.limits.search_results == 10 and s.limits.extract_batch_size == 20
    assert s.limits.round1_tiers == (1, 2) and 3 in s.limits.follow_up_tiers
    assert s.budget.credits > 0 and s.budget.time_s > 0 and s.cache_ttl_days == 7
