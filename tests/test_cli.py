from conftest import FakeLLM, FakeSearch, make_settings

from research_agent import cli, storage


def _patch(monkeypatch, db_holder):
    def clients(settings):
        conn = storage.connect(db_holder["path"])
        return FakeLLM(conn), FakeSearch()
    monkeypatch.setattr(cli, "default_clients", clients)
    monkeypatch.setattr(cli, "get_settings", lambda: make_settings())


def test_cli_yes_runs_and_exports_ris(monkeypatch, tmp_path, capsys):
    db = str(tmp_path / "t.db")
    _patch(monkeypatch, {"path": db})
    out = tmp_path / "s.ris"
    code = cli.main(["What changed for EGFR NSCLC since 2025?", "--yes", "--db", db, "--export", "ris",
                     "--out", str(out)])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    for part in ("BRIEF", "UPDATE BRIEF", "Stop reason: coverage_complete", "Tavily credits:", "TRACE", "prompt version"):
        assert part in captured.out
    assert "Planned queries" in captured.out and "[plan]" in captured.err
    assert out.read_text(encoding="utf-8").startswith("TY  - ")


def test_cli_without_yes_does_not_run_non_interactively(monkeypatch, tmp_path, capsys):
    db = str(tmp_path / "t.db")
    _patch(monkeypatch, {"path": db})
    assert cli.main(["What changed for EGFR NSCLC since 2025?", "--db", db, "--quiet"]) == 5
    assert "research --run 1" in capsys.readouterr().err
    assert cli.main(["--run", "1", "--db", db, "--quiet", "--no-trace"]) == 0


def test_cli_blocks_identifiers(monkeypatch, tmp_path, capsys):
    db = str(tmp_path / "t.db")
    _patch(monkeypatch, {"path": db})
    assert cli.main(["MRN 12345678 EGFR NSCLC update?", "--yes", "--db", db]) == 6
    assert "patient identifiers" in capsys.readouterr().err


def test_cli_blocks_identifiers_even_without_keys(monkeypatch, tmp_path, capsys):
    from research_agent.config import Settings
    monkeypatch.setattr(cli, "get_settings", lambda: Settings())
    assert cli.main(["My patient John Smith has EGFR NSCLC", "--db", str(tmp_path / "t.db")]) == 6
    assert "patient identifiers" in capsys.readouterr().err


def test_cli_missing_keys(monkeypatch, tmp_path, capsys):
    from research_agent.config import Settings
    monkeypatch.setattr(cli, "get_settings", lambda: Settings())
    assert cli.main(["q", "--db", str(tmp_path / "t.db")]) == 2
    assert "NEBIUS_API_KEY" in capsys.readouterr().err
