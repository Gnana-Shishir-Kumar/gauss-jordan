from conftest import FakeLLM, FakeSearch

from research_agent import cli


def test_cli_runs_with_fakes(monkeypatch, tmp_path, capsys, settings):
    monkeypatch.setattr(cli, "default_clients", lambda s: (FakeLLM(), FakeSearch()))
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    db = str(tmp_path / "t.db")
    assert cli.main(["What are the topic facts?", "--db", db]) == 0
    out = capsys.readouterr()
    assert "ANSWER" in out.out and "CITATIONS" in out.out and "TRACE" in out.out
    assert "[plan]" in out.err
    assert cli.main(["--show", "1", "--db", db, "--no-trace"]) == 0
    assert "TRACE" not in capsys.readouterr().out


def test_cli_missing_keys(monkeypatch, tmp_path, capsys):
    from research_agent.config import Settings

    monkeypatch.setattr(cli, "get_settings", lambda: Settings())
    assert cli.main(["q", "--db", str(tmp_path / "t.db")]) == 2
    assert "NEBIUS_API_KEY" in capsys.readouterr().err
