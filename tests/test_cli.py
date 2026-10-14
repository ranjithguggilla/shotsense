from pathlib import Path

from shotsense import cli


def test_run_command_dispatches(monkeypatch, tmp_path):
    seen = {}

    def fake_run(data, out, assets, quick=False):
        seen.update(data=data, out=out, assets=assets, quick=quick)

    monkeypatch.setattr("shotsense.pipeline.run", fake_run)
    cli.main(["run", "--data", "d.csv.gz", "--out", str(tmp_path), "--quick"])
    assert seen["data"] == Path("d.csv.gz") and seen["quick"] is True


def test_fetch_command_dispatches(monkeypatch):
    seen = {}
    monkeypatch.setattr("shotsense.extract.build", lambda clone, out: seen.update(clone=clone, out=out))
    cli.main(["fetch", "--clone", "c", "--out", "o.csv.gz"])
    assert seen == {"clone": Path("c"), "out": Path("o.csv.gz")}
