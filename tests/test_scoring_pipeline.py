import json

import joblib
import pytest

from shotsense import pipeline
from shotsense.app import run_shot
from shotsense.scoring import ShotInput, describe, explain


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory, small_shots):
    base = tmp_path_factory.mktemp("run")
    data = base / "shots.csv.gz"
    small_shots.to_csv(data, index=False, compression="gzip")
    metrics = pipeline.run(data, base / "out", base / "assets", quick=True)
    return base, metrics


def test_pipeline_writes_all_outputs(run_dir):
    base, metrics = run_dir
    assert (base / "out" / "metrics.json").exists()
    assert (base / "out" / "model.joblib").exists()
    assert (base / "assets" / "calibration.png").exists()
    assert json.loads((base / "out" / "metrics.json").read_text())["shipped"] == metrics["shipped"]


def test_split_is_by_match_with_no_overlap(run_dir):
    _, metrics = run_dir
    s = metrics["split"]
    assert s["train_matches"] > s["test_matches"] > 0


def test_richer_features_beat_the_distance_baseline(run_dir):
    _, m = run_dir
    assert m["test"]["boosting_defenders"]["log_loss"] < m["test"]["logit_distance"]["log_loss"]


def test_totals_are_close_to_goals(run_dir):
    _, m = run_dir
    assert 0.85 < m["oof_club"]["xg_over_goals"] < 1.15


def test_scoring_and_reasons(run_dir):
    base, _ = run_dir
    bundle = joblib.load(base / "out" / "model.joblib")
    close = explain(bundle, ShotInput(x=114, y=40, one_on_one=1))
    far = explain(bundle, ShotInput(x=82, y=20, under_pressure=1))
    assert close["xg"] > far["xg"]
    assert "expected goals" in close["text"]
    text, number = run_shot(bundle, 110, 40, "Head", "Normal", "From Corner", False, True, False, False)
    assert 0 < number["xG"] < 1 and isinstance(text, str)


def test_describe_handles_a_typical_chance():
    assert "average chance" in describe(0.1, [])
