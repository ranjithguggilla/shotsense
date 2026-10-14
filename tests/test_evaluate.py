import numpy as np

from shotsense import evaluate


def test_ece_is_zero_for_perfectly_calibrated_bins():
    p = np.repeat([0.2, 0.8], 500)
    y = np.concatenate([np.r_[np.ones(100), np.zeros(400)], np.r_[np.ones(400), np.zeros(100)]])
    assert evaluate.expected_calibration_error(y, p, bins=2) < 1e-9


def test_ece_detects_overconfidence():
    p = np.full(1000, 0.5)
    y = np.zeros(1000)
    assert evaluate.expected_calibration_error(y, p) > 0.4


def test_summarise_totals():
    y = np.array([1, 0, 0, 1])
    s = evaluate.summarise(y, np.array([0.7, 0.2, 0.3, 0.6]))
    assert s["goals"] == 2 and s["n_shots"] == 4
    assert abs(s["xg_total"] - 1.8) < 1e-9
    assert 0 < s["log_loss"] < 1


def test_paired_bootstrap_sign_and_interval():
    rng = np.random.default_rng(1)
    n = 3000
    p_true = rng.uniform(0.02, 0.6, n)
    y = rng.binomial(1, p_true)
    matches = np.repeat(np.arange(n // 10), 10)
    point, lo, hi = evaluate.paired_match_bootstrap(y, p_true, np.full(n, y.mean()), matches, n_boot=300)
    assert point < 0 and lo < point < hi and hi < 0


def test_paired_bootstrap_same_model_is_zero():
    y = np.array([0, 1, 0, 0, 1, 0])
    p = np.array([0.1, 0.6, 0.2, 0.3, 0.5, 0.1])
    point, lo, hi = evaluate.paired_match_bootstrap(y, p, p, np.array([1, 1, 2, 2, 3, 3]), n_boot=50)
    assert point == lo == hi == 0.0


def test_one_se_prefers_simpler_model_inside_the_margin():
    folds = {"simple": np.array([0.30, 0.31, 0.29, 0.30, 0.32]),
             "complex": np.array([0.29, 0.31, 0.28, 0.31, 0.31])}
    name, _ = evaluate.one_se_choice(folds, ["simple", "complex"])
    assert name == "simple"


def test_one_se_takes_the_better_model_when_the_gap_is_clear():
    folds = {"simple": np.array([0.30, 0.31, 0.30, 0.31, 0.30]),
             "complex": np.array([0.25, 0.26, 0.25, 0.26, 0.25])}
    name, _ = evaluate.one_se_choice(folds, ["simple", "complex"])
    assert name == "complex"


def test_bootstrap_metric_interval_contains_point():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.05, 0.5, 2000)
    y = rng.binomial(1, p)
    lo, hi = evaluate.bootstrap_metric(y, p, np.repeat(np.arange(200), 10),
                                       lambda a, b: float(np.mean(a)), n_boot=100)
    assert lo < y.mean() < hi
