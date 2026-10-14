import numpy as np
import pandas as pd

from shotsense import models
from shotsense.features import add_features, open_play_shots


def test_add_features_geometry(shots):
    f = add_features(shots.head(200))
    assert (f["distance"] > 0).all()
    assert (f["angle"] >= 0).all()
    assert set(f["is_header"].unique()) <= {0, 1}


def test_penalties_are_removed(shots):
    f = open_play_shots(shots)
    assert not (f["shot_type"] == "Penalty").any()
    assert len(f) < len(shots)


def test_missing_optional_columns_are_filled():
    df = pd.DataFrame([{"x": 100, "y": 40, "body_part": "Head", "shot_type": "Open Play"}])
    f = add_features(df)
    assert f["has_freeze_frame"].iloc[0] == 0.0
    assert f["is_header"].iloc[0] == 1
    assert np.isnan(f["keeper_depth"].iloc[0])


def test_every_candidate_fits_and_returns_probabilities(small_shots):
    df = open_play_shots(add_features(small_shots)).head(4000)
    for name in models.CANDIDATES:
        m = models.build_model(name).fit(df[models.FEATURES_FOR[name]], df["goal"])
        p = models.predict(m, name, df)
        assert p.shape == (len(df),)
        assert ((p > 0) & (p < 1)).all()


def test_closer_shots_score_higher(small_shots):
    df = open_play_shots(add_features(small_shots)).head(4000)
    m = models.build_model("logit_distance").fit(df[["distance"]], df["goal"])
    near = add_features(pd.DataFrame([{"x": 114, "y": 40, "body_part": "Right Foot", "shot_type": "Open Play"}]))
    far = add_features(pd.DataFrame([{"x": 85, "y": 40, "body_part": "Right Foot", "shot_type": "Open Play"}]))
    assert m.predict_proba(near[["distance"]])[0, 1] > m.predict_proba(far[["distance"]])[0, 1]


def test_unseen_category_does_not_break_the_pipeline(small_shots):
    df = open_play_shots(add_features(small_shots)).head(3000)
    name = "logit_context"
    m = models.build_model(name).fit(df[models.FEATURES_FOR[name]], df["goal"])
    odd = df.head(3).copy()
    odd["technique"] = "Scorpion"
    assert models.predict(m, name, odd).shape == (3,)
