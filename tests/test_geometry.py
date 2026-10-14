import math

from shotsense.geometry import (
    distance_to_goal,
    freeze_frame_features,
    in_shot_cone,
    shot_angle,
)


def test_distance_from_penalty_spot():
    assert math.isclose(distance_to_goal(108, 40), 12.0)


def test_angle_is_widest_in_front_of_goal_and_shrinks_with_distance():
    near = shot_angle(114, 40)
    far = shot_angle(90, 40)
    wide = shot_angle(114, 10)
    assert near > far > 0
    assert near > wide


def test_angle_on_goal_line_is_zero():
    assert shot_angle(120, 40) == 0.0
    assert shot_angle(121, 40) == 0.0


def test_cone_membership():
    assert in_shot_cone(100, 40, 112, 40)
    assert not in_shot_cone(100, 40, 112, 60)
    assert not in_shot_cone(100, 40, 95, 40)


def test_freeze_frame_features_counts_and_keeper():
    f = freeze_frame_features(100, 40, [(112, 40), (110, 70)], (118, 41))
    assert f["n_defenders_in_cone"] == 1.0
    assert math.isclose(f["keeper_depth"], 2.0)
    assert math.isclose(f["keeper_offset"], 1.0)
    assert f["has_freeze_frame"] == 1.0


def test_freeze_frame_without_keeper_gives_nan():
    f = freeze_frame_features(100, 40, [], None)
    assert math.isnan(f["keeper_depth"])
    assert math.isnan(f["nearest_defender"])
