from shotsense.extract import parse_events

META = {"match_id": 1, "home": "Home FC", "away": "Away FC",
        "tag": {"competition": "Test League", "group": "men_club"}}


def _shot(team, loc, outcome, minute, **extra):
    shot = {"type": {"name": "Open Play"}, "body_part": {"name": "Right Foot"},
            "technique": {"name": "Normal"}, "outcome": {"name": outcome},
            "statsbomb_xg": 0.1, **extra}
    return {"id": f"s{minute}", "type": {"name": "Shot"}, "period": 1, "minute": minute,
            "team": {"name": team}, "player": {"name": "P", "id": 9}, "location": loc,
            "play_pattern": {"name": "Regular Play"}, "shot": shot}


def _events():
    return [
        {"id": "a", "type": {"name": "Starting XI"}, "team": {"name": "Home FC"}},
        {"id": "b", "type": {"name": "Starting XI"}, "team": {"name": "Away FC"}},
        _shot("Home FC", [108, 40], "Goal", 10),
        _shot("Away FC", [100, 30], "Saved", 20),
        _shot("Home FC", [110, 38], "Off T", 30,
              freeze_frame=[
                  {"location": [118, 40], "teammate": False, "position": {"name": "Goalkeeper"}},
                  {"location": [112, 38.5], "teammate": False, "position": {"name": "Center Back"}},
                  {"location": [105, 50], "teammate": True, "position": {"name": "Center Forward"}},
              ]),
    ]


def test_one_row_per_shot_with_labels():
    rows = parse_events(_events(), META)
    assert [r["goal"] for r in rows] == [1, 0, 0]
    assert rows[0]["team"] == "Home FC" and rows[0]["is_home"] == 1
    assert rows[1]["is_home"] == 0


def test_game_state_is_taken_before_the_shot():
    rows = parse_events(_events(), META)
    assert rows[0]["goal_diff"] == 0
    assert rows[1]["goal_diff"] == -1
    assert rows[2]["goal_diff"] == 1


def test_freeze_frame_ignores_teammates():
    rows = parse_events(_events(), META)
    last = rows[2]
    assert last["has_freeze_frame"] == 1.0
    assert last["n_defenders_in_cone"] == 1.0
    assert rows[0]["has_freeze_frame"] == 0.0


def test_penalty_shootout_period_is_dropped():
    ev = _events()
    ev[2]["period"] = 5
    assert len(parse_events(ev, META)) == 2


def test_home_side_comes_from_lineups_not_match_list():
    meta = {**META, "home": "Spelled Differently"}
    rows = parse_events(_events(), meta)
    assert rows[0]["is_home"] == 1
