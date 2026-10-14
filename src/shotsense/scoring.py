"""Score a single shot and explain the number in plain language.

The explanation is model agnostic: each factor is swapped, one at a time, to what a
typical shot looks like, and the change in xG is reported. That works for the additive
models and for the boosted ones, and it needs no extra dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .features import add_features
from .models import FEATURES_FOR

PLAY_PATTERNS = ["Regular Play", "From Corner", "From Free Kick", "From Throw In",
                 "From Counter", "From Keeper", "From Goal Kick", "From Kick Off", "Other"]
BODY_PARTS = ["Right Foot", "Left Foot", "Head", "Other"]
TECHNIQUES = ["Normal", "Volley", "Half Volley", "Lob", "Overhead Kick", "Diving Header"]

# Reference shot used for "compared with a typical chance" explanations.
TYPICAL = {
    "x": 104.0, "y": 40.0, "body_part": "Right Foot", "technique": "Normal",
    "play_pattern": "Regular Play", "shot_type": "Open Play", "first_time": 0,
    "under_pressure": 0, "one_on_one": 0, "follows_dribble": 0, "aerial_won": 0,
    "assist_cross": 0, "assist_cut_back": 0, "assist_through": 0,
    "is_home": 1,
}

FACTORS = {
    "position on the pitch": ["x", "y"],
    "body part": ["body_part"],
    "technique": ["technique"],
    "how the move started": ["play_pattern"],
    "defender pressure": ["under_pressure"],
    "first-time strike": ["first_time"],
    "one-on-one with the keeper": ["one_on_one"],
    "cross into the box": ["assist_cross"],
    "cut-back": ["assist_cut_back"],
    "through ball": ["assist_through"],
    "set piece": ["shot_type"],
}


@dataclass
class ShotInput:
    x: float
    y: float
    body_part: str = "Right Foot"
    technique: str = "Normal"
    play_pattern: str = "Regular Play"
    shot_type: str = "Open Play"
    first_time: int = 0
    under_pressure: int = 0
    one_on_one: int = 0
    follows_dribble: int = 0
    assist_cross: int = 0
    assist_cut_back: int = 0
    assist_through: int = 0

    def as_frame(self) -> pd.DataFrame:
        row = {**TYPICAL, **self.__dict__}
        return add_features(pd.DataFrame([row]))


def score_frame(bundle: dict, frame: pd.DataFrame) -> float:
    name = bundle["name"]
    return float(bundle["model"].predict_proba(frame[FEATURES_FOR[name]])[0, 1])


def explain(bundle: dict, shot: ShotInput) -> dict:
    """Return the xG and a ranked list of what moved it away from a typical chance."""
    frame = shot.as_frame()
    xg = score_frame(bundle, frame)
    typical = ShotInput(x=TYPICAL["x"], y=TYPICAL["y"]).as_frame()
    reasons = []
    for label, cols in FACTORS.items():
        swapped = frame.copy()
        for c in cols:
            swapped[c] = typical[c].iloc[0]
        swapped = add_features(swapped.drop(columns=["distance", "angle", "log_distance", "abs_y",
                                                     "is_header", "is_free_kick"]))
        delta = xg - score_frame(bundle, swapped)
        if abs(delta) >= 0.005:
            reasons.append((label, delta))
    reasons.sort(key=lambda r: -abs(r[1]))
    return {"xg": xg, "reasons": reasons, "text": describe(xg, reasons)}


def describe(xg: float, reasons: list[tuple[str, float]]) -> str:
    chance = f"{xg:.2f} expected goals, about 1 goal in {1 / max(xg, 1e-6):.0f} shots like this."
    if not reasons:
        return chance + " This is close to an average chance."
    parts = []
    for label, delta in reasons[:4]:
        direction = "adds" if delta > 0 else "costs"
        parts.append(f"{label} {direction} {abs(delta):.2f}")
    return chance + " Against a typical chance, " + "; ".join(parts) + "."
