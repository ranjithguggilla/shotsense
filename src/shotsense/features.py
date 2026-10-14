"""Feature engineering for the shot model. Everything here is row-wise, so it is
safe to run before the train/test split; fitted preprocessing lives in the pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .geometry import distance_to_goal, shot_angle

NUMERIC_BASE = ["distance", "angle", "log_distance", "x", "abs_y"]
BINARY = ["is_header", "first_time", "under_pressure", "one_on_one", "follows_dribble",
          "aerial_won", "assist_cross", "assist_cut_back", "assist_through",
          "is_free_kick", "is_home"]
CATEGORICAL = ["play_pattern", "body_part", "technique"]
FREEZE = ["n_defenders_in_cone", "nearest_defender", "keeper_depth", "keeper_offset",
          "keeper_distance", "has_freeze_frame"]

# Feature sets, from the simplest to the richest.
SETS = {
    "distance": ["distance"],
    "geometry": ["distance", "angle"],
    "context": NUMERIC_BASE + BINARY + CATEGORICAL,
    "context_defenders": NUMERIC_BASE + BINARY + CATEGORICAL + FREEZE,
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["distance"] = [distance_to_goal(x, y) for x, y in zip(out["x"], out["y"], strict=True)]
    out["angle"] = [shot_angle(x, y) for x, y in zip(out["x"], out["y"], strict=True)]
    out["log_distance"] = np.log(out["distance"].clip(lower=1.0))
    out["abs_y"] = (out["y"] - 40.0).abs()
    out["is_header"] = (out["body_part"] == "Head").astype(int)
    out["is_free_kick"] = (out["shot_type"] == "Free Kick").astype(int)
    for col in ["assist_cross", "assist_cut_back", "assist_through"]:
        if col not in out:
            out[col] = 0
        out[col] = out[col].fillna(0).astype(int)
    for col in FREEZE:
        if col not in out:
            out[col] = np.nan if col != "has_freeze_frame" else 0.0
    out["has_freeze_frame"] = out["has_freeze_frame"].fillna(0.0)
    return out


def open_play_shots(df: pd.DataFrame) -> pd.DataFrame:
    """Penalties are a different problem (one number); the model is for the rest."""
    return df[df["shot_type"] != "Penalty"].reset_index(drop=True)
