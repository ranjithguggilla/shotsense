"""Pitch geometry on the 120 x 80 coordinate system used by the event data.

The attacking goal sits at x = 120 with posts at y = 36 and y = 44.
"""

from __future__ import annotations

import math

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
GOAL_X = 120.0
POST_LEFT = 36.0
POST_RIGHT = 44.0
GOAL_CENTRE_Y = 40.0
GOAL_WIDTH = POST_RIGHT - POST_LEFT


def distance_to_goal(x: float, y: float) -> float:
    """Straight-line distance from the shot location to the centre of the goal mouth."""
    return math.hypot(GOAL_X - x, GOAL_CENTRE_Y - y)


def shot_angle(x: float, y: float) -> float:
    """Visible angle of the goal mouth in radians (0 when the shooter is on the goal line)."""
    dx = GOAL_X - x
    if dx <= 0:
        return 0.0
    a1 = math.atan2(POST_LEFT - y, dx)
    a2 = math.atan2(POST_RIGHT - y, dx)
    return abs(a2 - a1)


def _sign(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    return (px - bx) * (ay - by) - (ax - bx) * (py - by)


def in_shot_cone(x: float, y: float, px: float, py: float) -> bool:
    """True when (px, py) lies inside the triangle between the ball and the two posts."""
    d1 = _sign(px, py, x, y, GOAL_X, POST_LEFT)
    d2 = _sign(px, py, GOAL_X, POST_LEFT, GOAL_X, POST_RIGHT)
    d3 = _sign(px, py, GOAL_X, POST_RIGHT, x, y)
    has_neg = d1 < 0 or d2 < 0 or d3 < 0
    has_pos = d1 > 0 or d2 > 0 or d3 > 0
    return not (has_neg and has_pos)


def freeze_frame_features(x: float, y: float, opponents: list[tuple[float, float]],
                          keeper: tuple[float, float] | None) -> dict[str, float]:
    """Summarise the positions of defenders around the shooter.

    opponents holds outfield defenders only. keeper is the goalkeeper location, if visible.
    """
    n_cone = sum(1 for ox, oy in opponents if in_shot_cone(x, y, ox, oy))
    nearest = min((math.hypot(ox - x, oy - y) for ox, oy in opponents), default=float("nan"))
    if keeper is None:
        k_off = k_lat = k_dist = float("nan")
    else:
        kx, ky = keeper
        k_off = GOAL_X - kx
        k_lat = abs(ky - GOAL_CENTRE_Y)
        k_dist = math.hypot(kx - x, ky - y)
    return {
        "n_defenders_in_cone": float(n_cone),
        "nearest_defender": nearest,
        "keeper_depth": k_off,
        "keeper_offset": k_lat,
        "keeper_distance": k_dist,
        "has_freeze_frame": 1.0,
    }
