"""Finishing skill with partial pooling.

Each shot is a Bernoulli trial whose log-odds are the model's xG log-odds plus a player
effect u, with u drawn from a normal distribution of spread tau. Tau is the thing to
learn: if it is close to zero, players are alike once shot quality is accounted for and
a hot season is mostly luck. The marginal likelihood is evaluated on a fixed grid of
u values, which keeps the maths exact enough and the code short.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logsumexp
from scipy.stats import norm

GRID = np.linspace(-1.5, 1.5, 121)
_EPS = 1e-6


def player_loglik(shots: pd.DataFrame, pred_col: str = "xg") -> pd.DataFrame:
    """One row per player: shots, goals, xg and the log-likelihood on the u grid."""
    rows = []
    for (pid, name), g in shots.groupby(["player_id", "player"]):
        p = np.clip(g[pred_col].to_numpy(), _EPS, 1 - _EPS)
        y = g["goal"].to_numpy()
        logit = np.log(p / (1 - p))
        pm = expit(logit[:, None] + GRID[None, :])
        ll = (y[:, None] * np.log(pm) + (1 - y[:, None]) * np.log(1 - pm)).sum(axis=0)
        rows.append({"player_id": pid, "player": name, "shots": len(g),
                     "goals": int(y.sum()), "xg": float(p.sum()), "ll": ll})
    return pd.DataFrame(rows)


def _log_prior(tau: float) -> np.ndarray:
    lp = norm.logpdf(GRID, scale=tau)
    return lp - logsumexp(lp)


def fit_tau(table: pd.DataFrame) -> float:
    ll = np.vstack(table["ll"].to_list())

    def nll(log_tau: float) -> float:
        lp = _log_prior(np.exp(log_tau))
        return -float(logsumexp(ll + lp[None, :], axis=1).sum())

    res = minimize_scalar(nll, bounds=(np.log(0.01), np.log(1.0)), method="bounded")
    return float(np.exp(res.x))


def posterior(table: pd.DataFrame, tau: float) -> pd.DataFrame:
    """Posterior mean and 90% interval of the log-odds effect, and the odds multiplier."""
    ll = np.vstack(table["ll"].to_list())
    logpost = ll + _log_prior(tau)[None, :]
    logpost -= logsumexp(logpost, axis=1, keepdims=True)
    w = np.exp(logpost)
    mean = (w * GRID[None, :]).sum(axis=1)
    cdf = np.cumsum(w, axis=1)
    lo = GRID[(cdf < 0.05).sum(axis=1).clip(0, len(GRID) - 1)]
    hi = GRID[(cdf < 0.95).sum(axis=1).clip(0, len(GRID) - 1)]
    out = table.drop(columns=["ll"]).copy()
    out["raw_ratio"] = out["goals"] / out["xg"]
    out["effect"] = mean
    out["odds_multiplier"] = np.exp(mean)
    out["lo"] = np.exp(lo)
    out["hi"] = np.exp(hi)
    out["goals_above_xg"] = out["goals"] - out["xg"]
    return out


def split_halves(shots: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Alternate matches (by id rank) into two interleaved halves of the season."""
    ids = np.sort(shots["match_id"].unique())
    first = set(ids[::2])
    mask = shots["match_id"].isin(first)
    return shots[mask], shots[~mask]
