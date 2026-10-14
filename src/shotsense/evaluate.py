"""Metrics and resampling helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

EPS = 1e-6


def clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def expected_calibration_error(y, p, bins: int = 10) -> float:
    """Equal-count ECE: bins hold the same number of shots, which suits a rare event."""
    y = np.asarray(y)
    p = np.asarray(p)
    order = np.argsort(p)
    ece = 0.0
    for idx in np.array_split(order, bins):
        if len(idx):
            ece += len(idx) / len(y) * abs(y[idx].mean() - p[idx].mean())
    return float(ece)


def summarise(y, p) -> dict[str, float]:
    p = clip(p)
    y = np.asarray(y)
    return {
        "n_shots": int(len(y)),
        "goals": int(y.sum()),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "ece": expected_calibration_error(y, p),
        "xg_total": float(p.sum()),
        "xg_over_goals": float(p.sum() / max(y.sum(), 1)),
    }


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    y = np.asarray(y)
    p = np.asarray(p)
    order = np.argsort(p)
    rows = []
    for idx in np.array_split(order, bins):
        rows.append({"mean_pred": p[idx].mean(), "observed": y[idx].mean(), "n": len(idx)})
    return pd.DataFrame(rows)


def _per_match_loss(y, p, match_ids):
    p = clip(p)
    ll = -(np.asarray(y) * np.log(p) + (1 - np.asarray(y)) * np.log(1 - p))
    frame = pd.DataFrame({"m": match_ids, "ll": ll})
    g = frame.groupby("m")["ll"].agg(["sum", "count"])
    return g["sum"].to_numpy(), g["count"].to_numpy()


def paired_match_bootstrap(y, p_a, p_b, match_ids, n_boot: int = 2000, seed: int = 11):
    """Mean log-loss difference (a minus b) with a 95% interval, resampling whole matches.

    Negative values mean model a is better.
    """
    sa, n = _per_match_loss(y, p_a, match_ids)
    sb, _ = _per_match_loss(y, p_b, match_ids)
    rng = np.random.default_rng(seed)
    k = len(n)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, k, k)
        diffs[i] = (sa[pick].sum() - sb[pick].sum()) / n[pick].sum()
    point = (sa.sum() - sb.sum()) / n.sum()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def bootstrap_metric(y, p, match_ids, fn, n_boot: int = 1000, seed: int = 5):
    """95% interval for fn(y, p) with whole-match resampling."""
    y = np.asarray(y)
    p = np.asarray(p)
    frame = pd.DataFrame({"m": match_ids, "i": np.arange(len(y))})
    groups = [g["i"].to_numpy() for _, g in frame.groupby("m")]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[j] for j in pick])
        vals.append(fn(y[idx], p[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def one_se_choice(fold_losses: dict[str, np.ndarray], order: list[str]) -> tuple[str, float]:
    """Simplest model whose mean loss is within one standard error of the best.

    The standard error is taken on the paired fold differences against the best model.
    order lists models from simplest to most complex.
    """
    means = {k: float(np.mean(v)) for k, v in fold_losses.items()}
    best = min(means, key=means.get)
    diff = {k: np.asarray(fold_losses[k]) - np.asarray(fold_losses[best]) for k in order}
    for name in order:
        d = diff[name]
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        if d.mean() <= se:
            return name, float(se)
    return best, 0.0
