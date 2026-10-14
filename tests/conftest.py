from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DATA = Path(__file__).resolve().parents[1] / "data" / "shots.csv.gz"


@pytest.fixture(scope="session")
def shots():
    return pd.read_csv(DATA)


@pytest.fixture(scope="session")
def small_shots(shots):
    """A match-level subsample that keeps every group, so end-to-end tests stay quick."""
    rng = np.random.default_rng(0)
    parts = []
    for _, g in shots.groupby("group"):
        ids = g["match_id"].unique()
        keep = rng.choice(ids, size=min(len(ids), 140), replace=False)
        parts.append(g[g["match_id"].isin(keep)])
    return pd.concat(parts, ignore_index=True)
