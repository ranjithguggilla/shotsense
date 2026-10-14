import numpy as np
import pandas as pd

from shotsense import players


def _simulate(tau, n_players=300, shots=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_players):
        u = rng.normal(0, tau)
        xg = rng.uniform(0.03, 0.4, shots)
        p = 1 / (1 + np.exp(-(np.log(xg / (1 - xg)) + u)))
        for g, x in zip(rng.binomial(1, p), xg, strict=True):
            rows.append({"player_id": pid, "player": f"p{pid}", "goal": int(g), "xg": x})
    return pd.DataFrame(rows)


def test_tau_is_small_when_players_are_identical():
    tau = players.fit_tau(players.player_loglik(_simulate(0.0)))
    assert tau < 0.1


def test_tau_recovers_a_real_spread():
    tau = players.fit_tau(players.player_loglik(_simulate(0.3)))
    assert 0.2 < tau < 0.4


def test_shrinkage_pulls_small_samples_harder():
    sim = _simulate(0.15)
    table = players.player_loglik(sim)
    post = players.posterior(table, 0.15)
    short = sim.groupby("player_id").head(5)
    post_short = players.posterior(players.player_loglik(short), 0.15)
    assert post_short["effect"].abs().mean() < post["effect"].abs().mean()


def test_interval_brackets_the_estimate():
    table = players.player_loglik(_simulate(0.2, n_players=30))
    post = players.posterior(table, 0.2)
    assert (post["lo"] <= post["odds_multiplier"]).all()
    assert (post["odds_multiplier"] <= post["hi"]).all()


def test_split_halves_are_disjoint_by_match():
    df = pd.DataFrame({"match_id": np.repeat(np.arange(10), 3), "goal": 0})
    a, b = players.split_halves(df)
    assert set(a["match_id"]).isdisjoint(set(b["match_id"]))
    assert len(a) + len(b) == len(df)
