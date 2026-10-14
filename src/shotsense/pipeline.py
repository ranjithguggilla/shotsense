"""End-to-end run: split, compare models, test, transfer, player study, charts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from . import charts, evaluate, players
from .features import add_features, open_play_shots
from .models import CANDIDATES, FEATURES_FOR, build_model, predict

SEED = 7
TEST_SHARE = 0.20
# Assumption, not a measurement: what one extra goal is worth to a club in a season.
VALUE_PER_GOAL = 1_000_000


def load_shots(path: Path) -> pd.DataFrame:
    return add_features(pd.read_csv(path))


def cross_validate(train: pd.DataFrame, names: list[str], folds: int = 5):
    gkf = GroupKFold(n_splits=folds)
    fold_loss = {n: [] for n in names}
    oof = {n: np.zeros(len(train)) for n in names}
    y = train["goal"].to_numpy()
    for tr, va in gkf.split(train, y, groups=train["match_id"]):
        for n in names:
            m = build_model(n, SEED).fit(train.iloc[tr][FEATURES_FOR[n]], y[tr])
            p = predict(m, n, train.iloc[va])
            oof[n][va] = p
            fold_loss[n].append(log_loss(y[va], evaluate.clip(p)))
    return {n: np.array(v) for n, v in fold_loss.items()}, oof


def intercept_shift(y, p, match_ids, seed: int = SEED):
    """Fit one log-odds offset on half the matches, score it on the other half."""
    ids = np.unique(match_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    fit_ids = set(ids[: len(ids) // 2])
    fit = np.array([m in fit_ids for m in match_ids])
    p = evaluate.clip(p)
    logit = np.log(p / (1 - p))
    shifts = np.linspace(-1.5, 1.5, 301)
    losses = [log_loss(y[fit], 1 / (1 + np.exp(-(logit[fit] + s)))) for s in shifts]
    s = float(shifts[int(np.argmin(losses))])
    shifted = 1 / (1 + np.exp(-(logit + s)))
    return s, {
        "before": float(log_loss(y[~fit], p[~fit])),
        "after": float(log_loss(y[~fit], shifted[~fit])),
    }


def run(data_path: Path, out_dir: Path, assets_dir: Path, quick: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    raw = load_shots(data_path)
    result: dict = {"seed": SEED}

    pens = raw[raw["shot_type"] == "Penalty"]
    result["penalties"] = {
        "n": int(len(pens)), "goals": int(pens["goal"].sum()),
        "conversion": float(pens["goal"].mean()),
    }
    shots = open_play_shots(raw)
    result["data"] = {
        "matches": int(shots["match_id"].nunique()), "shots_open_play": int(len(shots)),
        "by_group": {g: {"matches": int(d["match_id"].nunique()), "shots": int(len(d)),
                         "goals": int(d["goal"].sum()), "goal_rate": float(d["goal"].mean())}
                     for g, d in shots.groupby("group")},
        "freeze_frame_share": float(shots["has_freeze_frame"].mean()),
    }

    club = shots[shots["group"] == "men_club"].reset_index(drop=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SHARE, random_state=SEED)
    tr_idx, te_idx = next(gss.split(club, groups=club["match_id"]))
    train, test = club.iloc[tr_idx].reset_index(drop=True), club.iloc[te_idx].reset_index(drop=True)
    result["split"] = {"train_shots": len(train), "train_matches": int(train["match_id"].nunique()),
                       "test_shots": len(test), "test_matches": int(test["match_id"].nunique())}

    names = CANDIDATES
    fold_loss, _ = cross_validate(train, names)
    shipped, se = evaluate.one_se_choice(fold_loss, names)
    cv_rows = [{"model": n, "cv_log_loss": float(fold_loss[n].mean()),
                "cv_se": float(fold_loss[n].std(ddof=1) / np.sqrt(len(fold_loss[n])))}
               for n in names]
    result["cv"] = cv_rows
    result["shipped"] = shipped
    result["one_se_margin"] = se

    fitted = {n: build_model(n, SEED).fit(train[FEATURES_FOR[n]], train["goal"]) for n in names}
    ytest = test["goal"].to_numpy()
    test_pred = {n: predict(fitted[n], n, test) for n in names}
    result["test"] = {n: evaluate.summarise(ytest, test_pred[n]) for n in names}
    ref = test.dropna(subset=["sb_xg"])
    result["test_reference_statsbomb"] = evaluate.summarise(ref["goal"], ref["sb_xg"])
    comparisons = {}
    for other in names:
        if other == shipped:
            continue
        pt, lo, hi = evaluate.paired_match_bootstrap(
            ytest, test_pred[shipped], test_pred[other], test["match_id"], n_boot=500 if quick else 2000)
        comparisons[other] = {"diff": pt, "lo": lo, "hi": hi}
    result["shipped_vs_other_log_loss"] = comparisons
    result["shipped_auc_ci"] = evaluate.bootstrap_metric(
        ytest, test_pred[shipped], test["match_id"], roc_auc_score, n_boot=200 if quick else 1000)

    # Same model, two validation schemes: does the choice of split change the story?
    from sklearn.model_selection import KFold
    rows_cv = []
    for scheme, splitter, groups in [("random rows", KFold(5, shuffle=True, random_state=SEED), None),
                                     ("whole matches", GroupKFold(5), train["match_id"])]:
        losses = []
        for a, b in splitter.split(train, groups=groups):
            m = build_model(shipped, SEED).fit(train.iloc[a][FEATURES_FOR[shipped]], train.iloc[a]["goal"])
            losses.append(log_loss(train.iloc[b]["goal"], evaluate.clip(predict(m, shipped, train.iloc[b]))))
        rows_cv.append({"scheme": scheme, "log_loss": float(np.mean(losses))})
    result["split_scheme"] = rows_cv

    # Transfer to other competitions without refitting.
    transfer_rows = []
    cal_tables = {"Held-out league matches": evaluate.calibration_table(ytest, test_pred[shipped])}
    result["transfer"] = {}
    for group, label in [("men_national", "World Cup 2022"), ("women_club", "FA WSL")]:
        d = shots[shots["group"] == group].reset_index(drop=True)
        y = d["goal"].to_numpy()
        p = predict(fitted[shipped], shipped, d)
        pb = predict(fitted["logit_distance"], "logit_distance", d)
        entry = {"shipped": evaluate.summarise(y, p), "distance_only": evaluate.summarise(y, pb)}
        sref = d.dropna(subset=["sb_xg"])
        entry["statsbomb_reference"] = evaluate.summarise(sref["goal"], sref["sb_xg"])
        shift, eff = intercept_shift(y, p, d["match_id"].to_numpy())
        entry["offset_fit_on_half"] = {"log_odds_shift": shift, **eff}
        result["transfer"][label] = entry
        transfer_rows.append({"set": label, "xg_total": entry["shipped"]["xg_total"],
                              "goals": entry["shipped"]["goals"]})
        cal_tables[label] = evaluate.calibration_table(y, p)

    # Out-of-fold xG for every club shot, used for the player and game-state studies.
    oof = np.zeros(len(club))
    for a, b in GroupKFold(5).split(club, groups=club["match_id"]):
        m = build_model(shipped, SEED).fit(club.iloc[a][FEATURES_FOR[shipped]], club.iloc[a]["goal"])
        oof[b] = predict(m, shipped, club.iloc[b])
    club = club.assign(xg=oof)
    result["oof_club"] = evaluate.summarise(club["goal"], club["xg"])

    # Game state: do teams that trail or lead convert differently, given the chance?
    bins = pd.cut(club["goal_diff"], [-99, -2, -1, 0, 1, 99],
                  labels=["down 2+", "down 1", "level", "up 1", "up 2+"])
    gs = club.assign(state=bins).groupby("state", observed=True).agg(
        shots=("goal", "size"), goals=("goal", "sum"), xg=("xg", "sum"))
    gs["goals_over_xg"] = gs["goals"] / gs["xg"]
    result["game_state"] = {k: v for k, v in gs.reset_index().set_index("state").to_dict("index").items()}

    # Team level: does summed xG track goals over a season?
    team = club.groupby("team").agg(goals=("goal", "sum"), xg=("xg", "sum"))
    result["team_level"] = {"teams": int(len(team)),
                            "corr_goals_xg": float(np.corrcoef(team["goals"], team["xg"])[0, 1]),
                            "mae_goals": float((team["goals"] - team["xg"]).abs().mean())}

    # Finishing skill. The quick mode relaxes the shot thresholds so small samples still run.
    min_total, min_half = (10, 8) if quick else (30, 20)
    ll_all = players.player_loglik(club, "xg")
    tau = players.fit_tau(ll_all)
    qualified = players.posterior(ll_all[ll_all["shots"] >= min_total].reset_index(drop=True), tau)
    first, second = players.split_halves(club)
    ll_a, ll_b = players.player_loglik(first, "xg"), players.player_loglik(second, "xg")
    tau_a = players.fit_tau(ll_a)
    post_a = players.posterior(ll_a, tau_a)
    both = post_a.merge(ll_b.drop(columns=["ll"]), on=["player_id", "player"], suffixes=("_a", "_b"))
    both = both[(both["shots_a"] >= min_half) & (both["shots_b"] >= min_half)].reset_index(drop=True)
    both["excess_b"] = (both["goals_b"] - both["xg_b"]) / both["shots_b"]
    both["raw_a"] = (both["goals_a"] - both["xg_a"]) / both["shots_a"]
    r_raw = float(np.corrcoef(both["raw_a"], both["excess_b"])[0, 1])
    r_shr = float(np.corrcoef(both["effect"], both["excess_b"])[0, 1])
    k = 10
    short_raw = both.sort_values("raw_a", ascending=False).head(k)
    short_shr = both.sort_values("effect", ascending=False).head(k)

    def per100(df, suffix):
        return float((df[f"goals_{suffix}"].sum() - df[f"xg_{suffix}"].sum()) / df[f"shots_{suffix}"].sum() * 100)

    pool_b = per100(both, "b")
    shots_per_season = 80
    overpay = (per100(short_raw, "a") - per100(short_raw, "b")) / 100 * shots_per_season * VALUE_PER_GOAL
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(300 if quick else 1000):
        samp = both.iloc[rng.integers(0, len(both), len(both))]
        top = samp.sort_values("raw_a", ascending=False).head(k)
        boots.append((per100(top, "a") - per100(top, "b")) / 100 * shots_per_season * VALUE_PER_GOAL)
    lo_o, hi_o = np.percentile(boots, [2.5, 97.5])
    result["players"] = {
        "players_with_30_shots": int(len(qualified)),
        "tau": tau,
        "tau_first_half": tau_a,
        "odds_multiplier_1sd": float(np.exp(tau)),
        "split_half_players": int(len(both)),
        "corr_first_half_vs_second_raw": r_raw,
        "corr_first_half_vs_second_shrunk": r_shr,
        "shortlist_size": k,
        "goals_over_xg_per_100_shots": {
            "raw_top10_first_half": per100(short_raw, "a"),
            "raw_top10_second_half": per100(short_raw, "b"),
            "shrunk_top10_first_half": per100(short_shr, "a"),
            "shrunk_top10_second_half": per100(short_shr, "b"),
            "everyone_second_half": pool_b,
        },
        "value_per_goal_assumption": VALUE_PER_GOAL,
        "shots_per_season_assumption": shots_per_season,
        "overpayment_per_player_season_raw_shortlist": float(overpay),
        "overpayment_ci95": [float(lo_o), float(hi_o)],
    }
    cols = ["player", "shots", "goals", "xg", "raw_ratio", "odds_multiplier", "lo", "hi"]
    result["players"]["top_finishers"] = qualified.sort_values("raw_ratio", ascending=False).head(10)[
        cols].round(3).to_dict("records")
    result["players"]["top_by_shrunk"] = qualified.sort_values("effect", ascending=False).head(10)[
        cols].round(3).to_dict("records")
    qualified.sort_values("effect", ascending=False).to_csv(out_dir / "finishing_table.csv", index=False)

    # Charts.
    cv_df = pd.DataFrame(cv_rows)
    charts.model_comparison(cv_df, shipped, assets_dir / "model_comparison.png")
    charts.calibration(cal_tables, assets_dir / "calibration.png")
    xs = np.linspace(80, 120, 41)
    ys = np.linspace(0, 80, 81)
    grid = pd.DataFrame([{"x": x, "y": y} for x in xs for y in ys])
    for col, val in {"body_part": "Right Foot", "technique": "Normal", "play_pattern": "Regular Play",
                     "shot_type": "Open Play", "first_time": 0, "under_pressure": 0, "one_on_one": 0,
                     "follows_dribble": 0, "aerial_won": 0, "assist_cross": 0, "assist_cut_back": 0,
                     "assist_through": 0, "is_home": 1}.items():
        grid[col] = val
    grid = add_features(grid)
    gp = predict(fitted[shipped], shipped, grid).reshape(len(xs), len(ys))
    charts.pitch_map(gp, xs, ys, assets_dir / "xg_map.png")
    charts.shrinkage(qualified, assets_dir / "finishing_shrinkage.png")
    charts.split_half(both, r_raw, r_shr, assets_dir / "split_half.png")
    charts.transfer(pd.DataFrame(transfer_rows), assets_dir / "transfer.png")

    joblib.dump({"model": fitted[shipped], "name": shipped}, out_dir / "model.joblib")
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(result, fh, indent=2, default=float)
    return result


if __name__ == "__main__":
    import sys

    run(Path(sys.argv[1]), Path("outputs"), Path("assets"))
