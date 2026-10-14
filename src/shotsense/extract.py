"""Build the shot table from the public StatsBomb open-data repository.

The repository is large, so the extractor makes a blobless clone and streams only
the event files for the matches it needs through `git cat-file --batch`. Each event
file is parsed once and reduced to one row per shot.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .geometry import freeze_frame_features

REPO_URL = "https://github.com/statsbomb/open-data.git"

# (competition_id, season_id, label, group) pairs used in this project.
SELECTION = [
    (2, 27, "Premier League 2015/16", "men_club"),
    (12, 27, "Serie A 2015/16", "men_club"),
    (7, 27, "Ligue 1 2015/16", "men_club"),
    (11, 27, "La Liga 2015/16", "men_club"),
    (43, 106, "World Cup 2022", "men_national"),
    (37, 90, "FA WSL 2020/21", "women_club"),
    (37, 281, "FA WSL 2023/24", "women_club"),
]


def ensure_clone(path: Path) -> None:
    if (path / ".git").exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--no-checkout", "--depth", "1", "--filter=blob:none", REPO_URL, str(path)],
        check=True,
    )


def _show(path: Path, name: str) -> bytes:
    return subprocess.run(["git", "-C", str(path), "show", f"HEAD:{name}"], check=True,
                          capture_output=True).stdout


def list_matches(path: Path, competition: int, season: int) -> list[dict]:
    return json.loads(_show(path, f"data/matches/{competition}/{season}.json"))


def event_blob_ids(path: Path) -> dict[int, str]:
    out = subprocess.run(["git", "-C", str(path), "ls-tree", "-r", "HEAD", "data/events"],
                         check=True, capture_output=True, text=True).stdout
    ids = {}
    for line in out.splitlines():
        meta, name = line.split("\t")
        ids[int(Path(name).stem)] = meta.split()[2]
    return ids


def iter_blobs(path: Path, oids: list[str]):
    """Yield the bytes of each blob, in order, using one cat-file process."""
    proc = subprocess.Popen(["git", "-C", str(path), "cat-file", "--batch"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    assert proc.stdin and proc.stdout
    for oid in oids:
        proc.stdin.write(oid.encode() + b"\n")
        proc.stdin.flush()
        header = proc.stdout.readline().split()
        size = int(header[2])
        body = proc.stdout.read(size)
        proc.stdout.read(1)
        yield body
    proc.stdin.close()
    proc.wait()


def parse_events(events: list[dict], meta: dict) -> list[dict]:
    """Reduce one match's events to shot rows. Penalty shootouts are dropped."""
    by_id = {e["id"]: e for e in events}
    rows = []
    # Team names in the lineup events are the ones used by every other event, and the
    # first Starting XI is the home side. The match list sometimes spells names differently.
    lineups = [e["team"]["name"] for e in events if e["type"]["name"] == "Starting XI"]
    home, away = lineups[0], lineups[1]
    goals = {home: 0, away: 0}
    for ev in events:
        if ev["type"]["name"] != "Shot" or ev.get("period", 1) > 4:
            continue
        shot = ev["shot"]
        team = ev["team"]["name"]
        opp = away if team == home else home
        x, y = ev["location"][:2]
        row = {
            **meta["tag"],
            "match_id": meta["match_id"],
            "team": team,
            "opponent": opp,
            "is_home": int(team == home),
            "player": ev.get("player", {}).get("name"),
            "player_id": ev.get("player", {}).get("id"),
            "period": ev["period"],
            "minute": ev["minute"],
            "x": x,
            "y": y,
            "shot_type": shot["type"]["name"],
            "body_part": shot["body_part"]["name"],
            "technique": shot["technique"]["name"],
            "play_pattern": ev["play_pattern"]["name"],
            "first_time": int(bool(shot.get("first_time"))),
            "under_pressure": int(bool(ev.get("under_pressure"))),
            "one_on_one": int(bool(shot.get("one_on_one"))),
            "follows_dribble": int(bool(shot.get("follows_dribble"))),
            "open_goal": int(bool(shot.get("open_goal"))),
            "aerial_won": int(bool(shot.get("aerial_won"))),
            "deflected": int(bool(shot.get("deflected"))),
            "goal_diff": goals[team] - goals[opp],
            "sb_xg": shot.get("statsbomb_xg"),
            "outcome": shot["outcome"]["name"],
            "goal": int(shot["outcome"]["name"] == "Goal"),
        }
        kp = by_id.get(shot.get("key_pass_id"))
        if kp is not None and "pass" in kp:
            p = kp["pass"]
            row["assist_height"] = p.get("height", {}).get("name")
            row["assist_length"] = p.get("length")
            row["assist_kind"] = (p.get("technique", {}).get("name")
                                  or ("Cross" if p.get("cross") else None)
                                  or ("Cut back" if p.get("cut_back") else None)
                                  or ("Switch" if p.get("switch") else None)
                                  or ("Throw-in" if p.get("type", {}).get("name") == "Throw-in" else None)
                                  or "Pass")
            row["assist_cross"] = int(bool(p.get("cross")))
            row["assist_cut_back"] = int(bool(p.get("cut_back")))
            row["assist_through"] = int(p.get("technique", {}).get("name") == "Through Ball")
        ff = shot.get("freeze_frame")
        if ff:
            opponents = [tuple(f["location"]) for f in ff
                         if not f["teammate"] and f["position"]["name"] != "Goalkeeper"]
            keeper = next((tuple(f["location"]) for f in ff
                           if not f["teammate"] and f["position"]["name"] == "Goalkeeper"), None)
            row.update(freeze_frame_features(x, y, opponents, keeper))
        else:
            row["has_freeze_frame"] = 0.0
        rows.append(row)
        if row["goal"]:
            goals[team] += 1
    return rows


def build(clone: Path, out: Path, selection=SELECTION, log=print) -> pd.DataFrame:
    ensure_clone(clone)
    blob = event_blob_ids(clone)
    jobs = []
    for comp, season, label, group in selection:
        for m in list_matches(clone, comp, season):
            if m["match_id"] not in blob:
                continue
            jobs.append({
                "match_id": m["match_id"], "home": m["home_team"]["home_team_name"],
                "away": m["away_team"]["away_team_name"],
                "tag": {"competition": label, "group": group},
            })
    log(f"{len(jobs)} matches to read")
    rows: list[dict] = []
    stream = iter_blobs(clone, [blob[j["match_id"]] for j in jobs])
    for i, (job, body) in enumerate(zip(jobs, stream, strict=True), 1):
        rows.extend(parse_events(json.loads(body), job))
        if i % 100 == 0:
            log(f"  {i}/{len(jobs)} matches, {len(rows)} shots")
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, compression="gzip")
    log(f"wrote {len(df)} shots to {out}")
    return df


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]))
