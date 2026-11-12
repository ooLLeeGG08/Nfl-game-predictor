"""Precompute QB context for the current season's games (scheduled job; never run by the app).

    python scripts/fetch_qb.py && python scripts/build_qb.py

Reads data/qb_raw/ (QB stats, depth charts) and data/injury_snapshots/ (timestamped
injury reports we saved), and writes

    data/qb/upcoming.json  per game: expected starter per team, his injury status on the
                           latest report we saw before kickoff, his QB rating, the report
                           and depth-chart timestamps, and the v1 / v2 probabilities.
                           The app only reads this file.
    data/qb/v2_log.csv     append-only log of v1 and v2 before kickoff, for the
                           pre-registered end-of-season comparison (README).

Games that have kicked off are frozen: their last pre-kickoff entry is kept and
never rewritten, and nothing is logged for them any more.

For the current season the injury status comes ONLY from our timestamped snapshots,
never from nflverse's undated weekly file, so every status used was seen before kickoff.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import live  # noqa: E402
import qb  # noqa: E402
from data.fetch import CURRENT_SEASON, load_schedules  # noqa: E402

SNAP = ROOT / "data" / "injury_snapshots"
OUT_JSON = ROOT / "data" / "qb" / "upcoming.json"
LOG = ROOT / "data" / "qb" / "v2_log.csv"
LOG_COLS = ["logged_at", "game_id", "kickoff", "available", "report_as_of", "depth_as_of",
            "home_qb", "away_qb", "home_status", "away_status", "home_rating", "away_rating",
            "qb_rating_diff", "p_v1", "p_v2"]


def snapshot_injuries(season: int) -> pd.DataFrame:
    """QB rows of every saved snapshot, with date_modified = when we fetched it."""
    rows = []
    for f in sorted((SNAP / str(season)).glob("week*.parquet")):
        d = pd.read_parquet(f)
        d = d[d.position == "QB"]
        rows.append(pd.DataFrame({
            "season": d.season.astype(int), "week": d.week.astype(int), "team": d.team.replace(qb.FIX),
            "gsis_id": d.gsis_id, "report_status": d.report_status,
            "date_modified": pd.to_datetime(d.fetched_at, utc=True),
        }))
    cols = ["season", "week", "team", "gsis_id", "report_status", "date_modified"]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)


def report_times(season: int) -> dict[int, list[pd.Timestamp]]:
    """week -> fetch times of the saved snapshots of that week's report."""
    out: dict[int, list] = {}
    for f in sorted((SNAP / str(season)).glob("week*.parquet")):
        d = pd.read_parquet(f, columns=["week", "fetched_at"])
        if len(d):
            out.setdefault(int(d.week.iloc[0]), []).append(pd.Timestamp(d.fetched_at.iloc[0]).tz_convert("UTC"))
    return out


def latest_before(times, kick):
    before = [t for t in times if t < kick]
    return max(before) if before else None


def iso(t) -> str | None:
    return None if t is None or pd.isna(t) else pd.Timestamp(t).tz_convert("UTC").isoformat()


def build(now: datetime | None = None) -> dict:
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    sched = load_schedules(range(2002, CURRENT_SEASON + 1), allow_test=True)
    ps, inj, dep_w, dep_s = qb.load_sources(qb.QB_RAW, sched)
    # current season: injury status only from our timestamped snapshots
    inj = pd.concat([inj[inj.season < CURRENT_SEASON], snapshot_injuries(CURRENT_SEASON)], ignore_index=True)
    feats = qb.build_qb_features(sched, ps, inj, dep_w, dep_s)
    names = qb.qb_names()
    reports = report_times(CURRENT_SEASON)
    depth_times = {t: sorted(g.dt.unique()) for t, g in dep_s.groupby("team")}
    cur = inj[inj.season == CURRENT_SEASON].sort_values("date_modified")
    status_of: dict[tuple, list] = {}
    for r in cur.itertuples(index=False):
        status_of.setdefault((r.season, r.week, r.team, r.gsis_id), []).append((r.date_modified, r.report_status))

    # career starts before kickoff (actual starters of past games, 2002 on): known pre-game
    tg = qb.team_games(sched)
    actual = pd.concat([sched[["game_id", "home_qb_id"]].rename(columns={"home_qb_id": "qb"}),
                        sched[["game_id", "away_qb_id"]].rename(columns={"away_qb_id": "qb"})])
    actual = actual[actual.qb.notna()].merge(tg.drop_duplicates("game_id")[["game_id", "kick"]], on="game_id")
    starts_at = {q: pd.DatetimeIndex(sorted(g.kick)) for q, g in actual.groupby("qb")}

    # usual starter: who started most of the team's previous 4 games (ties: the more recent)
    team_starts = pd.concat([
        sched[["game_id", "home_team", "home_qb_id"]].set_axis(["game_id", "team", "qb"], axis=1),
        sched[["game_id", "away_team", "away_qb_id"]].set_axis(["game_id", "team", "qb"], axis=1),
    ])
    team_starts = team_starts[team_starts.qb.notna()].merge(tg.drop_duplicates("game_id")[["game_id", "kick"]], on="game_id")
    by_team = {t: g.sort_values("kick")[["kick", "qb"]].to_numpy() for t, g in team_starts.groupby("team")}

    def usual_starter(team, kick) -> str | None:
        a = by_team.get(team)
        if a is None:
            return None
        last4 = [q for k, q in a if k < kick][-4:]
        if not last4:
            return None
        return max(set(last4), key=lambda q: (last4.count(q), -last4[::-1].index(q)))

    def career_starts(q, kick) -> int | None:
        if not isinstance(q, str):
            return None
        t = starts_at.get(q)
        return 0 if t is None else int(t.searchsorted(kick, side="left"))  # starts before this kickoff

    snap = live.snapshot()
    games = snap.features[snap.features.season == CURRENT_SEASON]
    f = feats.set_index(["game_id", "side"])
    prev = json.loads(OUT_JSON.read_text())["games"] if OUT_JSON.exists() else {}

    out = {}
    for g in games.itertuples(index=False):
        kick = qb.kickoff_utc(pd.DataFrame([{"gameday": g.gameday, "gametime": g.gametime}])).iloc[0]
        if kick <= now:
            if g.game_id in prev:
                out[g.game_id] = {**prev[g.game_id], "frozen": True}
            continue
        rep_t = latest_before(reports.get(int(g.week), []), kick)
        rep_t = rep_t if rep_t is not None and rep_t <= now else None
        sides = {}
        for side, team in (("home", g.home_team), ("away", g.away_team)):
            r = f.loc[(g.game_id, side)]
            d_t = latest_before([pd.Timestamp(t) for t in depth_times.get(team, [])], kick)

            def status(q):
                """Status on the latest report we saw at or before rep_t; None = not listed."""
                if not isinstance(q, str) or rep_t is None:
                    return None
                seen = [st for t, st in status_of.get((int(g.season), int(g.week), team, q), []) if t <= rep_t]
                return seen[-1] if seen and isinstance(seen[-1], str) and seen[-1] else None

            replaces = None
            if r.starter_out and isinstance(r.lead_prev, str):
                replaces = {"qb_id": r.lead_prev, "name": names.get(r.lead_prev, r.lead_prev), "status": status(r.lead_prev)}
            sides[side] = {
                "team": team, "qb_id": r.expected, "name": names.get(r.expected, r.expected),
                "status": status(r.expected), "rating": round(float(r.qb_rating), 4),
                "plays_prior": int(r.qb_plays_prior), "career_starts": career_starts(r.expected, kick),
                "usual_starter": usual_starter(team, kick),
                "replaces": replaces, "depth_as_of": iso(d_t),
            }
        depth_ok = all(s["depth_as_of"] for s in sides.values())
        depth_as_of = min(s["depth_as_of"] for s in sides.values()) if depth_ok else None
        entry = {
            "week": int(g.week), "kickoff": iso(kick), "home": sides["home"], "away": sides["away"],
            "report_as_of": iso(rep_t), "depth_as_of": depth_as_of,
            "qb_rating_diff": round(sides["home"]["rating"] - sides["away"]["rating"], 6),
            "frozen": False,
        }
        entry["available"], entry["reason"] = live.qb_fresh(entry)
        row = pd.DataFrame([g._asdict()])
        entry["p_v1"] = round(float(live.home_win_prob(row)[0]), 6)
        row["qb_rating_diff"] = entry["qb_rating_diff"]
        entry["p_v2"] = round(float(live.home_win_prob_v2(row)[0]), 6)
        out[g.game_id] = entry
    return {"built_at": now.isoformat(), "season": CURRENT_SEASON, "stale_days": live.QB_STALE_DAYS,
            "model_v2": live.V2_VERSION, "games": out}


LOG_HORIZON_DAYS = 7  # log games kicking off within a week (the only ones with a current report)


def append_log(data: dict) -> int:
    """Append a row per game kicking off within LOG_HORIZON_DAYS whose QB inputs, statuses,
    injury report or probabilities changed since its last logged row. Never rewrites rows."""
    built = pd.Timestamp(data["built_at"])
    rows = []
    for gid, e in data["games"].items():
        if e["frozen"] or pd.Timestamp(e["kickoff"]) - built > pd.Timedelta(days=LOG_HORIZON_DAYS):
            continue
        rows.append({
            "logged_at": data["built_at"], "game_id": gid, "kickoff": e["kickoff"], "available": e["available"],
            "report_as_of": e["report_as_of"], "depth_as_of": e["depth_as_of"],
            "home_qb": e["home"]["qb_id"], "away_qb": e["away"]["qb_id"],
            "home_status": e["home"]["status"], "away_status": e["away"]["status"],
            "home_rating": e["home"]["rating"], "away_rating": e["away"]["rating"],
            "qb_rating_diff": e["qb_rating_diff"], "p_v1": e["p_v1"], "p_v2": e["p_v2"],
        })
    new = pd.DataFrame(rows, columns=LOG_COLS)
    if LOG.exists():
        old = pd.read_csv(LOG, dtype=str, keep_default_na=False)
        last = old.drop_duplicates("game_id", keep="last").set_index("game_id")
        key = [c for c in LOG_COLS if c not in ("logged_at", "depth_as_of")]
        fresh = new.astype(object).where(new.notna(), "").astype(str)  # same text form as the CSV
        changed = [i for i, r in fresh.iterrows()
                   if r.game_id not in last.index or any(r[c] != last.loc[r.game_id, c] for c in key if c != "game_id")]
        new = new.loc[changed]
    if len(new):
        new.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
    return len(new)


def main() -> None:
    data = build()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=1))
    n = append_log(data)
    live_games = [e for e in data["games"].values() if not e["frozen"]]
    print(f"wrote {OUT_JSON.relative_to(ROOT)}: {len(data['games'])} games "
          f"({sum(e['available'] for e in live_games)} upcoming with fresh QB data); logged {n} new rows")


if __name__ == "__main__":
    main()
