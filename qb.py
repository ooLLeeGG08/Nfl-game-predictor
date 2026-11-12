"""Point-in-time QB rating for the v2 model (``qb_rating_diff``).

Offline only: used by train_v2.py, evaluate_v2.py and scripts/build_qb.py. The
web app never imports this module; it reads the precomputed data/qb/qb_games.parquet.

For every team-game we decide, using only information available before kickoff:

* lead_prev   QB with the most plays for the team in its previous game (a past game).
* expected    the expected starter:
                  depth-chart QB1 in week 1 (no previous game this season),
                  otherwise lead_prev;
                  if that QB is Out/Doubtful on the final pre-game injury report,
                  the next QB on the depth chart who is not Out/Doubtful
                  (fallback: the team's other QB with the most recent plays).
* rating(q)   EPA per play (pass attempts + sacks + carries) over q's plays in
              games that kicked off before this one, current + previous 2 seasons,
              shrunk toward the league mean: (sum_epa + K*m) / (n + K), K = 200.
              m = league QB EPA/play over all seasons before this one.

Point-in-time sources:
    player stats  only games with kickoff < this kickoff
    injuries      final report row for (season, week, team, player); 2009-2024 rows
                  carry date_modified and must be < kickoff (enforced); 2025+ rows
                  have no timestamp (official pre-game designation, not verifiable)
    depth charts  2009-2024: one weekly chart, no timestamp (shown to be stale, not
                  post-game); 2025+: latest snapshot with dt < kickoff
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
QB_RAW = ROOT / "data" / "qb_raw"          # QB-only nflverse subsets (scripts/fetch_qb.py)
QB_GAMES = ROOT / "data" / "qb" / "qb_games.parquet"   # built by scripts/build_qb.py, read by the app

# Frozen v2 settings (also written to models/v2/config.json). Chosen before any
# evaluation and not tuned.
K_SHRINK = 200
FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
OUT = {"Out", "Doubtful"}


def kickoff_utc(s: pd.DataFrame) -> pd.Series:
    t = s.gametime.fillna("13:00")
    return (pd.to_datetime(s.gameday.astype(str).str[:10] + " " + t)
            .dt.tz_localize("America/New_York").dt.tz_convert("UTC"))


def load_sources(D: Path, sched: pd.DataFrame):
    """(player stats, injuries, weekly depth charts, depth snapshots), QB rows only."""
    first, last = int(sched.season.min()), int(sched.season.max())
    ps = pd.concat([pd.read_parquet(D / f"pstats_{y}.parquet") for y in range(max(first, 2002), last + 1)
                    if (D / f"pstats_{y}.parquet").exists()])
    ps = ps[ps.position == "QB"].copy()
    ps["plays"] = ps.attempts.fillna(0) + ps.sacks_suffered.fillna(0) + ps.carries.fillna(0)
    ps["epa"] = ps.passing_epa.fillna(0) + ps.rushing_epa.fillna(0)
    ps = ps[ps.plays > 0][["game_id", "team", "player_id", "plays", "epa"]]

    inj = []
    for y in range(max(first, 2009), last + 1):
        f = D / f"inj_{y}.parquet"
        if f.exists():
            i = pd.read_parquet(f)
            i = i[i.position == "QB"].copy()
            if "date_modified" not in i:
                i["date_modified"] = pd.NaT
            inj.append(i[["season", "week", "team", "gsis_id", "report_status", "date_modified"]])
    inj = pd.concat(inj)
    inj["team"] = inj.team.replace(FIX)
    inj["season"], inj["week"] = inj.season.astype(int), inj.week.astype(int)

    dep_w, dep_s = [], []
    for y in range(max(first, 2009), last + 1):
        f = D / f"depth_{y}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f)
        if "club_code" in d:  # weekly chart
            d = d[(d.position == "QB") & (d.formation == "Offense")].copy()
            d["team"] = d.club_code.replace(FIX)
            d["rank"] = d.depth_team.astype(int)
            dep_w.append(d[["season", "week", "team", "gsis_id", "rank"]])
        else:  # timestamped snapshots
            d = d[d.pos_abb == "QB"].copy()
            d["dt"] = pd.to_datetime(d.dt, utc=True)
            d = d.groupby(["dt", "team", "gsis_id"], as_index=False).pos_rank.min().rename(columns={"pos_rank": "rank"})
            dep_s.append(d)
    dep_w = pd.concat(dep_w) if dep_w else pd.DataFrame(columns=["season", "week", "team", "gsis_id", "rank"])
    dep_s = pd.concat(dep_s) if dep_s else pd.DataFrame(columns=["dt", "team", "gsis_id", "rank"])
    return ps, inj, dep_w, dep_s


def team_games(sched: pd.DataFrame) -> pd.DataFrame:
    s = sched.copy()
    s["kick"] = kickoff_utc(s)
    long = pd.concat([
        s[["game_id", "season", "week", "kick", "home_team"]].rename(columns={"home_team": "team"}).assign(side="home"),
        s[["game_id", "season", "week", "kick", "away_team"]].rename(columns={"away_team": "team"}).assign(side="away"),
    ])
    return long.sort_values(["team", "kick", "game_id"]).reset_index(drop=True)


def build_qb_features(sched: pd.DataFrame, ps, inj, dep_w, dep_s, audit: list | None = None,
                      _planted_leak: bool = False) -> pd.DataFrame:
    """One row per (game_id, side): expected starter and his rating before kickoff.

    `audit`, if given, collects (kickoff, source, timestamp) for every dated input used,
    so a test can check that nothing used is from after kickoff. `_planted_leak` is
    for tests only: it uses each game's own lead QB, which the leakage test must catch.
    """
    tg = team_games(sched)
    kick_of = tg.drop_duplicates("game_id").set_index("game_id").kick
    ps = ps[ps.game_id.isin(kick_of.index)].copy()
    ps["kick"] = ps.game_id.map(kick_of)
    season_of = sched.set_index("game_id").season
    ps["season"] = ps.game_id.map(season_of)

    # league mean QB EPA/play by season (used only for later seasons)
    by_season = ps.groupby("season").agg(e=("epa", "sum"), n=("plays", "sum"))
    cum = by_season.cumsum().shift(1)  # seasons strictly before
    league_mean = (cum.e / cum.n).to_dict()
    overall_first = by_season.e.iloc[0] / by_season.n.iloc[0]

    # per-QB play log, sorted by time, for fast "before kickoff" sums
    ps = ps.sort_values("kick")
    qb_log = {q: g[["kick", "season", "plays", "epa"]].to_numpy() for q, g in ps.groupby("player_id")}

    def rating(q, kick, season):
        m = league_mean.get(season, overall_first)
        if not isinstance(m, float) or np.isnan(m):
            m = overall_first
        if q is None or q not in qb_log:
            return m, 0.0
        a = qb_log[q]
        sel = (a[:, 0] < kick) & (a[:, 1] >= season - 2)
        n, e = a[sel, 2].sum(), a[sel, 3].sum()
        if audit is not None and sel.any():
            audit.append((kick, "pstats", a[sel, 0].max()))
        return (e + K_SHRINK * m) / (n + K_SHRINK), float(n)

    # team-game QB usage
    use = ps.groupby(["game_id", "team", "player_id"], as_index=False)[["plays", "epa"]].sum()
    # ties on plays: higher EPA, then player id (deterministic, independent of row order)
    lead = (use.sort_values(["game_id", "team", "plays", "epa", "player_id"], ascending=[True, True, False, False, True], kind="mergesort")
            .drop_duplicates(["game_id", "team"]).set_index(["game_id", "team"]).player_id)

    inj_final = inj.set_index(["season", "week", "team", "gsis_id"])
    dep_w_idx = {k: g.sort_values("rank").gsis_id.tolist() for k, g in dep_w.groupby(["season", "week", "team"])}
    dep_s_by_team = {t: g.sort_values("dt") for t, g in dep_s.groupby("team")}

    def depth_order(season, week, team, kick):
        if (season, week, team) in dep_w_idx:
            return dep_w_idx[(season, week, team)], None
        g = dep_s_by_team.get(team)
        if g is None:
            return [], None
        before = g[g.dt < kick]
        if before.empty:
            return [], None
        dt = before.dt.iloc[-1]
        snap = before[before.dt == dt].sort_values("rank")
        return snap.gsis_id.tolist(), dt

    def status(season, week, team, q, kick):
        try:
            r = inj_final.loc[(season, week, team, q)]
        except KeyError:
            return None
        if isinstance(r, pd.DataFrame):
            r = r.sort_values("date_modified").iloc[-1]
        dm = r.date_modified
        if pd.notna(dm):
            if dm >= kick:  # not available before kickoff: ignore
                return None
            if audit is not None:
                audit.append((kick, "injury", dm))
        return r.report_status

    rows = []
    for team, g in tg.groupby("team", sort=False):
        hist: list[tuple[str, str | None]] = []  # (game_id, lead qb) of past games
        recent_qbs: list[str] = []
        prev_season = None
        for r in g.itertuples(index=False):
            lead_prev = hist[-1][1] if hist else None
            if _planted_leak:
                lead_prev = lead.get((r.game_id, team))
            if audit is not None and hist:
                for gid, _ in hist[-16:]:  # every past game whose lead QB feeds this row
                    audit.append((r.kick, "prev_game", kick_of[gid]))
            order, dt = depth_order(r.season, r.week, team, r.kick)
            if audit is not None and dt is not None:
                audit.append((r.kick, "depth_snapshot", dt))
            new_season = prev_season is not None and r.season != prev_season or prev_season is None
            expected = order[0] if (new_season and order) else lead_prev
            if expected is None and order:
                expected = order[0]
            st = status(r.season, r.week, team, expected, r.kick) if expected else None
            was_out = st in OUT
            if was_out:
                alt = [q for q in order if q != expected and status(r.season, r.week, team, q, r.kick) not in OUT]
                if not alt:
                    alt = [q for q in reversed(recent_qbs) if q != expected]
                expected = alt[0] if alt else None
            rt, n_exp = rating(expected, r.kick, r.season)
            rows.append({
                "game_id": r.game_id, "side": r.side, "team": team, "kick": r.kick,
                "lead_prev": lead_prev, "expected": expected, "starter_out": bool(was_out),
                "qb_rating": float(rt), "qb_plays_prior": n_exp,
            })
            # after the game: record who actually led (known only after kickoff, used for later games)
            q = lead.get((r.game_id, team))
            hist.append((r.game_id, q))
            if q:
                recent_qbs.append(q)
            prev_season = r.season
    return pd.DataFrame(rows)


def to_game_features(qb: pd.DataFrame) -> pd.DataFrame:
    h = qb[qb.side == "home"].set_index("game_id")
    a = qb[qb.side == "away"].set_index("game_id")
    out = pd.DataFrame(index=h.index)
    out["qb_rating_diff"] = h.qb_rating - a.qb_rating
    out["home_expected"], out["away_expected"] = h.expected, a.expected
    return out.reset_index()


def qb_names(D: Path = QB_RAW) -> dict[str, str]:
    """gsis_id -> display name, from the player-stat and depth-chart files."""
    names: dict[str, str] = {}
    for f in sorted(D.glob("depth_*.parquet")) + sorted(D.glob("pstats_*.parquet")):
        d = pd.read_parquet(f)
        col = next(c for c in ("player_display_name", "full_name", "player_name") if c in d)
        idc = "player_id" if "player_id" in d else "gsis_id"
        names.update(dict(zip(d[idc], d[col])))
    return {k: v for k, v in names.items() if isinstance(k, str) and isinstance(v, str)}
